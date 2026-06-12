"""Unit tests for the M2 per-profile lock + reaper (#699 Phase 2 § 5).

Drives :class:`ProfileLockManager` and :class:`ProfileLockReaper`
against the same CAS-aware fake S3 used by the writer tests. No
network, no real boto3.

Coverage map:

* Acquire creates the lock blob with TTL.
* Acquire while held by a non-expired holder → ``LockConflictError``
  carrying the holder identity.
* Acquire when the existing blob has expired succeeds (CAS replaces
  the stale blob).
* Renew extends ``expires_at_ms`` and bumps ``renewal_count``.
* Renew when someone stole the lock → ``LockLostError``.
* Release deletes the lock object.
* Release on a missing object is best-effort (no raise) under
  ``quiet=True``.
* Reaper finds locks past ``expires_at + grace_seconds``.
* Reaper leaves still-live locks alone.
* Reaper.apply() deletes selected keys + returns counts.
* ``sweep()`` end-to-end: stale locks evicted, fresh ones untouched.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from mantis_agent.observability.profile_lock import (
    LockConflictError,
    LockLostError,
    ProfileLockBlob,
    ProfileLockManager,
    ProfileLockReaper,
)
from tests.test_profile_snapshotter_writer import _CASFakeS3


# ── Helpers ───────────────────────────────────────────────────────────


class _FakeClock:
    """Manual clock — tests advance time explicitly."""

    def __init__(self, t0: float = 1_000_000.0) -> None:
        self.now = t0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _PaginatedListS3(_CASFakeS3):
    """Adds ``list_objects_v2`` paginator support for the reaper."""

    def get_paginator(self, name: str) -> Any:
        assert name == "list_objects_v2"
        store = self._store

        class _Paginator:
            def paginate(self, *, Bucket: str, Prefix: str = ""):
                contents = [
                    {"Key": k}
                    for (b, k) in store
                    if b == Bucket and k.startswith(Prefix)
                ]
                yield {"Contents": contents}

        return _Paginator()

    def delete_object(self, *, Bucket: str, Key: str) -> dict:
        self._store.pop((Bucket, Key), None)
        return {}


def _mgr(s3: _CASFakeS3, *, clock: _FakeClock,
         ttl: int = 300, renewal: int = 60) -> ProfileLockManager:
    return ProfileLockManager(
        bucket="test-bucket",
        chrome_major=131,
        s3_client=s3,
        ttl_seconds=ttl,
        renewal_interval_seconds=renewal,
        clock=clock,
    )


def _reaper(s3: _PaginatedListS3, *, clock: _FakeClock,
            grace: int = 60) -> ProfileLockReaper:
    return ProfileLockReaper(
        bucket="test-bucket", s3_client=s3,
        grace_seconds=grace, clock=clock,
    )


# ── acquire ───────────────────────────────────────────────────────────


def test_acquire_creates_lock_blob_with_ttl() -> None:
    s3 = _CASFakeS3()
    clock = _FakeClock(1_700_000_000.0)
    mgr = _mgr(s3, clock=clock)

    held = mgr.acquire(
        tenant_id="acme", profile_id="user-1",
        holder_run_id="run-abc",
    )
    assert held.blob.holder_run_id == "run-abc"
    expected_now_ms = int(1_700_000_000.0 * 1000)
    assert held.blob.acquired_at_ms == expected_now_ms
    assert held.blob.expires_at_ms == expected_now_ms + 300 * 1000
    assert held.etag


def test_acquire_conflicts_with_live_holder() -> None:
    s3 = _CASFakeS3()
    clock = _FakeClock(1_700_000_000.0)
    mgr_a = _mgr(s3, clock=clock)
    mgr_a.acquire(
        tenant_id="acme", profile_id="user-1",
        holder_run_id="run-a",
    )

    # Second acquire within the TTL window should conflict.
    clock.advance(60)  # well inside the 300s TTL
    mgr_b = _mgr(s3, clock=clock)
    with pytest.raises(LockConflictError) as exc_info:
        mgr_b.acquire(
            tenant_id="acme", profile_id="user-1",
            holder_run_id="run-b",
        )
    assert exc_info.value.blob.holder_run_id == "run-a"


def test_acquire_replaces_expired_lock() -> None:
    s3 = _CASFakeS3()
    clock = _FakeClock(1_700_000_000.0)
    mgr_a = _mgr(s3, clock=clock)
    held_a = mgr_a.acquire(
        tenant_id="acme", profile_id="user-1",
        holder_run_id="run-a",
    )

    # Jump past TTL — the old holder went silent.
    clock.advance(500)
    mgr_b = _mgr(s3, clock=clock)
    held_b = mgr_b.acquire(
        tenant_id="acme", profile_id="user-1",
        holder_run_id="run-b",
    )
    assert held_b.blob.holder_run_id == "run-b"
    assert held_b.etag != held_a.etag


def test_acquire_uses_if_none_match_when_no_prior_lock() -> None:
    s3 = _CASFakeS3()
    clock = _FakeClock()
    mgr = _mgr(s3, clock=clock)

    mgr.acquire(
        tenant_id="acme", profile_id="user-1",
        holder_run_id="run-a",
    )
    lock_put = next(
        c for c in s3.put_calls if c["Key"].endswith("lock.json")
    )
    assert lock_put.get("IfNoneMatch") == "*"


# ── renew / release ───────────────────────────────────────────────────


def test_renew_extends_ttl_and_bumps_count() -> None:
    s3 = _CASFakeS3()
    clock = _FakeClock(1_700_000_000.0)
    mgr = _mgr(s3, clock=clock)

    held = mgr.acquire(
        tenant_id="acme", profile_id="user-1",
        holder_run_id="run-a",
    )
    original_expiry = held.blob.expires_at_ms

    clock.advance(60)
    renewed = mgr.renew(held)
    assert renewed.blob.renewal_count == held.blob.renewal_count + 1
    assert renewed.blob.expires_at_ms > original_expiry
    assert renewed.etag != held.etag


def test_renew_after_steal_raises_lock_lost() -> None:
    s3 = _CASFakeS3()
    clock = _FakeClock(1_700_000_000.0)
    mgr_a = _mgr(s3, clock=clock)
    held_a = mgr_a.acquire(
        tenant_id="acme", profile_id="user-1",
        holder_run_id="run-a",
    )

    # The reaper / another writer steals the lock — wipe + write new.
    key = next(k for (_, k) in s3._store if k.endswith("lock.json"))
    s3._store.pop(("test-bucket", key), None)
    clock.advance(60)
    mgr_b = _mgr(s3, clock=clock)
    mgr_b.acquire(
        tenant_id="acme", profile_id="user-1",
        holder_run_id="run-b",
    )

    # Original holder's renew should now fail with LockLostError.
    with pytest.raises(LockLostError):
        mgr_a.renew(held_a)


def test_release_deletes_lock_blob() -> None:
    s3 = _CASFakeS3()
    # delete_object is needed; reuse the paginated subclass for its impl.
    s3 = _PaginatedListS3()
    clock = _FakeClock()
    mgr = _mgr(s3, clock=clock)

    held = mgr.acquire(
        tenant_id="acme", profile_id="user-1",
        holder_run_id="run-a",
    )
    assert mgr.release(held) is True
    assert not any(k.endswith("lock.json") for (_, k) in s3._store)


def test_release_swallows_errors_under_quiet() -> None:
    class _DeleteFails(_PaginatedListS3):
        def delete_object(self, **kw):  # type: ignore[override]
            raise RuntimeError("simulated delete failure")

    s3 = _DeleteFails()
    clock = _FakeClock()
    mgr = _mgr(s3, clock=clock)
    held = mgr.acquire(
        tenant_id="acme", profile_id="user-1",
        holder_run_id="run-a",
    )
    # Default quiet=True → returns False, doesn't raise.
    assert mgr.release(held) is False
    # Explicit quiet=False propagates.
    with pytest.raises(RuntimeError):
        mgr.release(held, quiet=False)


# ── reaper ────────────────────────────────────────────────────────────


def test_reaper_finds_locks_past_grace() -> None:
    s3 = _PaginatedListS3()
    clock = _FakeClock(1_700_000_000.0)
    mgr = _mgr(s3, clock=clock, ttl=300)

    mgr.acquire(
        tenant_id="acme", profile_id="user-old",
        holder_run_id="run-old",
    )
    # Advance past TTL + grace.
    clock.advance(300 + 90)

    reaper = _reaper(s3, clock=clock, grace=60)
    decisions = reaper.find()
    assert len(decisions) == 1
    assert decisions[0].blob.holder_run_id == "run-old"
    assert decisions[0].reason == "grace_expired"


def test_reaper_leaves_live_locks_alone() -> None:
    s3 = _PaginatedListS3()
    clock = _FakeClock(1_700_000_000.0)
    mgr = _mgr(s3, clock=clock, ttl=300)

    mgr.acquire(
        tenant_id="acme", profile_id="user-live",
        holder_run_id="run-live",
    )
    # Still inside the TTL.
    clock.advance(60)

    reaper = _reaper(s3, clock=clock, grace=60)
    decisions = reaper.find()
    assert decisions == []


def test_reaper_sweep_deletes_stale_only() -> None:
    s3 = _PaginatedListS3()
    clock = _FakeClock(1_700_000_000.0)

    # One stale, one live.
    mgr_stale = _mgr(s3, clock=clock, ttl=300)
    mgr_stale.acquire(
        tenant_id="acme", profile_id="user-stale",
        holder_run_id="run-stale",
    )
    clock.advance(500)  # stale expires here
    mgr_live = _mgr(s3, clock=clock, ttl=300)
    mgr_live.acquire(
        tenant_id="acme", profile_id="user-live",
        holder_run_id="run-live",
    )

    reaper = _reaper(s3, clock=clock, grace=60)
    summary = reaper.sweep()
    assert summary["evicted"] == 1
    assert summary["skipped"] == 0

    # Only the live lock remains.
    remaining_keys = [k for (_, k) in s3._store if k.endswith("lock.json")]
    assert len(remaining_keys) == 1
    assert "user-live" in remaining_keys[0]


def test_reaper_ignores_corrupt_lock_blobs() -> None:
    s3 = _PaginatedListS3()
    clock = _FakeClock(1_700_000_000.0)

    # Inject a corrupt lock.json directly.
    s3.seed(
        Bucket="test-bucket",
        Key="snapshots/acme/user-x/131/lock.json",
        Body=b"not-valid-json",
    )
    reaper = _reaper(s3, clock=clock, grace=60)
    decisions = reaper.find()
    assert decisions == []
    summary = reaper.sweep()
    assert summary["evicted"] == 0


# ── blob schema ───────────────────────────────────────────────────────


def test_lock_blob_round_trips() -> None:
    raw = json.dumps({
        "version": 1,
        "holder_run_id": "run-1",
        "holder_host": "modal",
        "holder_host_run_id": "fc-aabb",
        "acquired_at_ms": 1_700_000_000_000,
        "renewed_at_ms": 1_700_000_060_000,
        "expires_at_ms": 1_700_000_300_000,
        "renewal_count": 2,
        "extra_field_for_forward_compat": "ignored",
    }).encode("utf-8")
    blob = ProfileLockBlob.model_validate_json(raw)
    assert blob.renewal_count == 2
    # Round-trip preserves extras under extra=allow.
    assert "extra_field_for_forward_compat" in blob.model_dump()
