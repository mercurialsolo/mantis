"""Unit tests for the M3 brain wire-up helpers.

Drives every entry point under both env-gate states (off + on) so we
catch any path that forgets to short-circuit when
``MANTIS_PROFILE_SNAPSHOT_BUCKET`` isn't set. Uses the CAS-aware fake
S3 from the M2 writer tests for the env-on paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mantis_agent.observability.snapshot_lifecycle import (
    is_phase2_configured,
    maybe_acquire_snapshot_lock,
    maybe_capture_snapshot,
    maybe_force_release_lock_by_keys,
    maybe_release_snapshot_lock,
)

# Reuse the writer-test fixtures.
from tests.test_profile_snapshotter_writer import _CASFakeS3 as _BaseCAS
from tests.test_profile_snapshotter_loader import _make_profile_dir


class _CASFakeS3(_BaseCAS):
    """``_CASFakeS3`` + ``delete_object`` (release/force-release need it)."""

    def delete_object(self, *, Bucket: str, Key: str) -> dict:
        self._store.pop((Bucket, Key), None)
        return {}


# ── env-gate off (production default) ────────────────────────────────


@pytest.fixture
def env_off(monkeypatch):
    monkeypatch.delenv("MANTIS_PROFILE_SNAPSHOT_BUCKET", raising=False)
    return None


def test_is_phase2_configured_false_when_unset(env_off) -> None:
    assert is_phase2_configured() is False


def test_acquire_returns_none_when_env_off(env_off) -> None:
    assert maybe_acquire_snapshot_lock(
        tenant_id="acme", profile_id="user-1", run_id="r-1",
    ) is None


def test_release_is_noop_when_lock_is_none(env_off) -> None:
    # Must not raise.
    maybe_release_snapshot_lock(None)


def test_force_release_is_noop_when_env_off(env_off) -> None:
    maybe_force_release_lock_by_keys(tenant_id="acme", profile_id="user-1")


def test_capture_returns_none_when_env_off(env_off, tmp_path: Path) -> None:
    src = _make_profile_dir(tmp_path)
    assert maybe_capture_snapshot(
        tenant_id="acme", profile_id="user-1", source_profile_dir=src,
    ) is None


# ── env-gate on (Phase 2 deploy) ─────────────────────────────────────


@pytest.fixture
def env_on(monkeypatch, tmp_path: Path):
    """Configure the env block + inject a fake S3 into ProfileSnapshotter.

    ``ProfileSnapshotter.from_env()`` builds a real boto3 client, which
    we can't talk to in unit tests. Monkey-patch the factory to return
    a snapshotter wired to a ``_CASFakeS3`` instance instead.
    """
    monkeypatch.setenv("MANTIS_PROFILE_SNAPSHOT_BUCKET", "test-bucket")
    monkeypatch.setenv("MANTIS_PROFILE_SNAPSHOT_S3_ENDPOINT", "http://test")
    monkeypatch.setenv("MANTIS_PROFILE_SNAPSHOT_S3_REGION", "test")
    monkeypatch.setenv("MANTIS_CHROME_MAJOR_VERSION", "131")

    s3 = _CASFakeS3()
    from mantis_agent.observability.profile_snapshotter import (
        ProfileSnapshotter,
    )
    original = ProfileSnapshotter.from_env

    def _fake_from_env() -> ProfileSnapshotter:
        return ProfileSnapshotter(
            bucket="test-bucket", chrome_major=131, s3_client=s3,
        )

    monkeypatch.setattr(
        ProfileSnapshotter, "from_env",
        classmethod(lambda cls: _fake_from_env()),
    )
    yield s3
    # monkeypatch undoes the classmethod patch via cleanup; nothing to do.
    _ = original


def test_is_phase2_configured_true_when_set(env_on) -> None:
    assert is_phase2_configured() is True


def test_acquire_returns_handle_when_env_on(env_on) -> None:
    held = maybe_acquire_snapshot_lock(
        tenant_id="acme", profile_id="user-1", run_id="r-1",
    )
    assert held is not None
    assert held.blob.holder_run_id == "r-1"
    assert held.blob.holder_host == "modal"  # default fallback


def test_acquire_conflict_raises_for_caller_to_handle(env_on) -> None:
    """A second holder must see :class:`LockConflictError`."""
    from mantis_agent.observability.profile_lock import LockConflictError
    maybe_acquire_snapshot_lock(
        tenant_id="acme", profile_id="user-1", run_id="r-a",
    )
    with pytest.raises(LockConflictError) as exc_info:
        maybe_acquire_snapshot_lock(
            tenant_id="acme", profile_id="user-1", run_id="r-b",
        )
    assert exc_info.value.blob.holder_run_id == "r-a"


def test_release_round_trips_via_handle(env_on) -> None:
    held = maybe_acquire_snapshot_lock(
        tenant_id="acme", profile_id="user-1", run_id="r-1",
    )
    maybe_release_snapshot_lock(held)
    # After release another holder can acquire.
    other = maybe_acquire_snapshot_lock(
        tenant_id="acme", profile_id="user-1", run_id="r-2",
    )
    assert other is not None
    assert other.blob.holder_run_id == "r-2"


def test_force_release_clears_lock_blob_without_handle(env_on) -> None:
    """The API-side terminal path can release without carrying the
    handle across the API/executor boundary."""
    s3 = env_on
    held = maybe_acquire_snapshot_lock(
        tenant_id="acme", profile_id="user-x", run_id="r-1",
    )
    assert held is not None
    # Force-release as if from a different process.
    maybe_force_release_lock_by_keys(tenant_id="acme", profile_id="user-x")
    # Lock blob is gone.
    keys = [k for (_, k) in s3._store if k.endswith("lock.json")]
    assert keys == []


def test_capture_captured_on_happy_path(env_on, tmp_path: Path) -> None:
    src = _make_profile_dir(tmp_path)
    summary = maybe_capture_snapshot(
        tenant_id="acme", profile_id="user-1", source_profile_dir=src,
    )
    assert summary is not None
    assert summary["outcome"] == "captured"
    assert summary["archive_sha256"]
    assert summary["archive_size_bytes"] > 0


def test_capture_skipped_when_source_missing(env_on, tmp_path: Path) -> None:
    summary = maybe_capture_snapshot(
        tenant_id="acme", profile_id="user-1",
        source_profile_dir=tmp_path / "does-not-exist",
    )
    assert summary == {"outcome": "skipped", "reason": "no_source_dir"}


def test_capture_skipped_when_source_is_none(env_on) -> None:
    summary = maybe_capture_snapshot(
        tenant_id="acme", profile_id="user-1", source_profile_dir=None,
    )
    assert summary == {"outcome": "skipped", "reason": "no_source_dir"}


def test_capture_deduplicates_identical_content(
    env_on, tmp_path: Path,
) -> None:
    src = _make_profile_dir(tmp_path)
    first = maybe_capture_snapshot(
        tenant_id="acme", profile_id="user-1", source_profile_dir=src,
    )
    second = maybe_capture_snapshot(
        tenant_id="acme", profile_id="user-1", source_profile_dir=src,
    )
    assert first is not None and second is not None
    assert first["outcome"] == "captured"
    assert second["outcome"] == "deduplicated"


def test_capture_never_raises_on_writer_exception(env_on, monkeypatch, tmp_path: Path) -> None:
    """Even if the writer raises an unexpected error, capture returns
    a structured ``skipped`` rather than bubbling to the caller."""
    src = _make_profile_dir(tmp_path)
    from mantis_agent.observability.profile_snapshotter import (
        ProfileSnapshotter,
    )

    def _boom(self: Any, **kw: Any) -> None:
        raise RuntimeError("simulated writer failure")

    monkeypatch.setattr(ProfileSnapshotter, "capture", _boom)
    summary = maybe_capture_snapshot(
        tenant_id="acme", profile_id="user-1", source_profile_dir=src,
    )
    assert summary is not None
    assert summary["outcome"] == "skipped"
    assert "raised:" in summary["reason"]


def test_acquire_holder_host_uses_mantis_host_env(env_on, monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_HOST", "e2b-east")
    held = maybe_acquire_snapshot_lock(
        tenant_id="acme", profile_id="user-host", run_id="r-1",
    )
    assert held is not None
    assert held.blob.holder_host == "e2b-east"
