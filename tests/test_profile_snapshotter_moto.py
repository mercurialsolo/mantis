"""End-to-end snapshot pipeline tests against moto's in-memory S3.

The four real-B2 integration tests in
``test_profile_snapshotter_b2_live.py`` are skip-gated on
``MANTIS_PROFILE_SNAPSHOT_BUCKET`` because they hit real Backblaze
B2. That means CI runs zero integration tests for the snapshot
pipeline today — a regression in the writer/loader/lock interaction
ships green.

This file runs the SAME end-to-end flows against moto's in-memory S3
mock so CI always has a baseline. The tests are NOT a substitute for
the live B2 tests — those exercise the B2-specific
NotImplemented-on-conditional-PUT fallback path (see
``feedback_b2_no_conditional_put.md``). Moto implements conditional
PUT correctly, so it exercises the strict-CAS branch.

Run locally:

    pytest tests/test_profile_snapshotter_moto.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterator

import pytest


# Skip the whole module gracefully when moto isn't installed (dev
# extras might not include it yet on older checkouts).
moto = pytest.importorskip("moto")
boto3 = pytest.importorskip("boto3")
pytest.importorskip("zstandard")


@pytest.fixture
def s3_client() -> Iterator:
    """A moto-backed S3 client + a freshly-created test bucket."""
    from moto import mock_aws

    with mock_aws():
        client = boto3.client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
        )
        client.create_bucket(Bucket="moto-snapshots-test")
        yield client


def _make_profile(root: Path) -> Path:
    """Build a realistic Chrome profile dir with a Cookies SQLite db."""
    profile = root / "src-profile"
    (profile / "Default").mkdir(parents=True)
    (profile / "Default" / "Preferences").write_text(
        json.dumps({"profile": {"name": "moto-test"}}),
        encoding="utf-8",
    )
    with sqlite3.connect(profile / "Default" / "Cookies") as conn:
        conn.execute(
            "CREATE TABLE cookies (host TEXT, name TEXT, value TEXT)"
        )
        conn.executemany(
            "INSERT INTO cookies VALUES (?, ?, ?)",
            [
                ("example.com", "session", "abc"),
                ("example.com", "csrf", "deadbeef"),
            ],
        )
        conn.commit()
    return profile


# ── Writer + Loader round trip ────────────────────────────────────────


def test_capture_then_load_round_trips_cookies(s3_client, tmp_path: Path) -> None:
    """End-to-end happy path — capture → load → assert cookies intact."""
    from mantis_agent.observability.profile_snapshotter import (
        ProfileSnapshotter,
    )

    snap = ProfileSnapshotter(
        bucket="moto-snapshots-test",
        chrome_major=131,
        s3_client=s3_client,
    )
    src = _make_profile(tmp_path)

    cap = snap.capture(
        tenant_id="acme", profile_id="user-1",
        source_profile_dir=src,
    )
    assert cap.outcome == "captured", cap.reason
    assert cap.archive_sha256
    assert cap.archive_size_bytes > 0

    target = tmp_path / "loaded"
    result = snap.load(
        tenant_id="acme", profile_id="user-1",
        local_profile_dir=target,
    )
    assert result.outcome == "loaded", result.reason
    assert (target / "Default" / "Cookies").exists()

    with sqlite3.connect(target / "Default" / "Cookies") as conn:
        rows = conn.execute(
            "SELECT host, name, value FROM cookies ORDER BY host, name"
        ).fetchall()
    assert rows == [
        ("example.com", "csrf", "deadbeef"),
        ("example.com", "session", "abc"),
    ]


def test_recapture_dedups_identical_content(s3_client, tmp_path: Path) -> None:
    from mantis_agent.observability.profile_snapshotter import (
        ProfileSnapshotter,
    )

    snap = ProfileSnapshotter(
        bucket="moto-snapshots-test", chrome_major=131,
        s3_client=s3_client,
    )
    src = _make_profile(tmp_path)
    first = snap.capture(
        tenant_id="acme", profile_id="user-1",
        source_profile_dir=src,
    )
    second = snap.capture(
        tenant_id="acme", profile_id="user-1",
        source_profile_dir=src,
    )
    assert first.outcome == "captured"
    assert second.outcome == "deduplicated"
    assert second.archive_sha256 == first.archive_sha256


def test_load_no_snapshot_returns_no_snapshot(s3_client, tmp_path: Path) -> None:
    from mantis_agent.observability.profile_snapshotter import (
        ProfileSnapshotter,
    )

    snap = ProfileSnapshotter(
        bucket="moto-snapshots-test", chrome_major=131,
        s3_client=s3_client,
    )
    result = snap.load(
        tenant_id="acme", profile_id="never-snapshotted",
        local_profile_dir=tmp_path / "target",
    )
    assert result.outcome == "no_snapshot"
    assert result.reason == ""


def test_load_isolates_snapshots_by_chrome_major(
    s3_client, tmp_path: Path,
) -> None:
    """Snapshots for different Chrome major versions land under distinct
    prefixes — the reader for v130 sees no snapshot if only v131 was
    written."""
    from mantis_agent.observability.profile_snapshotter import (
        ProfileSnapshotter,
    )

    # Write with chrome_major=131.
    src = _make_profile(tmp_path)
    writer = ProfileSnapshotter(
        bucket="moto-snapshots-test", chrome_major=131,
        s3_client=s3_client,
    )
    writer.capture(
        tenant_id="acme", profile_id="user-1",
        source_profile_dir=src,
    )

    # Read with chrome_major=130 → no snapshot in the v130 prefix.
    reader = ProfileSnapshotter(
        bucket="moto-snapshots-test", chrome_major=130,
        s3_client=s3_client,
    )
    target = tmp_path / "target"
    result = reader.load(
        tenant_id="acme", profile_id="user-1",
        local_profile_dir=target,
    )
    assert result.outcome == "no_snapshot"


# ── Lock manager ──────────────────────────────────────────────────────


def test_lock_acquire_conflict_release_round_trip(
    s3_client, tmp_path: Path,
) -> None:
    """Round-trip the lock manager end-to-end."""
    from mantis_agent.observability.profile_lock import (
        LockConflictError, ProfileLockManager,
    )

    a = ProfileLockManager(
        bucket="moto-snapshots-test", chrome_major=131,
        s3_client=s3_client, ttl_seconds=300,
    )
    b = ProfileLockManager(
        bucket="moto-snapshots-test", chrome_major=131,
        s3_client=s3_client, ttl_seconds=300,
    )

    held = a.acquire(
        tenant_id="acme", profile_id="lock-test",
        holder_run_id="run-a",
    )
    assert held.blob.holder_run_id == "run-a"

    # Second acquire conflicts (lock still fresh).
    with pytest.raises(LockConflictError) as exc_info:
        b.acquire(
            tenant_id="acme", profile_id="lock-test",
            holder_run_id="run-b",
        )
    assert exc_info.value.blob.holder_run_id == "run-a"

    # Release frees the lock.
    a.release(held)
    other = b.acquire(
        tenant_id="acme", profile_id="lock-test",
        holder_run_id="run-b",
    )
    assert other.blob.holder_run_id == "run-b"
    b.release(other)


def test_lock_reaper_evicts_stale_locks(s3_client) -> None:
    """The reaper sweeps lock blobs past expires_at + grace."""
    from mantis_agent.observability.profile_lock import (
        ProfileLockManager, ProfileLockReaper,
    )

    # Manual clock so we don't have to wait 5 minutes for the lock
    # to expire naturally.
    now = [1_700_000_000.0]

    mgr = ProfileLockManager(
        bucket="moto-snapshots-test", chrome_major=131,
        s3_client=s3_client, ttl_seconds=60,
        clock=lambda: now[0],
    )
    mgr.acquire(
        tenant_id="acme", profile_id="lock-stale",
        holder_run_id="run-old",
    )

    # Jump past TTL + grace.
    now[0] += 60 + 90

    reaper = ProfileLockReaper(
        bucket="moto-snapshots-test", s3_client=s3_client,
        grace_seconds=60,
        clock=lambda: now[0],
    )
    summary = reaper.sweep()
    assert summary["evicted"] == 1


# ── M3 brain wire-up ──────────────────────────────────────────────────


def test_snapshot_lifecycle_capture_against_moto(
    s3_client, tmp_path: Path, monkeypatch,
) -> None:
    """The M3 ``maybe_capture_snapshot`` helper round-trips against
    a moto-backed snapshotter when the env block is set."""
    # M3 wire-up lives on the brain-wireup branch; skip cleanly when
    # this test runs on a branch that's older than M3.
    _lc = pytest.importorskip(
        "mantis_agent.observability.snapshot_lifecycle",
        reason="snapshot_lifecycle is from M3 — skip on M2-and-earlier branches",
    )

    monkeypatch.setenv("MANTIS_PROFILE_SNAPSHOT_BUCKET", "moto-snapshots-test")
    monkeypatch.setenv("MANTIS_PROFILE_SNAPSHOT_S3_REGION", "us-east-1")
    monkeypatch.setenv("MANTIS_CHROME_MAJOR_VERSION", "131")

    # Wire the moto S3 client into from_env.
    from mantis_agent.observability.profile_snapshotter import (
        ProfileSnapshotter,
    )

    def _fake_from_env() -> ProfileSnapshotter:
        return ProfileSnapshotter(
            bucket="moto-snapshots-test", chrome_major=131,
            s3_client=s3_client,
        )

    monkeypatch.setattr(
        ProfileSnapshotter, "from_env",
        classmethod(lambda cls: _fake_from_env()),
    )

    src = _make_profile(tmp_path)
    summary = _lc.maybe_capture_snapshot(
        tenant_id="acme", profile_id="user-1",
        source_profile_dir=src,
    )
    assert summary is not None
    assert summary["outcome"] == "captured"
    # And the bucket now has the archive + manifest + pointer.
    resp = s3_client.list_objects_v2(
        Bucket="moto-snapshots-test",
        Prefix="snapshots/acme/user-1/",
    )
    keys = sorted(o["Key"] for o in (resp.get("Contents") or []))
    assert any(k.endswith("latest.json") for k in keys)
    assert any(k.endswith(".tar.zst") for k in keys)
    assert any(k.endswith(".manifest.json") for k in keys)
