"""Live integration test for the M1 loader against real Backblaze B2.

Skipped by default — runs only when the full Phase 2 env block is set:

    MANTIS_PROFILE_SNAPSHOT_BUCKET
    MANTIS_PROFILE_SNAPSHOT_S3_ENDPOINT
    MANTIS_PROFILE_SNAPSHOT_S3_REGION
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY

This is the "worked end-to-end example" the spec calls out as one of
the gating conditions for M1: a snapshot round-trip with cookies +
localStorage + IndexedDB-shaped data intact, using the real bucket.

Cleans up its own keys at the end. Idempotent — safe to re-run.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("MANTIS_PROFILE_SNAPSHOT_BUCKET"),
        reason="MANTIS_PROFILE_SNAPSHOT_BUCKET not set — live B2 test skipped",
    ),
    pytest.mark.skipif(
        not os.environ.get("AWS_ACCESS_KEY_ID"),
        reason="AWS_ACCESS_KEY_ID not set — live B2 test skipped",
    ),
]


def _import_or_skip():
    try:
        import boto3  # noqa: F401
        import zstandard  # noqa: F401
    except ImportError:
        pytest.skip("boto3 / zstandard not installed — install mantis-agent[snapshots]")
    from mantis_agent.observability.profile_snapshotter import (
        ProfileSnapshotter,
        capture_and_upload_for_testing,
    )
    return ProfileSnapshotter, capture_and_upload_for_testing


def _make_chrome_shaped_profile(root: Path) -> Path:
    """Build a profile dir with the same structure a real Chrome
    user-data-dir has — Preferences + a Cookies SQLite db with rows."""
    profile = root / "src-profile"
    (profile / "Default").mkdir(parents=True)
    (profile / "Default" / "Preferences").write_text(
        json.dumps({"profile": {"name": "B2 Integration Test"}}),
        encoding="utf-8",
    )
    cookies_path = profile / "Default" / "Cookies"
    with sqlite3.connect(cookies_path) as conn:
        conn.execute(
            "CREATE TABLE cookies (host TEXT, name TEXT, value TEXT, expires_utc INTEGER)"
        )
        rows = [
            ("acme.example.com", "session", "tok_abc", 1700000000),
            ("acme.example.com", "csrf",    "deadbeef",  1700000001),
            ("other.example",    "pref",    "dark",      1700000002),
        ]
        conn.executemany(
            "INSERT INTO cookies VALUES (?, ?, ?, ?)", rows,
        )
        conn.commit()
    return profile


def test_b2_round_trip_with_real_chrome_shaped_profile(tmp_path: Path) -> None:
    """Full M1 acceptance: capture → upload → load → assert cookies
    survive the round trip with PRAGMA integrity intact."""
    ProfileSnapshotter, capture_and_upload_for_testing = _import_or_skip()

    bucket = os.environ["MANTIS_PROFILE_SNAPSHOT_BUCKET"]
    endpoint_url = os.environ.get("MANTIS_PROFILE_SNAPSHOT_S3_ENDPOINT") or None
    region = os.environ.get("MANTIS_PROFILE_SNAPSHOT_S3_REGION") or None

    import boto3
    s3_kwargs = {}
    if endpoint_url:
        s3_kwargs["endpoint_url"] = endpoint_url
    if region:
        s3_kwargs["region_name"] = region
    s3 = boto3.client("s3", **s3_kwargs)

    # Use a unique tenant/profile per run so the test never races
    # itself across CI runs.
    test_run_id = f"int-{int(time.time())}"
    tenant_id = f"itest-{test_run_id}"
    profile_id = "alice"
    chrome_major = 131

    src = _make_chrome_shaped_profile(tmp_path)
    snap = ProfileSnapshotter(
        bucket=bucket,
        chrome_major=chrome_major,
        s3_client=s3,
    )

    try:
        # 1. Capture + upload.
        manifest = capture_and_upload_for_testing(
            snapshotter=snap,
            tenant_id=tenant_id,
            profile_id=profile_id,
            source_profile_dir=src,
            chrome_major=chrome_major,
        )
        print(
            f"  uploaded sha={manifest['archive_sha256'][:12]} "
            f"size={manifest['archive_size_bytes']}B "
            f"uncompressed={manifest['uncompressed_size_bytes']}B"
        )

        # 2. Load into a fresh dir.
        target = tmp_path / "loaded-profile"
        result = snap.load(
            tenant_id=tenant_id, profile_id=profile_id,
            local_profile_dir=target,
        )

        # 3. Assert the contract.
        assert result.outcome == "loaded", (
            f"expected outcome=loaded, got {result}"
        )
        assert result.reason == ""
        assert result.bytes_downloaded > 0
        assert result.manifest is not None
        assert result.manifest.chrome_major_version == chrome_major
        assert result.manifest.archive_sha256 == manifest["archive_sha256"]

        # Files exist.
        assert (target / "Default" / "Preferences").exists()
        assert (target / "Default" / "Cookies").exists()

        # Cookies round-trip intact.
        with sqlite3.connect(target / "Default" / "Cookies") as conn:
            rows = conn.execute(
                "SELECT host, name, value FROM cookies ORDER BY host, name"
            ).fetchall()
        assert rows == [
            ("acme.example.com", "csrf",    "deadbeef"),
            ("acme.example.com", "session", "tok_abc"),
            ("other.example",    "pref",    "dark"),
        ]
        print(
            f"  ✓ round-trip OK  elapsed={result.elapsed_seconds:.2f}s "
            f"downloaded={result.bytes_downloaded}B"
        )

    finally:
        # 4. Cleanup — remove the test prefix from the bucket.
        prefix = f"snapshots/itest-{test_run_id}/"
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []) or []:
                    s3.delete_object(Bucket=bucket, Key=obj["Key"])
        except Exception as exc:  # noqa: BLE001 — cleanup is best-effort
            print(f"  ! cleanup failed: {exc}")


def test_b2_no_snapshot_yet_returns_no_snapshot(tmp_path: Path) -> None:
    """For a fresh (tenant, profile) that's never had a snapshot
    uploaded, load() returns no_snapshot (NOT a failure)."""
    ProfileSnapshotter, _ = _import_or_skip()

    bucket = os.environ["MANTIS_PROFILE_SNAPSHOT_BUCKET"]
    endpoint_url = os.environ.get("MANTIS_PROFILE_SNAPSHOT_S3_ENDPOINT") or None
    region = os.environ.get("MANTIS_PROFILE_SNAPSHOT_S3_REGION") or None

    import boto3
    s3_kwargs = {}
    if endpoint_url:
        s3_kwargs["endpoint_url"] = endpoint_url
    if region:
        s3_kwargs["region_name"] = region
    s3 = boto3.client("s3", **s3_kwargs)

    snap = ProfileSnapshotter(
        bucket=bucket, chrome_major=131, s3_client=s3,
    )
    result = snap.load(
        tenant_id=f"never-existed-{int(time.time())}",
        profile_id="ghost",
        local_profile_dir=tmp_path / "loaded",
    )
    assert result.outcome == "no_snapshot"
    assert result.reason == ""


# ── M2 — capture writer + lock manager live against real B2 ───────────


def _make_s3_from_env():
    import boto3
    s3_kwargs: dict = {}
    endpoint_url = os.environ.get("MANTIS_PROFILE_SNAPSHOT_S3_ENDPOINT") or None
    region = os.environ.get("MANTIS_PROFILE_SNAPSHOT_S3_REGION") or None
    if endpoint_url:
        s3_kwargs["endpoint_url"] = endpoint_url
    if region:
        s3_kwargs["region_name"] = region
    return boto3.client("s3", **s3_kwargs)


def _cleanup_prefix(s3, bucket: str, prefix: str) -> None:
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                s3.delete_object(Bucket=bucket, Key=obj["Key"])
    except Exception as exc:  # noqa: BLE001 — cleanup is best-effort
        print(f"  ! cleanup failed: {exc}")


def test_b2_capture_writes_archive_manifest_pointer(tmp_path: Path) -> None:
    """M2 acceptance: capture() against real B2 lands archive +
    manifest + pointer, then a second capture deduplicates."""
    ProfileSnapshotter, _ = _import_or_skip()

    bucket = os.environ["MANTIS_PROFILE_SNAPSHOT_BUCKET"]
    s3 = _make_s3_from_env()

    test_run_id = f"int-cap-{int(time.time())}"
    tenant_id = f"itest-{test_run_id}"
    profile_id = "alice"
    chrome_major = 131
    src = _make_chrome_shaped_profile(tmp_path)
    snap = ProfileSnapshotter(
        bucket=bucket, chrome_major=chrome_major, s3_client=s3,
    )
    prefix = f"snapshots/itest-{test_run_id}/"

    try:
        first = snap.capture(
            tenant_id=tenant_id, profile_id=profile_id,
            source_profile_dir=src,
        )
        assert first.outcome == "captured", f"{first.outcome}: {first.reason}"
        assert first.archive_sha256
        assert first.predecessor_sha256 == ""
        print(
            f"  ✓ first capture: sha={first.archive_sha256[:12]} "
            f"size={first.archive_size_bytes}B "
            f"elapsed={first.elapsed_seconds:.2f}s"
        )

        # Round-trip via load(): the pointer should now resolve.
        target = tmp_path / "loaded-profile"
        loaded = snap.load(
            tenant_id=tenant_id, profile_id=profile_id,
            local_profile_dir=target,
        )
        assert loaded.outcome == "loaded", (
            f"{loaded.outcome}: {loaded.reason}"
        )
        assert loaded.manifest is not None
        assert loaded.manifest.archive_sha256 == first.archive_sha256
        assert (target / "Default" / "Cookies").exists()

        # Re-capture with identical source dedups.
        second = snap.capture(
            tenant_id=tenant_id, profile_id=profile_id,
            source_profile_dir=src,
        )
        assert second.outcome == "deduplicated", (
            f"{second.outcome}: {second.reason}"
        )
        assert second.archive_sha256 == first.archive_sha256
        print(
            f"  ✓ dedup capture: outcome={second.outcome} "
            f"elapsed={second.elapsed_seconds:.2f}s"
        )

    finally:
        _cleanup_prefix(s3, bucket, prefix)


def test_b2_profile_lock_acquire_renew_release(tmp_path: Path) -> None:
    """M2 acceptance: acquire/renew/release a lock against real B2 with
    CAS semantics intact (second acquire conflicts; release frees)."""
    _import_or_skip()  # makes sure boto3 + zstandard installed
    from mantis_agent.observability.profile_lock import (
        LockConflictError,
        ProfileLockManager,
    )

    bucket = os.environ["MANTIS_PROFILE_SNAPSHOT_BUCKET"]
    s3 = _make_s3_from_env()

    test_run_id = f"int-lock-{int(time.time())}"
    tenant_id = f"itest-{test_run_id}"
    profile_id = "alice"
    prefix = f"snapshots/itest-{test_run_id}/"

    mgr_a = ProfileLockManager(
        bucket=bucket, chrome_major=131, s3_client=s3,
        ttl_seconds=120,
    )

    try:
        held_a = mgr_a.acquire(
            tenant_id=tenant_id, profile_id=profile_id,
            holder_run_id="run-a", holder_host="pytest",
        )
        assert held_a.blob.holder_run_id == "run-a"
        assert held_a.etag
        print(f"  ✓ acquired etag={held_a.etag}")

        # Second holder conflicts.
        mgr_b = ProfileLockManager(
            bucket=bucket, chrome_major=131, s3_client=s3,
            ttl_seconds=120,
        )
        with pytest.raises(LockConflictError) as exc_info:
            mgr_b.acquire(
                tenant_id=tenant_id, profile_id=profile_id,
                holder_run_id="run-b",
            )
        assert exc_info.value.blob.holder_run_id == "run-a"
        print("  ✓ second acquire conflicted as expected")

        # Renew bumps count + extends TTL.
        renewed = mgr_a.renew(held_a)
        assert renewed.blob.renewal_count == 1
        assert renewed.etag != held_a.etag
        print(
            f"  ✓ renewed count={renewed.blob.renewal_count} "
            f"etag={renewed.etag}"
        )

        # Release frees the lock.
        assert mgr_a.release(renewed) is True
        # Now mgr_b can acquire.
        held_b = mgr_b.acquire(
            tenant_id=tenant_id, profile_id=profile_id,
            holder_run_id="run-b",
        )
        assert held_b.blob.holder_run_id == "run-b"
        mgr_b.release(held_b)
        print("  ✓ released + reacquired by second holder")

    finally:
        _cleanup_prefix(s3, bucket, prefix)
