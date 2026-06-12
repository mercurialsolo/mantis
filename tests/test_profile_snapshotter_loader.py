"""Unit tests for the M1 read-path loader (#699 Phase 2).

Drives ``ProfileSnapshotter.load()`` against a fake S3 client. No
network, no real boto3 — every fixture builds its own in-memory
key/value store quack-compatible with boto3's get_object surface.

Each test pins a single loader contract:

* The happy path round-trips bytes.
* Each strict-refusal path falls back to fresh Chrome + wipes the
  target dir + carries the right ``reason`` on the result.
* Network / S3 errors degrade silently (no exception bubble; the
  brain just gets a fresh fallback).

The integration test against real B2 lives in
``tests/integration/test_profile_snapshotter_b2_live.py`` and is
gated behind ``MANTIS_PROFILE_SNAPSHOT_BUCKET``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mantis_agent.observability.profile_snapshotter import (
    ProfileSnapshotter,
    capture_and_upload_for_testing,
)


# ── Fake S3 client ────────────────────────────────────────────────────


class _FakeS3:
    """Minimal in-memory S3 shim.

    Only implements ``get_object`` + ``put_object`` — that's all the
    loader and the test helper need. The error surface is shaped like
    botocore's: missing key → ``ClientError``-like exception with
    ``response['Error']['Code'] == 'NoSuchKey'``.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], bytes] = {}
        self.get_calls: list[tuple[str, str]] = []
        self.put_calls: list[tuple[str, str]] = []
        self.raise_on_get: Exception | None = None

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_kw) -> dict:
        self.put_calls.append((Bucket, Key))
        if isinstance(Body, str):
            Body = Body.encode("utf-8")
        self._store[(Bucket, Key)] = bytes(Body)
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        self.get_calls.append((Bucket, Key))
        if self.raise_on_get is not None:
            raise self.raise_on_get
        body = self._store.get((Bucket, Key))
        if body is None:
            err = _FakeClientError(
                "NoSuchKey",
                f"The specified key does not exist: {Key}",
            )
            raise err
        return {"Body": _FakeBody(body)}


class _FakeBody:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def read(self) -> bytes:
        return self._raw


class _FakeClientError(Exception):
    """boto3-shape ClientError. Carries ``.response['Error']['Code']``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.response = {"Error": {"Code": code, "Message": message}}


# ── Profile-dir helpers ───────────────────────────────────────────────


def _make_profile_dir(root: Path, *, with_cookies: bool = True) -> Path:
    """Build a small Chrome-shaped profile dir for tests."""
    profile = root / "src-profile"
    (profile / "Default").mkdir(parents=True)
    (profile / "Default" / "Preferences").write_text(
        json.dumps({"profile": {"name": "Test"}}),
        encoding="utf-8",
    )
    if with_cookies:
        cookies_path = profile / "Default" / "Cookies"
        with sqlite3.connect(cookies_path) as conn:
            conn.execute("CREATE TABLE cookies (host TEXT, name TEXT, value TEXT)")
            conn.execute(
                "INSERT INTO cookies VALUES (?, ?, ?)",
                ("example.com", "session", "abc123"),
            )
            conn.commit()
    return profile


def _snapshotter(s3: _FakeS3, *, chrome_major: int = 131,
                 allow_hot: bool = False) -> ProfileSnapshotter:
    return ProfileSnapshotter(
        bucket="test-bucket",
        chrome_major=chrome_major,
        s3_client=s3,
        allow_hot_mode=allow_hot,
    )


# ── Happy path ────────────────────────────────────────────────────────


def test_load_round_trips_bytes(tmp_path: Path) -> None:
    """Capture → upload → load → assert the profile contents match."""
    s3 = _FakeS3()
    snap = _snapshotter(s3, chrome_major=131)
    src = _make_profile_dir(tmp_path)

    capture_and_upload_for_testing(
        snapshotter=snap,
        tenant_id="acme",
        profile_id="alice",
        source_profile_dir=src,
        chrome_major=131,
    )

    target = tmp_path / "loaded-profile"
    result = snap.load(
        tenant_id="acme", profile_id="alice", local_profile_dir=target,
    )

    assert result.outcome == "loaded", result
    assert result.reason == ""
    assert result.manifest is not None
    assert result.bytes_downloaded > 0
    # The target dir now contains the same shape as the source.
    assert (target / "Default" / "Preferences").exists()
    assert (target / "Default" / "Cookies").exists()
    # And the cookie row round-trips.
    with sqlite3.connect(target / "Default" / "Cookies") as conn:
        row = conn.execute("SELECT host, name, value FROM cookies").fetchone()
    assert row == ("example.com", "session", "abc123")


def test_load_with_no_pointer_returns_no_snapshot(tmp_path: Path) -> None:
    """First-ever load for a (tenant, profile) — distinct from a failure."""
    s3 = _FakeS3()
    snap = _snapshotter(s3)
    target = tmp_path / "loaded-profile"

    result = snap.load(
        tenant_id="acme", profile_id="alice", local_profile_dir=target,
    )

    assert result.outcome == "no_snapshot"
    assert result.reason == ""
    assert result.manifest is None


# ── Strict-refusal paths ──────────────────────────────────────────────


def test_sha_mismatch_refuses_load(tmp_path: Path) -> None:
    """The manifest's archive_sha256 disagrees with the actual bytes →
    refuse the load and fresh-fallback."""
    s3 = _FakeS3()
    snap = _snapshotter(s3)
    src = _make_profile_dir(tmp_path)
    capture_and_upload_for_testing(
        snapshotter=snap, tenant_id="acme", profile_id="alice",
        source_profile_dir=src, chrome_major=131,
    )
    # Corrupt the archive (find the .tar.zst key, replace bytes).
    archive_keys = [
        k for (_b, k) in s3._store.keys()  # noqa: SLF001
        if k.endswith(".tar.zst")
    ]
    s3._store[("test-bucket", archive_keys[0])] = b"corrupted bytes"  # noqa: SLF001

    target = tmp_path / "loaded"
    result = snap.load(
        tenant_id="acme", profile_id="alice", local_profile_dir=target,
    )
    assert result.outcome == "fresh_fallback"
    assert result.reason == "sha_mismatch"
    assert target.exists()
    assert not list(target.iterdir())


def test_chrome_major_mismatch_refuses_load(tmp_path: Path) -> None:
    """Defense-in-depth: the prefix already segments by chrome_major,
    but a manifest written into the wrong prefix (manual tamper, or a
    botched migration) must still be refused.

    Manually layer a v131 manifest into the v128 prefix and load with
    a v128 loader. The prefix lookup succeeds; the manifest's claimed
    version disagrees with the loader's pin → strict refusal."""
    s3 = _FakeS3()
    loader = _snapshotter(s3, chrome_major=128)
    prefix = "snapshots/acme/alice/128/"

    # Build a manifest claiming chrome_major=131 even though it's in
    # the 128 prefix.
    fake_sha = "c" * 64
    manifest = {
        "version": 1, "schema": "computer-plane.profile-snapshot",
        "tenant_id": "acme", "profile_id": "alice",
        "chrome_major_version": 131,  # ← intentionally wrong
        "archive_sha256": fake_sha,
        "archive_size_bytes": 1024,
        "uncompressed_size_bytes": 2048,
        "captured_at_ms": 0, "mode": "cold",
    }
    s3.put_object(Bucket="test-bucket",
                  Key=f"{prefix}profile-cccccccccccc.manifest.json",
                  Body=json.dumps(manifest).encode())
    s3.put_object(Bucket="test-bucket",
                  Key=f"{prefix}profile-cccccccccccc.tar.zst",
                  Body=b"never gets read")
    s3.put_object(Bucket="test-bucket", Key=f"{prefix}latest.json",
                  Body=json.dumps({
                      "version": 1,
                      "active_sha256_prefix": "cccccccccccc",
                      "active_archive_key": f"{prefix}profile-cccccccccccc.tar.zst",
                      "active_manifest_key": f"{prefix}profile-cccccccccccc.manifest.json",
                      "flipped_at_ms": 0,
                  }).encode())

    target = tmp_path / "loaded"
    result = loader.load(
        tenant_id="acme", profile_id="alice", local_profile_dir=target,
    )
    assert result.outcome == "fresh_fallback"
    assert result.reason == "chrome_major_mismatch"


def test_archive_too_large_refuses(tmp_path: Path) -> None:
    """Bucket has a manifest claiming a 12 GB archive; policy ceiling
    is 8 GiB. Refuse before even downloading."""
    s3 = _FakeS3()
    snap = ProfileSnapshotter(
        bucket="test-bucket", chrome_major=131, s3_client=s3,
        max_archive_bytes=100,  # tiny ceiling for the test
    )
    # Build a manifest by hand — bypass the helper so we can lie about size.
    fake_sha = "a" * 64
    manifest = {
        "version": 1, "schema": "computer-plane.profile-snapshot",
        "tenant_id": "acme", "profile_id": "alice",
        "chrome_major_version": 131,
        "archive_sha256": fake_sha,
        "archive_size_bytes": 1024,  # exceeds the test ceiling
        "uncompressed_size_bytes": 2048,
        "captured_at_ms": 0, "mode": "cold",
    }
    prefix = "snapshots/acme/alice/131/"
    s3.put_object(Bucket="test-bucket", Key=f"{prefix}profile-aaaaaaaaaaaa.manifest.json",
                  Body=json.dumps(manifest).encode())
    s3.put_object(Bucket="test-bucket", Key=f"{prefix}profile-aaaaaaaaaaaa.tar.zst",
                  Body=b"never gets read")
    s3.put_object(Bucket="test-bucket", Key=f"{prefix}latest.json", Body=json.dumps({
        "version": 1,
        "active_sha256_prefix": "aaaaaaaaaaaa",
        "active_archive_key": f"{prefix}profile-aaaaaaaaaaaa.tar.zst",
        "active_manifest_key": f"{prefix}profile-aaaaaaaaaaaa.manifest.json",
        "flipped_at_ms": 0,
    }).encode())

    target = tmp_path / "loaded"
    result = snap.load(
        tenant_id="acme", profile_id="alice", local_profile_dir=target,
    )
    assert result.outcome == "fresh_fallback"
    assert result.reason == "too_large"
    # The archive was NEVER read — the ceiling check fires before download.
    archive_reads = [
        (b, k) for (b, k) in s3.get_calls if k.endswith(".tar.zst")
    ]
    assert archive_reads == []


def test_hot_mode_refused_by_default(tmp_path: Path) -> None:
    """Default policy: refuse hot-mode snapshots. The spec calls hot
    mode an opt-in for callers who explicitly accept the WAL risk."""
    s3 = _FakeS3()
    snap = _snapshotter(s3, allow_hot=False)
    src = _make_profile_dir(tmp_path)
    capture_and_upload_for_testing(
        snapshotter=snap, tenant_id="acme", profile_id="alice",
        source_profile_dir=src, chrome_major=131, mode="hot",
    )
    target = tmp_path / "loaded"
    result = snap.load(
        tenant_id="acme", profile_id="alice", local_profile_dir=target,
    )
    assert result.outcome == "fresh_fallback"
    assert result.reason == "hot_mode_disabled"


def test_hot_mode_allowed_when_opted_in(tmp_path: Path) -> None:
    """Same snapshot, but loader was constructed with allow_hot=True →
    loads cleanly."""
    s3 = _FakeS3()
    writer = _snapshotter(s3, allow_hot=False)  # writer policy irrelevant
    src = _make_profile_dir(tmp_path)
    capture_and_upload_for_testing(
        snapshotter=writer, tenant_id="acme", profile_id="alice",
        source_profile_dir=src, chrome_major=131, mode="hot",
    )
    loader = _snapshotter(s3, allow_hot=True)
    target = tmp_path / "loaded"
    result = loader.load(
        tenant_id="acme", profile_id="alice", local_profile_dir=target,
    )
    assert result.outcome == "loaded"


def test_corrupted_sqlite_refuses_load(tmp_path: Path) -> None:
    """Inject a non-SQLite blob in place of the Cookies DB. Loader
    extracts, then PRAGMA integrity_check refuses."""
    s3 = _FakeS3()
    snap = _snapshotter(s3)
    src = tmp_path / "src-profile"
    (src / "Default").mkdir(parents=True)
    (src / "Default" / "Cookies").write_bytes(b"this is not a sqlite db")

    capture_and_upload_for_testing(
        snapshotter=snap, tenant_id="acme", profile_id="alice",
        source_profile_dir=src, chrome_major=131,
    )
    target = tmp_path / "loaded"
    result = snap.load(
        tenant_id="acme", profile_id="alice", local_profile_dir=target,
    )
    assert result.outcome == "fresh_fallback"
    assert result.reason == "integrity_check_failed"


def test_pointer_with_unknown_version_refuses(tmp_path: Path) -> None:
    """latest.json carries version=2; loader is pinned to v1."""
    s3 = _FakeS3()
    snap = _snapshotter(s3)
    prefix = "snapshots/acme/alice/131/"
    s3.put_object(Bucket="test-bucket", Key=f"{prefix}latest.json", Body=json.dumps({
        "version": 2,
        "active_sha256_prefix": "deadbeefcafe",
        "active_archive_key": f"{prefix}archive",
        "active_manifest_key": f"{prefix}manifest",
        "flipped_at_ms": 0,
    }).encode())
    target = tmp_path / "loaded"
    result = snap.load(
        tenant_id="acme", profile_id="alice", local_profile_dir=target,
    )
    assert result.outcome == "fresh_fallback"
    assert result.reason == "manifest_version_mismatch"


def test_garbled_pointer_refuses(tmp_path: Path) -> None:
    """latest.json is not valid JSON."""
    s3 = _FakeS3()
    snap = _snapshotter(s3)
    s3.put_object(
        Bucket="test-bucket",
        Key="snapshots/acme/alice/131/latest.json",
        Body=b"not json at all",
    )
    result = snap.load(
        tenant_id="acme", profile_id="alice",
        local_profile_dir=tmp_path / "loaded",
    )
    assert result.outcome == "fresh_fallback"
    assert result.reason == "manifest_invalid"


def test_missing_manifest_after_pointer_refuses(tmp_path: Path) -> None:
    """latest.json points at a manifest that doesn't exist."""
    s3 = _FakeS3()
    snap = _snapshotter(s3)
    prefix = "snapshots/acme/alice/131/"
    s3.put_object(Bucket="test-bucket", Key=f"{prefix}latest.json", Body=json.dumps({
        "version": 1,
        "active_sha256_prefix": "deadbeefcafe",
        "active_archive_key": f"{prefix}archive",
        "active_manifest_key": f"{prefix}missing-manifest.json",
        "flipped_at_ms": 0,
    }).encode())
    target = tmp_path / "loaded"
    result = snap.load(
        tenant_id="acme", profile_id="alice", local_profile_dir=target,
    )
    assert result.outcome == "fresh_fallback"
    assert result.reason == "no_manifest"


def test_missing_archive_after_pointer_refuses(tmp_path: Path) -> None:
    """Manifest exists but its archive key is gone."""
    s3 = _FakeS3()
    snap = _snapshotter(s3)
    prefix = "snapshots/acme/alice/131/"
    fake_sha = "b" * 64
    manifest = {
        "version": 1, "schema": "computer-plane.profile-snapshot",
        "tenant_id": "acme", "profile_id": "alice",
        "chrome_major_version": 131,
        "archive_sha256": fake_sha,
        "archive_size_bytes": 1024,
        "uncompressed_size_bytes": 2048,
        "captured_at_ms": 0, "mode": "cold",
    }
    s3.put_object(Bucket="test-bucket", Key=f"{prefix}profile-bbbbbbbbbbbb.manifest.json",
                  Body=json.dumps(manifest).encode())
    s3.put_object(Bucket="test-bucket", Key=f"{prefix}latest.json", Body=json.dumps({
        "version": 1,
        "active_sha256_prefix": "bbbbbbbbbbbb",
        "active_archive_key": f"{prefix}profile-bbbbbbbbbbbb.tar.zst",
        "active_manifest_key": f"{prefix}profile-bbbbbbbbbbbb.manifest.json",
        "flipped_at_ms": 0,
    }).encode())
    # Note: NO archive uploaded.
    target = tmp_path / "loaded"
    result = snap.load(
        tenant_id="acme", profile_id="alice", local_profile_dir=target,
    )
    assert result.outcome == "fresh_fallback"
    assert result.reason == "no_archive"


# ── Network / S3 failure modes (don't bubble; return fresh fallback) ──


def test_s3_credential_error_returns_fresh(tmp_path: Path) -> None:
    """boto3 raises an auth error → don't crash the brain, return
    fresh-fallback (the caller's existing Chrome will boot empty)."""
    s3 = _FakeS3()
    snap = _snapshotter(s3)
    s3.raise_on_get = _FakeClientError("AccessDenied", "creds broken")

    result = snap.load(
        tenant_id="acme", profile_id="alice",
        local_profile_dir=tmp_path / "loaded",
    )
    # Loader treats unknown-cause failures as no_snapshot — see the
    # docstring on _read_bytes. Either no_snapshot or fresh_fallback
    # is acceptable; brain code branches the same way.
    assert result.outcome in {"no_snapshot", "fresh_fallback"}


def test_load_requires_tenant_id_and_profile_id() -> None:
    s3 = _FakeS3()
    snap = _snapshotter(s3)
    with pytest.raises(ValueError):
        snap.load(tenant_id="", profile_id="x",
                  local_profile_dir=Path("/tmp/anywhere"))
    with pytest.raises(ValueError):
        snap.load(tenant_id="x", profile_id="",
                  local_profile_dir=Path("/tmp/anywhere"))


def test_constructor_validates_args() -> None:
    s3 = _FakeS3()
    with pytest.raises(ValueError):
        ProfileSnapshotter(bucket="", chrome_major=131, s3_client=s3)
    with pytest.raises(ValueError):
        ProfileSnapshotter(bucket="b", chrome_major=0, s3_client=s3)
    with pytest.raises(ValueError):
        ProfileSnapshotter(bucket="b", chrome_major=131, s3_client=None)


# ── from_env ──────────────────────────────────────────────────────────


def test_from_env_raises_when_bucket_missing(monkeypatch) -> None:
    monkeypatch.delenv("MANTIS_PROFILE_SNAPSHOT_BUCKET", raising=False)
    with pytest.raises(RuntimeError, match="BUCKET"):
        ProfileSnapshotter.from_env()


def test_from_env_raises_when_chrome_major_missing(monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_PROFILE_SNAPSHOT_BUCKET", "b")
    monkeypatch.delenv("MANTIS_CHROME_MAJOR_VERSION", raising=False)
    with pytest.raises(RuntimeError, match="CHROME_MAJOR"):
        ProfileSnapshotter.from_env()


# ── Idempotency ──


def test_repeat_load_is_idempotent(tmp_path: Path) -> None:
    """Loading twice produces the same target state — no leftover
    files from the first load."""
    s3 = _FakeS3()
    snap = _snapshotter(s3)
    src = _make_profile_dir(tmp_path)
    capture_and_upload_for_testing(
        snapshotter=snap, tenant_id="acme", profile_id="alice",
        source_profile_dir=src, chrome_major=131,
    )
    target = tmp_path / "loaded"
    r1 = snap.load(tenant_id="acme", profile_id="alice", local_profile_dir=target)
    files1 = sorted(p.relative_to(target) for p in target.rglob("*") if p.is_file())
    r2 = snap.load(tenant_id="acme", profile_id="alice", local_profile_dir=target)
    files2 = sorted(p.relative_to(target) for p in target.rglob("*") if p.is_file())
    assert r1.outcome == "loaded"
    assert r2.outcome == "loaded"
    assert files1 == files2
