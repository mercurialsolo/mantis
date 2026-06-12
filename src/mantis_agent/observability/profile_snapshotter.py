"""Profile snapshot loader for Phase 2 host backends (#699).

The canonical spec lives at
``docs/reference/computer-plane-profile-snapshots.md``. This module
is M1 of three milestones — it ships only the **read path** plus a
clearly-labelled test helper for populating snapshots into the
bucket. M2 ships the production writer; M3 ships operator surfaces.

Why read-only first: a M1-only deploy can be exercised against a
real bucket using snapshots populated manually (via the test
helper). This lets us validate the contract end-to-end — sha256
verification, SQLite integrity check, fresh-Chrome fallback,
chrome_major mismatch — before any production writer can corrupt a
tenant's profile.

The loader is deliberately permissive on its way DOWN to the
profile bytes and strict on the way IN:

* Bucket / network failures → log WARNING, return ``LoadResult(
  outcome="fresh_fallback")``. The brain proceeds with an empty
  Chrome rather than crashing the run.
* sha256 mismatch → strict refusal. Never load bytes the writer
  didn't sign off on. Fresh fallback + WARNING.
* chrome_major mismatch → strict refusal. Same.
* SQLite integrity check failure → strict refusal. Same.

Everything that surfaces a fresh-fallback path is logged at
WARNING level so Modal / Truss / B2 observability still see it
(``feedback_warning_level_for_modal_observability``).

This module does NOT acquire profile locks. The lock is part of M2.
M1 lets us exercise the read path against pre-populated test
snapshots; it does not yet replace the live profile-reuse contract.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import sqlite3
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Wire contracts (mirror the spec doc §1 + §2) ───────────────────────


class SnapshotManifest(BaseModel):
    """Schema for ``profile-<sha>.manifest.json``. Pinned to v1.

    Spec section: ``computer-plane-profile-snapshots.md § 1
    "On-disk format → Manifest"``.

    Forward-compat: unknown fields are accepted (extra='allow'). The
    loader only consumes the named fields; everything else is
    advisory.
    """

    model_config = {"extra": "allow"}

    version: int = Field(..., ge=1)
    schema_: str = Field(alias="schema", default="computer-plane.profile-snapshot")
    tenant_id: str = Field(..., min_length=1)
    profile_id: str = Field(..., min_length=1)
    chrome_major_version: int = Field(..., ge=1)
    archive_sha256: str = Field(..., min_length=64, max_length=64)
    archive_size_bytes: int = Field(..., ge=0)
    uncompressed_size_bytes: int = Field(..., ge=0)
    captured_at_ms: int = Field(..., ge=0)
    mode: Literal["cold", "hot"] = "cold"
    chrome_uptime_seconds_at_capture: int = 0
    predecessor_sha256: str = ""
    notes: str = ""

    # ``captured_by`` and ``captured_in`` are dicts — read but unvalidated.
    captured_by: dict[str, Any] = Field(default_factory=dict)
    captured_in: dict[str, Any] = Field(default_factory=dict)


class LatestPointer(BaseModel):
    """Schema for ``latest.json``. Pinned to v1.

    Spec section: ``§ 2 "Object-store layout → latest.json"``.
    """

    model_config = {"extra": "allow"}

    version: int = Field(..., ge=1)
    active_sha256_prefix: str = Field(..., min_length=8)
    active_archive_key: str = Field(..., min_length=1)
    active_manifest_key: str = Field(..., min_length=1)
    flipped_at_ms: int = Field(..., ge=0)
    flipped_from_sha256_prefix: str = ""


# ── Outcome of a load() call ──────────────────────────────────────────


LoadOutcome = Literal[
    "loaded",                 # Snapshot successfully extracted into target
    "no_snapshot",            # No latest.json — fresh Chrome (not an error)
    "fresh_fallback",         # Failure path — fresh Chrome with WARNING
]


_FAILURE_REASONS = frozenset({
    "bucket_unreachable",          # boto3 raised
    "no_manifest",                 # latest.json points at a missing manifest
    "no_archive",                  # latest.json points at a missing archive
    "manifest_invalid",            # JSON parse / schema validation failed
    "manifest_version_mismatch",   # version != 1
    "sha_mismatch",                # downloaded archive disagrees with manifest
    "chrome_major_mismatch",       # captured for a different Chrome major
    "extract_failed",              # tar/zstd decode raised
    "integrity_check_failed",      # SQLite PRAGMA integrity_check returned !ok
    "too_large",                   # archive_size_bytes exceeds policy ceiling
    "hot_mode_disabled",           # snapshot is hot mode + policy refuses
})


@dataclass(frozen=True)
class LoadResult:
    """Structured outcome the brain can branch on after ``load()``."""

    outcome: LoadOutcome
    reason: str = ""              # one of _FAILURE_REASONS, or empty
    manifest: Optional[SnapshotManifest] = None
    elapsed_seconds: float = 0.0
    bytes_downloaded: int = 0


# ── Configuration knobs ───────────────────────────────────────────────


# Policy ceiling. Spec § 3 — profiles above this get rejected by the
# writer and refused by the loader (so a misbehaving writer that
# slipped through can't unintentionally explode the target dir).
_MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB

# Chrome user-data-dir SQLite databases the loader integrity-checks
# after extract. Failing any one is a strict refusal. These are the
# four databases Chrome treats as authoritative; LocalStorage's
# leveldb is checked separately (just an existence sanity).
_SQLITE_DBS_TO_VERIFY = (
    "Default/Cookies",
    "Default/History",
    "Default/Web Data",
    "Default/Login Data",
)


# ── Loader ────────────────────────────────────────────────────────────


class ProfileSnapshotter:
    """M1 read-path loader.

    Construction is DI-friendly: a ``boto3`` S3 client (or anything
    quack-compatible) is passed in, so unit tests can drive against
    a fake without touching real network.

    Production wire-up (in the worker bring-up path) — typical
    construction lives in ``setup_env`` once M2 ships the writer hook;
    for M1 it's invoked by the brain explicitly before Chrome boot:

        snapshotter = ProfileSnapshotter.from_env()
        result = snapshotter.load(
            tenant_id=..., profile_id=...,
            local_profile_dir=Path("/data/chrome-profile/..."),
        )
        if result.outcome == "loaded":
            # Chrome boots against the loaded profile
            ...
        elif result.outcome in ("no_snapshot", "fresh_fallback"):
            # Chrome boots fresh
            ...

    The loader does NOT mutate state outside ``local_profile_dir``.
    """

    def __init__(
        self,
        *,
        bucket: str,
        chrome_major: int,
        s3_client: Any,
        allow_hot_mode: bool = False,
        max_archive_bytes: int = _MAX_ARCHIVE_BYTES,
        clock: Any = None,
    ) -> None:
        if not bucket:
            raise ValueError("ProfileSnapshotter requires a non-empty bucket")
        if chrome_major <= 0:
            raise ValueError(
                f"chrome_major must be a positive int, got {chrome_major!r}"
            )
        if s3_client is None:
            raise ValueError("ProfileSnapshotter requires an S3 client")
        self._bucket = bucket
        self._chrome_major = int(chrome_major)
        self._s3 = s3_client
        self._allow_hot_mode = bool(allow_hot_mode)
        self._max_archive_bytes = int(max_archive_bytes)
        self._clock = clock or time.monotonic

    @classmethod
    def from_env(cls) -> "ProfileSnapshotter":
        """Build from the ``MANTIS_PROFILE_SNAPSHOT_*`` env vars.

        Reads:
        * ``MANTIS_PROFILE_SNAPSHOT_BUCKET``
        * ``MANTIS_PROFILE_SNAPSHOT_S3_ENDPOINT``
        * ``MANTIS_PROFILE_SNAPSHOT_S3_REGION``
        * ``MANTIS_CHROME_MAJOR_VERSION`` (the deploy-time pin)
        * AWS standard env (``AWS_ACCESS_KEY_ID`` /
          ``AWS_SECRET_ACCESS_KEY``) via boto3's default chain

        Raises ``RuntimeError`` if any required var is missing.
        """
        import os

        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "ProfileSnapshotter.from_env requires boto3 — install with "
                "``pip install mantis-agent[snapshots]``"
            ) from exc

        bucket = os.environ.get("MANTIS_PROFILE_SNAPSHOT_BUCKET", "").strip()
        endpoint_url = os.environ.get(
            "MANTIS_PROFILE_SNAPSHOT_S3_ENDPOINT", ""
        ).strip()
        region = os.environ.get(
            "MANTIS_PROFILE_SNAPSHOT_S3_REGION", ""
        ).strip()
        chrome_major_raw = os.environ.get(
            "MANTIS_CHROME_MAJOR_VERSION", ""
        ).strip()

        if not bucket:
            raise RuntimeError("MANTIS_PROFILE_SNAPSHOT_BUCKET is not set")
        if not chrome_major_raw:
            raise RuntimeError("MANTIS_CHROME_MAJOR_VERSION is not set")
        try:
            chrome_major = int(chrome_major_raw)
        except ValueError as exc:
            raise RuntimeError(
                f"MANTIS_CHROME_MAJOR_VERSION must be int, got "
                f"{chrome_major_raw!r}"
            ) from exc

        kwargs: dict[str, Any] = {}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        if region:
            kwargs["region_name"] = region
        s3 = boto3.client("s3", **kwargs)

        return cls(bucket=bucket, chrome_major=chrome_major, s3_client=s3)

    # ── Public surface ──

    def load(
        self,
        *,
        tenant_id: str,
        profile_id: str,
        local_profile_dir: Path,
    ) -> LoadResult:
        """Read the latest snapshot for ``(tenant_id, profile_id)`` into
        ``local_profile_dir``.

        On any failure path the target dir is WIPED and the caller
        gets ``LoadResult(outcome="fresh_fallback", reason=...)`` —
        Chrome can boot against the empty directory and the brain
        carries on. The loader never raises; the only thing that
        propagates out is the structured outcome.
        """
        if not tenant_id or not profile_id:
            raise ValueError(
                "load() requires non-empty tenant_id + profile_id"
            )

        t0 = self._clock()
        local_profile_dir = Path(local_profile_dir)
        prefix = self._prefix(tenant_id, profile_id)

        # 1. Resolve latest.json
        pointer_blob = self._read_text(prefix + "latest.json")
        if pointer_blob is None:
            # Distinct from a failure — no pointer = legitimately empty.
            return LoadResult(
                outcome="no_snapshot",
                elapsed_seconds=self._clock() - t0,
            )

        try:
            pointer = LatestPointer.model_validate_json(pointer_blob)
        except Exception as exc:  # noqa: BLE001 — strict refusal
            logger.warning(
                "profile snapshot: latest.json parse failed "
                "tenant=%s profile=%s (%s)", tenant_id, profile_id, exc,
            )
            return self._fresh_fallback(
                local_profile_dir, "manifest_invalid",
                started_at=t0,
            )
        if pointer.version != 1:
            logger.warning(
                "profile snapshot: latest.json version=%d (expected 1) "
                "tenant=%s profile=%s",
                pointer.version, tenant_id, profile_id,
            )
            return self._fresh_fallback(
                local_profile_dir, "manifest_version_mismatch",
                started_at=t0,
            )

        # 2. Manifest
        manifest_blob = self._read_text(pointer.active_manifest_key)
        if manifest_blob is None:
            logger.warning(
                "profile snapshot: manifest missing at %s "
                "tenant=%s profile=%s",
                pointer.active_manifest_key, tenant_id, profile_id,
            )
            return self._fresh_fallback(
                local_profile_dir, "no_manifest", started_at=t0,
            )
        try:
            manifest = SnapshotManifest.model_validate_json(manifest_blob)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "profile snapshot: manifest parse failed "
                "tenant=%s profile=%s (%s)", tenant_id, profile_id, exc,
            )
            return self._fresh_fallback(
                local_profile_dir, "manifest_invalid",
                started_at=t0,
            )

        # 3. Pre-flight checks before paying for the archive download
        if manifest.version != 1:
            logger.warning(
                "profile snapshot: manifest version=%d (expected 1) "
                "tenant=%s profile=%s",
                manifest.version, tenant_id, profile_id,
            )
            return self._fresh_fallback(
                local_profile_dir, "manifest_version_mismatch",
                started_at=t0, manifest=manifest,
            )
        if manifest.chrome_major_version != self._chrome_major:
            logger.warning(
                "profile snapshot: chrome_major mismatch "
                "snapshot=%d loader=%d tenant=%s profile=%s",
                manifest.chrome_major_version, self._chrome_major,
                tenant_id, profile_id,
            )
            return self._fresh_fallback(
                local_profile_dir, "chrome_major_mismatch",
                started_at=t0, manifest=manifest,
            )
        if manifest.archive_size_bytes > self._max_archive_bytes:
            logger.warning(
                "profile snapshot: archive exceeds policy ceiling "
                "size=%d max=%d tenant=%s profile=%s",
                manifest.archive_size_bytes, self._max_archive_bytes,
                tenant_id, profile_id,
            )
            return self._fresh_fallback(
                local_profile_dir, "too_large",
                started_at=t0, manifest=manifest,
            )
        if manifest.mode == "hot" and not self._allow_hot_mode:
            logger.warning(
                "profile snapshot: hot-mode snapshot refused by policy "
                "tenant=%s profile=%s", tenant_id, profile_id,
            )
            return self._fresh_fallback(
                local_profile_dir, "hot_mode_disabled",
                started_at=t0, manifest=manifest,
            )

        # 4. Download archive
        archive_bytes = self._read_bytes(pointer.active_archive_key)
        if archive_bytes is None:
            logger.warning(
                "profile snapshot: archive missing at %s "
                "tenant=%s profile=%s",
                pointer.active_archive_key, tenant_id, profile_id,
            )
            return self._fresh_fallback(
                local_profile_dir, "no_archive",
                started_at=t0, manifest=manifest,
            )

        # 5. sha256 verification — STRICT
        computed = hashlib.sha256(archive_bytes).hexdigest()
        if computed != manifest.archive_sha256:
            logger.warning(
                "profile snapshot: sha256 mismatch "
                "tenant=%s profile=%s "
                "expected=%s computed=%s",
                tenant_id, profile_id,
                manifest.archive_sha256, computed,
            )
            return self._fresh_fallback(
                local_profile_dir, "sha_mismatch",
                started_at=t0, manifest=manifest,
                bytes_downloaded=len(archive_bytes),
            )

        # 6. Extract
        try:
            self._extract(archive_bytes, local_profile_dir)
        except Exception as exc:  # noqa: BLE001 — strict refusal
            logger.warning(
                "profile snapshot: extract failed "
                "tenant=%s profile=%s (%s)", tenant_id, profile_id, exc,
            )
            return self._fresh_fallback(
                local_profile_dir, "extract_failed",
                started_at=t0, manifest=manifest,
                bytes_downloaded=len(archive_bytes),
            )

        # 7. SQLite integrity check — STRICT
        bad_db = self._integrity_check(local_profile_dir)
        if bad_db is not None:
            logger.warning(
                "profile snapshot: SQLite integrity check failed "
                "db=%s tenant=%s profile=%s",
                bad_db, tenant_id, profile_id,
            )
            return self._fresh_fallback(
                local_profile_dir, "integrity_check_failed",
                started_at=t0, manifest=manifest,
                bytes_downloaded=len(archive_bytes),
            )

        return LoadResult(
            outcome="loaded",
            manifest=manifest,
            elapsed_seconds=self._clock() - t0,
            bytes_downloaded=len(archive_bytes),
        )

    # ── Internals ──

    def _prefix(self, tenant_id: str, profile_id: str) -> str:
        return (
            f"snapshots/{_safe(tenant_id)}/{_safe(profile_id)}/"
            f"{self._chrome_major}/"
        )

    def _read_text(self, key: str) -> Optional[str]:
        body = self._read_bytes(key)
        if body is None:
            return None
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def _read_bytes(self, key: str) -> Optional[bytes]:
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 — any S3 failure → None
            # Differentiate "not found" (silent return → caller turns
            # into no_snapshot / no_archive / no_manifest) from
            # "actually broken" (network / auth) which we log.
            if _is_not_found(exc):
                return None
            logger.warning(
                "profile snapshot: S3 read failed key=%s bucket=%s (%s)",
                key, self._bucket, exc,
            )
            return None
        try:
            return resp["Body"].read()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "profile snapshot: S3 read-body failed key=%s (%s)",
                key, exc,
            )
            return None

    def _extract(self, archive: bytes, target: Path) -> None:
        """Decompress + un-tar ``archive`` into ``target``.

        Wipe-and-replace: the target dir is removed (if present) then
        the archive's tree is extracted fresh. Chrome cares about
        consistency of the whole directory; partial overlays cause
        torn reads.
        """
        import zstandard as zstd  # imported lazily — optional dep

        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(archive)
        with tarfile.open(fileobj=io.BytesIO(decompressed), mode="r:") as tf:
            tf.extractall(target, filter="data")

    def _integrity_check(self, profile_dir: Path) -> Optional[str]:
        """Return the path of the FIRST DB that fails integrity, or None.

        Missing DB files are treated as "fresh profile sub-state" —
        skipped silently. Only files that exist + fail SQLite's
        ``PRAGMA integrity_check`` count as a failure.
        """
        for rel in _SQLITE_DBS_TO_VERIFY:
            path = profile_dir / rel
            if not path.exists() or path.is_dir():
                continue
            try:
                with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                    cur = conn.execute("PRAGMA integrity_check")
                    row = cur.fetchone()
            except sqlite3.DatabaseError as exc:
                logger.warning(
                    "profile snapshot: SQLite open failed "
                    "path=%s (%s)", path, exc,
                )
                return rel
            except sqlite3.OperationalError as exc:
                # Locked / not a database — count as failed integrity
                logger.warning(
                    "profile snapshot: SQLite operational failure "
                    "path=%s (%s)", path, exc,
                )
                return rel
            if not row or str(row[0]).lower() != "ok":
                return rel
        return None

    def _fresh_fallback(
        self,
        local_profile_dir: Path,
        reason: str,
        *,
        started_at: float,
        manifest: Optional[SnapshotManifest] = None,
        bytes_downloaded: int = 0,
    ) -> LoadResult:
        """Wipe the target dir + return a fresh-fallback outcome.

        Every fail path runs through here so the brain's downstream
        Chrome boot is guaranteed-clean. The reason field is the only
        observable signal of "why didn't we load the snapshot"; the
        WARNING log line is the operator-visible counterpart.
        """
        try:
            if local_profile_dir.exists():
                shutil.rmtree(local_profile_dir)
            local_profile_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "profile snapshot: fresh-fallback wipe failed "
                "path=%s (%s)", local_profile_dir, exc,
            )
        return LoadResult(
            outcome="fresh_fallback",
            reason=reason,
            manifest=manifest,
            elapsed_seconds=self._clock() - started_at,
            bytes_downloaded=bytes_downloaded,
        )


# ── Capture (M2 writer) ───────────────────────────────────────────────


CaptureOutcome = Literal[
    "captured",        # New snapshot written and pointer flipped
    "deduplicated",    # Identical archive_sha already canonical (no-op)
    "too_large",       # Source exceeded policy ceiling — refused
    "empty_source",    # Source dir didn't exist or had no files
    "upload_failed",   # Archive / manifest upload didn't complete
    "pointer_race",    # CAS retries exhausted; archive uploaded, pointer not flipped
    "exception",       # Unhandled error inside capture
]


@dataclass(frozen=True)
class CaptureResult:
    """Structured outcome of a ``capture()`` call.

    Mirrors :class:`LoadResult`'s shape so brain code can treat the
    two outcomes symmetrically (every plan run loads at boot, captures
    at teardown).
    """

    outcome: CaptureOutcome
    reason: str = ""
    archive_sha256: str = ""
    archive_size_bytes: int = 0
    uncompressed_size_bytes: int = 0
    elapsed_seconds: float = 0.0
    manifest_key: str = ""
    archive_key: str = ""
    predecessor_sha256: str = ""


# How many times to retry the ``latest.json`` conditional PUT before
# giving up. Spec § 2 — 3 attempts is the empirical sweet spot for B2
# under typical concurrent-writer pressure (two reapers, one brain).
_POINTER_FLIP_RETRIES = 3


def _capture(self, *, tenant_id: str, profile_id: str,
             source_profile_dir: Path,
             mode: Literal["cold", "hot"] = "cold",
             chrome_uptime_seconds_at_capture: int = 0,
             notes: str = "",
             captured_in: Optional[dict[str, Any]] = None) -> "CaptureResult":
    """Production writer — spec § 1 + § 2.

    Tars + zstd-9 compresses ``source_profile_dir`` into an in-memory
    archive, uploads the archive + sibling manifest, then flips
    ``latest.json`` via S3 conditional PUT (compare-and-swap on ETag).

    Hot mode raises ``NotImplementedError`` until the cold-mode test
    harness lands — spec § 3 deliberately defers hot mode.

    Idempotency:

    * If the computed ``archive_sha256`` matches the currently
      pointed-at snapshot, the upload is a content-addressed no-op:
      we don't re-upload, and the pointer flip is also skipped. We
      return ``outcome="deduplicated"`` so the caller can log it as
      free.
    * If the conditional PUT fails after ``_POINTER_FLIP_RETRIES``
      attempts, the archive + manifest STAY in the bucket (visible
      via listing). The pointer just didn't flip. Operator can flip
      manually via the M3 CLI.

    Failure modes that don't raise (return CaptureResult instead):

    * Source dir doesn't exist → ``empty_source``
    * Source dir empty (tar of nothing) → ``empty_source``
    * Compressed size exceeds policy ceiling → ``too_large``
    * Archive PUT raises → ``upload_failed``
    * Pointer CAS retries exhausted → ``pointer_race``

    Caller decides whether to surface these as a hard error or just
    log + carry on (the brain's hot path typically logs + carries on:
    a missed snapshot doesn't break the run, just costs a fresh-boot
    next time).
    """
    import zstandard as zstd

    if mode == "hot":
        raise NotImplementedError(
            "hot-mode capture is deferred — see "
            "docs/reference/computer-plane-profile-snapshots.md § 3. "
            "Use cold mode (the default) until the WAL-checkpoint "
            "test harness lands."
        )

    t0 = self._clock()
    source = Path(source_profile_dir)
    if not source.is_dir():
        return CaptureResult(
            outcome="empty_source",
            reason=f"source dir does not exist: {source}",
            elapsed_seconds=self._clock() - t0,
        )

    # 1. Tar the directory contents into an in-memory buffer.
    buf = io.BytesIO()
    file_count = 0
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for path in sorted(source.rglob("*")):
            rel = path.relative_to(source)
            tf.add(path, arcname=str(rel))
            if path.is_file():
                file_count += 1
    if file_count == 0:
        return CaptureResult(
            outcome="empty_source",
            reason="source dir contains no files",
            elapsed_seconds=self._clock() - t0,
        )
    uncompressed_size = buf.tell()

    # 2. zstd-9 compress.
    cctx = zstd.ZstdCompressor(level=9)
    compressed = cctx.compress(buf.getvalue())
    archive_sha256 = hashlib.sha256(compressed).hexdigest()
    sha_prefix = archive_sha256[:12]
    archive_size = len(compressed)

    # 3. Policy ceiling — refuse if too large.
    if archive_size > self._max_archive_bytes:
        return CaptureResult(
            outcome="too_large",
            reason=(
                f"archive {archive_size}B exceeds policy ceiling "
                f"{self._max_archive_bytes}B"
            ),
            archive_sha256=archive_sha256,
            archive_size_bytes=archive_size,
            uncompressed_size_bytes=uncompressed_size,
            elapsed_seconds=self._clock() - t0,
        )

    # 4. Dedup check — if the current pointer already points at this
    # sha, we have nothing new to upload.
    prefix = self._prefix(tenant_id, profile_id)
    archive_key = f"{prefix}profile-{sha_prefix}.tar.zst"
    manifest_key = f"{prefix}profile-{sha_prefix}.manifest.json"
    pointer_key = f"{prefix}latest.json"
    predecessor_sha = ""
    existing_pointer_etag = ""
    try:
        existing = self._s3.get_object(Bucket=self._bucket, Key=pointer_key)
        existing_pointer_etag = (existing.get("ETag") or "").strip('"')
        existing_body = existing["Body"].read()
        existing_pointer = LatestPointer.model_validate_json(existing_body)
        predecessor_sha = existing_pointer.active_sha256_prefix
        if existing_pointer.active_sha256_prefix == sha_prefix:
            return CaptureResult(
                outcome="deduplicated",
                reason="archive sha matches current latest pointer",
                archive_sha256=archive_sha256,
                archive_size_bytes=archive_size,
                uncompressed_size_bytes=uncompressed_size,
                manifest_key=manifest_key,
                archive_key=archive_key,
                predecessor_sha256=predecessor_sha,
                elapsed_seconds=self._clock() - t0,
            )
    except Exception as exc:  # noqa: BLE001
        if not _is_not_found(exc):
            logger.warning(
                "profile snapshot: pointer pre-read failed "
                "tenant=%s profile=%s (%s)",
                tenant_id, profile_id, exc,
            )
        # No existing pointer → first-ever snapshot. predecessor stays
        # empty, etag stays empty, conditional PUT below uses
        # If-None-Match: *.

    # 5. Upload archive + manifest. Archive first so a successful
    # manifest never points at a missing archive.
    try:
        self._s3.put_object(
            Bucket=self._bucket, Key=archive_key, Body=compressed,
            ContentType="application/zstd",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "profile snapshot: archive upload failed "
            "tenant=%s profile=%s key=%s (%s)",
            tenant_id, profile_id, archive_key, exc,
        )
        return CaptureResult(
            outcome="upload_failed", reason=f"archive PUT: {exc}",
            archive_sha256=archive_sha256,
            archive_size_bytes=archive_size,
            uncompressed_size_bytes=uncompressed_size,
            elapsed_seconds=self._clock() - t0,
        )

    manifest = {
        "version": 1,
        "schema": "computer-plane.profile-snapshot",
        "tenant_id": tenant_id,
        "profile_id": profile_id,
        "chrome_major_version": self._chrome_major,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size,
        "uncompressed_size_bytes": uncompressed_size,
        "captured_at_ms": int(time.time() * 1000),
        "mode": mode,
        "chrome_uptime_seconds_at_capture": int(
            chrome_uptime_seconds_at_capture
        ),
        "predecessor_sha256": predecessor_sha,
        "notes": notes or "",
        "captured_by": {
            "host": "modal",  # M2 will accept this as a kwarg in M3
            "writer_version": "snapshotter-0.2.0-m2",
        },
        "captured_in": captured_in or {},
    }
    try:
        self._s3.put_object(
            Bucket=self._bucket, Key=manifest_key,
            Body=json.dumps(manifest, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "profile snapshot: manifest upload failed "
            "tenant=%s profile=%s key=%s (%s)",
            tenant_id, profile_id, manifest_key, exc,
        )
        return CaptureResult(
            outcome="upload_failed", reason=f"manifest PUT: {exc}",
            archive_sha256=archive_sha256,
            archive_size_bytes=archive_size,
            uncompressed_size_bytes=uncompressed_size,
            archive_key=archive_key,
            elapsed_seconds=self._clock() - t0,
        )

    # 6. Conditional pointer flip via ETag CAS.
    flipped = self._flip_pointer(
        pointer_key=pointer_key,
        new_sha_prefix=sha_prefix,
        new_archive_key=archive_key,
        new_manifest_key=manifest_key,
        expected_etag=existing_pointer_etag,
        predecessor_sha_prefix=predecessor_sha,
    )
    if not flipped:
        return CaptureResult(
            outcome="pointer_race",
            reason=(
                f"conditional PUT on {pointer_key} exhausted "
                f"{_POINTER_FLIP_RETRIES} retries; archive+manifest "
                f"uploaded under sha={sha_prefix} but pointer not flipped"
            ),
            archive_sha256=archive_sha256,
            archive_size_bytes=archive_size,
            uncompressed_size_bytes=uncompressed_size,
            manifest_key=manifest_key,
            archive_key=archive_key,
            predecessor_sha256=predecessor_sha,
            elapsed_seconds=self._clock() - t0,
        )

    return CaptureResult(
        outcome="captured",
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size,
        uncompressed_size_bytes=uncompressed_size,
        manifest_key=manifest_key,
        archive_key=archive_key,
        predecessor_sha256=predecessor_sha,
        elapsed_seconds=self._clock() - t0,
    )


def _flip_pointer(self, *,
                  pointer_key: str,
                  new_sha_prefix: str,
                  new_archive_key: str,
                  new_manifest_key: str,
                  expected_etag: str,
                  predecessor_sha_prefix: str) -> bool:
    """Conditional PUT of ``latest.json`` with ETag CAS.

    Returns True on success, False if all retries hit a CAS conflict.
    """
    for attempt in range(_POINTER_FLIP_RETRIES):
        pointer_body = json.dumps({
            "version": 1,
            "active_sha256_prefix": new_sha_prefix,
            "active_archive_key": new_archive_key,
            "active_manifest_key": new_manifest_key,
            "flipped_at_ms": int(time.time() * 1000),
            "flipped_from_sha256_prefix": predecessor_sha_prefix,
        }, indent=2).encode("utf-8")
        kwargs: dict[str, Any] = {
            "Bucket": self._bucket,
            "Key": pointer_key,
            "Body": pointer_body,
            "ContentType": "application/json",
        }
        # B2 + S3 both support IfMatch/IfNoneMatch on PutObject. boto3
        # forwards these as request headers. ``expected_etag=""`` means
        # we expect the key not to exist yet (first snapshot).
        if expected_etag:
            kwargs["IfMatch"] = expected_etag
        else:
            kwargs["IfNoneMatch"] = "*"
        try:
            self._s3.put_object(**kwargs)
            return True
        except Exception as exc:  # noqa: BLE001
            if not _is_cas_conflict(exc):
                logger.warning(
                    "profile snapshot: pointer PUT raised non-CAS "
                    "error attempt=%d/%d (%s)",
                    attempt + 1, _POINTER_FLIP_RETRIES, exc,
                )
                return False
        # CAS conflict — re-read the pointer to pick up the new ETag,
        # then retry. If the re-read shows the predecessor sha already
        # matches ours, treat as deduplicated success.
        try:
            existing = self._s3.get_object(
                Bucket=self._bucket, Key=pointer_key,
            )
            expected_etag = (existing.get("ETag") or "").strip('"')
            try:
                cur = LatestPointer.model_validate_json(
                    existing["Body"].read()
                )
                if cur.active_sha256_prefix == new_sha_prefix:
                    return True  # someone else won the race with the same sha
                predecessor_sha_prefix = cur.active_sha256_prefix
            except Exception:  # noqa: BLE001 — fall through to retry
                pass
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "profile snapshot: pointer re-read failed during retry "
                "attempt=%d (%s)", attempt + 1, exc,
            )
            return False
    return False


# Bind the new methods onto ``ProfileSnapshotter``. Defining them at
# module scope (rather than inline in the class body) keeps the
# loader-only PR #865 diff small AND keeps M2's writer concerns
# visually grouped under the # ── Capture (M2 writer) ── banner.
ProfileSnapshotter.capture = _capture  # type: ignore[attr-defined]
ProfileSnapshotter._flip_pointer = _flip_pointer  # type: ignore[attr-defined]


def _is_cas_conflict(exc: Exception) -> bool:
    """True iff the S3 exception is a CAS conflict (PreconditionFailed
    or 412)."""
    err = getattr(exc, "response", {}) or {}
    code = (err.get("Error") or {}).get("Code", "")
    if code in {"PreconditionFailed", "412"}:
        return True
    msg = str(exc).lower()
    if "preconditionfailed" in msg or "412" in msg:
        return True
    return False


# ── Helpers ──────────────────────────────────────────────────────────


def _safe(value: str) -> str:
    """Filesystem-safe key segment. Mirrors the run-state-store rule."""
    return "".join(
        c if c.isalnum() or c in {"-", "_"} else "_" for c in value
    )


def _is_not_found(exc: Exception) -> bool:
    """True iff the S3 exception means the key doesn't exist."""
    # boto3's ClientError carries response['Error']['Code'] which is
    # 'NoSuchKey' or '404' depending on the backend. Check both.
    err = getattr(exc, "response", {}) or {}
    code = (err.get("Error") or {}).get("Code", "")
    if code in {"NoSuchKey", "404", "NoSuchBucket"}:
        return True
    msg = str(exc).lower()
    if "nosuchkey" in msg or "not found" in msg or "404" in msg:
        return True
    return False


# ── Test helper (NOT for production use) ───────────────────────────────


def capture_and_upload_for_testing(
    *,
    snapshotter: ProfileSnapshotter,
    tenant_id: str,
    profile_id: str,
    source_profile_dir: Path,
    chrome_major: int,
    mode: Literal["cold", "hot"] = "cold",
) -> dict[str, Any]:
    """**TEST HELPER** — production writer ships in M2.

    Tars + zstds ``source_profile_dir``, uploads the archive +
    manifest, flips ``latest.json`` directly (no conditional PUT;
    M2 adds that). Returns the resulting manifest dict for
    assertion in tests.

    DO NOT call this from production worker code. It has no lock,
    no conditional pointer flip, no retry budget, no observability.
    Use it from tests + the one-off "populate a snapshot to validate
    the loader" workflow described in the spec's M1 acceptance
    criterion.
    """
    import zstandard as zstd  # lazy

    source_profile_dir = Path(source_profile_dir)
    if not source_profile_dir.is_dir():
        raise ValueError(
            f"source_profile_dir must be an existing directory: "
            f"{source_profile_dir}"
        )

    # 1. Build tarball into memory.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        # Add the directory CONTENTS, not the directory itself, so
        # extract → target produces the same shape source had.
        for path in sorted(source_profile_dir.rglob("*")):
            rel = path.relative_to(source_profile_dir)
            tf.add(path, arcname=str(rel))
    uncompressed_size = buf.tell()
    buf.seek(0)

    cctx = zstd.ZstdCompressor(level=9)
    compressed = cctx.compress(buf.getvalue())
    archive_sha256 = hashlib.sha256(compressed).hexdigest()
    sha_prefix = archive_sha256[:12]

    # 2. Manifest.
    manifest = {
        "version": 1,
        "schema": "computer-plane.profile-snapshot",
        "tenant_id": tenant_id,
        "profile_id": profile_id,
        "chrome_major_version": chrome_major,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": len(compressed),
        "uncompressed_size_bytes": uncompressed_size,
        "captured_at_ms": int(time.time() * 1000),
        "mode": mode,
        "chrome_uptime_seconds_at_capture": 0,
        "predecessor_sha256": "",
        "notes": "test-helper",
        "captured_by": {
            "host": "test",
            "writer_version": "snapshotter-0.1.0-m1-test-helper",
        },
        "captured_in": {},
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

    # 3. Upload archive + manifest.
    prefix = snapshotter._prefix(tenant_id, profile_id)  # noqa: SLF001
    archive_key = f"{prefix}profile-{sha_prefix}.tar.zst"
    manifest_key = f"{prefix}profile-{sha_prefix}.manifest.json"
    snapshotter._s3.put_object(  # noqa: SLF001
        Bucket=snapshotter._bucket, Key=archive_key, Body=compressed,
    )
    snapshotter._s3.put_object(  # noqa: SLF001
        Bucket=snapshotter._bucket, Key=manifest_key, Body=manifest_bytes,
        ContentType="application/json",
    )

    # 4. Pointer flip (raw PUT — M2 will replace with conditional PUT).
    pointer = {
        "version": 1,
        "active_sha256_prefix": sha_prefix,
        "active_archive_key": archive_key,
        "active_manifest_key": manifest_key,
        "flipped_at_ms": int(time.time() * 1000),
        "flipped_from_sha256_prefix": "",
    }
    snapshotter._s3.put_object(  # noqa: SLF001
        Bucket=snapshotter._bucket, Key=f"{prefix}latest.json",
        Body=json.dumps(pointer, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    return manifest


__all__ = [
    "CaptureOutcome",
    "CaptureResult",
    "LatestPointer",
    "LoadOutcome",
    "LoadResult",
    "ProfileSnapshotter",
    "SnapshotManifest",
    "capture_and_upload_for_testing",
]
