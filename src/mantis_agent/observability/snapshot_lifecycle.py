"""M3 brain wire-up — Phase 2 snapshot capture + per-profile lock around
the plan lifecycle.

This module is the brain-side companion to ``profile_snapshotter`` and
``profile_lock``. M1 wired the LOAD path into Modal's
``_chrome_profile_dir``; M3 adds the CAPTURE path at plan terminal
and the cross-host lock at submit time.

Every helper is env-gated on ``MANTIS_PROFILE_SNAPSHOT_BUCKET``: when
the bucket isn't configured, every entry point is a no-op. Production
behaviour stays exactly what it was before M1+M2 landed until an
operator flips the env block on.

Failure semantics — capture and lock release MUST NOT propagate
exceptions. They run alongside the existing Phase 1 file-lock + Modal
Volume write paths, which are independent of object-store
availability. The only thing M3 raises out of is a lock CONFLICT at
acquire — that's a real concurrent-writer signal the caller may want
to surface as a 409.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .profile_lock import _AcquiredLock

logger = logging.getLogger(__name__)


# Phase 2 lock TTL — generous to avoid renewal complexity for the
# first wire-up. The Phase 1 file lock has ``stale_after_seconds=4h``;
# we mirror that here so a crashed-without-release run gets reaped on
# the same cadence. The cross-host reaper (M2 ``ProfileLockReaper``)
# sweeps stale locks at this TTL + a 60s grace by default.
_DEFAULT_LOCK_TTL_SECONDS = 4 * 3600


def is_phase2_configured() -> bool:
    """True iff the snapshot env block is set.

    All M1/M2/M3 helpers short-circuit when this is False — production
    behaviour stays unchanged until an operator flips the bucket on
    in the Modal / Baseten secret.
    """
    return bool(os.environ.get("MANTIS_PROFILE_SNAPSHOT_BUCKET", "").strip())


def maybe_acquire_snapshot_lock(
    *,
    tenant_id: str,
    profile_id: str,
    run_id: str,
    holder_host: str = "",
    ttl_seconds: int = _DEFAULT_LOCK_TTL_SECONDS,
) -> Optional["_AcquiredLock"]:
    """Acquire the Phase 2 object-store lock for a run.

    Returns ``None`` when:

    * Phase 2 isn't configured (env-gate off).
    * The snapshot package isn't installed in this image (M1's import
      gate). The Phase 1 file lock continues to gate concurrent
      submissions on this host; the only loss is cross-host
      coordination, which M3 promises but didn't yet exist before.
    * The lock service is unreachable. Same fallback — log warning,
      let the Phase 1 lock carry the contract.

    Raises :class:`LockConflictError` ONLY when the lock is currently
    held by a DIFFERENT host. Caller may want to surface this as a
    409 — but for Modal-only deployments the Phase 1 file lock
    already caught same-host races, so a Phase 2 conflict here means
    "another backend (E2B / Daytona) is holding this profile."
    """
    if not is_phase2_configured():
        return None
    try:
        from .profile_lock import ProfileLockManager
        from .profile_snapshotter import ProfileSnapshotter
    except ImportError as exc:
        logger.warning(
            "snapshot_lifecycle: profile_lock import failed, "
            "skipping Phase 2 lock acquire: %s", exc,
        )
        return None
    try:
        snap = ProfileSnapshotter.from_env()
    except Exception as exc:  # noqa: BLE001 — env/network failure
        logger.warning(
            "snapshot_lifecycle: ProfileSnapshotter.from_env failed, "
            "skipping Phase 2 lock acquire tenant=%s profile=%s (%s)",
            tenant_id, profile_id, exc,
        )
        return None
    # Reuse the snapshotter's bucket + s3 client + chrome_major — the
    # lock blob lives in the same prefix as ``latest.json`` so it
    # discoverable via the same listing call the reaper uses.
    mgr = ProfileLockManager(
        bucket=snap._bucket,           # noqa: SLF001 — known reuse
        chrome_major=snap._chrome_major,  # noqa: SLF001
        s3_client=snap._s3,            # noqa: SLF001
        ttl_seconds=ttl_seconds,
    )
    try:
        return mgr.acquire(
            tenant_id=tenant_id,
            profile_id=profile_id,
            holder_run_id=run_id,
            holder_host=holder_host or _default_host_tag(),
        )
    except Exception as exc:  # noqa: BLE001
        # Includes LockConflictError. Re-raise the conflict so the
        # caller can branch on it; everything else degrades silently.
        from .profile_lock import LockConflictError
        if isinstance(exc, LockConflictError):
            raise
        logger.warning(
            "snapshot_lifecycle: lock acquire failed tenant=%s "
            "profile=%s run=%s (%s)", tenant_id, profile_id, run_id, exc,
        )
        return None


def maybe_release_snapshot_lock(
    lock: Optional["_AcquiredLock"],
) -> None:
    """Release the lock if one was acquired. No-op when ``None``.

    Use this from the SAME container that acquired the lock (the
    handle carries the ETag for CAS). For cross-container release
    (acquire on the API, release on the executor's terminal detector,
    or vice-versa) use :func:`maybe_force_release_lock_by_keys`.
    """
    if lock is None:
        return
    try:
        from .profile_lock import ProfileLockManager
        from .profile_snapshotter import ProfileSnapshotter
        snap = ProfileSnapshotter.from_env()
        mgr = ProfileLockManager(
            bucket=snap._bucket,           # noqa: SLF001
            chrome_major=snap._chrome_major,  # noqa: SLF001
            s3_client=snap._s3,            # noqa: SLF001
        )
        mgr.release(lock, quiet=True)
    except Exception as exc:  # noqa: BLE001 — release is best-effort
        logger.warning(
            "snapshot_lifecycle: lock release failed key=%s (%s)",
            getattr(lock, "key", "?"), exc,
        )


def maybe_force_release_lock_by_keys(
    *,
    tenant_id: str,
    profile_id: str,
) -> None:
    """Unconditionally delete the lock blob for ``(tenant, profile)``.

    The API container acquires the lock at submit and releases at
    terminal — but acquire and release happen on different requests
    (potentially on different replicas). Threading the
    :class:`_AcquiredLock` handle across that boundary requires
    persisting the ETag somewhere; this helper sidesteps that by
    just deleting the key.

    Use ONLY from the container that owns the API surface for the
    profile (i.e. the controller that just wrote the terminal status).
    The cross-host reaper will catch any release we drop on the floor
    via the TTL anyway.
    """
    if not is_phase2_configured():
        return
    try:
        from .profile_snapshotter import ProfileSnapshotter, _safe
        snap = ProfileSnapshotter.from_env()
        key = (
            f"snapshots/{_safe(tenant_id)}/{_safe(profile_id)}/"
            f"{snap._chrome_major}/lock.json"  # noqa: SLF001
        )
        snap._s3.delete_object(  # noqa: SLF001
            Bucket=snap._bucket, Key=key,  # noqa: SLF001
        )
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.warning(
            "snapshot_lifecycle: force-release failed tenant=%s "
            "profile=%s (%s)", tenant_id, profile_id, exc,
        )


def maybe_capture_snapshot(
    *,
    tenant_id: str,
    profile_id: str,
    source_profile_dir: Any,  # str or Path
    notes: str = "",
    captured_in: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Capture the post-run profile dir into the snapshot bucket.

    Fired at plan terminal — Chrome has already been closed by the
    executor, the user-data-dir is in a clean cold state. Capture is
    cold-mode (the only mode M2 ships); hot mode is M3's remaining
    follow-up.

    Returns a structured summary dict (with ``outcome`` ∈ {captured /
    deduplicated / too_large / empty_source / upload_failed /
    pointer_race / skipped}) so callers can log it. Returns ``None``
    when Phase 2 isn't configured.

    Never raises — the M2 writer already converts every failure path
    into a structured ``CaptureResult``, and this wrapper degrades any
    unexpected exception to a WARNING + ``skipped`` outcome.
    """
    if not is_phase2_configured():
        return None
    src = Path(source_profile_dir) if source_profile_dir else None
    if src is None or not src.is_dir():
        # Nothing to capture (Chrome never booted, or the dir was
        # already cleaned up). Log at DEBUG so it doesn't flood Modal
        # logs on every run that runs against a fresh profile.
        logger.debug(
            "snapshot_lifecycle: capture skipped — no source dir "
            "tenant=%s profile=%s src=%s",
            tenant_id, profile_id, src,
        )
        return {"outcome": "skipped", "reason": "no_source_dir"}
    try:
        from .profile_snapshotter import ProfileSnapshotter
    except ImportError as exc:
        logger.warning(
            "snapshot_lifecycle: profile_snapshotter import failed, "
            "skipping capture: %s", exc,
        )
        return {"outcome": "skipped", "reason": f"import_failed:{exc}"}
    try:
        snap = ProfileSnapshotter.from_env()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "snapshot_lifecycle: from_env failed tenant=%s "
            "profile=%s (%s)", tenant_id, profile_id, exc,
        )
        return {"outcome": "skipped", "reason": f"init_failed:{exc}"}
    try:
        result = snap.capture(
            tenant_id=tenant_id,
            profile_id=profile_id,
            source_profile_dir=src,
            mode="cold",
            notes=notes,
            captured_in=captured_in or {},
        )
    except NotImplementedError:
        # Hot mode wasn't enabled, OR the writer raised on an unknown
        # mode kwarg. The cold path shouldn't raise NotImplementedError;
        # surface as skipped so the run isn't disrupted.
        logger.warning(
            "snapshot_lifecycle: capture NotImplementedError "
            "tenant=%s profile=%s — treating as skipped",
            tenant_id, profile_id,
        )
        return {"outcome": "skipped", "reason": "not_implemented"}
    except Exception as exc:  # noqa: BLE001 — never break terminal
        logger.warning(
            "snapshot_lifecycle: capture raised tenant=%s "
            "profile=%s (%s)", tenant_id, profile_id, exc,
        )
        return {"outcome": "skipped", "reason": f"raised:{exc}"}

    summary = {
        "outcome": result.outcome,
        "reason": result.reason,
        "archive_sha256": result.archive_sha256,
        "archive_size_bytes": result.archive_size_bytes,
        "uncompressed_size_bytes": result.uncompressed_size_bytes,
        "predecessor_sha256": result.predecessor_sha256,
        "elapsed_seconds": result.elapsed_seconds,
    }
    # WARNING-level so the outcome is visible in Modal app logs
    # (feedback_warning_level_for_modal_observability).
    logger.warning(
        "profile_snapshot capture tenant=%s profile=%s outcome=%s "
        "size=%dB elapsed=%.2fs",
        tenant_id, profile_id, result.outcome,
        result.archive_size_bytes, result.elapsed_seconds,
    )
    return summary


# ── helpers ───────────────────────────────────────────────────────────


def _default_host_tag() -> str:
    """Best-effort host identifier for the lock blob's ``holder_host``.

    Uses ``MANTIS_HOST`` when set, else falls back to ``modal`` (the
    only host that today honours this env block). E2B / Daytona
    deploys should set ``MANTIS_HOST`` in their secret so cross-host
    conflicts surface a useful holder identity.
    """
    return (os.environ.get("MANTIS_HOST") or "").strip() or "modal"


__all__ = [
    "is_phase2_configured",
    "maybe_acquire_snapshot_lock",
    "maybe_release_snapshot_lock",
    "maybe_force_release_lock_by_keys",
    "maybe_capture_snapshot",
]
