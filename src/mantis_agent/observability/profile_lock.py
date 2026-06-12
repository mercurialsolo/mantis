"""Per-profile lock manager for Phase 2 (#699 spec § 5).

Phase 1's profile lock (#342) is a file on the Modal Volume. It dies
with the executor that wrote it and isn't reachable from a Phase 2
host backend (E2B, Daytona) that doesn't mount the volume. The Phase
2 lock lives in the object store next to the snapshot pointer.

Design choices the spec § 5 made (recap):

* **Object-store CAS**, not Redis / Modal Dict — every Phase 2 host
  already has S3 credentials for the snapshot bucket. No new
  operational dependency.
* **5-minute TTL, 60s renewal interval.** Holder renews every minute;
  reaper sweeps every 5 minutes; grace period 60s before forced
  eviction. These are tunable per tenant via constructor kwargs.
* **Lock object next to the snapshot pointer** — same prefix as
  ``latest.json``. Discoverable via the same listing as snapshots.

This module ships:

* :class:`ProfileLock` — the lock blob schema + helpers.
* :class:`ProfileLockManager` — acquire / renew / release.
* :class:`ProfileLockReaper` — sweep / evict orphaned locks.

The manager and reaper share an S3 client. The brain's hot path
constructs a manager; a scheduled function (Modal cron) constructs a
reaper and calls ``sweep()``.

Failure semantics:

* Lock conflict (someone else holds it) → :class:`LockConflictError`
  carrying the holder's identity for caller visibility.
* CAS retries exhausted on renew → :class:`LockLostError` — the
  caller MUST treat the run as dirty (its lock was stolen).
* Object-store errors during release → log warning, don't raise.
  Release is best-effort; the reaper TTL handles a missed release.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Defaults sourced from spec § 5.
_DEFAULT_TTL_SECONDS = 300            # 5 min
_DEFAULT_RENEWAL_SECONDS = 60         # 60 s
_DEFAULT_REAPER_GRACE_SECONDS = 60    # extra slack before forced evict


class ProfileLockBlob(BaseModel):
    """Schema for ``lock.json`` — spec § 5."""

    model_config = {"extra": "allow"}

    version: int = Field(..., ge=1)
    holder_run_id: str = Field(..., min_length=1)
    holder_host: str = ""
    holder_host_run_id: str = ""
    acquired_at_ms: int = Field(..., ge=0)
    renewed_at_ms: int = Field(..., ge=0)
    expires_at_ms: int = Field(..., ge=0)
    renewal_count: int = 0


# ── Errors ────────────────────────────────────────────────────────────


class LockConflictError(Exception):
    """Lock is held by someone else AND still within its TTL."""

    def __init__(self, blob: ProfileLockBlob) -> None:
        super().__init__(
            f"profile lock held by run_id={blob.holder_run_id!r} "
            f"expires_at_ms={blob.expires_at_ms}"
        )
        self.blob = blob


class LockLostError(Exception):
    """Renewal CAS failed — someone else stole the lock.

    The holder MUST treat its run as dirty (the next acquire on the
    profile will pay the snapshot re-load cost; the in-flight changes
    against the profile may have been clobbered by whoever stole the
    lock).
    """


# ── Lock handle ───────────────────────────────────────────────────────


@dataclass
class _AcquiredLock:
    """Internal handle the manager hands back from acquire().

    Carries the ETag of the lock we own so subsequent renews + releases
    can compare-and-swap against it.
    """

    blob: ProfileLockBlob
    etag: str
    bucket: str
    key: str


# ── Manager ───────────────────────────────────────────────────────────


class ProfileLockManager:
    """Acquire / renew / release a per-profile lock.

    Constructed once per (tenant, profile, run) — the lock is bound to
    a ``run_id`` so two concurrent runs against the same profile see
    the holder identity on the conflict path.

    Construction is DI-friendly: pass any boto3-compatible S3 client.
    Unit tests pass a fake; prod passes a real boto3 client.
    """

    def __init__(
        self,
        *,
        bucket: str,
        chrome_major: int,
        s3_client: Any,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        renewal_interval_seconds: int = _DEFAULT_RENEWAL_SECONDS,
        clock: Any = None,
    ) -> None:
        if not bucket:
            raise ValueError("ProfileLockManager requires a non-empty bucket")
        if chrome_major <= 0:
            raise ValueError(
                f"chrome_major must be a positive int, got {chrome_major!r}"
            )
        if s3_client is None:
            raise ValueError("ProfileLockManager requires an S3 client")
        self._bucket = bucket
        self._chrome_major = int(chrome_major)
        self._s3 = s3_client
        self._ttl_seconds = int(ttl_seconds)
        self._renewal_seconds = int(renewal_interval_seconds)
        self._clock = clock or time.time

    # ── lifecycle ──

    def acquire(
        self,
        *,
        tenant_id: str,
        profile_id: str,
        holder_run_id: str,
        holder_host: str = "",
        holder_host_run_id: str = "",
    ) -> _AcquiredLock:
        """Take the lock. Raises :class:`LockConflictError` when held."""
        key = self._lock_key(tenant_id, profile_id)
        existing, existing_etag = self._read(key)
        now_ms = int(self._clock() * 1000)
        if existing is not None and existing.expires_at_ms > now_ms:
            raise LockConflictError(existing)

        # Either no lock, or expired. Build our own blob.
        blob = ProfileLockBlob(
            version=1,
            holder_run_id=holder_run_id,
            holder_host=holder_host,
            holder_host_run_id=holder_host_run_id,
            acquired_at_ms=now_ms,
            renewed_at_ms=now_ms,
            expires_at_ms=now_ms + self._ttl_seconds * 1000,
            renewal_count=0,
        )
        new_etag = self._write(
            key, blob, expected_etag=existing_etag,
        )
        if new_etag is None:
            # CAS conflict — someone else took it between read and write.
            # Re-read to surface the holder.
            current, _ = self._read(key)
            if current is None or current.expires_at_ms <= now_ms:
                # The lock disappeared / expired by the time we re-read.
                # Don't loop forever — surface as conflict with a
                # synthetic blob so the caller knows it has to retry.
                raise LockConflictError(
                    ProfileLockBlob(
                        version=1, holder_run_id="<unknown>",
                        acquired_at_ms=0, renewed_at_ms=0,
                        expires_at_ms=0, renewal_count=0,
                    )
                )
            raise LockConflictError(current)
        return _AcquiredLock(blob=blob, etag=new_etag, bucket=self._bucket, key=key)

    def renew(self, lock: _AcquiredLock) -> _AcquiredLock:
        """Extend the TTL. Raises :class:`LockLostError` if someone
        stole the lock between our last write and now."""
        now_ms = int(self._clock() * 1000)
        new_blob = ProfileLockBlob(
            **{
                **lock.blob.model_dump(),
                "renewed_at_ms": now_ms,
                "expires_at_ms": now_ms + self._ttl_seconds * 1000,
                "renewal_count": lock.blob.renewal_count + 1,
            }
        )
        new_etag = self._write(
            lock.key, new_blob, expected_etag=lock.etag,
        )
        if new_etag is None:
            raise LockLostError(
                f"renewal CAS failed for lock key={lock.key}; "
                f"holder_run_id={lock.blob.holder_run_id} renewal_count="
                f"{lock.blob.renewal_count}"
            )
        return _AcquiredLock(
            blob=new_blob, etag=new_etag,
            bucket=lock.bucket, key=lock.key,
        )

    def release(self, lock: _AcquiredLock, *, quiet: bool = True) -> bool:
        """Delete the lock. ``quiet=True`` (default) swallows errors —
        the reaper handles missed releases via TTL.

        Returns True iff the delete actually fired; False on error
        when ``quiet=True``.
        """
        try:
            self._s3.delete_object(Bucket=lock.bucket, Key=lock.key)
            return True
        except Exception as exc:  # noqa: BLE001
            if quiet:
                logger.warning(
                    "profile lock: release failed (best-effort) "
                    "key=%s holder=%s (%s)",
                    lock.key, lock.blob.holder_run_id, exc,
                )
                return False
            raise

    # ── plumbing ──

    def _lock_key(self, tenant_id: str, profile_id: str) -> str:
        from .profile_snapshotter import _safe
        return (
            f"snapshots/{_safe(tenant_id)}/{_safe(profile_id)}/"
            f"{self._chrome_major}/lock.json"
        )

    def _read(self, key: str) -> tuple[Optional[ProfileLockBlob], str]:
        """Return (blob, etag). (None, "") if the key doesn't exist."""
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            from .profile_snapshotter import _is_not_found
            if _is_not_found(exc):
                return None, ""
            logger.warning(
                "profile lock: read failed key=%s (%s)", key, exc,
            )
            return None, ""
        etag = (resp.get("ETag") or "").strip('"')
        try:
            body = resp["Body"].read()
            blob = ProfileLockBlob.model_validate_json(body)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "profile lock: parse failed key=%s (%s)", key, exc,
            )
            return None, etag
        return blob, etag

    def _write(
        self, key: str, blob: ProfileLockBlob, *, expected_etag: str,
    ) -> Optional[str]:
        """Conditional PUT. Returns the new ETag on success; None on
        CAS conflict OR transient failure (caller treats as
        conflict)."""
        kwargs: dict[str, Any] = {
            "Bucket": self._bucket,
            "Key": key,
            "Body": blob.model_dump_json(indent=2).encode("utf-8"),
            "ContentType": "application/json",
        }
        if expected_etag:
            kwargs["IfMatch"] = expected_etag
        else:
            kwargs["IfNoneMatch"] = "*"
        try:
            resp = self._s3.put_object(**kwargs)
        except Exception as exc:  # noqa: BLE001
            from .profile_snapshotter import _is_cas_conflict
            if _is_cas_conflict(exc):
                return None
            logger.warning(
                "profile lock: write raised non-CAS error key=%s (%s)",
                key, exc,
            )
            return None
        return (resp.get("ETag") or "").strip('"')


# ── Reaper ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReapDecision:
    """One reaper sweep candidate."""

    bucket: str
    key: str
    blob: ProfileLockBlob
    reason: str  # "ttl_expired" | "grace_expired"


class ProfileLockReaper:
    """Sweep orphaned locks across all tenants.

    Run as a scheduled function (Modal cron, every 5 minutes per spec).
    Lists every ``lock.json`` under ``snapshots/`` and forcibly deletes
    those past ``expires_at_ms + grace_seconds``.

    Never terminates running workloads — only releases the lock. The
    orphaned holder, if any, discovers the eviction on its next
    renewal attempt (renew → CAS fail → LockLostError).
    """

    def __init__(
        self,
        *,
        bucket: str,
        s3_client: Any,
        grace_seconds: int = _DEFAULT_REAPER_GRACE_SECONDS,
        clock: Any = None,
    ) -> None:
        if not bucket:
            raise ValueError("ProfileLockReaper requires a non-empty bucket")
        if s3_client is None:
            raise ValueError("ProfileLockReaper requires an S3 client")
        self._bucket = bucket
        self._s3 = s3_client
        self._grace_seconds = int(grace_seconds)
        self._clock = clock or time.time

    # ── api ──

    def find(self, *, prefix: str = "snapshots/") -> list[ReapDecision]:
        """List ``lock.json`` files and return those past
        ``expires_at_ms + grace_seconds``.

        ``prefix`` is configurable so tests can scope the sweep to a
        unique tenant prefix.
        """
        now_ms = int(self._clock() * 1000)
        deadline = now_ms - self._grace_seconds * 1000

        decisions: list[ReapDecision] = []
        for key in self._list_lock_keys(prefix=prefix):
            blob, _ = self._read(key)
            if blob is None:
                # Either missing (deleted between list+read) or corrupt.
                # Corrupt locks aren't reaped — operator-side decision.
                continue
            if blob.expires_at_ms < deadline:
                decisions.append(ReapDecision(
                    bucket=self._bucket, key=key, blob=blob,
                    reason="grace_expired",
                ))
        return decisions

    def apply(self, decisions: Iterable[ReapDecision]) -> dict[str, int]:
        """Delete the named locks. Returns summary counts.

        Per-decision failures are best-effort: logged + skipped, never
        raised. The next sweep retries.
        """
        evicted = 0
        skipped = 0
        for d in decisions:
            try:
                self._s3.delete_object(Bucket=d.bucket, Key=d.key)
                evicted += 1
                logger.warning(
                    "profile lock reaper: evicted holder=%s key=%s "
                    "reason=%s",
                    d.blob.holder_run_id, d.key, d.reason,
                )
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                logger.warning(
                    "profile lock reaper: delete failed key=%s (%s)",
                    d.key, exc,
                )
        return {"evicted": evicted, "skipped": skipped}

    def sweep(self, *, prefix: str = "snapshots/") -> dict[str, int]:
        """Convenience — find + apply in one call."""
        decisions = self.find(prefix=prefix)
        result = self.apply(decisions)
        result["scanned"] = len(decisions)
        return result

    # ── plumbing ──

    def _list_lock_keys(self, *, prefix: str) -> list[str]:
        keys: list[str] = []
        try:
            paginator = self._s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(
                Bucket=self._bucket, Prefix=prefix,
            ):
                for obj in page.get("Contents", []) or []:
                    key = obj.get("Key", "")
                    if key.endswith("/lock.json"):
                        keys.append(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "profile lock reaper: list failed prefix=%s (%s)",
                prefix, exc,
            )
        return keys

    def _read(self, key: str) -> tuple[Optional[ProfileLockBlob], str]:
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            from .profile_snapshotter import _is_not_found
            if _is_not_found(exc):
                return None, ""
            logger.warning(
                "profile lock reaper: read failed key=%s (%s)", key, exc,
            )
            return None, ""
        etag = (resp.get("ETag") or "").strip('"')
        try:
            body = resp["Body"].read()
            blob = ProfileLockBlob.model_validate_json(body)
            return blob, etag
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "profile lock reaper: parse failed key=%s (%s)", key, exc,
            )
            return None, etag


# ── helpers ───────────────────────────────────────────────────────────


def new_holder_run_id() -> str:
    """Convenience — generate a unique holder id for tests and
    interactive use. Production callers should pass an explicit
    ``run_id`` (e.g. the API-side run_id)."""
    return f"holder-{uuid.uuid4().hex[:12]}"


__all__ = [
    "LockConflictError",
    "LockLostError",
    "ProfileLockBlob",
    "ProfileLockManager",
    "ProfileLockReaper",
    "ReapDecision",
    "new_holder_run_id",
]
