"""Tier 2 idempotency-key cache.

When a caller sends the same ``Idempotency-Key`` header twice for the same
tenant, the server returns the original ``run_id`` instead of starting a
new run. Useful for retried POSTs where the network failed but the run
was actually accepted.

Storage: per-process dict + JSON sidecar in the data volume so a replica
restart preserves the cache. The sidecar layout is:

    $MANTIS_DATA_DIR/idempotency/<tenant_id>/<key_hash>.json

Each entry expires after ``DEFAULT_TTL_SECONDS`` (24h). Expired entries
are pruned lazily on read.

For multi-replica strict idempotency, swap this for a Redis KV
backed by SET NX EX semantics. The interface (``get`` / ``store``) is
designed to make that swap a one-file change.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("mantis_agent.idempotency")

DEFAULT_TTL_SECONDS = 24 * 60 * 60


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class CachedRun:
    run_id: str
    response: dict[str, Any]
    stored_at: float

    def is_fresh(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> bool:
        return (time.time() - self.stored_at) < ttl_seconds


class IdempotencyCache:
    """Per-tenant idempotency-key store, keyed by SHA-256 of the user key."""

    def __init__(self, root_dir: Path | str | None = None) -> None:
        if root_dir is None:
            root_dir = os.environ.get("MANTIS_IDEMPOTENCY_DIR")
        if root_dir is None:
            data = Path(os.environ.get("MANTIS_DATA_DIR", "/workspace/mantis-data"))
            root_dir = data / "idempotency"
        self._root = Path(root_dir)
        self._lock = threading.Lock()
        self._mem: dict[tuple[str, str], CachedRun] = {}

    def _path_for(self, tenant_id: str, key_hash: str) -> Path:
        d = self._root / tenant_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key_hash}.json"

    def get(self, tenant_id: str, key: str) -> CachedRun | None:
        if not key:
            return None
        kh = _hash_key(key)
        cache_key = (tenant_id, kh)
        with self._lock:
            cached = self._mem.get(cache_key)
            if cached is not None and cached.is_fresh():
                return cached
            # Try sidecar
            path = self._path_for(tenant_id, kh)
            if not path.exists():
                return None
            try:
                raw = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                return None
            cached = CachedRun(
                run_id=raw.get("run_id", ""),
                response=raw.get("response") or {},
                stored_at=float(raw.get("stored_at", 0.0)),
            )
            if not cached.is_fresh():
                # Prune lazily
                try:
                    path.unlink()
                except OSError:
                    pass
                self._mem.pop(cache_key, None)
                return None
            self._mem[cache_key] = cached
            return cached

    def store(self, tenant_id: str, key: str, run_id: str, response: dict[str, Any]) -> None:
        if not key or not run_id:
            return
        kh = _hash_key(key)
        cache_key = (tenant_id, kh)
        cached = CachedRun(run_id=run_id, response=response, stored_at=time.time())
        with self._lock:
            self._mem[cache_key] = cached
            try:
                self._path_for(tenant_id, kh).write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "response": response,
                            "stored_at": cached.stored_at,
                        }
                    )
                )
            except OSError as exc:
                logger.warning(
                    "idempotency: failed to write sidecar for tenant=%s key=%s: %s",
                    tenant_id,
                    kh,
                    exc,
                )


# Module-level singleton.
_CACHE: IdempotencyCache | None = None


def get_idempotency_cache() -> IdempotencyCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = IdempotencyCache()
    return _CACHE


def reset_idempotency_cache() -> None:
    """Test helper."""
    global _CACHE
    _CACHE = None
