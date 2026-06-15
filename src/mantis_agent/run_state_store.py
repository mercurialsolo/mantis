"""Cross-replica run-state store for small hot JSON sidecars.

The Modal CUA API runs as an ASGI ``@app.function(...)`` with no
``max_containers=1`` pin, so submissions can land on container A while
the poll for the same ``run_id`` lands on container B. The sidecar
files used by ``_do_action`` — ``status.json``, ``viewer.json``,
``augur.json``, ``pause_request.json`` — were originally stored only on
the ``/data`` Modal Volume. Volume writes are durable but propagation
across replicas is eventually consistent: a poll inside that window
legitimately sees ``None`` and 404s the run_id.

This module introduces a small key-value store that sits in front of
the disk sidecars and is shared across replicas at write time. In
production it is backed by a ``modal.Dict``; in tests (or off-Modal
contexts) a plain ``dict`` works the same way.

Layout — one entry per ``(tenant_id, run_id, kind)``::

    "<tenant>/<run_id>/<kind>"  →  {<json-blob>}

The store is a *cache* layered on top of disk: writes go to both the
backing object and disk (via the caller-supplied disk writer), so
unrelated code paths that still read the file directly (the queue scan
in ``modal_cua_server.get_queue``, the lifecycle phase response, etc.)
keep working. Reads consult the cache first; missing keys fall back to
the disk read.

The cache is bounded so a runaway tenant cannot exhaust container
memory or the modal.Dict shard. ``put`` evicts the oldest ~25% when
the in-process LRU mirror crosses ``max_entries`` so we drop entries
in batches rather than on every write.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


# Sidecar kinds — keep the set explicit so a typo doesn't silently
# create a new namespace.
KIND_STATUS = "status"
KIND_VIEWER = "viewer"
KIND_AUGUR = "augur"
KIND_PAUSE_REQUEST = "pause_request"
# Phase 1.5 (#846): per-session computer-plane record. Keyed by the
# router-minted ``session_id``, not the ``run_id`` — one run can
# theoretically open and close multiple sessions (e.g. reaper kills
# one and the brain transparently re-creates), and we want both
# records discoverable for forensics.
KIND_SESSION = "session"
# mantis-server pause/resume (#909-followup): the paused run's resolved
# micro-suite + original payload, mirrored off the /data Volume so a resume
# landing on a different replica reads them immediately (no eventual-consistency
# miss → no re-decompose / "pause_state missing").
KIND_PAUSE_STATE = "pause_state"
KIND_RESUME_PAYLOAD = "resume_payload"

_VALID_KINDS = frozenset({
    KIND_STATUS, KIND_VIEWER, KIND_AUGUR, KIND_PAUSE_REQUEST, KIND_SESSION,
    KIND_PAUSE_STATE, KIND_RESUME_PAYLOAD,
})


def _safe(value: str) -> str:
    """Filesystem-safe scope segment. Mirrors ``server_utils.safe_state_key``
    without importing FastAPI machinery."""
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in value)


def _entry_key(tenant_id: str, run_id: str, kind: str) -> str:
    if kind not in _VALID_KINDS:
        raise ValueError(f"unknown run-state kind: {kind!r}")
    return f"{_safe(tenant_id)}/{_safe(run_id)}/{kind}"


class RunStateStore:
    """Dependency-injected backing for the run-state cache.

    The backing object only needs ``get``/``__setitem__``/``__delitem__``
    plus iteration over ``keys()``. ``modal.Dict`` satisfies this in
    prod; a plain ``dict`` works for tests. We never call ``len()`` on
    the backing — ``modal.Dict`` raises ``TypeError`` on that
    (see ``feedback_modal_dict_no_len``).
    """

    def __init__(self, backing: Any, *, max_entries: int = 4096) -> None:
        self._d = backing
        # Per-container LRU mirror — bounds growth without iterating
        # the backing object on every write.
        self._lru: "OrderedDict[str, None]" = OrderedDict()
        self.max_entries = max(64, int(max_entries))

    @classmethod
    def from_name(cls, dict_name: str, *, create_if_missing: bool = True,
                  max_entries: int = 4096) -> "RunStateStore":
        """Build against a named ``modal.Dict``. ``modal`` is imported
        lazily so this module stays importable off-Modal (tests use the
        plain-``dict`` constructor instead)."""
        import modal  # noqa: PLC0415 — lazy so the module imports without modal

        backing = modal.Dict.from_name(dict_name, create_if_missing=create_if_missing)
        return cls(backing, max_entries=max_entries)

    # ── reads ──

    def get(self, tenant_id: str, run_id: str, kind: str) -> Optional[dict]:
        key = _entry_key(tenant_id, run_id, kind)
        try:
            raw = self._d.get(key)
        except Exception as exc:  # noqa: BLE001 — store fault ≠ run fault
            logger.warning("RunStateStore.get failed key=%s (%s)", key, exc)
            return None
        if raw is None:
            return None
        if not isinstance(raw, dict):
            logger.warning("RunStateStore.get non-dict value key=%s type=%s",
                           key, type(raw).__name__)
            return None
        return raw

    # ── writes ──

    def put(self, tenant_id: str, run_id: str, kind: str, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError(
                f"RunStateStore.put value must be a dict, got {type(value).__name__}"
            )
        key = _entry_key(tenant_id, run_id, kind)
        try:
            self._d[key] = dict(value)  # defensive copy — backing may not
        except Exception as exc:  # noqa: BLE001 — store fault ≠ run fault
            logger.warning("RunStateStore.put failed key=%s (%s)", key, exc)
            return
        self._touch(key)

    def delete(self, tenant_id: str, run_id: str, kind: str) -> None:
        key = _entry_key(tenant_id, run_id, kind)
        try:
            del self._d[key]
        except KeyError:
            pass
        except Exception as exc:  # noqa: BLE001 — store fault ≠ run fault
            logger.warning("RunStateStore.delete failed key=%s (%s)", key, exc)
        self._lru.pop(key, None)

    # ── LRU bookkeeping ──

    def _touch(self, key: str) -> None:
        self._lru.pop(key, None)
        self._lru[key] = None
        if len(self._lru) > self.max_entries:
            drop = max(1, self.max_entries // 4)
            for _ in range(drop):
                try:
                    evicted, _ = self._lru.popitem(last=False)
                except KeyError:
                    break
                try:
                    del self._d[evicted]
                except KeyError:
                    pass
                except Exception:  # noqa: BLE001 — best-effort eviction
                    pass


# ── No-op store for environments without a backing object ──


class NullRunStateStore:
    """Fallback when no shared backing is configured.

    Always misses on read, drops on write. Lets the disk fallback do
    all the work without forcing callers to branch on store presence.
    """

    def get(self, tenant_id: str, run_id: str, kind: str) -> Optional[dict]:
        return None

    def put(self, tenant_id: str, run_id: str, kind: str, value: dict) -> None:
        return None

    def delete(self, tenant_id: str, run_id: str, kind: str) -> None:
        return None


# ── Helpers for the disk-mirrored read path ──


def read_with_store(
    store: Any,
    *,
    tenant_id: str,
    run_id: str,
    kind: str,
    disk_reader: Callable[[], Optional[dict]],
) -> Optional[dict]:
    """Cache-first read; fall back to ``disk_reader`` on miss.

    On a disk hit we backfill the cache so subsequent reads on the
    same replica skip the file read. Disk reads must already validate
    JSON shape; the helper just forwards their return value.
    """
    cached = store.get(tenant_id, run_id, kind)
    if cached is not None:
        return cached
    on_disk = disk_reader()
    if on_disk is not None:
        try:
            store.put(tenant_id, run_id, kind, on_disk)
        except Exception:  # noqa: BLE001 — backfill is best-effort
            pass
    return on_disk


def list_active_keys(store: Any) -> Iterable[str]:
    """Iterate the store's keys (useful in tests/diagnostics).

    Lazy iteration — caller can break early. We catch the typical
    store-fault and yield nothing rather than propagating.
    """
    try:
        keys = store._d.keys()  # noqa: SLF001 — diagnostic helper
    except Exception:  # noqa: BLE001
        return
    for k in keys:
        yield k
