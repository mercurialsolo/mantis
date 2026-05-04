"""Per-tenant extraction cache.

Persists successful ``extract_data`` results keyed by URL so subsequent
runs (or subsequent loop iterations within one run) can short-circuit the
expensive deep-extract Claude call when an already-extracted URL comes
back into view.

Two operating modes, controlled per-request via ``cache_read`` and
``cache_write`` flags on ``PredictRequest``:

- **read** — at run start, populate the runner's seen-URL set from the
  cache; in the ``extract_data`` step, peek ``env.current_url`` BEFORE
  the Claude deep-extract. On hit, emit the cached lead summary as a
  successful extraction without spending any tokens.
- **write** — on every viable extraction, store the (url → summary +
  fields) entry in the cache. Persists at end of run.

Cache file layout::

    <data_root>/tenants/<tenant_id>/cache/<cache_key>.json

with shape::

    {
      "version": 1,
      "entries": {
        "<url>": {
          "summary":       "VIABLE | Title: ... | ...",
          "extracted_at":  "2026-05-04T18:30:00+00:00",
          "fields":        { /* extracted_fields dict, optional */ }
        }
      }
    }

Entries older than ``cache_ttl_seconds`` are treated as stale and
re-extracted (and the entry overwritten). TTL is enforced at lookup
time, not at load — stale entries linger on disk until a write
overwrites them.

Concurrency: writes happen at end-of-run on a single thread; reads
during a run are also single-threaded (the runner serializes step
execution). No locking needed within one container.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("mantis_agent.extraction.cache")


CACHE_VERSION = 1


@dataclass
class CacheEntry:
    """One cached extraction. Persisted as a JSON object under the URL key."""

    summary: str
    extracted_at: str  # ISO 8601 UTC
    fields: dict[str, str] = field(default_factory=dict)
    item_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "extracted_at": self.extracted_at,
            "fields": dict(self.fields),
            "item_label": self.item_label,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CacheEntry:
        return cls(
            summary=str(raw.get("summary") or ""),
            extracted_at=str(raw.get("extracted_at") or ""),
            fields={str(k): str(v) for k, v in (raw.get("fields") or {}).items()},
            item_label=str(raw.get("item_label") or ""),
        )

    def is_fresh(self, ttl_seconds: int, *, now: datetime | None = None) -> bool:
        if ttl_seconds <= 0:
            return True
        if not self.extracted_at:
            return False
        try:
            ts = datetime.fromisoformat(self.extracted_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        cutoff = (now or datetime.now(timezone.utc)) - _seconds(ttl_seconds)
        return ts >= cutoff


def _seconds(n: int):
    from datetime import timedelta
    return timedelta(seconds=int(n))


class ExtractionCache:
    """File-backed extraction cache, scoped to (tenant_id, cache_key).

    Constructed by the server when a request opts into caching. The
    runner consults it via ``get`` before extraction and ``put`` after
    a viable extraction; ``save`` is called once at end of run.

    Disabled cache (read=False AND write=False) is represented by NOT
    constructing one; callers should pass ``None`` to the runner instead.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        read_enabled: bool = True,
        write_enabled: bool = True,
        ttl_seconds: int = 86400,
    ) -> None:
        self.path = Path(path)
        self.read_enabled = bool(read_enabled)
        self.write_enabled = bool(write_enabled)
        self.ttl_seconds = max(0, int(ttl_seconds))
        self._entries: dict[str, CacheEntry] = {}
        self._dirty = False
        self._lock = threading.Lock()
        self._load()

    # ── persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("extraction cache unreadable at %s: %s", self.path, exc)
            return
        if not isinstance(data, dict):
            return
        entries = data.get("entries") or {}
        if not isinstance(entries, dict):
            return
        for url, raw in entries.items():
            if not isinstance(raw, dict):
                continue
            self._entries[str(url)] = CacheEntry.from_dict(raw)
        logger.info(
            "extraction cache loaded: %d entries from %s", len(self._entries), self.path,
        )

    def save(self) -> None:
        if not self._dirty:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": CACHE_VERSION,
                "entries": {url: e.to_dict() for url, e in self._entries.items()},
            }
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self.path)
            self._dirty = False
        logger.info(
            "extraction cache saved: %d entries to %s",
            len(self._entries), self.path,
        )

    # ── lookup / mutation ─────────────────────────────────────────────

    def get(self, url: str) -> CacheEntry | None:
        """Return a fresh cached entry for ``url``, or ``None``.

        Stale entries (older than ``ttl_seconds``) are treated as misses
        but are NOT evicted — they get overwritten on the next ``put``.
        """
        if not self.read_enabled or not url:
            return None
        entry = self._entries.get(url)
        if entry is None:
            return None
        if not entry.is_fresh(self.ttl_seconds):
            return None
        return entry

    def put(
        self,
        url: str,
        summary: str,
        *,
        fields: dict[str, str] | None = None,
        item_label: str = "",
    ) -> None:
        """Record a viable extraction. No-op when write is disabled."""
        if not self.write_enabled or not url:
            return
        entry = CacheEntry(
            summary=summary,
            extracted_at=datetime.now(timezone.utc).isoformat(),
            fields=dict(fields or {}),
            item_label=item_label,
        )
        with self._lock:
            self._entries[url] = entry
            self._dirty = True

    def known_urls(self) -> list[str]:
        """Return URLs that currently have a fresh entry.

        Useful at run start to seed the runner's ``_seen_urls`` set so
        the deep-extract dedup short-circuits even before a cache check
        on the URL itself fires.
        """
        return [url for url, e in self._entries.items() if e.is_fresh(self.ttl_seconds)]

    # ── stats ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, Any]:
        fresh = sum(1 for e in self._entries.values() if e.is_fresh(self.ttl_seconds))
        return {
            "total": len(self._entries),
            "fresh": fresh,
            "stale": len(self._entries) - fresh,
            "ttl_seconds": self.ttl_seconds,
            "read_enabled": self.read_enabled,
            "write_enabled": self.write_enabled,
            "path": str(self.path),
        }
