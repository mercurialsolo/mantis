"""Coordinate cache for the grounding pipeline (#117, step 1).

Each :class:`ClaudeGrounding` call takes ~5-10s and costs ~$0.003 — and
listing-card layouts repeat across pages, so the same visual region +
description pair gets re-grounded constantly. This cache keys results by
``(perceptual_hash(crop) + description_hash)`` so a hit short-circuits
the whole network round-trip.

Design choices:

* **Bounded.** The default cap (1024 entries) is large enough to cover a
  realistic per-domain working set without memory creep during long runs.
  Eviction is plain LRU — sufficient for the access pattern (recently
  visited cards keep getting clicked).
* **TTL'd.** Entries auto-expire after ``ttl_seconds`` (default 1 h) so
  layouts that change between sessions don't return stale coordinates.
* **Process-local.** No disk persistence. Adding it would let cache
  poisoning leak across tenants — out of scope for this PR.
* **Hash crops, not full screenshots.** A click target near (x, y) only
  needs the surrounding ~80px region to identify the layout. Cropping
  before hashing gives the cache a meaningful "this card looks the same"
  signal even when other parts of the screenshot differ (header banners,
  page numbers, scroll position).

Wired separately from the grounders so existing :class:`ClaudeGrounding`
calls keep working unchanged. A follow-on PR will add an opt-in
``cache=`` constructor arg on the grounders themselves.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .grounding import GroundingResult
from .loop_detector import phash_64

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


# Default crop half-size. The cache hashes a 2*CROP_HALF px box around
# (initial_x, initial_y). Big enough to capture a listing card or button
# row; small enough that scrolling doesn't shift content out of frame.
DEFAULT_CROP_HALF: int = 80
DEFAULT_MAX_ENTRIES: int = 1024
DEFAULT_TTL_SECONDS: float = 3600.0


@dataclass
class _CacheEntry:
    result: GroundingResult
    expires_at: float


class GroundingCache:
    """Bounded TTL cache keyed by (frame-region hash, description hash).

    Hit/miss counters are public so callers can dashboard them via
    Prometheus or log them at run end. Counters never reset implicitly —
    callers can call :meth:`reset_counters` between runs if they want
    per-run telemetry.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        crop_half: int = DEFAULT_CROP_HALF,
    ) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1 (got {max_entries})")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0 (got {ttl_seconds})")
        if crop_half < 8:
            raise ValueError(f"crop_half must be >= 8 (got {crop_half})")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.crop_half = crop_half
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.expirations: int = 0

    # ── Key construction ────────────────────────────────────────────────

    def make_key(
        self,
        screenshot: "Image.Image",
        description: str,
        initial_x: int | None = None,
        initial_y: int | None = None,
    ) -> str:
        """Build a cache key from the screenshot + click target.

        Crops a ``crop_half * 2`` box around ``(initial_x, initial_y)`` (or
        the image center when coordinates are missing) and dHashes it.
        Description is SHA1'd to a short hex prefix so the key stays
        bounded regardless of description length.
        """
        ix = initial_x if initial_x is not None else screenshot.width // 2
        iy = initial_y if initial_y is not None else screenshot.height // 2
        try:
            crop = self._crop_around(screenshot, ix, iy)
            frame_hash = phash_64(crop)
        except Exception as exc:  # noqa: BLE001 — degrade rather than crash
            logger.debug("crop/phash failed: %s — falling back to full hash", exc)
            frame_hash = phash_64(screenshot)
        desc_hash = hashlib.sha1((description or "").encode("utf-8")).hexdigest()[:12]
        return f"{frame_hash}:{desc_hash}"

    def _crop_around(
        self, screenshot: "Image.Image", x: int, y: int
    ) -> "Image.Image":
        half = self.crop_half
        left = max(0, x - half)
        upper = max(0, y - half)
        right = min(screenshot.width, x + half)
        lower = min(screenshot.height, y + half)
        if right <= left or lower <= upper:
            return screenshot
        return screenshot.crop((left, upper, right, lower))

    # ── Get / put ──────────────────────────────────────────────────────

    def get(self, key: str) -> GroundingResult | None:
        """Look up a cache entry. ``None`` on miss or expiry."""
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        if entry.expires_at <= time.time():
            # Expired — drop it so stale layouts don't keep returning.
            del self._entries[key]
            self.expirations += 1
            self.misses += 1
            return None
        # Touch for LRU.
        self._entries.move_to_end(key)
        self.hits += 1
        return entry.result

    def put(self, key: str, result: GroundingResult) -> None:
        """Store a result. Evicts the oldest entry when at capacity."""
        if key in self._entries:
            # Update in place + bump recency.
            self._entries[key] = _CacheEntry(
                result=result, expires_at=time.time() + self.ttl_seconds
            )
            self._entries.move_to_end(key)
            return
        if len(self._entries) >= self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1
        self._entries[key] = _CacheEntry(
            result=result, expires_at=time.time() + self.ttl_seconds
        )

    # ── Convenience wrapper ─────────────────────────────────────────────

    def lookup_or_compute(
        self,
        screenshot: "Image.Image",
        description: str,
        compute_fn,
        initial_x: int | None = None,
        initial_y: int | None = None,
    ) -> GroundingResult:
        """Cache-aware ground call.

        ``compute_fn`` is a 0-arg callable that performs the actual
        grounding (the expensive Claude API call, OS-Atlas inference, etc.)
        and returns a :class:`GroundingResult`. Use partial / lambda to
        bind the args:

            cache.lookup_or_compute(
                screenshot, description,
                lambda: claude_grounder.ground(screenshot, description, ix, iy),
                initial_x=ix, initial_y=iy,
            )
        """
        key = self.make_key(screenshot, description, initial_x, initial_y)
        cached = self.get(key)
        if cached is not None:
            return cached
        result = compute_fn()
        self.put(key, result)
        return result

    # ── Diagnostics ────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._entries)

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    def clear(self) -> None:
        self._entries.clear()
        self.reset_counters()


__all__ = ["GroundingCache", "DEFAULT_CROP_HALF", "DEFAULT_MAX_ENTRIES", "DEFAULT_TTL_SECONDS"]
