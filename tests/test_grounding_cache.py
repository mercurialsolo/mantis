"""Tests for #117 step 1 — GroundingCache."""

from __future__ import annotations

import time

import pytest
from PIL import Image

from mantis_agent.grounding import GroundingResult
from mantis_agent.grounding_cache import (
    DEFAULT_CROP_HALF,
    DEFAULT_MAX_ENTRIES,
    DEFAULT_TTL_SECONDS,
    GroundingCache,
)


def _solid_image(color: tuple[int, int, int] = (128, 128, 128), w: int = 256, h: int = 256) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def _result(x: int = 100, y: int = 100, conf: float = 0.9) -> GroundingResult:
    return GroundingResult(x=x, y=y, confidence=conf, description="cached")


# ── Constructor validation ──────────────────────────────────────────────


def test_invalid_max_entries_raises() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        GroundingCache(max_entries=0)


def test_invalid_ttl_raises() -> None:
    with pytest.raises(ValueError, match="ttl_seconds"):
        GroundingCache(ttl_seconds=0)


def test_invalid_crop_half_raises() -> None:
    with pytest.raises(ValueError, match="crop_half"):
        GroundingCache(crop_half=4)


def test_default_constants_are_reasonable() -> None:
    """Sanity: defaults make sense for the intended workload."""
    assert DEFAULT_CROP_HALF >= 32  # large enough for a card
    assert DEFAULT_MAX_ENTRIES >= 256  # covers a multi-page session
    assert DEFAULT_TTL_SECONDS >= 60  # not so short layouts churn


# ── Key construction ────────────────────────────────────────────────────


def test_make_key_is_deterministic_for_same_inputs() -> None:
    cache = GroundingCache()
    img = _solid_image()
    k1 = cache.make_key(img, "click first listing", 100, 100)
    k2 = cache.make_key(img.copy(), "click first listing", 100, 100)
    assert k1 == k2


def test_make_key_differs_when_description_differs() -> None:
    cache = GroundingCache()
    img = _solid_image()
    k1 = cache.make_key(img, "click first listing", 100, 100)
    k2 = cache.make_key(img, "click second listing", 100, 100)
    assert k1 != k2


def test_make_key_uses_local_crop_not_full_image() -> None:
    """Two screenshots that differ in a far-away region but match around
    the click target should produce the same key."""
    cache = GroundingCache(crop_half=40)
    img_a = _solid_image()
    # Modify a region far from the click target on img_b only.
    img_b = img_a.copy()
    for x in range(0, 30):
        for y in range(0, 30):
            img_b.putpixel((x, y), (255, 0, 0))
    # Click target at (200, 200) — far from the modified top-left region.
    k_a = cache.make_key(img_a, "click", 200, 200)
    k_b = cache.make_key(img_b, "click", 200, 200)
    assert k_a == k_b


def test_make_key_handles_missing_coords() -> None:
    """When initial_x/y are None, the cache should fall back to image center
    rather than crash."""
    cache = GroundingCache()
    img = _solid_image()
    key = cache.make_key(img, "click", None, None)
    assert isinstance(key, str)
    assert len(key) > 0


# ── Get / put ───────────────────────────────────────────────────────────


def test_miss_then_hit() -> None:
    cache = GroundingCache()
    assert cache.get("k") is None
    assert cache.misses == 1
    cache.put("k", _result(50, 60))
    out = cache.get("k")
    assert out is not None
    assert out.x == 50 and out.y == 60
    assert cache.hits == 1


def test_put_overwrites_existing_entry() -> None:
    cache = GroundingCache()
    cache.put("k", _result(1, 1))
    cache.put("k", _result(99, 99))
    out = cache.get("k")
    assert out is not None
    assert out.x == 99


def test_lru_eviction_when_at_capacity() -> None:
    cache = GroundingCache(max_entries=2)
    cache.put("a", _result(1, 1))
    cache.put("b", _result(2, 2))
    # Touch "a" so "b" becomes the LRU.
    cache.get("a")
    cache.put("c", _result(3, 3))
    assert cache.size == 2
    assert cache.get("a") is not None
    assert cache.get("b") is None  # evicted
    assert cache.get("c") is not None
    assert cache.evictions == 1


def test_ttl_expiry_drops_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = GroundingCache(ttl_seconds=10.0)
    cache.put("k", _result())
    # Advance time past expiry.
    real_time = time.time()
    monkeypatch.setattr(
        "mantis_agent.grounding_cache.time.time",
        lambda: real_time + 100.0,
    )
    assert cache.get("k") is None
    assert cache.expirations == 1
    assert cache.size == 0  # expired entries get cleaned up on access


# ── lookup_or_compute ──────────────────────────────────────────────────


def test_lookup_or_compute_invokes_compute_fn_on_miss() -> None:
    cache = GroundingCache()
    img = _solid_image()
    calls = []

    def compute() -> GroundingResult:
        calls.append(None)
        return _result(123, 456)

    out = cache.lookup_or_compute(img, "click", compute, 50, 50)
    assert out.x == 123 and out.y == 456
    assert len(calls) == 1


def test_lookup_or_compute_skips_compute_fn_on_hit() -> None:
    cache = GroundingCache()
    img = _solid_image()
    calls = []

    def compute() -> GroundingResult:
        calls.append(None)
        return _result(123, 456)

    cache.lookup_or_compute(img, "click", compute, 50, 50)
    cache.lookup_or_compute(img.copy(), "click", compute, 50, 50)
    assert len(calls) == 1
    assert cache.hits == 1


def test_lookup_or_compute_caches_distinct_descriptions_separately() -> None:
    cache = GroundingCache()
    img = _solid_image()
    a_calls = []
    b_calls = []

    cache.lookup_or_compute(
        img, "click first", lambda: (a_calls.append(None), _result(1, 1))[1], 50, 50
    )
    cache.lookup_or_compute(
        img, "click second", lambda: (b_calls.append(None), _result(2, 2))[1], 50, 50
    )
    cache.lookup_or_compute(
        img, "click first", lambda: (a_calls.append(None), _result(1, 1))[1], 50, 50
    )
    # First description was a hit on second call; second description was a miss.
    assert len(a_calls) == 1
    assert len(b_calls) == 1


# ── Diagnostics ─────────────────────────────────────────────────────────


def test_hit_rate_reports_accurate_fraction() -> None:
    cache = GroundingCache()
    cache.put("k", _result())
    cache.get("k")     # hit
    cache.get("k")     # hit
    cache.get("miss")  # miss
    assert cache.hit_rate() == pytest.approx(2 / 3)


def test_hit_rate_zero_before_any_access() -> None:
    cache = GroundingCache()
    assert cache.hit_rate() == 0.0


def test_reset_counters_zeros_metrics_only() -> None:
    cache = GroundingCache()
    cache.put("k", _result())
    cache.get("k")
    assert cache.hits == 1
    cache.reset_counters()
    assert cache.hits == 0
    assert cache.misses == 0
    # Entry survives — only counters reset.
    assert cache.size == 1


def test_clear_drops_entries_and_zeros_counters() -> None:
    cache = GroundingCache()
    cache.put("k", _result())
    cache.get("k")
    cache.clear()
    assert cache.size == 0
    assert cache.hits == 0


# ── Robustness ──────────────────────────────────────────────────────────


def test_crop_failure_falls_back_to_full_image_hash() -> None:
    """A degenerate crop (out-of-bounds coords) should not raise — the key
    should still be deterministic via the full-image fallback."""
    cache = GroundingCache()
    img = _solid_image(w=64, h=64)
    # Way out of bounds; _crop_around clamps but worth ensuring no raise.
    key = cache.make_key(img, "click", 10000, 10000)
    assert isinstance(key, str)
    assert len(key) > 0
