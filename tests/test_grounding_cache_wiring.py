"""Tests for #117 step 2 — GroundingCache wired into ClaudeGrounding / LLMGrounding.

We don't hit the real APIs — instead we verify that:
  * When a cache is set, ground() goes through cache.lookup_or_compute
  * Cache hits short-circuit the remote call
  * When no cache is set, behavior is unchanged
"""

from __future__ import annotations

from unittest.mock import patch

from PIL import Image

from mantis_agent.grounding import ClaudeGrounding, GroundingResult, LLMGrounding
from mantis_agent.grounding_cache import GroundingCache


def _img() -> Image.Image:
    return Image.new("RGB", (256, 256), (128, 128, 128))


# ── ClaudeGrounding ─────────────────────────────────────────────────────


def test_claude_grounding_no_cache_constructor_default() -> None:
    g = ClaudeGrounding(api_key="x")
    assert g.cache is None


def test_claude_grounding_accepts_cache() -> None:
    cache = GroundingCache()
    g = ClaudeGrounding(api_key="x", cache=cache)
    assert g.cache is cache


def test_claude_grounding_cache_hit_skips_remote_call() -> None:
    """Two ground() calls with identical inputs: the second must hit the
    cache and skip _ground_remote entirely."""
    cache = GroundingCache()
    g = ClaudeGrounding(api_key="x", cache=cache)
    img = _img()

    with patch.object(
        g, "_ground_remote",
        return_value=GroundingResult(x=42, y=43, confidence=0.9, description="d"),
    ) as remote:
        out1 = g.ground(img, "click first listing", 100, 100)
        out2 = g.ground(img.copy(), "click first listing", 100, 100)

    assert remote.call_count == 1
    assert out1.x == 42 and out1.y == 43
    assert out2.x == 42 and out2.y == 43
    assert cache.hits == 1
    assert cache.misses == 1


def test_claude_grounding_cache_miss_caches_for_next_call() -> None:
    cache = GroundingCache()
    g = ClaudeGrounding(api_key="x", cache=cache)
    img = _img()

    with patch.object(
        g, "_ground_remote",
        return_value=GroundingResult(x=10, y=20, confidence=0.7, description="d"),
    ):
        g.ground(img, "click", 50, 50)

    # Second call with the same description hits cache regardless of
    # the same/copied image.
    with patch.object(
        g, "_ground_remote",
        side_effect=AssertionError("should not be called on hit"),
    ):
        out = g.ground(img, "click", 50, 50)
    assert out.x == 10 and out.y == 20


def test_claude_grounding_no_cache_calls_remote_directly() -> None:
    g = ClaudeGrounding(api_key="x")  # no cache
    img = _img()

    with patch.object(
        g, "_ground_remote",
        return_value=GroundingResult(x=1, y=2, confidence=0.5, description="d"),
    ) as remote:
        g.ground(img, "click", 50, 50)
        g.ground(img.copy(), "click", 50, 50)

    # No cache → both calls go to _ground_remote.
    assert remote.call_count == 2


def test_claude_grounding_skips_cache_when_no_api_key() -> None:
    """No API key means _ground_remote returns a fallback every time —
    caching that would pollute the cache with junk. Skip the cache."""
    cache = GroundingCache()
    g = ClaudeGrounding(api_key="", cache=cache)
    img = _img()

    g.ground(img, "click", 50, 50)
    # Cache state untouched.
    assert cache.size == 0
    assert cache.hits == 0
    assert cache.misses == 0


def test_claude_grounding_skips_cache_when_no_description() -> None:
    """Empty description gives the cache nothing to key on. Skip."""
    cache = GroundingCache()
    g = ClaudeGrounding(api_key="x", cache=cache)
    img = _img()

    with patch.object(
        g, "_ground_remote",
        return_value=GroundingResult(x=1, y=2, confidence=0.3, description=""),
    ):
        g.ground(img, "", 50, 50)
    assert cache.size == 0


def test_claude_grounding_distinct_descriptions_dont_share_cache() -> None:
    cache = GroundingCache()
    g = ClaudeGrounding(api_key="x", cache=cache)
    img = _img()

    with patch.object(
        g, "_ground_remote",
        side_effect=[
            GroundingResult(x=10, y=10, confidence=0.9, description="first"),
            GroundingResult(x=20, y=20, confidence=0.9, description="second"),
        ],
    ) as remote:
        a = g.ground(img, "click first", 50, 50)
        b = g.ground(img, "click second", 50, 50)

    assert remote.call_count == 2  # both miss
    assert a.x == 10
    assert b.x == 20


# ── LLMGrounding ────────────────────────────────────────────────────────


def test_llm_grounding_accepts_cache() -> None:
    cache = GroundingCache()
    g = LLMGrounding(cache=cache)
    assert g.cache is cache


def test_llm_grounding_cache_hit_skips_remote_call() -> None:
    cache = GroundingCache()
    g = LLMGrounding(cache=cache)
    img = _img()

    with patch.object(
        g, "_ground_remote",
        return_value=GroundingResult(x=99, y=88, confidence=0.8, description="d"),
    ) as remote:
        g.ground(img, "click", 50, 50)
        g.ground(img, "click", 50, 50)

    assert remote.call_count == 1
    assert cache.hits == 1


def test_llm_grounding_no_cache_calls_remote_directly() -> None:
    g = LLMGrounding()  # no cache
    img = _img()

    with patch.object(
        g, "_ground_remote",
        return_value=GroundingResult(x=1, y=2, confidence=0.5, description="d"),
    ) as remote:
        g.ground(img, "click", 50, 50)
        g.ground(img, "click", 50, 50)

    assert remote.call_count == 2


# ── Integration: hit-rate counter for ops dashboards ───────────────────


def test_hit_rate_after_realistic_session() -> None:
    """Simulate a multi-page session where the same listing-card
    descriptions repeat: hit rate should be > 50%."""
    cache = GroundingCache()
    g = ClaudeGrounding(api_key="x", cache=cache)
    img = _img()

    descriptions = ["click first listing", "click second listing", "click first listing"] * 5
    with patch.object(
        g, "_ground_remote",
        return_value=GroundingResult(x=42, y=43, confidence=0.9, description="d"),
    ):
        for desc in descriptions:
            g.ground(img, desc, 100, 100)

    # 15 calls; 2 unique → 2 misses, 13 hits.
    assert cache.misses == 2
    assert cache.hits == 13
    assert cache.hit_rate() > 0.85
