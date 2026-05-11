"""Tests for issue #259 — adaptive content-settle in deep-extract.

Replaces the four fixed-duration ``time.sleep(...)`` calls in
``_extract_listing_data_deep`` (claude_step.py) with pixel-diff
polling. Same shape as ``_adaptive_submit_settle`` in
``_runner_helpers.py`` — bounded poll with early exit on a
stability signal.

Two layers:

1. ``screenshot_pixel_diff_fraction`` — primitive pixel-diff
   helper. Cheap downsample-and-compare; returns the fraction of
   pixels that changed beyond a brightness threshold.
2. ``adaptive_content_settle(env, max_seconds=..., ...)`` —
   bounded poll. Exits when two consecutive screenshots stabilize
   under ``diff_threshold``. Respects the ``min_seconds`` floor
   (initial render always needs some wall time) and the
   ``max_seconds`` cap (worst-case stays at the original fixed-
   sleep duration).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from PIL import Image

from mantis_agent.gym._runner_helpers import (
    adaptive_content_settle,
    screenshot_pixel_diff_fraction,
)


# ── Pixel-diff primitive ────────────────────────────────────────────


def _solid(color: int, size: tuple[int, int] = (320, 200)) -> Image.Image:
    """Solid grayscale image."""
    return Image.new("L", size, color).convert("RGB")


def test_pixel_diff_identical_images_is_zero() -> None:
    img = _solid(128)
    assert screenshot_pixel_diff_fraction(img, img) == 0.0


def test_pixel_diff_completely_different_images_is_near_one() -> None:
    """Black vs white — every pixel crosses any reasonable
    brightness threshold."""
    a = _solid(0)
    b = _solid(255)
    assert screenshot_pixel_diff_fraction(a, b) > 0.95


def test_pixel_diff_small_brightness_change_is_zero() -> None:
    """A 1-step brightness shift across the whole image must be
    below the noise threshold — JPEG re-encoding or cursor-blink
    artifacts shouldn't count as change."""
    a = _solid(128)
    b = _solid(129)
    assert screenshot_pixel_diff_fraction(a, b) < 0.01


def test_pixel_diff_size_mismatch_returns_conservative_one() -> None:
    """Different-sized screenshots can't be compared element-wise.
    Return 1.0 (assume changed) so the caller keeps polling."""
    a = _solid(128, size=(320, 200))
    b = _solid(128, size=(640, 400))
    assert screenshot_pixel_diff_fraction(a, b) == 1.0


def test_pixel_diff_handles_bad_input_gracefully() -> None:
    """Non-image input shouldn't crash — return conservative 1.0."""
    assert screenshot_pixel_diff_fraction(None, _solid(128)) == 1.0
    assert screenshot_pixel_diff_fraction(_solid(128), None) == 1.0


# ── Adaptive settle ─────────────────────────────────────────────────


class _StableEnv:
    """Env stub that returns the same screenshot every time —
    settle should exit at the floor."""

    def __init__(self, color: int = 128) -> None:
        self._img = _solid(color)
        self.screenshot_count = 0

    def screenshot(self) -> Image.Image:
        self.screenshot_count += 1
        return self._img


class _ChangingEnv:
    """Env stub that returns a different screenshot every call —
    settle should pay max_seconds because the page never settles."""

    def __init__(self) -> None:
        self._step = 0
        self.screenshot_count = 0

    def screenshot(self) -> Image.Image:
        self.screenshot_count += 1
        # Alternate between black and white — guaranteed full
        # pixel-diff between consecutive calls.
        color = 0 if self._step % 2 == 0 else 255
        self._step += 1
        return _solid(color)


class _SettlingEnv:
    """Env stub that changes for the first 2 polls then stabilizes
    — settle should exit shortly after the page calms down."""

    def __init__(self) -> None:
        self._step = 0
        self.screenshot_count = 0

    def screenshot(self) -> Image.Image:
        self.screenshot_count += 1
        if self._step < 2:
            color = 0 if self._step % 2 == 0 else 255
        else:
            color = 200  # stable from step 2 onward
        self._step += 1
        return _solid(color)


def test_settle_stable_page_exits_at_min_seconds(monkeypatch) -> None:
    """Stable page (consecutive screenshots identical) → settle
    returns shortly after min_seconds. Tolerance for poll_seconds
    because the loop takes one poll to confirm stability."""
    env = _StableEnv()
    start = time.time()
    elapsed = adaptive_content_settle(
        env, min_seconds=0.1, max_seconds=1.0, poll_seconds=0.1,
    )
    wall = time.time() - start
    # At least the min floor; well under the max cap.
    assert elapsed >= 0.1
    assert elapsed < 0.5
    assert wall < 0.5


def test_settle_changing_page_pays_max_seconds() -> None:
    """Page that keeps changing — settle pays the full max cap."""
    env = _ChangingEnv()
    elapsed = adaptive_content_settle(
        env, min_seconds=0.05, max_seconds=0.4, poll_seconds=0.1,
    )
    # Hit the cap (within tolerance for poll granularity).
    assert elapsed >= 0.4 - 0.15


def test_settle_settling_page_exits_when_stable() -> None:
    """Page that changes for 2 polls then stabilizes → settle
    exits a poll or two after stabilization, well before max."""
    env = _SettlingEnv()
    elapsed = adaptive_content_settle(
        env, min_seconds=0.05, max_seconds=2.0, poll_seconds=0.1,
    )
    # Should exit somewhere around min + 2 polls + stabilization = ~0.4s
    assert elapsed < 1.5


def test_settle_handles_screenshot_exception(monkeypatch) -> None:
    """env.screenshot() raising falls through to the fixed-sleep
    equivalent — better to over-pay than to halt the deep-extract."""
    env = MagicMock()
    env.screenshot.side_effect = RuntimeError("CDP glitch")
    elapsed = adaptive_content_settle(
        env, min_seconds=0.05, max_seconds=0.3, poll_seconds=0.1,
    )
    # Paid roughly max_seconds (with tolerance).
    assert elapsed >= 0.2


def test_settle_respects_min_seconds_floor() -> None:
    """Even with a perfectly-stable page, the floor still applies
    — first render needs some wall time regardless of pixel-diff."""
    env = _StableEnv()
    elapsed = adaptive_content_settle(
        env, min_seconds=0.3, max_seconds=1.0, poll_seconds=0.1,
    )
    assert elapsed >= 0.3


def test_settle_returns_actual_elapsed() -> None:
    """Telemetry contract: returned value reflects real wall-clock
    spent, not the configured max_seconds — callers add this to
    runner.costs['gpu_seconds']."""
    env = _StableEnv()
    elapsed = adaptive_content_settle(
        env, min_seconds=0.1, max_seconds=2.0, poll_seconds=0.1,
    )
    # Must be the real elapsed, not just max_seconds.
    assert elapsed < 1.0


# ── Validation of arguments ────────────────────────────────────────


def test_settle_rejects_invalid_diff_threshold() -> None:
    env = _StableEnv()
    with pytest.raises(ValueError):
        adaptive_content_settle(
            env, min_seconds=0.1, max_seconds=1.0,
            poll_seconds=0.1, diff_threshold=-0.1,
        )


def test_settle_rejects_min_greater_than_max() -> None:
    env = _StableEnv()
    with pytest.raises(ValueError):
        adaptive_content_settle(
            env, min_seconds=2.0, max_seconds=1.0, poll_seconds=0.1,
        )
