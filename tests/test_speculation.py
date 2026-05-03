"""Tests for #118 step 1 — speculative brain inference primitive."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import pytest
from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.speculation import (
    Speculation,
    _hamming_distance,
    frames_close_enough,
    shutdown_executor,
    start,
)


# ── Fakes ───────────────────────────────────────────────────────────────


@dataclass
class _FakeResult:
    action: Action


class _BlockingBrain:
    """Brain that blocks on a barrier so tests can inspect state mid-call."""

    def __init__(self, sleep_s: float = 0.0) -> None:
        self.calls: list[dict] = []
        self.sleep_s = sleep_s
        self.barrier = threading.Event()
        self.result_value = _FakeResult(action=Action(ActionType.CLICK, {"x": 1, "y": 1}))

    def think(self, **kw: Any) -> _FakeResult:
        self.calls.append(kw)
        if self.sleep_s:
            time.sleep(self.sleep_s)
        self.barrier.set()
        return self.result_value


class _RaisingBrain:
    def think(self, **kw: Any) -> Any:
        raise RuntimeError("brain exploded")


def _img(color: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    return Image.new("RGB", (32, 32), color)


def _white() -> Image.Image:
    return _img((255, 255, 255))


def _black() -> Image.Image:
    return _img((0, 0, 0))


@pytest.fixture(autouse=True)
def _shutdown_after_each_test():
    """Ensure the module-level executor is torn down between tests so
    one test's pending futures can't bleed into another."""
    yield
    shutdown_executor(wait=True)


# ── _hamming_distance ──────────────────────────────────────────────────


def test_hamming_zero_for_equal_strings() -> None:
    assert _hamming_distance("abcd1234", "abcd1234") == 0


def test_hamming_counts_bit_differences() -> None:
    # 0xff vs 0x00 → 8 bits differ.
    assert _hamming_distance("ff", "00") == 8


def test_hamming_max_for_empty_or_mismatched_lengths() -> None:
    assert _hamming_distance("", "abc") == 64
    assert _hamming_distance("abc", "abcd") == 64


def test_hamming_max_for_invalid_hex() -> None:
    """Defensive: a malformed hash (non-hex chars) returns max distance
    rather than crashing the validator."""
    assert _hamming_distance("zzzz", "abcd") == 64


# ── frames_close_enough ─────────────────────────────────────────────────


def test_frames_close_enough_zero_distance_passes() -> None:
    assert frames_close_enough("abcdef0123456789a", "abcdef0123456789a") is True


def test_frames_close_enough_default_strict() -> None:
    """Default tolerance is 0 — any single-bit difference fails."""
    h1 = "0" * 17
    h2 = "1" + "0" * 16
    assert frames_close_enough(h1, h2) is False


def test_frames_close_enough_loosened_threshold() -> None:
    h1 = "0" * 17
    h2 = "f" + "0" * 16  # 4 bits different
    assert frames_close_enough(h1, h2, max_hamming_distance=4) is True
    assert frames_close_enough(h1, h2, max_hamming_distance=3) is False


# ── Speculation lifecycle ──────────────────────────────────────────────


def test_start_kicks_off_brain_think() -> None:
    brain = _BlockingBrain(sleep_s=0.01)
    spec = start(brain, frames=[_white()], task="test")
    assert isinstance(spec, Speculation)
    # Wait for the worker to actually invoke the brain.
    brain.barrier.wait(timeout=2.0)
    assert len(brain.calls) == 1


def test_speculation_result_returns_brain_output() -> None:
    brain = _BlockingBrain()
    spec = start(brain, frames=[_white()], task="test")
    result = spec.result(timeout=2.0)
    assert result is brain.result_value


def test_speculation_passes_args_through() -> None:
    brain = _BlockingBrain()
    history = [Action(ActionType.CLICK, {"x": 0, "y": 0})]
    spec = start(
        brain,
        frames=[_white(), _black()],
        task="MY TASK",
        action_history=history,
        screen_size=(800, 600),
    )
    spec.result(timeout=2.0)
    call = brain.calls[0]
    assert call["task"] == "MY TASK"
    assert call["action_history"] is history
    assert call["screen_size"] == (800, 600)
    assert len(call["frames"]) == 2


def test_speculation_propagates_brain_exceptions() -> None:
    brain = _RaisingBrain()
    spec = start(brain, frames=[_white()], task="test")
    with pytest.raises(RuntimeError, match="brain exploded"):
        spec.result(timeout=2.0)


def test_speculation_done_reports_completion() -> None:
    brain = _BlockingBrain()
    spec = start(brain, frames=[_white()], task="test")
    spec.result(timeout=2.0)
    assert spec.done() is True


# ── Validation ─────────────────────────────────────────────────────────


def test_is_valid_true_for_identical_post_frame() -> None:
    brain = _BlockingBrain()
    img = _white()
    spec = start(brain, frames=[img], task="test")
    spec.result(timeout=2.0)  # ensure the call completed
    # Same image → same dHash → valid.
    assert spec.is_valid(img.copy()) is True
    assert spec.stats.valid_on_check is True


def test_is_valid_false_for_different_post_frame() -> None:
    brain = _BlockingBrain()
    spec = start(brain, frames=[_white()], task="test")
    spec.result(timeout=2.0)
    assert spec.is_valid(_black()) is False
    assert spec.stats.valid_on_check is False


def test_is_valid_records_distance() -> None:
    brain = _BlockingBrain()
    spec = start(brain, frames=[_white()], task="test")
    spec.result(timeout=2.0)
    spec.is_valid(_black())
    assert spec.stats.distance_on_check is not None
    assert spec.stats.distance_on_check > 0


def test_is_valid_with_custom_validator() -> None:
    """Custom validator: always-true predicate makes any post-frame valid."""
    brain = _BlockingBrain()
    spec = start(brain, frames=[_white()], task="test")
    spec.result(timeout=2.0)
    assert spec.is_valid(_black(), validator=lambda a, b: True) is True


def test_is_valid_handles_corrupt_frame_safely() -> None:
    """A frame that can't be hashed (e.g. None) must not crash is_valid."""
    brain = _BlockingBrain()
    spec = start(brain, frames=[_white()], task="test")
    spec.result(timeout=2.0)
    # Pass a non-image; phash_64 should fail and is_valid should return False.
    assert spec.is_valid(None) is False  # type: ignore[arg-type]


# ── Empty frames ────────────────────────────────────────────────────────


def test_start_with_empty_frames_falls_back_safely() -> None:
    """No frames means the speculation can't validate later — but
    starting it shouldn't raise. The validator will fail on any post
    frame so the runner discards correctly."""
    brain = _BlockingBrain()
    spec = start(brain, frames=[], task="test")
    spec.result(timeout=2.0)
    assert spec.is_valid(_white()) is False


# ── Stats ──────────────────────────────────────────────────────────────


def test_stats_records_submitted_at() -> None:
    brain = _BlockingBrain()
    before = time.monotonic()
    spec = start(brain, frames=[_white()], task="test")
    after = time.monotonic()
    assert before <= spec.stats.submitted_at <= after


def test_stats_records_frame_hash_at_start() -> None:
    brain = _BlockingBrain()
    spec = start(brain, frames=[_white()], task="test")
    assert spec.stats.frame_hash_at_start != ""
    # 17 hex chars — matches loop_detector.phash_64 format.
    assert len(spec.stats.frame_hash_at_start) == 17


# ── Cancellation ────────────────────────────────────────────────────────


def test_cancel_returns_bool() -> None:
    """Cancel returns whether cancellation succeeded. We don't assert the
    value because the worker may have already started; we just verify the
    method works without error."""
    brain = _BlockingBrain()
    spec = start(brain, frames=[_white()], task="test")
    out = spec.cancel()
    assert isinstance(out, bool)
