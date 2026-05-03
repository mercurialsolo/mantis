"""Tests for #118 step 2 — SpeculativeBrain wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.speculation import shutdown_executor
from mantis_agent.speculative_brain import BRAIN_USED_ATTR, SpeculativeBrain


# ── Fakes ───────────────────────────────────────────────────────────────


@dataclass
class _FakeResult:
    action: Action
    raw_output: str = ""
    n: int = 0


class _CountingBrain:
    """Records every think() call and returns a numbered result."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def think(
        self,
        frames: Any = None,
        task: str = "",
        action_history: Any = None,
        screen_size: tuple[int, int] = (1920, 1080),
    ) -> _FakeResult:
        n = len(self.calls)
        self.calls.append({"task": task, "screen_size": screen_size})
        return _FakeResult(
            action=Action(ActionType.CLICK, {"x": 1, "y": 1}),
            raw_output=f"call-{n}",
            n=n,
        )


class _RaisingBrain:
    """First call raises, subsequent calls return normally — lets us
    exercise the speculative-exception → sync-fallback path."""

    def __init__(self) -> None:
        self.calls = 0

    def load(self) -> None: ...

    def think(self, **kw: Any) -> _FakeResult:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("first call boom")
        return _FakeResult(action=Action(ActionType.WAIT, {}))


def _white() -> Image.Image:
    return Image.new("RGB", (32, 32), (255, 255, 255))


def _black() -> Image.Image:
    return Image.new("RGB", (32, 32), (0, 0, 0))


@pytest.fixture(autouse=True)
def _shutdown_executor_after_test() -> None:
    yield
    shutdown_executor(wait=True)


# ── First call always synchronous ───────────────────────────────────────


def test_first_call_is_synchronous() -> None:
    brain = _CountingBrain()
    wrapper = SpeculativeBrain(brain)
    result = wrapper.think(frames=[_white()], task="t")
    # The visible call to inner.think — speculation was queued but not consumed.
    assert getattr(result, BRAIN_USED_ATTR) == "synchronous"
    assert wrapper.synchronous_starts == 1
    assert wrapper.hits == 0


# ── Validated speculation — hit ─────────────────────────────────────────


def test_consecutive_calls_with_identical_frames_hit_speculation() -> None:
    brain = _CountingBrain()
    wrapper = SpeculativeBrain(brain)
    # First call: synchronous, queues speculation against frames=[_white()].
    wrapper.think(frames=[_white()], task="t")
    # Second call: same frames → speculation valid → hit.
    result = wrapper.think(frames=[_white()], task="t")
    assert getattr(result, BRAIN_USED_ATTR) == "speculative"
    assert wrapper.hits == 1


def test_multiple_consecutive_hits_with_stable_screen() -> None:
    brain = _CountingBrain()
    wrapper = SpeculativeBrain(brain)
    wrapper.think(frames=[_white()], task="t")  # sync 1
    wrapper.think(frames=[_white()], task="t")  # spec hit 1
    wrapper.think(frames=[_white()], task="t")  # spec hit 2
    wrapper.think(frames=[_white()], task="t")  # spec hit 3
    assert wrapper.hits == 3
    assert wrapper.synchronous_starts == 1


# ── Invalidated speculation — miss ──────────────────────────────────────


def test_changed_frame_invalidates_speculation() -> None:
    brain = _CountingBrain()
    wrapper = SpeculativeBrain(brain)
    wrapper.think(frames=[_white()], task="t")  # sync; queues against white
    # Second call with a black frame — invalidated.
    result = wrapper.think(frames=[_black()], task="t")
    assert getattr(result, BRAIN_USED_ATTR) == "synchronous"
    assert wrapper.misses == 1


def test_empty_frames_treated_as_invalidation() -> None:
    """Some loops can produce a frameless step (loading state). Speculation
    can't validate, so fall through to sync — never crash."""
    brain = _CountingBrain()
    wrapper = SpeculativeBrain(brain)
    wrapper.think(frames=[_white()], task="t")
    result = wrapper.think(frames=[], task="t")
    assert getattr(result, BRAIN_USED_ATTR) == "synchronous"
    assert wrapper.misses == 1


# ── Mixed pattern ───────────────────────────────────────────────────────


def test_mixed_hit_miss_pattern() -> None:
    brain = _CountingBrain()
    wrapper = SpeculativeBrain(brain)
    wrapper.think(frames=[_white()], task="t")  # sync (start)
    wrapper.think(frames=[_white()], task="t")  # hit
    wrapper.think(frames=[_black()], task="t")  # miss → sync
    wrapper.think(frames=[_black()], task="t")  # hit (matches prior black)
    assert wrapper.hits == 2
    assert wrapper.misses == 1
    assert wrapper.synchronous_starts == 1


# ── hit_rate ────────────────────────────────────────────────────────────


def test_hit_rate_initially_zero() -> None:
    wrapper = SpeculativeBrain(_CountingBrain())
    assert wrapper.hit_rate() == 0.0


def test_hit_rate_reflects_counters() -> None:
    brain = _CountingBrain()
    wrapper = SpeculativeBrain(brain)
    # 1 sync + 3 hits → 3/4 = 0.75
    wrapper.think(frames=[_white()], task="t")
    wrapper.think(frames=[_white()], task="t")
    wrapper.think(frames=[_white()], task="t")
    wrapper.think(frames=[_white()], task="t")
    assert wrapper.hit_rate() == pytest.approx(3 / 4)


# ── reset ───────────────────────────────────────────────────────────────


def test_reset_clears_counters_and_pending() -> None:
    brain = _CountingBrain()
    wrapper = SpeculativeBrain(brain)
    wrapper.think(frames=[_white()], task="t")
    wrapper.think(frames=[_white()], task="t")
    assert wrapper.hits >= 1
    wrapper.reset()
    assert wrapper.hits == 0
    assert wrapper.misses == 0
    assert wrapper.synchronous_starts == 0
    # First post-reset call must be synchronous (no pending).
    result = wrapper.think(frames=[_white()], task="t")
    assert getattr(result, BRAIN_USED_ATTR) == "synchronous"


# ── Brain protocol conformance ─────────────────────────────────────────


def test_speculative_brain_satisfies_brain_protocol() -> None:
    from mantis_agent.brain_protocol import Brain

    wrapper = SpeculativeBrain(_CountingBrain())
    assert isinstance(wrapper, Brain)


def test_load_forwards_to_inner() -> None:
    brain = _CountingBrain()
    wrapper = SpeculativeBrain(brain)
    wrapper.load()
    assert brain.loaded is True


def test_think_passes_args_through_to_inner() -> None:
    brain = _CountingBrain()
    wrapper = SpeculativeBrain(brain)
    wrapper.think(
        frames=[_white()],
        task="MY TASK",
        action_history=[Action(ActionType.CLICK, {"x": 0, "y": 0})],
        screen_size=(800, 600),
    )
    # First call ran synchronously — args should propagate.
    assert brain.calls[0]["task"] == "MY TASK"
    assert brain.calls[0]["screen_size"] == (800, 600)


# ── Exception handling ─────────────────────────────────────────────────


def test_speculative_exception_falls_back_to_sync() -> None:
    """If the speculative future raised, the wrapper must fall back to a
    fresh synchronous call rather than re-raising."""
    brain = _RaisingBrain()
    wrapper = SpeculativeBrain(brain)
    # First sync call raises — that propagates (caller's responsibility).
    with pytest.raises(RuntimeError):
        wrapper.think(frames=[_white()], task="t")
    # The wrapper queued a speculation that will also raise. Next call
    # should detect the speculative failure and fall through to a
    # NEW sync call (which succeeds).
    result = wrapper.think(frames=[_white()], task="t")
    # We don't assert brain_used here — depending on timing the second
    # call may be a sync-after-speculative-error or a sync-no-pending.
    # What matters: it didn't raise.
    assert result is not None


# ── Custom validator ───────────────────────────────────────────────────


def test_loosened_validator_accepts_close_frames() -> None:
    """A validator that always says-true makes every post-frame valid —
    useful for workflows that tolerate animations."""
    brain = _CountingBrain()
    wrapper = SpeculativeBrain(brain, validator=lambda a, b: True)
    wrapper.think(frames=[_white()], task="t")  # sync
    result = wrapper.think(frames=[_black()], task="t")  # would normally miss
    assert getattr(result, BRAIN_USED_ATTR) == "speculative"


# ── End-to-end: simulated session metrics ──────────────────────────────


def test_realistic_session_hit_rate_for_stable_screen() -> None:
    """Simulate a workflow where the screen stabilizes (page loaded; agent
    is doing form fills with no visual delta between consecutive views).
    Most calls after the first should be speculative hits."""
    brain = _CountingBrain()
    wrapper = SpeculativeBrain(brain)
    # 11 calls, all on the same frame.
    wrapper.think(frames=[_white()], task="t")  # sync start
    for _ in range(10):
        wrapper.think(frames=[_white()], task="t")
    # 10 hits + 1 sync start = ~91% hit rate.
    assert wrapper.hits == 10
    assert wrapper.synchronous_starts == 1
    assert wrapper.misses == 0
    assert wrapper.hit_rate() > 0.9


def test_alternating_frames_produces_mostly_misses() -> None:
    """Sanity check the inverse: if every frame is different, every prior
    speculation invalidates. The wrapper degrades to ~all sync calls."""
    brain = _CountingBrain()
    wrapper = SpeculativeBrain(brain)
    # Pattern: white, black, white, black, ...
    wrapper.think(frames=[_white()], task="t")  # sync start
    for i in range(10):
        wrapper.think(frames=[_white() if i % 2 else _black()], task="t")
    # Every post-start call invalidates the prior speculation.
    assert wrapper.hits == 0
    assert wrapper.misses == 10
