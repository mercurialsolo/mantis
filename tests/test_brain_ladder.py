"""Tests for #124 — BrainLadder primary/fallback wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from mantis_agent.actions import Action, ActionType
from mantis_agent.brain_ladder import BRAIN_USED_ATTR, BrainLadder, LadderAttempt


# ── Fakes ───────────────────────────────────────────────────────────────


@dataclass
class _FakeResult:
    action: Action
    raw_output: str = ""
    thinking: str = ""


class _FakeBrain:
    """Records every think() call so tests can assert routing."""

    def __init__(self, name: str, raises: bool = False) -> None:
        self.name = name
        self.calls: list[dict] = []
        self.loaded = False
        self.raises = raises

    def load(self) -> None:
        if self.raises:
            raise RuntimeError(f"{self.name} load failed")
        self.loaded = True

    def think(
        self,
        frames: Any = None,
        task: str = "",
        action_history: Any = None,
        screen_size: tuple[int, int] = (1920, 1080),
    ) -> _FakeResult:
        self.calls.append(
            {"task": task, "action_history": action_history, "screen_size": screen_size}
        )
        return _FakeResult(
            action=Action(ActionType.CLICK, {"x": 1, "y": 1}),
            raw_output=f"from-{self.name}",
        )


# ── Default routing ─────────────────────────────────────────────────────


def test_default_routes_to_primary() -> None:
    p, f = _FakeBrain("primary"), _FakeBrain("fallback")
    ladder = BrainLadder(primary=p, fallback=f)
    ladder.think(frames=[], task="test")
    assert len(p.calls) == 1
    assert len(f.calls) == 0


def test_force_fallback_routes_next_call_to_fallback() -> None:
    p, f = _FakeBrain("primary"), _FakeBrain("fallback")
    ladder = BrainLadder(primary=p, fallback=f)
    ladder.force_fallback()
    ladder.think(frames=[], task="test")
    assert len(p.calls) == 0
    assert len(f.calls) == 1


def test_force_fallback_is_idempotent() -> None:
    p, f = _FakeBrain("primary"), _FakeBrain("fallback")
    ladder = BrainLadder(primary=p, fallback=f)
    ladder.force_fallback()
    ladder.force_fallback()
    ladder.think(frames=[], task="test")
    assert len(f.calls) == 1


# ── One-shot vs sticky behavior ─────────────────────────────────────────


def test_one_shot_fallback_reverts_after_single_call() -> None:
    """Default behavior: fallback is used for exactly one think() call."""
    p, f = _FakeBrain("primary"), _FakeBrain("fallback")
    ladder = BrainLadder(primary=p, fallback=f)
    ladder.force_fallback()
    ladder.think(frames=[], task="first")   # → fallback
    ladder.think(frames=[], task="second")  # → primary again
    ladder.think(frames=[], task="third")   # → primary
    assert len(p.calls) == 2
    assert len(f.calls) == 1


def test_sticky_fallback_stays_until_reset() -> None:
    p, f = _FakeBrain("primary"), _FakeBrain("fallback")
    ladder = BrainLadder(primary=p, fallback=f, sticky_fallback=True)
    ladder.force_fallback()
    ladder.think(frames=[], task="first")
    ladder.think(frames=[], task="second")
    ladder.think(frames=[], task="third")
    assert len(p.calls) == 0
    assert len(f.calls) == 3
    ladder.reset()
    ladder.think(frames=[], task="fourth")
    assert len(p.calls) == 1


def test_reset_drops_fallback_flag() -> None:
    p, f = _FakeBrain("primary"), _FakeBrain("fallback")
    ladder = BrainLadder(primary=p, fallback=f, sticky_fallback=True)
    ladder.force_fallback()
    assert ladder.is_using_fallback is True
    ladder.reset()
    assert ladder.is_using_fallback is False


# ── Attribution ─────────────────────────────────────────────────────────


def test_result_carries_brain_used_primary() -> None:
    p, f = _FakeBrain("primary"), _FakeBrain("fallback")
    ladder = BrainLadder(primary=p, fallback=f)
    result = ladder.think(frames=[], task="test")
    assert getattr(result, BRAIN_USED_ATTR) == "primary"


def test_result_carries_brain_used_fallback() -> None:
    p, f = _FakeBrain("primary"), _FakeBrain("fallback")
    ladder = BrainLadder(primary=p, fallback=f)
    ladder.force_fallback()
    result = ladder.think(frames=[], task="test")
    assert getattr(result, BRAIN_USED_ATTR) == "fallback"


def test_attempts_log_records_each_call() -> None:
    p, f = _FakeBrain("primary"), _FakeBrain("fallback")
    ladder = BrainLadder(primary=p, fallback=f)
    ladder.think(frames=[], task="a")
    ladder.force_fallback()
    ladder.think(frames=[], task="b")
    ladder.think(frames=[], task="c")  # back to primary
    assert len(ladder.attempts) == 3
    assert [a.brain_used for a in ladder.attempts] == ["primary", "fallback", "primary"]


def test_attempts_durations_are_non_negative() -> None:
    p, f = _FakeBrain("primary"), _FakeBrain("fallback")
    ladder = BrainLadder(primary=p, fallback=f)
    ladder.think(frames=[], task="a")
    ladder.think(frames=[], task="b")
    for a in ladder.attempts:
        assert isinstance(a, LadderAttempt)
        assert a.duration >= 0.0


def test_primary_and_fallback_call_counts() -> None:
    p, f = _FakeBrain("primary"), _FakeBrain("fallback")
    ladder = BrainLadder(primary=p, fallback=f, sticky_fallback=True)
    ladder.think(frames=[], task="a")
    ladder.force_fallback()
    ladder.think(frames=[], task="b")
    ladder.think(frames=[], task="c")
    assert ladder.primary_calls() == 1
    assert ladder.fallback_calls() == 2


def test_reset_attempts_clears_log_only() -> None:
    p, f = _FakeBrain("primary"), _FakeBrain("fallback")
    ladder = BrainLadder(primary=p, fallback=f, sticky_fallback=True)
    ladder.force_fallback()
    ladder.think(frames=[], task="a")
    assert len(ladder.attempts) == 1
    ladder.reset_attempts()
    assert len(ladder.attempts) == 0
    # Fallback flag unchanged.
    assert ladder.is_using_fallback is True


# ── load() behavior ─────────────────────────────────────────────────────


def test_load_initializes_both_brains() -> None:
    p, f = _FakeBrain("primary"), _FakeBrain("fallback")
    ladder = BrainLadder(primary=p, fallback=f)
    ladder.load()
    assert p.loaded is True
    assert f.loaded is True


def test_load_swallows_fallback_load_failure() -> None:
    """The fallback may need lazy init (e.g. requires network on first
    call). A load failure shouldn't block primary use."""
    p = _FakeBrain("primary")
    f = _FakeBrain("fallback", raises=True)
    ladder = BrainLadder(primary=p, fallback=f)
    # Should NOT raise.
    ladder.load()
    assert p.loaded is True


def test_load_propagates_primary_failure() -> None:
    """A primary load failure is fatal — no point continuing."""
    p = _FakeBrain("primary", raises=True)
    f = _FakeBrain("fallback")
    ladder = BrainLadder(primary=p, fallback=f)
    with pytest.raises(RuntimeError, match="primary"):
        ladder.load()


# ── Brain protocol conformance ──────────────────────────────────────────


def test_brain_ladder_satisfies_brain_protocol() -> None:
    from mantis_agent.brain_protocol import Brain

    p, f = _FakeBrain("primary"), _FakeBrain("fallback")
    ladder = BrainLadder(primary=p, fallback=f)
    assert isinstance(ladder, Brain)


def test_think_passes_through_args_unmodified() -> None:
    """The ladder is transparent — it must not transform frames / task /
    action_history / screen_size on the way through."""
    p, f = _FakeBrain("primary"), _FakeBrain("fallback")
    ladder = BrainLadder(primary=p, fallback=f)
    history = [Action(ActionType.CLICK, {"x": 0, "y": 0})]
    ladder.think(
        frames=["frame_marker"],
        task="MY TASK",
        action_history=history,
        screen_size=(800, 600),
    )
    assert p.calls[0]["task"] == "MY TASK"
    assert p.calls[0]["action_history"] is history
    assert p.calls[0]["screen_size"] == (800, 600)
