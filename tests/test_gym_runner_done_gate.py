"""Runner integration for #303 done-acceptance gate.

Drives the runner with a fake brain that emits a bad ``done(success=True)``,
then asserts the gate rejected it (substituted WAIT, recorded reason in the
trajectory step, incremented per-reason counter on RunResult).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
from mantis_agent.gym.done_gate import (
    REJECT_EMPTY_SUMMARY,
    REJECT_NO_DELTA_AFTER_WAITS,
)
from mantis_agent.gym.runner import GymRunner


@dataclass
class _BrainResult:
    action: Action
    thinking: str = ""
    predicted_outcome: str = ""


class _ScriptedBrain:
    """Plays back a list of actions; once exhausted emits a final failure done."""

    def __init__(self, actions: list[Action]) -> None:
        self.actions = list(actions)
        self.calls = 0

    def think(
        self,
        frames: Any,
        task: str,
        action_history: Any = None,
        screen_size: tuple[int, int] = (100, 100),
    ) -> _BrainResult:
        if self.calls < len(self.actions):
            a = self.actions[self.calls]
            self.calls += 1
            return _BrainResult(a)
        # Out-of-script: emit a real failure done so the run terminates.
        self.calls += 1
        return _BrainResult(
            Action(ActionType.DONE, {"success": False, "summary": "scripted end"}),
        )


class _StableEnv(GymEnvironment):
    """Always returns the same screenshot — frame_hash is stable."""

    def __init__(self) -> None:
        self._frame = Image.new("RGB", (100, 100), color=(128, 128, 128))

    @property
    def screen_size(self) -> tuple[int, int]:
        return (100, 100)

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        return GymObservation(screenshot=self._frame, extras={"url": "https://x.test"})

    def step(self, action: Action) -> GymResult:
        return GymResult(
            GymObservation(screenshot=self._frame),
            reward=0.0,
            done=False,
            info={"url": "https://x.test"},
        )

    def close(self) -> None:
        pass


def test_done_gate_rejects_empty_summary_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First done() has empty summary — gate rejects, second done(success=False)
    terminates the run normally."""
    monkeypatch.delenv("MANTIS_DONE_GATE", raising=False)

    brain = _ScriptedBrain([
        Action(ActionType.DONE, {"success": True, "summary": ""}),
    ])
    env = _StableEnv()
    result = GymRunner(brain, env, max_steps=5).run("task")

    # The first done was rejected; the next inference returned the
    # scripted-end failure done which terminated the run.
    assert result.done_rejections_by_reason.get(REJECT_EMPTY_SUMMARY) == 1
    # Trajectory should have at least one step with done_rejected_reason set
    # (the substituted WAIT after the rejection).
    rejected_steps = [
        t for t in result.trajectory if t.done_rejected_reason
    ]
    assert len(rejected_steps) == 1
    assert rejected_steps[0].done_rejected_reason == REJECT_EMPTY_SUMMARY
    assert rejected_steps[0].action.action_type == ActionType.WAIT


def test_done_gate_rejects_no_delta_after_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three waits with stable frame, then done(success=True) — gate rejects."""
    monkeypatch.delenv("MANTIS_DONE_GATE", raising=False)

    brain = _ScriptedBrain([
        Action(ActionType.WAIT, {"seconds": 1.0}),
        Action(ActionType.WAIT, {"seconds": 1.0}),
        Action(ActionType.WAIT, {"seconds": 1.0}),
        Action(ActionType.DONE, {"success": True, "summary": "Page loaded."}),
    ])
    env = _StableEnv()
    result = GymRunner(brain, env, max_steps=10).run("task")

    assert (
        result.done_rejections_by_reason.get(REJECT_NO_DELTA_AFTER_WAITS) == 1
    )


def test_done_gate_disabled_via_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """MANTIS_DONE_GATE=disabled is the ablation toggle."""
    monkeypatch.setenv("MANTIS_DONE_GATE", "disabled")

    brain = _ScriptedBrain([
        Action(ActionType.DONE, {"success": True, "summary": ""}),
    ])
    env = _StableEnv()
    result = GymRunner(brain, env, max_steps=5).run("task")

    # Gate disabled — no rejection counted; the empty-summary done was accepted
    # and terminated the run successfully.
    assert result.done_rejections_by_reason == {}
    assert result.success is True
    assert result.termination_reason == "done"


def test_done_gate_accepts_well_formed_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A done() with a real summary on the first step should pass the gate."""
    monkeypatch.delenv("MANTIS_DONE_GATE", raising=False)

    brain = _ScriptedBrain([
        Action(ActionType.DONE, {"success": True, "summary": "Logged in successfully and navigated to dashboard."}),
    ])
    env = _StableEnv()
    result = GymRunner(brain, env, max_steps=5).run("task")

    # Empty-summary check passes (summary is non-empty);
    # plan check skipped (no plan);
    # form check skipped (no force_fill);
    # wait/progress checks skipped (not enough history).
    # Note: the existing model-based verifier may still reject, but the brain
    # in this test has no detector wired, so verify_done returns None and the
    # done is accepted.
    assert result.done_rejections_by_reason == {}


def test_done_rejected_reason_does_not_leak_to_subsequent_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pending_done_rejected_reason`` is one-shot — the substituted-WAIT
    step carries it; the next normal step must not."""
    monkeypatch.delenv("MANTIS_DONE_GATE", raising=False)

    brain = _ScriptedBrain([
        Action(ActionType.DONE, {"success": True, "summary": ""}),
        Action(ActionType.CLICK, {"x": 50, "y": 50}),
        Action(ActionType.DONE, {"success": False, "summary": "stop"}),
    ])
    env = _StableEnv()
    result = GymRunner(brain, env, max_steps=5).run("task")

    rejected = [t for t in result.trajectory if t.done_rejected_reason]
    assert len(rejected) == 1, (
        f"expected exactly one step with done_rejected_reason, got "
        f"{[(t.step, t.done_rejected_reason) for t in result.trajectory]}"
    )
    assert rejected[0].action.action_type == ActionType.WAIT
    # The CLICK that follows must NOT inherit the reason.
    click_steps = [
        t for t in result.trajectory if t.action.action_type == ActionType.CLICK
    ]
    assert click_steps and click_steps[0].done_rejected_reason == ""


def test_done_gate_skipped_when_success_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``done(success=False)`` is a real failure done — the gate must not
    rewrite it into a WAIT just because the summary is empty."""
    monkeypatch.delenv("MANTIS_DONE_GATE", raising=False)

    brain = _ScriptedBrain([
        Action(ActionType.DONE, {"success": False, "summary": ""}),
    ])
    env = _StableEnv()
    result = GymRunner(brain, env, max_steps=5).run("task")

    assert result.done_rejections_by_reason == {}
    assert result.termination_reason == "done"
    # success=False with no completed work → run is unsuccessful, but we
    # accepted the failure-done immediately rather than looping.
    assert result.success is False


def test_done_gate_respects_max_done_rejections_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After ``max_done_rejections`` (=2) rejections, the next bad done
    must be accepted — bound on the gate so it never traps a real done."""
    monkeypatch.delenv("MANTIS_DONE_GATE", raising=False)

    brain = _ScriptedBrain([
        Action(ActionType.DONE, {"success": True, "summary": ""}),
        Action(ActionType.DONE, {"success": True, "summary": ""}),
        Action(ActionType.DONE, {"success": True, "summary": ""}),
        Action(ActionType.DONE, {"success": True, "summary": ""}),
    ])
    env = _StableEnv()
    result = GymRunner(brain, env, max_steps=10).run("task")

    # Exactly two rejections; the third bad done is accepted.
    assert (
        result.done_rejections_by_reason.get(REJECT_EMPTY_SUMMARY) == 2
    ), result.done_rejections_by_reason
    assert result.termination_reason == "done"
    assert result.success is True


def test_done_gate_well_formed_done_terminates_successfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity strengthening of the existing accept-path test — verify the
    run actually terminated with success=True, not just that the counter
    stayed empty."""
    monkeypatch.delenv("MANTIS_DONE_GATE", raising=False)

    brain = _ScriptedBrain([
        Action(ActionType.DONE, {
            "success": True,
            "summary": "Logged in successfully and navigated to dashboard.",
        }),
    ])
    env = _StableEnv()
    result = GymRunner(brain, env, max_steps=5).run("task")

    assert result.done_rejections_by_reason == {}
    assert result.termination_reason == "done"
    assert result.success is True


def test_trajectory_step_to_dict_round_trips_done_rejected_reason() -> None:
    from mantis_agent.gym.runner import (
        TrajectoryStep,
        _trajectory_step_from_dict,
        _trajectory_step_to_dict,
    )

    original = TrajectoryStep(
        step=1,
        action=Action(ActionType.WAIT, {"seconds": 1.0}),
        thinking="",
        reward=0.0,
        done=False,
        inference_time=0.0,
        done_rejected_reason=REJECT_EMPTY_SUMMARY,
    )
    payload = _trajectory_step_to_dict(original)
    assert payload["done_rejected_reason"] == REJECT_EMPTY_SUMMARY
    rehydrated = _trajectory_step_from_dict(payload)
    assert rehydrated.done_rejected_reason == REJECT_EMPTY_SUMMARY
