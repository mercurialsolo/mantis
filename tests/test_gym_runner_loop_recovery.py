"""Runner integration for #302 — LoopRecoveryPolicy wired into GymRunner.

Drives the runner with a scripted brain that emits 3 byte-equal
clicks on a static env so the soft-loop detector trips, then asserts
the policy substituted the 3rd click for a TAB or RETURN.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pytest
from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
from mantis_agent.gym.loop_recovery import (
    REASON_PRESS_RETURN_FOR_SUBMIT,
    REASON_TAB_TO_NEXT_FIELD,
)
from mantis_agent.gym.runner import GymRunner


@dataclass
class _BrainResult:
    action: Action
    thinking: str = ""
    predicted_outcome: str = ""


class _ScriptedBrain:
    def __init__(self, actions: list[Action]) -> None:
        self.actions = list(actions)
        self.calls = 0

    def think(
        self,
        frames: Any,
        task: str,
        action_history: Any = None,
        screen_size: tuple[int, int] = (200, 200),
    ) -> _BrainResult:
        if self.calls < len(self.actions):
            a = self.actions[self.calls]
            self.calls += 1
            return _BrainResult(a)
        self.calls += 1
        return _BrainResult(
            Action(ActionType.DONE, {"success": False, "summary": "scripted end"}),
        )


def _noisy_img(seed: int, size: tuple[int, int] = (200, 200)) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", size, color=(0, 0, 0))
    pixels = img.load()
    for _ in range(200):
        x, y = rng.randrange(size[0]), rng.randrange(size[1])
        pixels[x, y] = (255, 255, 255)
    return img


class _StaticEnvWithFocus(GymEnvironment):
    """Env that always returns the same screenshot AND surfaces a
    persistent focused_input — simulates a brain that keeps clicking
    a focused field that the runner's form controller can't match."""

    def __init__(self, focused_label: str | None = "captcha") -> None:
        self._frame = _noisy_img(seed=42)
        self._focused = (
            {"name": focused_label, "placeholder": focused_label}
            if focused_label else None
        )

    @property
    def screen_size(self) -> tuple[int, int]:
        return (200, 200)

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        return GymObservation(screenshot=self._frame)

    def step(self, action: Action) -> GymResult:
        info: dict = {}
        if self._focused is not None:
            info["focused_input"] = dict(self._focused)
        return GymResult(
            GymObservation(screenshot=self._frame), 0.0, False, info=info,
        )

    def close(self) -> None:
        pass


# ── Rule 1: focused click without pending value → Tab ─────────────────


def test_loop_recovery_forces_tab_on_focused_click_no_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3 byte-equal clicks on a focused field with no plan value match
    → soft-loop detector trips → policy forces Tab."""
    monkeypatch.delenv("MANTIS_LOOP_RECOVERY", raising=False)

    # soft_loop_window=3: the policy needs 3 samples in the detector
    # before it can declare a loop. Emit 4 clicks so the 4th has the
    # required history to trip the policy.
    click = Action(ActionType.CLICK, {"x": 100, "y": 100})
    brain = _ScriptedBrain([click, click, click, click])
    env = _StaticEnvWithFocus(focused_label="captcha")
    result = GymRunner(brain, env, max_steps=6).run("solve the captcha")

    assert result.loop_recoveries_by_reason.get(REASON_TAB_TO_NEXT_FIELD, 0) >= 1
    recovered_steps = [
        t for t in result.trajectory
        if t.loop_recovery_reason == REASON_TAB_TO_NEXT_FIELD
    ]
    assert len(recovered_steps) >= 1
    assert recovered_steps[0].action.action_type == ActionType.KEY_PRESS
    assert recovered_steps[0].action.params == {"keys": "Tab"}


# ── Rule 2: submit-shaped click loop → Return ──────────────────────────


def test_loop_recovery_forces_return_on_submit_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3 byte-equal clicks on a static env (no focused field) with a
    submit-shaped task → policy forces Return."""
    monkeypatch.delenv("MANTIS_LOOP_RECOVERY", raising=False)

    click = Action(
        ActionType.CLICK, {"x": 150, "y": 200},
        reasoning="Click the submit button.",
    )
    brain = _ScriptedBrain([click, click, click, click])
    env = _StaticEnvWithFocus(focused_label=None)  # no focus
    result = GymRunner(brain, env, max_steps=6).run("Submit the form.")

    assert result.loop_recoveries_by_reason.get(REASON_PRESS_RETURN_FOR_SUBMIT, 0) >= 1
    recovered = [
        t for t in result.trajectory
        if t.loop_recovery_reason == REASON_PRESS_RETURN_FOR_SUBMIT
    ]
    assert len(recovered) == 1
    assert recovered[0].action.action_type == ActionType.KEY_PRESS
    assert recovered[0].action.params == {"keys": "Return"}


# ── Ablation toggle ────────────────────────────────────────────────────


def test_loop_recovery_disabled_via_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANTIS_LOOP_RECOVERY", "disabled")

    click = Action(ActionType.CLICK, {"x": 100, "y": 100})
    brain = _ScriptedBrain([click, click, click, click])
    env = _StaticEnvWithFocus(focused_label="captcha")
    result = GymRunner(brain, env, max_steps=6).run("solve")

    assert result.loop_recoveries_by_reason == {}
    for t in result.trajectory:
        assert t.loop_recovery_reason == ""


# ── No false-positives on non-loop runs ─────────────────────────────────


def test_loop_recovery_does_not_fire_on_diverse_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diverse clicks at different coords don't trigger soft-loop;
    policy should never fire."""
    monkeypatch.delenv("MANTIS_LOOP_RECOVERY", raising=False)

    brain = _ScriptedBrain([
        Action(ActionType.CLICK, {"x": 50, "y": 50}),
        Action(ActionType.CLICK, {"x": 100, "y": 100}),
        Action(ActionType.CLICK, {"x": 150, "y": 150}),
    ])
    env = _StaticEnvWithFocus(focused_label="email")
    result = GymRunner(brain, env, max_steps=5).run("Click around.")

    assert result.loop_recoveries_by_reason == {}


# ── Trajectory round-trip ──────────────────────────────────────────────


def test_trajectory_step_round_trips_loop_recovery_reason() -> None:
    from mantis_agent.gym.runner import (
        TrajectoryStep,
        _trajectory_step_from_dict,
        _trajectory_step_to_dict,
    )

    original = TrajectoryStep(
        step=1,
        action=Action(ActionType.KEY_PRESS, {"keys": "Tab"}),
        thinking="",
        reward=0.0,
        done=False,
        inference_time=0.0,
        loop_recovery_reason=REASON_TAB_TO_NEXT_FIELD,
    )
    payload = _trajectory_step_to_dict(original)
    assert payload["loop_recovery_reason"] == REASON_TAB_TO_NEXT_FIELD
    rehydrated = _trajectory_step_from_dict(payload)
    assert rehydrated.loop_recovery_reason == REASON_TAB_TO_NEXT_FIELD
