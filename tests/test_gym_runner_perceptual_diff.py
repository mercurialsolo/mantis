"""Runner integration for #293 — perceptual-diff verifier wired into GymRunner.

Drives the runner with a scripted brain that emits a high-risk click,
plays back an env that produces an identical post-action frame, and
asserts:

* ``trajectory[k].action_effect_observed is False`` when the page didn't
  change.
* ``run_result.perceptual_summary`` reports the no-effect count.
* The feedback string on the next step gets a ``WARNING: ... no
  observed effect`` line so the brain has a signal.
* Toggle ``MANTIS_PERCEPTUAL_VERIFY=disabled`` short-circuits the check.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pytest
from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
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
        # End of script — terminate the loop.
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


class _StaticEnv(GymEnvironment):
    """Env that always returns the same screenshot — simulates a click
    that produced no visible effect (overlay absorbed it)."""

    def __init__(self) -> None:
        self._frame = _noisy_img(seed=42)

    @property
    def screen_size(self) -> tuple[int, int]:
        return (200, 200)

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        return GymObservation(screenshot=self._frame)

    def step(self, action: Action) -> GymResult:
        return GymResult(
            GymObservation(screenshot=self._frame), 0.0, False, info={},
        )

    def close(self) -> None:
        pass


class _ChangingEnv(GymEnvironment):
    """Env whose screenshot changes on every step — simulates a click
    that actually moved the page forward."""

    def __init__(self) -> None:
        self._step = 0
        self._frames = [_noisy_img(seed=i) for i in range(10)]

    @property
    def screen_size(self) -> tuple[int, int]:
        return (200, 200)

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        self._step = 0
        return GymObservation(screenshot=self._frames[0])

    def step(self, action: Action) -> GymResult:
        self._step += 1
        idx = min(self._step, len(self._frames) - 1)
        return GymResult(
            GymObservation(screenshot=self._frames[idx]), 0.0, False, info={},
        )

    def close(self) -> None:
        pass


def _high_risk_click() -> Action:
    return Action(
        ActionType.CLICK, {"x": 100, "y": 100},
        reasoning="Click the submit button.",
    )


def _benign_click() -> Action:
    return Action(
        ActionType.CLICK, {"x": 100, "y": 100},
        reasoning="Click the next event card.",
    )


# ── Silent-failure detection ───────────────────────────────────────────


def test_high_risk_action_no_effect_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """High-risk click on a static env → action_effect_observed=False
    AND perceptual_summary.no_effect == 1."""
    monkeypatch.delenv("MANTIS_PERCEPTUAL_VERIFY", raising=False)

    runner = GymRunner(
        brain=_ScriptedBrain([_high_risk_click()]),
        env=_StaticEnv(),
        max_steps=3,
    )
    result = runner.run("task")

    high_risk_steps = [
        t for t in result.trajectory if t.action.action_type == ActionType.CLICK
    ]
    assert len(high_risk_steps) == 1
    assert high_risk_steps[0].action_effect_observed is False
    assert result.perceptual_summary == {"checked": 1, "no_effect": 1}


def test_warning_injected_into_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the verifier fires, the feedback string carries a WARNING the
    brain will see on the next step."""
    monkeypatch.delenv("MANTIS_PERCEPTUAL_VERIFY", raising=False)

    runner = GymRunner(
        brain=_ScriptedBrain([_high_risk_click()]),
        env=_StaticEnv(),
        max_steps=3,
    )
    result = runner.run("task")
    click_step = next(
        t for t in result.trajectory
        if t.action.action_type == ActionType.CLICK
    )
    assert "WARNING" in click_step.feedback
    assert "no observed effect" in click_step.feedback


# ── Action-class gating ─────────────────────────────────────────────────


def test_benign_click_skipped_by_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal click (no submit/confirm/buy keyword) → verifier skips,
    action_effect_observed is None, perceptual_summary stays empty."""
    monkeypatch.delenv("MANTIS_PERCEPTUAL_VERIFY", raising=False)

    runner = GymRunner(
        brain=_ScriptedBrain([_benign_click()]),
        env=_StaticEnv(),
        max_steps=3,
    )
    result = runner.run("task")
    click_step = next(
        t for t in result.trajectory
        if t.action.action_type == ActionType.CLICK
    )
    assert click_step.action_effect_observed is None
    assert result.perceptual_summary == {}


# ── Real effect ────────────────────────────────────────────────────────


def test_high_risk_click_with_real_effect_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the env's post-step frame differs, the verifier records
    effect_observed=True and no WARNING is injected."""
    monkeypatch.delenv("MANTIS_PERCEPTUAL_VERIFY", raising=False)

    runner = GymRunner(
        brain=_ScriptedBrain([_high_risk_click()]),
        env=_ChangingEnv(),
        max_steps=3,
    )
    result = runner.run("task")
    click_step = next(
        t for t in result.trajectory
        if t.action.action_type == ActionType.CLICK
    )
    assert click_step.action_effect_observed is True
    assert "WARNING" not in click_step.feedback
    assert result.perceptual_summary == {"checked": 1, "no_effect": 0}


# ── Ablation toggle ────────────────────────────────────────────────────


def test_ablation_toggle_disables_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MANTIS_PERCEPTUAL_VERIFY=disabled → action_effect_observed stays
    None, perceptual_summary stays empty, no WARNING in feedback."""
    monkeypatch.setenv("MANTIS_PERCEPTUAL_VERIFY", "disabled")

    runner = GymRunner(
        brain=_ScriptedBrain([_high_risk_click()]),
        env=_StaticEnv(),
        max_steps=3,
    )
    result = runner.run("task")
    click_step = next(
        t for t in result.trajectory
        if t.action.action_type == ActionType.CLICK
    )
    assert click_step.action_effect_observed is None
    assert result.perceptual_summary == {}
    assert "WARNING" not in click_step.feedback


# ── Trajectory round-trip ───────────────────────────────────────────────


def test_trajectory_step_round_trips_action_effect_observed() -> None:
    from mantis_agent.gym.runner import (
        TrajectoryStep,
        _trajectory_step_from_dict,
        _trajectory_step_to_dict,
    )

    original = TrajectoryStep(
        step=1,
        action=Action(ActionType.CLICK, {"x": 1, "y": 1}),
        thinking="",
        reward=0.0,
        done=False,
        inference_time=0.0,
        action_effect_observed=False,
    )
    payload = _trajectory_step_to_dict(original)
    assert payload["action_effect_observed"] is False
    rehydrated = _trajectory_step_from_dict(payload)
    assert rehydrated.action_effect_observed is False
