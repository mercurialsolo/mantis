"""GymRunner integration for #291 — predicates evaluated per step.

Drives the runner with a fake brain that emits a structured ``expected``
JSON block, then asserts the trajectory carries per-predicate booleans
plus a ``world_model_error`` reward component.
"""

from __future__ import annotations

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
    """Emits a fixed sequence of (action, predicted_outcome) tuples."""

    def __init__(self, script: list[tuple[Action, str]]) -> None:
        self.script = list(script)
        self.calls = 0

    def think(
        self,
        frames: Any,
        task: str,
        action_history: Any = None,
        screen_size: tuple[int, int] = (100, 100),
    ) -> _BrainResult:
        if self.calls >= len(self.script):
            action = Action(ActionType.DONE, {"success": True, "summary": "done"})
            self.calls += 1
            return _BrainResult(action)
        action, pred = self.script[self.calls]
        self.calls += 1
        return _BrainResult(action, predicted_outcome=pred)


class _ScriptedEnv(GymEnvironment):
    """Plays back a fixed sequence of (url, title) post-step observations."""

    def __init__(self, observations: list[tuple[str, str]]) -> None:
        self.observations = list(observations)
        self.calls = 0
        # Two distinct images so frame_hash differs on transition.
        self._white = Image.new("RGB", (100, 100), color=(255, 255, 255))
        self._black = Image.new("RGB", (100, 100), color=(0, 0, 0))

    @property
    def screen_size(self) -> tuple[int, int]:
        return (100, 100)

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        self.calls = 0
        return GymObservation(screenshot=self._white)

    def step(self, action: Action) -> GymResult:
        if self.calls >= len(self.observations):
            url, title = self.observations[-1] if self.observations else ("", "")
        else:
            url, title = self.observations[self.calls]
        self.calls += 1
        # Alternate frame so frame_changed predicates evaluate True.
        screenshot = self._black if self.calls % 2 == 1 else self._white
        return GymResult(
            GymObservation(screenshot=screenshot),
            reward=0.0,
            done=False,
            info={"url": url, "title": title},
        )

    def close(self) -> None:
        pass


def _click() -> Action:
    return Action(ActionType.CLICK, {"x": 1, "y": 1})


def test_runner_evaluates_predicates_and_emits_world_model_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Brain predicts url_changed; env confirms — error component should be 0."""
    monkeypatch.delenv("MANTIS_PREDICATE_VERIFY", raising=False)

    brain = _ScriptedBrain([
        (_click(), '{"expected": ["url_changed", "title_changed"]}'),
    ])
    env = _ScriptedEnv([("https://x.test/b", "Page B")])

    result = GymRunner(brain, env, max_steps=2).run("task")

    # First step is the click; second is the auto-DONE the scripted brain emits.
    step = result.trajectory[0]
    assert step.predicate_results, "predicate_results should be populated"
    kinds = [r["predicate"] for r in step.predicate_results]
    assert "url_changed" in kinds
    assert "title_changed" in kinds
    # Both predicates were True (URL/title changed from "" to a non-empty value).
    assert all(r["result"] is True for r in step.predicate_results)
    # All-correct => world_model_error contribution == 0 (no penalty).
    assert step.reward_components.get("world_model_error", 0.0) == 0.0


def test_runner_penalizes_wrong_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MANTIS_PREDICATE_VERIFY", raising=False)

    brain = _ScriptedBrain([
        # Brain predicts URL will contain '/checkout', but env stays on /home.
        (_click(), '{"expected": ["url_contains:/checkout"]}'),
    ])
    env = _ScriptedEnv([("https://x.test/home", "Home")])

    result = GymRunner(brain, env, max_steps=2).run("task")
    step = result.trajectory[0]
    assert step.predicate_results[0]["result"] is False
    # All-wrong => error == 1.0, weighted by 0.05.
    assert step.reward_components["world_model_error"] == pytest.approx(-0.05)


def test_runner_skips_when_brain_emits_no_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MANTIS_PREDICATE_VERIFY", raising=False)

    brain = _ScriptedBrain([(_click(), "")])  # no prediction
    env = _ScriptedEnv([("https://x.test/a", "A")])

    result = GymRunner(brain, env, max_steps=2).run("task")
    step = result.trajectory[0]
    assert step.predicate_results == []
    assert "world_model_error" not in step.reward_components


def test_runner_disabled_via_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """MANTIS_PREDICATE_VERIFY=disabled is the ablation toggle."""
    monkeypatch.setenv("MANTIS_PREDICATE_VERIFY", "disabled")

    brain = _ScriptedBrain([
        (_click(), '{"expected": ["url_changed"]}'),
    ])
    env = _ScriptedEnv([("https://x.test/b", "B")])

    result = GymRunner(brain, env, max_steps=2).run("task")
    step = result.trajectory[0]
    # Predicates parsed but not evaluated — no per-step results, no reward
    # contribution. predicted_outcome is still recorded for distillation.
    assert step.predicate_results == []
    assert "world_model_error" not in step.reward_components
    assert step.predicted_outcome == '{"expected": ["url_changed"]}'


def test_runner_unevaluable_predicate_skipped_from_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A best-effort predicate (no DOM) returns None and is excluded
    from the world_model_error denominator."""
    monkeypatch.delenv("MANTIS_PREDICATE_VERIFY", raising=False)

    brain = _ScriptedBrain([
        (_click(),
         '{"expected": ["url_changed", "modal_opens"]}'),
    ])
    env = _ScriptedEnv([("https://x.test/b", "B")])

    result = GymRunner(brain, env, max_steps=2).run("task")
    step = result.trajectory[0]
    results_by_kind = {r["predicate"]: r["result"] for r in step.predicate_results}
    assert results_by_kind["url_changed"] is True
    assert results_by_kind["modal_opens"] is None
    # Only url_changed contributes — it was correct, so error == 0.
    assert step.reward_components.get("world_model_error", 0.0) == 0.0


def test_trajectory_step_to_dict_round_trips_predicate_results() -> None:
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
        predicate_results=[
            {"predicate": "url_changed", "result": True, "reason": "/a -> /b"},
            {"predicate": "modal_opens", "result": None, "reason": "no DOM"},
        ],
    )
    payload = _trajectory_step_to_dict(original)
    rehydrated = _trajectory_step_from_dict(payload)
    assert rehydrated.predicate_results == original.predicate_results
