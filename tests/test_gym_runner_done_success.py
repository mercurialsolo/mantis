"""GymRunner terminal success should follow done(success=...)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
from mantis_agent.gym.runner import GymRunner


@dataclass
class _BrainResult:
    action: Action
    thinking: str = ""
    predicted_outcome: str = ""


class _DoneBrain:
    def __init__(self, success: bool) -> None:
        self.success = success

    def think(
        self,
        frames: Any,
        task: str,
        action_history: Any = None,
        screen_size: tuple[int, int] = (100, 100),
    ) -> _BrainResult:
        return _BrainResult(
            Action(ActionType.DONE, {"success": self.success, "summary": "done"})
        )


class _Env(GymEnvironment):
    @property
    def screen_size(self) -> tuple[int, int]:
        return (100, 100)

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        return GymObservation(screenshot=Image.new("RGB", self.screen_size))

    def step(self, action: Action) -> GymResult:
        return GymResult(self.reset(""), reward=0.0, done=False, info={})

    def close(self) -> None:
        pass


def test_done_success_false_returns_failed_result() -> None:
    result = GymRunner(_DoneBrain(False), _Env(), max_steps=1).run("task")

    assert result.termination_reason == "done"
    assert result.success is False


def test_done_success_true_returns_successful_result() -> None:
    result = GymRunner(_DoneBrain(True), _Env(), max_steps=1).run("task")

    assert result.termination_reason == "done"
    assert result.success is True
