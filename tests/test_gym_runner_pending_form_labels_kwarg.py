"""Test for #306 follow-up — GymRunner.run accepts cross-layer
``pending_form_labels`` kwarg, and the done-gate honours it.

When ``MicroPlanRunner`` spawns an inner ``GymRunner`` via
``Holo3StepHandler``, the inner runner has its own ``FormController``
with values extracted from the inner sub-step's intent. The outer
runner may track values pending across the whole plan that the inner
can't see. The new kwarg lets the outer pass that state in so the
inner done-gate rejects ``done(success=True)`` while outer values
remain unconsumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
from mantis_agent.gym.done_gate import REJECT_PENDING_FORM_VALUES
from mantis_agent.gym.runner import GymRunner


@dataclass
class _BrainResult:
    action: Action
    thinking: str = ""
    predicted_outcome: str = ""


class _DoneBrain:
    def think(
        self,
        frames: Any,
        task: str,
        action_history: Any = None,
        screen_size: tuple[int, int] = (100, 100),
    ) -> _BrainResult:
        return _BrainResult(
            Action(
                ActionType.DONE,
                {"success": True, "summary": "all done"},
            )
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


def test_outer_pending_labels_reject_done_at_inner_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inner brain emits done(success=True). Outer caller passed
    ``pending_form_labels=["password"]`` — the inner done-gate must
    reject as ``pending_form_values`` even though the inner runner's
    own FormController is empty."""
    monkeypatch.delenv("MANTIS_DONE_GATE", raising=False)
    monkeypatch.delenv("MANTIS_FORM_CONTROLLER", raising=False)

    runner = GymRunner(_DoneBrain(), _Env(), max_steps=3)
    result = runner.run(
        "task with no creds in text",
        pending_form_labels=["password"],
    )

    # Gate rejects repeatedly until max_done_rejections — what matters
    # is at least one rejection with the right reason code.
    assert (
        result.done_rejections_by_reason.get(REJECT_PENDING_FORM_VALUES, 0) >= 1
    )
    rejected_steps = [
        t for t in result.trajectory if t.done_rejected_reason
    ]
    assert len(rejected_steps) >= 1
    assert rejected_steps[0].done_rejected_reason == REJECT_PENDING_FORM_VALUES


def test_default_none_falls_back_to_inner_form_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the kwarg isn't passed, the inner gate sources from the
    inner FormController as before — backward-compat for /v1/cua."""
    monkeypatch.delenv("MANTIS_DONE_GATE", raising=False)
    monkeypatch.delenv("MANTIS_FORM_CONTROLLER", raising=False)

    runner = GymRunner(_DoneBrain(), _Env(), max_steps=3)
    result = runner.run("task with no creds in text")

    # No outer hint, no inner pending → gate accepts the well-formed
    # done(success=True, summary="all done").
    assert REJECT_PENDING_FORM_VALUES not in result.done_rejections_by_reason


def test_empty_outer_pending_labels_treated_as_no_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit empty list is semantically "outer says nothing pending"
    and should NOT trigger the rejection — matches the FormController
    semantic (empty pending list = no rejection)."""
    monkeypatch.delenv("MANTIS_DONE_GATE", raising=False)
    monkeypatch.delenv("MANTIS_FORM_CONTROLLER", raising=False)

    runner = GymRunner(_DoneBrain(), _Env(), max_steps=3)
    result = runner.run(
        "task",
        pending_form_labels=[],
    )

    assert REJECT_PENDING_FORM_VALUES not in result.done_rejections_by_reason
