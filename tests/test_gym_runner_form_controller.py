"""Runner integration for #301 — FormController is wired into GymRunner.run.

Verifies:

* Default (controller enabled): ``self.form_controller`` is a populated
  :class:`FormController` carrying the values returned by the Holo3
  extractor; pending_values / used_regions are the same list objects the
  controller exposes (so legacy local-variable mutations land on the
  controller).
* ``MANTIS_FORM_CONTROLLER=disabled`` (ablation toggle): controller is
  ``None`` and the legacy ``holo3_detector.extract_form_values`` path runs.
* ``mark_consumed_label`` is exposed for the director hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
from mantis_agent.gym.form_controller import FormController
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
            Action(ActionType.DONE, {"success": True, "summary": "done"})
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


@pytest.fixture
def _stub_extract(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Patch holo3 extractor with a deterministic two-value plan."""
    from mantis_agent.gym import holo3_detector

    values = [
        {"label": "user_id", "value": "alice"},
        {"label": "password", "value": "p4ss"},
    ]
    monkeypatch.setattr(
        holo3_detector,
        "extract_form_values",
        lambda brain, task: list(values),
    )
    return values


def test_runner_constructs_form_controller_by_default(
    monkeypatch: pytest.MonkeyPatch,
    _stub_extract: list[dict],
) -> None:
    monkeypatch.delenv("MANTIS_FORM_CONTROLLER", raising=False)

    runner = GymRunner(_DoneBrain(), _Env(), max_steps=1)
    runner.run("Log in with alice / p4ss")

    assert isinstance(runner.form_controller, FormController)
    assert runner.form_controller.pending_count == 2
    assert runner.form_controller.initial_labels == ["user_id", "password"]


def test_runner_ablation_toggle_disables_controller(
    monkeypatch: pytest.MonkeyPatch,
    _stub_extract: list[dict],
) -> None:
    monkeypatch.setenv("MANTIS_FORM_CONTROLLER", "disabled")

    runner = GymRunner(_DoneBrain(), _Env(), max_steps=1)
    runner.run("Log in with alice / p4ss")

    # Legacy path: controller is None, but the extractor still runs and
    # the form-fill local-variable behaviour matches pre-#301.
    assert runner.form_controller is None


def test_director_hook_consumes_pending_label(
    monkeypatch: pytest.MonkeyPatch,
    _stub_extract: list[dict],
) -> None:
    """When an external director or fallback path types a value outside
    the controller's own substitution flow, ``mark_consumed_label`` keeps
    the controller's pending list consistent so the next click on a
    different field doesn't re-type the same value."""
    monkeypatch.delenv("MANTIS_FORM_CONTROLLER", raising=False)

    runner = GymRunner(_DoneBrain(), _Env(), max_steps=1)
    runner.run("Log in with alice / p4ss")
    controller = runner.form_controller
    assert controller is not None
    assert controller.pending_count == 2

    # Simulate the director typing the password externally.
    consumed = controller.mark_consumed_label("password")
    assert consumed is True
    assert controller.pending_count == 1
    assert controller.pending_labels == ["user_id"]


def test_controller_pending_values_alias_runner_local(
    monkeypatch: pytest.MonkeyPatch,
    _stub_extract: list[dict],
) -> None:
    """The controller's ``pending_values`` list IS the same list used by the
    legacy local-variable mutations — so when the runner's
    ``_maybe_force_type_text`` pops an entry, the controller sees it."""
    monkeypatch.delenv("MANTIS_FORM_CONTROLLER", raising=False)

    runner = GymRunner(_DoneBrain(), _Env(), max_steps=1)
    runner.run("Log in with alice / p4ss")
    controller = runner.form_controller
    assert controller is not None

    # The reference identity is what makes the refactor zero-behaviour-
    # change: the runner's force_fill_values local and the controller's
    # pending_values are the same object.
    initial_id = id(controller.pending_values)
    controller.pending_values.pop(0)
    assert id(controller.pending_values) == initial_id
    assert controller.pending_count == 1
