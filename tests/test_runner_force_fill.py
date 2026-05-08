from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
from mantis_agent.gym.runner import GymRunner


def test_repeated_form_click_types_next_plan_value() -> None:
    values = [{"label": "zip code", "value": "33101"}]
    used: list[tuple[int, int]] = []
    previous = Action(ActionType.CLICK, {"x": 167, "y": 444})
    current = Action(ActionType.CLICK, {"x": 169, "y": 445})

    forced = GymRunner._maybe_force_type_after_repeated_form_click(
        current,
        [previous],
        values,
        used,
        "Click the ZIP Code field and type 33101.",
    )

    assert forced is not None
    assert forced.action_type == ActionType.TYPE
    assert forced.params == {"text": "33101"}
    assert values == []
    assert used == [(169, 445)]


def test_repeated_non_form_click_does_not_force_type() -> None:
    values = [{"label": "zip code", "value": "33101"}]
    previous = Action(ActionType.CLICK, {"x": 167, "y": 444})
    current = Action(ActionType.CLICK, {"x": 169, "y": 445})

    forced = GymRunner._maybe_force_type_after_repeated_form_click(
        current,
        [previous],
        values,
        [],
        "Open the next boat listing.",
    )

    assert forced is None
    assert values == [{"label": "zip code", "value": "33101"}]


@dataclass
class _BrainResult:
    action: Action
    thinking: str = ""
    predicted_outcome: str = ""


class _ClickBrain:
    def __init__(self) -> None:
        self.calls = 0

    def think(
        self,
        frames: Any,
        task: str,
        action_history: Any = None,
        screen_size: tuple[int, int] = (100, 100),
    ) -> _BrainResult:
        self.calls += 1
        return _BrainResult(Action(ActionType.CLICK, {"x": 169, "y": 445}))


class _RecordingEnv(GymEnvironment):
    def __init__(self) -> None:
        self.actions: list[Action] = []

    @property
    def screen_size(self) -> tuple[int, int]:
        return (100, 100)

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        return GymObservation(
            screenshot=Image.new("RGB", self.screen_size),
            extras={"url": "https://www.boattrader.com/search/index/"},
        )

    def step(self, action: Action) -> GymResult:
        self.actions.append(action)
        return GymResult(
            self.reset(""),
            reward=0.0,
            done=False,
            info={"url": "https://www.boattrader.com/search/index/"},
        )

    def close(self) -> None:
        pass


def test_single_field_force_fill_commits_and_finishes(monkeypatch) -> None:
    from mantis_agent.gym import holo3_detector

    monkeypatch.setattr(
        holo3_detector,
        "extract_form_values",
        lambda brain, task: [{"label": "zip code", "value": "33101"}],
    )
    monkeypatch.setattr(
        holo3_detector,
        "detect_focused_field",
        lambda *args, **kwargs: {
            "focused": True,
            "label": "zip code",
            "type": "text",
        },
    )

    brain = _ClickBrain()
    env = _RecordingEnv()
    result = GymRunner(brain, env, max_steps=5).run(
        "Click the ZIP Code field, type 33101, press Tab, and finish when "
        "33101 is visible in the field."
    )

    assert result.success is True
    assert result.termination_reason == "done"
    assert brain.calls == 1
    assert [a.action_type for a in env.actions] == [
        ActionType.CLICK,
        ActionType.KEY_PRESS,
        ActionType.TYPE,
        ActionType.KEY_PRESS,
    ]
    assert env.actions[0].params == {"x": 169, "y": 445}
    assert env.actions[1].params == {"keys": "ctrl+a"}
    assert env.actions[2].params == {"text": "33101"}
    assert env.actions[3].params == {"keys": "Tab"}


def test_force_fill_does_not_auto_finish_submit_or_login_tasks() -> None:
    assert not GymRunner._force_fill_should_finish_task(
        task="Type the password and submit the login form.",
        initial_value_count=1,
        pending_value_count=0,
        submitted=False,
    )


def test_repeated_top_click_redirects_forward_task_to_scroll() -> None:
    previous = Action(ActionType.CLICK, {"x": 263, "y": 63})
    current = Action(ActionType.CLICK, {"x": 275, "y": 64})

    redirected = GymRunner._maybe_redirect_repeated_top_click(
        current,
        [previous],
        "Click the visible Search button to submit the form.",
    )

    assert redirected is not None
    assert redirected.action_type == ActionType.SCROLL
    assert redirected.params == {"direction": "down", "amount": 350}


def test_repeated_top_click_guard_ignores_non_forward_tasks() -> None:
    previous = Action(ActionType.CLICK, {"x": 263, "y": 63})
    current = Action(ActionType.CLICK, {"x": 275, "y": 64})

    redirected = GymRunner._maybe_redirect_repeated_top_click(
        current,
        [previous],
        "Inspect the page header.",
    )

    assert redirected is None
