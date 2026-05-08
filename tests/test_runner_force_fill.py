from __future__ import annotations

from mantis_agent.actions import Action, ActionType
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
