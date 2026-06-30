"""type-stall fix: env.step(CLICK) reports when a click focused a text field.

The /v1/cua brain stalled on search/login/composer because clicking an input
is visually silent — the runner read "no observed effect" and the brain
wait()/re-clicked instead of typing. The env now probes activeElement after a
click and surfaces ``focused_input`` so the runner can tell the brain to type
(and so loop-recovery's submit-Return path, gated on focused_input is None,
no longer misfires).
"""

from __future__ import annotations

import pytest
from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymObservation
from mantis_agent.gym.xdotool_env import XdotoolGymEnv


@pytest.fixture
def click_env(monkeypatch: pytest.MonkeyPatch) -> XdotoolGymEnv:
    env = XdotoolGymEnv.__new__(XdotoolGymEnv)
    env._human_speed = False
    env._settle_time = 0.0
    monkeypatch.setattr(XdotoolGymEnv, "_execute_action", lambda self, a: None)
    monkeypatch.setattr(
        XdotoolGymEnv, "_capture",
        lambda self: GymObservation(screenshot=Image.new("RGB", (32, 32))),
    )
    monkeypatch.setattr("mantis_agent.gym.xdotool_env.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.xdotool_env.adaptive_settle.is_enabled", lambda: False,
    )
    return env


def _click() -> Action:
    return Action(ActionType.CLICK, {"x": 270, "y": 35})


def test_click_into_field_sets_focused_input(click_env, monkeypatch) -> None:
    monkeypatch.setattr(
        XdotoolGymEnv, "_active_field_probe",
        lambda self: {"has_field": True, "text": "machine"},
    )
    result = click_env.step(_click())
    fi = result.info.get("focused_input")
    assert isinstance(fi, dict)
    assert fi["focused_via_click"] is True
    assert fi["value"] == "machine"


def test_click_not_into_field_no_focused_input(click_env, monkeypatch) -> None:
    monkeypatch.setattr(
        XdotoolGymEnv, "_active_field_probe", lambda self: {"has_field": False},
    )
    result = click_env.step(_click())
    assert "focused_input" not in result.info


def test_click_probe_none_no_focused_input(click_env, monkeypatch) -> None:
    """CDP unavailable → probe None → no focused_input (no false signal)."""
    monkeypatch.setattr(XdotoolGymEnv, "_active_field_probe", lambda self: None)
    result = click_env.step(_click())
    assert "focused_input" not in result.info


def test_click_probe_exception_is_swallowed(click_env, monkeypatch) -> None:
    def _boom(self):
        raise RuntimeError("cdp down")
    monkeypatch.setattr(XdotoolGymEnv, "_active_field_probe", _boom)
    result = click_env.step(_click())  # must not raise
    assert "focused_input" not in result.info
