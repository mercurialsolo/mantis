"""Tests for #123 — curriculum re-injection on soft-loop nudge.

We exercise GymRunner._curriculum_refresher in isolation since the full
runner loop pulls in heavy deps. The refresher is a pure helper.
"""

from __future__ import annotations

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.runner import GymRunner


def test_refresher_empty_when_no_history() -> None:
    assert GymRunner._curriculum_refresher([], None) == ""


def test_refresher_for_repeated_click_on_focused_input_returns_form_technique() -> None:
    history = [Action(ActionType.CLICK, {"x": 100, "y": 100})] * 3
    focused = {"placeholder": "search", "empty": True}
    out = GymRunner._curriculum_refresher(history, focused)
    # The chrome_forms technique mentions type_text — a robust signature.
    assert "type_text" in out or "form" in out.lower(), out


def test_refresher_for_scroll_returns_non_empty_or_safely_empty() -> None:
    history = [Action(ActionType.SCROLL, {"direction": "down", "amount": 3})] * 3
    out = GymRunner._curriculum_refresher(history, None)
    # We only require that this doesn't crash; the curriculum's TF-IDF may
    # or may not match a scroll-specific hint depending on registered topics.
    assert isinstance(out, str)


def test_refresher_for_alt_left_keypress_returns_navigation_or_empty() -> None:
    history = [Action(ActionType.KEY_PRESS, {"keys": "alt+left"})] * 3
    out = GymRunner._curriculum_refresher(history, None)
    assert isinstance(out, str)


def test_refresher_for_unknown_action_type_returns_empty() -> None:
    # WAIT isn't covered — should return "" without exception.
    history = [Action(ActionType.WAIT, {"seconds": 1})]
    assert GymRunner._curriculum_refresher(history, None) == ""


def test_refresher_handles_curriculum_import_failure_gracefully(monkeypatch) -> None:
    """If the curriculum module is broken/missing, refresher returns ''."""
    import sys

    # Force ImportError by stashing the module.
    saved = sys.modules.pop("mantis_agent.curriculum", None)
    monkeypatch.setitem(sys.modules, "mantis_agent.curriculum", None)
    try:
        history = [Action(ActionType.CLICK, {"x": 1, "y": 1})]
        out = GymRunner._curriculum_refresher(history, None)
        assert out == ""
    finally:
        if saved is not None:
            sys.modules["mantis_agent.curriculum"] = saved
        else:
            sys.modules.pop("mantis_agent.curriculum", None)
