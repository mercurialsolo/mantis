"""Regression tests for #574 — Holo3 click(0,0) silent dispatch.

Before this fix, ``_parse_holo3_action`` returned
``Action(CLICK, {x:0, y:0})`` whenever the model emitted ``click()``
without coordinates (or with malformed values that ``_safe_int``
coerced to 0). On listings/loop plans, that dispatched to the page
origin → wrong-page extract → DUPLICATE-URL loop → halt with 0 leads.

Empirical 2026-05-21: boattrader run ``20260521_205442_b3428c17``
logged dozens of ``[click] (0,0) grounding=NO`` lines after CF
cleared. The fix: parse-fail (None) when x/y are missing or both
zero; brain's normal fallthrough emits ``Action(WAIT, 1s)`` instead.
"""

from __future__ import annotations

from mantis_agent.actions import ActionType
from mantis_agent.brain_holo3 import Holo3Brain


SCREEN = (1280, 720)


def _parse(text: str):
    return Holo3Brain._parse_holo3_action(
        Holo3Brain.__new__(Holo3Brain), text, SCREEN,
    )


# ── click — invalid coord variants must return None ────────────────


def test_click_with_no_args_returns_none() -> None:
    assert _parse("Action: click()") is None


def test_click_with_empty_dict_returns_none() -> None:
    assert _parse("Action: click({})") is None


def test_click_with_only_button_arg_returns_none() -> None:
    assert _parse("Action: click({'button': 'left'})") is None


def test_click_with_x_only_returns_none() -> None:
    assert _parse("Action: click({'x': 500})") is None


def test_click_with_y_only_returns_none() -> None:
    assert _parse("Action: click({'y': 300})") is None


def test_click_with_explicit_zero_zero_returns_none() -> None:
    # Page origin is never a legitimate click target — reject to avoid
    # the DUPLICATE-URL loop on listings plans.
    assert _parse("Action: click({'x': 0, 'y': 0})") is None


def test_click_with_malformed_xy_returns_none() -> None:
    # _safe_int(default=-1) returns -1 for non-parseable; we reject < 0.
    assert _parse("Action: click({'x': 'abc', 'y': 'def'})") is None


# ── click — valid coords still dispatch ─────────────────────────────


def test_click_with_valid_coords_returns_click_action() -> None:
    action = _parse("Action: click({'x': 500, 'y': 300})")
    assert action is not None
    assert action.action_type == ActionType.CLICK
    # Coords go through _model_coords_to_screen scaling; verify they're
    # non-zero (the actual scaled values depend on model-image dims).
    assert action.params.get("x") != 0 or action.params.get("y") != 0


def test_click_with_x_zero_y_nonzero_dispatches() -> None:
    # Only BOTH-zero is rejected; (0, 300) is a legitimate left-edge click.
    action = _parse("Action: click({'x': 0, 'y': 300})")
    assert action is not None
    assert action.action_type == ActionType.CLICK


def test_click_with_x_nonzero_y_zero_dispatches() -> None:
    action = _parse("Action: click({'x': 500, 'y': 0})")
    assert action is not None
    assert action.action_type == ActionType.CLICK


# ── double_click — same coverage ───────────────────────────────────


def test_double_click_with_no_args_returns_none() -> None:
    assert _parse("Action: double_click()") is None


def test_double_click_with_zero_zero_returns_none() -> None:
    assert _parse("Action: double_click({'x': 0, 'y': 0})") is None


def test_double_click_with_valid_coords_dispatches() -> None:
    action = _parse("Action: double_click({'x': 500, 'y': 300})")
    assert action is not None
    assert action.action_type == ActionType.DOUBLE_CLICK


# ── non-click actions unaffected ───────────────────────────────────


def test_scroll_unaffected_by_xy_check() -> None:
    # scroll uses direction + amount, not x/y. Must still parse.
    action = _parse("Action: scroll({'direction': 'down', 'amount': 5})")
    assert action is not None
    assert action.action_type == ActionType.SCROLL


def test_type_text_unaffected() -> None:
    action = _parse("Action: type_text({'text': 'hello'})")
    assert action is not None
    assert action.action_type == ActionType.TYPE


def test_done_unaffected() -> None:
    action = _parse("Action: done({'success': true, 'summary': 'ok'})")
    assert action is not None
    assert action.action_type == ActionType.DONE
