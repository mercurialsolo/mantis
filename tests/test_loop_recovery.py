"""Tests for #302 — LoopRecoveryPolicy.

Deterministic via direct calls to ``decide_recovery``. Covers each
rule, the ablation toggle, and the "no rule applies" fallthrough.
"""

from __future__ import annotations

import pytest

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.loop_recovery import (
    REASON_CODES,
    REASON_PRESS_RETURN_FOR_SUBMIT,
    REASON_TAB_TO_NEXT_FIELD,
    REASON_TYPE_PENDING_VALUE,
    LoopRecoveryDecision,
    decide_recovery,
    is_enabled,
)


def _click(x: int = 100, y: int = 200, *, reasoning: str = "") -> Action:
    return Action(ActionType.CLICK, {"x": x, "y": y}, reasoning=reasoning)


def _common(**overrides) -> dict:
    """Default kwargs for decide_recovery — tests override per case."""
    defaults: dict = {
        "action": _click(),
        "action_history": [_click(), _click(), _click()],
        "focused_input": None,
        "pending_form_values": [],
        "recent_frame_hashes": ["abc", "abc", "abc"],
        "task": "do a thing",
        "soft_loop_window": 3,
    }
    defaults.update(overrides)
    return defaults


# ── Toggle ─────────────────────────────────────────────────────────────


def test_is_enabled_default_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANTIS_LOOP_RECOVERY", raising=False)
    assert is_enabled() is True


def test_is_enabled_disabled_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_LOOP_RECOVERY", "disabled")
    assert is_enabled() is False


def test_disabled_toggle_returns_no_forced_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANTIS_LOOP_RECOVERY", "disabled")
    decision = decide_recovery(**_common(
        focused_input={"name": "email"},
        pending_form_values=[{"label": "email", "value": "x@y.com"}],
    ))
    assert decision.forced_action is None
    assert bool(decision) is False


# ── Rule 1a: focused-click with pending value → type ──────────────────


def test_focused_click_with_pending_value_forces_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MANTIS_LOOP_RECOVERY", raising=False)
    decision = decide_recovery(**_common(
        focused_input={"name": "username"},
        pending_form_values=[{"label": "username", "value": "alice"}],
    ))
    assert decision.reason == REASON_TYPE_PENDING_VALUE
    assert decision.forced_action is not None
    assert decision.forced_action.action_type == ActionType.TYPE
    assert decision.forced_action.params == {"text": "alice"}


def test_focused_click_with_empty_value_falls_through_to_tab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pending entry has empty value → can't type, fall through to Tab."""
    monkeypatch.delenv("MANTIS_LOOP_RECOVERY", raising=False)
    decision = decide_recovery(**_common(
        focused_input={"name": "username"},
        pending_form_values=[{"label": "username", "value": ""}],
    ))
    assert decision.reason == REASON_TAB_TO_NEXT_FIELD


# ── Rule 1b: focused-click without pending value → Tab ────────────────


def test_focused_click_no_pending_values_forces_tab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MANTIS_LOOP_RECOVERY", raising=False)
    decision = decide_recovery(**_common(
        focused_input={"name": "captcha"},
        pending_form_values=[],
    ))
    assert decision.reason == REASON_TAB_TO_NEXT_FIELD
    assert decision.forced_action is not None
    assert decision.forced_action.action_type == ActionType.KEY_PRESS
    assert decision.forced_action.params == {"keys": "Tab"}


# ── Rule 1c: submit-shaped focused-click loop → Return ────────────────


def test_focused_submit_shaped_click_forces_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A form input is focused and the brain loops on a submit-shaped
    click it can't land. Press Return so the focused field's form submits
    via the keyboard, instead of Tabbing focus off the field."""
    monkeypatch.delenv("MANTIS_LOOP_RECOVERY", raising=False)
    decision = decide_recovery(**_common(
        action=_click(reasoning="Click the Contact Seller button to submit the form."),
        focused_input={"name": "email"},
        pending_form_values=[],
        recent_frame_hashes=["aaa", "aaa", "aaa"],
        task="Submit the seller contact form.",
    ))
    assert decision.reason == REASON_PRESS_RETURN_FOR_SUBMIT
    assert decision.forced_action is not None
    assert decision.forced_action.action_type == ActionType.KEY_PRESS
    assert decision.forced_action.params == {"keys": "Return"}


def test_focused_submit_shaped_click_changing_frame_falls_through_to_tab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit-shaped focused click but the frame is changing → the click
    is doing something, so fall through to the milder Tab nudge."""
    monkeypatch.delenv("MANTIS_LOOP_RECOVERY", raising=False)
    decision = decide_recovery(**_common(
        action=_click(reasoning="Click submit."),
        focused_input={"name": "email"},
        pending_form_values=[],
        recent_frame_hashes=["aaa", "bbb", "ccc"],
        task="Submit the form.",
    ))
    assert decision.reason == REASON_TAB_TO_NEXT_FIELD


# ── Rule 2: submit-shaped click with frozen frame → Return ────────────


def test_submit_click_frozen_frame_forces_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MANTIS_LOOP_RECOVERY", raising=False)
    decision = decide_recovery(**_common(
        action=_click(reasoning="Click the submit button."),
        focused_input=None,
        recent_frame_hashes=["aaa", "aaa", "aaa"],
        task="Submit the form.",
    ))
    assert decision.reason == REASON_PRESS_RETURN_FOR_SUBMIT
    assert decision.forced_action is not None
    assert decision.forced_action.action_type == ActionType.KEY_PRESS
    assert decision.forced_action.params == {"keys": "Return"}


def test_submit_click_with_changing_frame_no_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the frame is changing the click is producing visible effect —
    no recovery needed even though it's submit-shaped."""
    monkeypatch.delenv("MANTIS_LOOP_RECOVERY", raising=False)
    decision = decide_recovery(**_common(
        action=_click(reasoning="Click the submit button."),
        focused_input=None,
        recent_frame_hashes=["aaa", "bbb", "ccc"],
        task="Submit the form.",
    ))
    assert decision.forced_action is None


def test_submit_click_task_hint_alone_triggers_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Brain's reasoning is empty but the run-level task mentions
    'submit' — recovery still fires."""
    monkeypatch.delenv("MANTIS_LOOP_RECOVERY", raising=False)
    decision = decide_recovery(**_common(
        action=_click(reasoning=""),
        focused_input=None,
        recent_frame_hashes=["aaa", "aaa", "aaa"],
        task="Log in to staff-crm with credentials.",
    ))
    assert decision.reason == REASON_PRESS_RETURN_FOR_SUBMIT


def test_non_submit_shaped_click_no_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated click on a listing card (no submit context) should
    NOT trigger Return — that'd press Enter on a card and likely break
    the flow."""
    monkeypatch.delenv("MANTIS_LOOP_RECOVERY", raising=False)
    decision = decide_recovery(**_common(
        action=_click(reasoning="Click the first event card."),
        focused_input=None,
        recent_frame_hashes=["aaa", "aaa", "aaa"],
        task="Find an event and observe its details.",
    ))
    assert decision.forced_action is None


# ── No applicable rule ────────────────────────────────────────────────


def test_non_click_action_no_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANTIS_LOOP_RECOVERY", raising=False)
    decision = decide_recovery(**_common(
        action=Action(ActionType.SCROLL, {"direction": "down", "amount": 3}),
    ))
    assert decision.forced_action is None


def test_submit_shaped_click_with_short_frame_window_no_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Need at least ``soft_loop_window`` frame hashes to declare stable."""
    monkeypatch.delenv("MANTIS_LOOP_RECOVERY", raising=False)
    decision = decide_recovery(**_common(
        action=_click(reasoning="Click submit."),
        focused_input=None,
        recent_frame_hashes=["aaa"],  # only 1 hash, window=3
        task="Submit.",
    ))
    assert decision.forced_action is None


def test_submit_shaped_click_with_empty_hash_no_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty hash in the window disqualifies — same policy as the
    perceptual-diff verifier's missing-frame fallthrough."""
    monkeypatch.delenv("MANTIS_LOOP_RECOVERY", raising=False)
    decision = decide_recovery(**_common(
        action=_click(reasoning="Click submit."),
        focused_input=None,
        recent_frame_hashes=["", "aaa", "aaa"],
        task="Submit.",
    ))
    assert decision.forced_action is None


# ── Public surface lock ────────────────────────────────────────────────


def test_reason_codes_tuple_locked() -> None:
    """Codes round-trip into TrajectoryStep.loop_recovery_reason and the
    /v1/cua API surface — renames are a breaking change."""
    assert REASON_CODES == (
        "type_pending_value",
        "tab_to_next_field",
        "press_return_for_submit",
    )


def test_decision_truthy_when_action_present() -> None:
    d = LoopRecoveryDecision(forced_action=_click(), reason="x")
    assert bool(d) is True


def test_decision_falsy_when_no_action() -> None:
    d = LoopRecoveryDecision(forced_action=None)
    assert bool(d) is False
