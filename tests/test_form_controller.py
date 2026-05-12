"""Tests for #301 — FormController.

Covers the controller's stateful surface: pending-list semantics, the
``mark_consumed_label`` external-mover hook, used-region bookkeeping,
the submit latch, and delegation to the existing ``GymRunner`` static
helpers (so the controller and the legacy direct calls stay in sync).
"""

from __future__ import annotations

from typing import Any

import pytest

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.form_controller import FormController


# ── Construction & read-only views ─────────────────────────────────────


def test_default_construction_is_empty() -> None:
    c = FormController()
    assert c.has_pending is False
    assert c.pending_count == 0
    assert c.consumed_count == 0
    assert c.initial_labels == []
    assert c.pending_labels == []
    assert c.submitted is False


def test_seeded_construction_records_initial_labels() -> None:
    c = FormController(
        pending_values=[
            {"label": "user_id", "value": "alice"},
            {"label": "password", "value": "p4ss"},
        ],
        initial_labels=["user_id", "password"],
    )
    assert c.has_pending is True
    assert c.pending_count == 2
    assert c.consumed_count == 0
    assert c.pending_labels == ["user_id", "password"]


def test_consumed_count_tracks_pending() -> None:
    c = FormController(
        pending_values=[
            {"label": "a", "value": "1"},
            {"label": "b", "value": "2"},
        ],
        initial_labels=["a", "b"],
    )
    c.pending_values.pop(0)
    assert c.consumed_count == 1
    assert c.pending_labels == ["b"]


# ── from_task factory delegates to extractor ───────────────────────────


def test_from_task_invokes_holo3_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    from mantis_agent.gym import holo3_detector

    captured: dict[str, Any] = {}

    def fake_extract(brain: Any, task: str) -> list[dict]:
        captured["brain"] = brain
        captured["task"] = task
        return [
            {"label": "Email", "value": "x@y.com"},
            {"label": "Password", "value": "secret"},
        ]

    monkeypatch.setattr(holo3_detector, "extract_form_values", fake_extract)

    c = FormController.from_task(brain="STUB", task="Log in with x@y.com / secret")
    assert captured["brain"] == "STUB"
    assert captured["task"] == "Log in with x@y.com / secret"
    assert c.pending_count == 2
    # initial_labels snapshotted from the values
    assert c.initial_labels == ["Email", "Password"]


# ── mark_consumed_label external hook (#301 acceptance bullet 6) ───────


def test_mark_consumed_label_drops_matching_entry() -> None:
    c = FormController(
        pending_values=[
            {"label": "user_id", "value": "alice"},
            {"label": "password", "value": "p4ss"},
        ],
        initial_labels=["user_id", "password"],
    )
    assert c.mark_consumed_label("password") is True
    assert c.pending_labels == ["user_id"]


def test_mark_consumed_label_substring_match() -> None:
    c = FormController(
        pending_values=[{"label": "User Email", "value": "x@y.com"}],
        initial_labels=["User Email"],
    )
    # Director hook may report the field as just "email"; substring match
    # both directions handles the normal label-shape mismatches.
    assert c.mark_consumed_label("email") is True
    assert c.pending_count == 0


def test_mark_consumed_label_case_insensitive() -> None:
    c = FormController(
        pending_values=[{"label": "PASSWORD", "value": "p4ss"}],
        initial_labels=["PASSWORD"],
    )
    assert c.mark_consumed_label("password") is True


def test_mark_consumed_label_returns_false_on_no_match() -> None:
    c = FormController(
        pending_values=[{"label": "user_id", "value": "alice"}],
        initial_labels=["user_id"],
    )
    assert c.mark_consumed_label("captcha") is False
    assert c.pending_count == 1


def test_mark_consumed_label_ignores_blank_label() -> None:
    c = FormController(
        pending_values=[{"label": "x", "value": "y"}],
        initial_labels=["x"],
    )
    assert c.mark_consumed_label("") is False
    assert c.mark_consumed_label("   ") is False
    assert c.pending_count == 1


def test_mark_consumed_label_drops_only_first_match() -> None:
    """Two entries with the same label — only the first is dropped per call."""
    c = FormController(
        pending_values=[
            {"label": "field", "value": "1"},
            {"label": "field", "value": "2"},
        ],
        initial_labels=["field", "field"],
    )
    assert c.mark_consumed_label("field") is True
    assert c.pending_count == 1
    assert c.pending_values[0]["value"] == "2"


# ── Used-region bookkeeping ────────────────────────────────────────────


def test_mark_used_region_appends_int_tuple() -> None:
    c = FormController()
    c.mark_used_region(100, 200)
    c.mark_used_region(300.7, 400.2)  # type: ignore[arg-type]
    assert c.used_regions == [(100, 200), (300, 400)]


# ── submitted latch ────────────────────────────────────────────────────


def test_mark_submitted_latches() -> None:
    c = FormController()
    assert c.submitted is False
    c.mark_submitted()
    assert c.submitted is True
    c.mark_submitted()  # idempotent
    assert c.submitted is True


# ── Delegation to GymRunner static helpers ─────────────────────────────


def test_maybe_substitute_repeated_click_consumes_value() -> None:
    """Confirms the controller's repeated-click delegate matches the legacy
    static-helper test (test_runner_force_fill::test_repeated_form_click_…)."""
    c = FormController(
        pending_values=[{"label": "zip code", "value": "33101"}],
        initial_labels=["zip code"],
    )
    previous = Action(ActionType.CLICK, {"x": 167, "y": 444})
    current = Action(ActionType.CLICK, {"x": 169, "y": 445})

    forced = c.maybe_substitute_repeated_click(
        current, [previous],
        "Click the ZIP Code field and type 33101.",
    )

    assert forced is not None
    assert forced.action_type == ActionType.TYPE
    assert forced.params == {"text": "33101"}
    assert c.pending_count == 0
    assert c.used_regions == [(169, 445)]


def test_should_finish_task_true_for_one_value_consumed() -> None:
    c = FormController(
        pending_values=[],
        initial_labels=["zip code"],
    )
    assert c.should_finish_task("Type 33101 into the zip code field") is True


def test_should_finish_task_false_when_still_pending() -> None:
    c = FormController(
        pending_values=[{"label": "zip code", "value": "33101"}],
        initial_labels=["zip code"],
    )
    assert c.should_finish_task("Type 33101 into the zip code field") is False


def test_should_finish_task_false_for_login() -> None:
    """Multi-field auth tasks are excluded — submit/navigation comes after."""
    c = FormController(
        pending_values=[],
        initial_labels=["password"],
    )
    assert c.should_finish_task("Log in with credentials") is False


def test_finish_task_actions_returns_tab_commit_for_zip() -> None:
    actions = FormController().finish_task_actions("Type 33101 into the zip code field")
    assert len(actions) == 1
    assert actions[0].action_type == ActionType.KEY_PRESS
    assert actions[0].params == {"keys": "Tab"}


def test_finish_task_actions_returns_return_then_tab_for_radius() -> None:
    actions = FormController().finish_task_actions("Set the radius to 35 miles")
    assert [a.action_type for a in actions] == [
        ActionType.KEY_PRESS, ActionType.KEY_PRESS,
    ]
    assert [a.params["keys"] for a in actions] == ["Return", "Tab"]


def test_finish_task_actions_empty_for_unknown_task() -> None:
    assert FormController().finish_task_actions("look at the page") == []
