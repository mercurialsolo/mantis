"""Tests for #303 — DoneAcceptanceGate predicates.

Pure unit tests for the deterministic gate. The runner-integration test
that exercises the WAIT-substitute path lives in
tests/test_gym_runner_done_gate.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.done_gate import (
    REJECT_CODES,
    REJECT_EMPTY_SUMMARY,
    REJECT_NO_DELTA_AFTER_WAITS,
    REJECT_NO_PROGRESS_IN_WINDOW,
    REJECT_PENDING_FORM_VALUES,
    REJECT_PLAN_STEPS_INCOMPLETE,
    REJECT_SUMMARY_MISSING_FIELDS,
    DoneAcceptanceDecision,
    check_done_acceptance,
)


# ── Stub Plan / PlanStep so the gate doesn't need the gym.plans module ──


@dataclass
class _Step:
    action: str = "click"


@dataclass
class _Plan:
    steps: list[_Step]


def _wait(seconds: float = 1.0) -> Action:
    return Action(ActionType.WAIT, {"seconds": seconds})


def _click(x: int = 1, y: int = 1) -> Action:
    return Action(ActionType.CLICK, {"x": x, "y": y})


# ── Decision shape ─────────────────────────────────────────────────────


def test_decision_truthy_when_accepted() -> None:
    d = check_done_acceptance(summary="Logged in and saved record.")
    assert bool(d) is True
    assert d.accept is True
    assert d.reason == ""


def test_decision_falsy_when_rejected() -> None:
    d = check_done_acceptance(summary="")
    assert bool(d) is False
    assert d.accept is False
    assert d.reason == REJECT_EMPTY_SUMMARY


# ── empty_summary ──────────────────────────────────────────────────────


@pytest.mark.parametrize("summary", ["", "   ", "\n\t  ", None])
def test_empty_summary_rejected(summary: str | None) -> None:
    d = check_done_acceptance(summary=summary or "")
    assert d.reason == REJECT_EMPTY_SUMMARY


def test_non_empty_summary_passes_first_gate() -> None:
    d = check_done_acceptance(summary="Filled the form and submitted.")
    assert d.accept is True


# ── plan_steps_incomplete ──────────────────────────────────────────────


def test_plan_complete_passes() -> None:
    plan = _Plan(steps=[_Step(), _Step(), _Step()])
    d = check_done_acceptance(
        summary="Done.", plan=plan, plan_step_idx=2,  # last step (idx 2 of 3)
    )
    assert d.accept is True


def test_plan_step_short_of_last_rejected() -> None:
    plan = _Plan(steps=[_Step(), _Step(), _Step()])
    d = check_done_acceptance(
        summary="Done.", plan=plan, plan_step_idx=0,  # first of three
    )
    assert d.reason == REJECT_PLAN_STEPS_INCOMPLETE
    assert "2 step(s) remaining" in d.detail


def test_plan_step_one_short_rejected() -> None:
    plan = _Plan(steps=[_Step(), _Step()])
    d = check_done_acceptance(
        summary="Done.", plan=plan, plan_step_idx=0,
    )
    assert d.reason == REJECT_PLAN_STEPS_INCOMPLETE


def test_no_plan_skips_plan_check() -> None:
    d = check_done_acceptance(summary="Done.", plan=None, plan_step_idx=99)
    assert d.accept is True


def test_empty_plan_steps_skipped() -> None:
    """A Plan with no steps shouldn't reject — gate has no signal."""
    d = check_done_acceptance(summary="Done.", plan=_Plan(steps=[]))
    assert d.accept is True


# ── pending_form_values ────────────────────────────────────────────────


def test_pending_form_values_rejected() -> None:
    d = check_done_acceptance(
        summary="Logged in.",
        pending_form_labels=["password"],
    )
    assert d.reason == REJECT_PENDING_FORM_VALUES
    assert "password" in d.detail


def test_pending_form_values_truncated_in_detail() -> None:
    labels = [f"field{i}" for i in range(7)]
    d = check_done_acceptance(
        summary="Done.", pending_form_labels=labels,
    )
    assert d.reason == REJECT_PENDING_FORM_VALUES
    assert "+4 more" in d.detail  # 7 total, 3 shown


def test_empty_pending_form_labels_passes() -> None:
    d = check_done_acceptance(summary="Done.", pending_form_labels=[])
    assert d.accept is True


def test_pending_form_labels_with_only_blanks_passes() -> None:
    d = check_done_acceptance(
        summary="Done.", pending_form_labels=["", "", None],  # type: ignore[list-item]
    )
    assert d.accept is True


# ── summary_missing_required_fields ────────────────────────────────────


def test_summary_includes_all_required_fields_passes() -> None:
    d = check_done_acceptance(
        summary="Year: 2020 | Make: Honda | Model: Civic | URL: https://x",
        required_summary_fields=["year", "make", "model"],
    )
    assert d.accept is True


def test_summary_missing_fields_rejected() -> None:
    d = check_done_acceptance(
        summary="Year: 2020 | URL: https://x",
        required_summary_fields=["year", "make", "model"],
    )
    assert d.reason == REJECT_SUMMARY_MISSING_FIELDS
    assert "make" in d.detail and "model" in d.detail


def test_required_fields_case_insensitive() -> None:
    d = check_done_acceptance(
        summary="YEAR: 2020 | Make: Honda | MODEL: Civic",
        required_summary_fields=["year", "make", "model"],
    )
    assert d.accept is True


# ── no_observed_delta_after_waits ──────────────────────────────────────


def test_three_waits_stable_frame_rejected() -> None:
    d = check_done_acceptance(
        summary="Done.",
        recent_actions=[_wait(), _wait(), _wait()],
        recent_frame_hashes=["abc", "abc", "abc"],
    )
    assert d.reason == REJECT_NO_DELTA_AFTER_WAITS


def test_three_waits_changing_frame_passes() -> None:
    d = check_done_acceptance(
        summary="Done.",
        recent_actions=[_wait(), _wait(), _wait()],
        recent_frame_hashes=["abc", "def", "ghi"],
    )
    assert d.accept is True


def test_two_waits_one_click_not_rejected_for_waits() -> None:
    """Mixed action sequence with non-wait at the end should pass the
    wait-loop predicate (it may still trip other predicates)."""
    d = check_done_acceptance(
        summary="Done.",
        recent_actions=[_wait(), _wait(), _click()],
        recent_frame_hashes=["abc", "abc", "abc"],
    )
    # No-wait predicate doesn't fire; no-progress predicate needs window=5.
    assert d.accept is True


def test_waits_with_empty_frame_hashes_skip_check() -> None:
    """Without frame hashes we can't tell if anything moved — don't reject."""
    d = check_done_acceptance(
        summary="Done.",
        recent_actions=[_wait(), _wait(), _wait()],
        recent_frame_hashes=[],
    )
    assert d.accept is True


# ── no_progress_in_window ──────────────────────────────────────────────


def test_five_steps_no_url_or_frame_change_rejected() -> None:
    d = check_done_acceptance(
        summary="Done.",
        recent_actions=[_click(), _click(), _click(), _click(), _click()],
        recent_frame_hashes=["x"] * 5,
        recent_urls=["https://x.test/page"] * 5,
    )
    assert d.reason == REJECT_NO_PROGRESS_IN_WINDOW


def test_five_steps_with_url_change_passes_progress_check() -> None:
    d = check_done_acceptance(
        summary="Done.",
        recent_actions=[_click()] * 5,
        recent_frame_hashes=["x"] * 5,
        recent_urls=[
            "https://x.test/a", "https://x.test/a", "https://x.test/a",
            "https://x.test/a", "https://x.test/b",
        ],
    )
    assert d.accept is True


def test_under_progress_window_skipped() -> None:
    d = check_done_acceptance(
        summary="Done.",
        recent_actions=[_click(), _click()],
        recent_frame_hashes=["x", "x"],
        recent_urls=["https://x.test/a", "https://x.test/a"],
    )
    assert d.accept is True


# ── First-rejection-wins ordering ───────────────────────────────────────


def test_empty_summary_takes_priority_over_other_rejections() -> None:
    plan = _Plan(steps=[_Step(), _Step()])
    d = check_done_acceptance(
        summary="",
        plan=plan, plan_step_idx=0,
        pending_form_labels=["password"],
    )
    assert d.reason == REJECT_EMPTY_SUMMARY


def test_plan_check_takes_priority_over_form_values() -> None:
    plan = _Plan(steps=[_Step(), _Step()])
    d = check_done_acceptance(
        summary="Done.",
        plan=plan, plan_step_idx=0,
        pending_form_labels=["password"],
    )
    assert d.reason == REJECT_PLAN_STEPS_INCOMPLETE


# ── Documented benchmark patterns ───────────────────────────────────────


def test_run_009_010_pattern_done_after_only_waits_rejected() -> None:
    """Holo3 emits done(success=True, summary='') after a string of waits."""
    d = check_done_acceptance(
        summary="",
        recent_actions=[_wait(), _wait(), _wait(), _wait()],
        recent_frame_hashes=["x"] * 4,
    )
    assert d.reason == REJECT_EMPTY_SUMMARY


def test_run_023_pattern_login_only_with_pending_credentials_rejected() -> None:
    """Holo3 claims downstream success while password was never typed."""
    d = check_done_acceptance(
        summary="Updated lead industry to Space Exploration.",
        pending_form_labels=["password"],
    )
    assert d.reason == REJECT_PENDING_FORM_VALUES


def test_per_step_done_when_confusion_rejected_via_plan_check() -> None:
    """Model treats a step-local 'Done when' as whole-task completion."""
    plan = _Plan(steps=[_Step(), _Step(), _Step()])
    d = check_done_acceptance(
        summary="Logged in.",  # only completed step 1
        plan=plan, plan_step_idx=0,
    )
    assert d.reason == REJECT_PLAN_STEPS_INCOMPLETE


# ── Decision class re-export sanity ─────────────────────────────────────


def test_decision_is_dataclass_with_expected_fields() -> None:
    d = DoneAcceptanceDecision(False, "x", "y")
    assert d.accept is False
    assert d.reason == "x"
    assert d.detail == "y"


# ── Public surface lock ────────────────────────────────────────────────


def test_reject_codes_tuple_locked() -> None:
    """Codes round-trip into TrajectoryStep.done_rejected_reason and the
    /v1/cua API surface — renaming or removing any of these is a breaking
    change. New codes append to the tuple. Update this test deliberately."""
    assert REJECT_CODES == (
        "empty_summary",
        "plan_steps_incomplete",
        "pending_form_values",
        "summary_missing_required_fields",
        "no_observed_delta_after_waits",
        "no_progress_in_window",
    )


# ── _hashes_stable empty-slot policy (review fix) ──────────────────────


def test_no_delta_skipped_when_any_window_hash_empty() -> None:
    """All hashes in the window must be non-empty for the predicate to
    fire — otherwise we'd declare stability based on an unpopulated slot."""
    d = check_done_acceptance(
        summary="Done.",
        recent_actions=[_wait(), _wait(), _wait()],
        recent_frame_hashes=["abc", "abc", ""],  # last slot empty
    )
    assert d.accept is True


def test_no_delta_skipped_when_first_window_hash_empty() -> None:
    """Symmetric to the above: an empty first slot also disqualifies."""
    d = check_done_acceptance(
        summary="Done.",
        recent_actions=[_wait(), _wait(), _wait()],
        recent_frame_hashes=["", "abc", "abc"],
    )
    assert d.accept is True


def test_no_delta_fires_when_full_window_identical_and_truthy() -> None:
    d = check_done_acceptance(
        summary="Done.",
        recent_actions=[_wait(), _wait(), _wait()],
        recent_frame_hashes=["abc", "abc", "abc"],
    )
    assert d.reason == REJECT_NO_DELTA_AFTER_WAITS
