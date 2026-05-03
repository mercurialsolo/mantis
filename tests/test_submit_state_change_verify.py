"""Tests for the submit-step state-change verifier.

Surfaced by the staffcrm v4 verify (run 20260503_121500_61da9464):
the runner reported step 3 (login click) as success because the click
fired, but the post-step diff showed "no change" — the browser stayed
on the login page, and step 4 retried scrolling looking for "Leads"
on a still-login screenshot.

This module verifies the runner now demotes form-shape `submit` /
`select_option` steps that report ok but produced zero observable
state change. Pure-observational: uses the #121 ``step_snapshot.diff``
infrastructure, no regex / heuristic / extra LLM call.
"""

from __future__ import annotations


import pytest

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.step_snapshot import StepDiff, StepStateSnapshot


# ── Helpers: simulate the runner's verifier without instantiating the runner ──


def _verifier(
    pre: StepStateSnapshot,
    post: StepStateSnapshot,
    *,
    step_type: str,
    initial_success: bool = True,
) -> tuple[bool, str]:
    """Re-implement the runner's verifier branch in 12 lines.

    The actual verifier lives inside :class:`MicroPlanRunner.run` and
    can't be exercised in isolation without a full runner. This tiny
    fixture mirrors the same predicate so tests can drive it across
    every diff state.
    """
    from mantis_agent.gym.step_snapshot import diff as compute_diff

    success = initial_success
    data = ""
    if success and step_type in ("submit", "select_option"):
        delta = compute_diff(pre, post)
        if not delta.has_changes:
            success = False
            data = ":no_state_change"
    return success, data


# ── No-change submits get demoted ──────────────────────────────────────


def test_submit_with_no_change_is_demoted_to_failure() -> None:
    """The exact staffcrm v4 failure mode: login click reported ok, but
    URL / focus / scroll all unchanged."""
    pre = StepStateSnapshot(url="https://x.test/login")
    post = StepStateSnapshot(url="https://x.test/login")
    success, data = _verifier(pre, post, step_type="submit")
    assert success is False
    assert "no_state_change" in data


def test_select_option_with_no_change_also_demoted() -> None:
    """Same rule applies to dropdown selects — silently accepting a
    select that didn't take effect leads to the same retry-pointless-
    over-bad-state failure."""
    pre = StepStateSnapshot(url="https://x.test/edit")
    post = StepStateSnapshot(url="https://x.test/edit")
    success, _ = _verifier(pre, post, step_type="select_option")
    assert success is False


# ── State-change cases keep the success ────────────────────────────────


def test_submit_with_url_change_keeps_success() -> None:
    """Login that actually navigates to /dashboard."""
    pre = StepStateSnapshot(url="https://x.test/login")
    post = StepStateSnapshot(url="https://x.test/dashboard")
    success, data = _verifier(pre, post, step_type="submit")
    assert success is True
    assert data == ""


def test_submit_with_focus_change_only_keeps_success() -> None:
    """Opening a modal: URL stays the same, focus moves to the modal's
    first input. That's a real state change — don't demote."""
    pre = StepStateSnapshot(focused_input_signature="empty")
    post = StepStateSnapshot(focused_input_signature="modal-field-1")
    success, _ = _verifier(pre, post, step_type="submit")
    assert success is True


def test_submit_with_scroll_change_keeps_success() -> None:
    """A submit that scrolls content into view (e.g. expanding an
    accordion) is real progress."""
    pre = StepStateSnapshot(scroll_signature="empty")
    post = StepStateSnapshot(scroll_signature="scr-12-3")
    success, _ = _verifier(pre, post, step_type="submit")
    assert success is True


def test_submit_with_viewport_advance_keeps_success() -> None:
    pre = StepStateSnapshot(viewport_stage=0)
    post = StepStateSnapshot(viewport_stage=1)
    success, _ = _verifier(pre, post, step_type="submit")
    assert success is True


def test_submit_with_extraction_added_keeps_success() -> None:
    """Save action that produces a confirmation panel with extracted data."""
    pre = StepStateSnapshot(last_extracted_url="")
    post = StepStateSnapshot(last_extracted_url="https://x.test/saved/123")
    success, _ = _verifier(pre, post, step_type="submit")
    assert success is True


# ── fill_field is exempt from the verifier ─────────────────────────────


def test_fill_field_with_no_change_is_not_demoted() -> None:
    """fill_field is explicitly skipped — typing into a field doesn't
    necessarily change any of the snapshot signals (the field just gains
    focus, which may or may not register depending on env)."""
    pre = StepStateSnapshot()
    post = StepStateSnapshot()
    success, data = _verifier(pre, post, step_type="fill_field")
    assert success is True
    assert data == ""


def test_navigate_with_no_change_is_not_demoted() -> None:
    """Navigate steps have their own success criterion (URL load), not
    governed by the submit verifier."""
    pre = StepStateSnapshot()
    post = StepStateSnapshot()
    success, _ = _verifier(pre, post, step_type="navigate")
    assert success is True


def test_click_step_is_not_demoted() -> None:
    """Listings-style click — the runner has separate page-exhaustion /
    duplicate signals for these. Don't double-handle."""
    pre = StepStateSnapshot()
    post = StepStateSnapshot()
    success, _ = _verifier(pre, post, step_type="click")
    assert success is True


# ── Already-failed steps are not affected ──────────────────────────────


def test_already_failed_submit_stays_failed() -> None:
    """A submit that the handler already marked failed (form_target_not_found)
    isn't second-guessed by the verifier — the handler had richer context."""
    pre = StepStateSnapshot()
    post = StepStateSnapshot()
    success, data = _verifier(
        pre, post, step_type="submit", initial_success=False,
    )
    assert success is False
    assert "no_state_change" not in data


# ── Compatibility with StepResult dataclass ────────────────────────────


def test_step_result_round_trips_with_demoted_data() -> None:
    """StepResult.data field accepts the appended ":no_state_change" tag."""
    sr = StepResult(
        step_index=3,
        intent="Click the login button",
        success=False,
        data="submit:Sign In:no_state_change",
    )
    payload = sr.to_dict()
    assert payload["data"] == "submit:Sign In:no_state_change"
    assert payload["success"] is False


# ── Integration: every StepDiff field is meaningful ────────────────────


@pytest.mark.parametrize(
    "kwargs,expected_has_changes",
    [
        ({}, False),
        ({"url_changed": True, "changed_fields": ["url: a → b"]}, True),
        ({"page_changed": True, "changed_fields": ["page: 1 → 2"]}, True),
        ({"viewport_changed": True, "changed_fields": ["viewport_stage: 0 → 1"]}, True),
        ({"focus_changed": True, "changed_fields": ["focused_input"]}, True),
        ({"scroll_changed": True, "changed_fields": ["scroll_state"]}, True),
        ({"extraction_added": True, "changed_fields": ["last_extracted: X"]}, True),
        ({"new_urls_seen": True, "changed_fields": ["seen_urls +1"]}, True),
    ],
)
def test_diff_has_changes_reflects_every_field(kwargs, expected_has_changes) -> None:
    """Sanity: each individual signal flips has_changes, so the verifier
    won't false-positive on a real state change in any of the dimensions."""
    delta = StepDiff(**kwargs)
    assert delta.has_changes is expected_has_changes
