"""Audit batch — runner plumbing + scroll recovery delta.

* ``_pending_form_labels`` collects fill_field labels from the
  remaining plan so the Holo3 sub-runner's done-gate can refuse
  premature completion (audit item 2).
* Scroll recovery requires actual viewport / scroll-state delta
  instead of blindly advancing on any failed scroll. ``brain_loop_
  exhausted`` keeps the same step so the rewriter can affect retry
  (audit item 3).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.run_executor import _pending_form_labels
from mantis_agent.gym.step_recovery import StepRecoveryPolicy
from mantis_agent.gym.step_snapshot import StepStateSnapshot
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan


# ── _pending_form_labels ────────────────────────────────────────────


def test_pending_form_labels_collects_fill_fields_from_remaining_plan() -> None:
    """Every fill_field step at or after ``current_step_index`` lands
    in the list, in plan order, deduped. Other step types are
    skipped (only fill_field has a meaningful label for the done-gate)."""
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(intent="navigate", type="navigate"),
        MicroIntent(intent="fill user", type="fill_field",
                    params={"label": "User ID"}),
        MicroIntent(intent="fill pass", type="fill_field",
                    params={"label": "Password"}),
        MicroIntent(intent="click submit", type="submit"),
        MicroIntent(intent="fill notes", type="fill_field",
                    params={"label": "Notes"}),
    ])
    # From step 0: every fill_field downstream.
    out = _pending_form_labels(plan, 0)
    assert out == ["User ID", "Password", "Notes"]


def test_pending_form_labels_starts_at_current_step() -> None:
    """Steps BEFORE current_step_index are skipped — they've already
    run and the values are already filled. The gate only cares about
    fields whose values aren't in yet."""
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(intent="fill user", type="fill_field",
                    params={"label": "User ID"}),  # already done
        MicroIntent(intent="fill pass", type="fill_field",
                    params={"label": "Password"}),  # current step
        MicroIntent(intent="click submit", type="submit"),
    ])
    out = _pending_form_labels(plan, current_step_index=1)
    assert out == ["Password"]


def test_pending_form_labels_dedupes_by_label() -> None:
    """A plan that re-uses the same field (e.g. confirm-password) only
    contributes one entry — the gate cares about distinct labels."""
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(intent="fill pwd", type="fill_field",
                    params={"label": "Password"}),
        MicroIntent(intent="fill confirm", type="fill_field",
                    params={"label": "Password"}),
    ])
    out = _pending_form_labels(plan, 0)
    assert out == ["Password"]


def test_pending_form_labels_drops_empty_labels() -> None:
    """fill_field steps without a label (or whitespace-only) are
    skipped — there's nothing for the gate to anchor on."""
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(intent="fill", type="fill_field", params={"label": ""}),
        MicroIntent(intent="fill", type="fill_field",
                    params={"label": "   "}),
        MicroIntent(intent="fill", type="fill_field",
                    params={"label": "Email"}),
    ])
    assert _pending_form_labels(plan, 0) == ["Email"]


def test_pending_form_labels_empty_when_no_fill_field_steps() -> None:
    """Plans with no fill_field steps (extraction-only flows) return
    an empty list — the gate gets nothing to reject completion on."""
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(intent="nav", type="navigate"),
        MicroIntent(intent="extract", type="extract_data"),
    ])
    assert _pending_form_labels(plan, 0) == []


# ── Scroll recovery requires actual delta ───────────────────────────


def _snap(viewport_stage: int = 0, scroll_sig: str = "x") -> StepStateSnapshot:
    """Build a minimal StepStateSnapshot. Other fields don't matter
    for the scroll-recovery branch under test."""
    return StepStateSnapshot(
        url="", current_page=1, focused_input_signature="",
        viewport_stage=viewport_stage, scroll_signature=scroll_sig,
    )


def _runner_with_snap(
    pre_snap: StepStateSnapshot | None,
    post_snap: StepStateSnapshot | None,
    *,
    monkeypatch=None,
) -> MagicMock:
    """Runner stub with the attributes step_recovery's scroll branch
    reads. ``post_snap`` is what ``step_snapshot.capture`` returns —
    we monkeypatch the module-level function so the test doesn't have
    to wire the full snapshot capture path."""
    runner = MagicMock()
    runner._pre_step_snapshot = pre_snap
    runner._scroll_state = {}
    runner._viewport_stage = (
        post_snap.viewport_stage if post_snap is not None else 0
    )
    runner._step_failure_history = {}
    runner._recovery_attempts_per_step = {}
    runner._total_recovery_attempts = 0
    if monkeypatch is not None and post_snap is not None:
        from mantis_agent.gym import step_snapshot as _snap_mod
        monkeypatch.setattr(_snap_mod, "capture", lambda _r: post_snap)
    return runner


def test_scroll_recovery_advances_when_viewport_stage_changed(monkeypatch) -> None:
    """The canonical happy path — the brain scrolled, viewport_stage
    incremented, even though it forgot to call done(). Recovery
    accepts the implicit completion and advances to the next step."""
    pre = _snap(viewport_stage=0, scroll_sig="a")
    post = _snap(viewport_stage=1, scroll_sig="a")
    runner = _runner_with_snap(pre, post, monkeypatch=monkeypatch)
    policy = StepRecoveryPolicy(parent=runner)

    plan = MicroPlan(domain="x", steps=[
        MicroIntent(intent="scroll", type="scroll"),
    ])
    step = plan.steps[0]
    result = StepResult(step_index=0, intent="scroll", success=False)
    outcome = policy.handle_failure(
        step=step, step_result=result, plan=plan,
        step_index=0, step_retry_counts={}, max_retries=2,
        loop_counters={}, listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.step_index == 1  # advanced
    assert outcome.halt_reason == "scroll_no_done"


def test_scroll_recovery_keeps_step_when_viewport_unchanged(monkeypatch) -> None:
    """The bug case the audit flagged — scroll returned failure AND
    the viewport didn't move. The old code advanced anyway, treating
    a stuck scroll as success. The fix keeps the step so retry / a
    different verb can actually fix it."""
    same = _snap(viewport_stage=0, scroll_sig="a")
    runner = _runner_with_snap(same, same, monkeypatch=monkeypatch)
    policy = StepRecoveryPolicy(parent=runner)

    plan = MicroPlan(domain="x", steps=[
        MicroIntent(intent="scroll", type="scroll"),
    ])
    step = plan.steps[0]
    result = StepResult(step_index=0, intent="scroll", success=False)
    outcome = policy.handle_failure(
        step=step, step_result=result, plan=plan,
        step_index=0, step_retry_counts={}, max_retries=2,
        loop_counters={}, listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.step_index == 0  # same step
    assert outcome.halt_reason == "scroll_no_delta"


def test_scroll_recovery_brain_loop_keeps_step_regardless_of_delta() -> None:
    """``brain_loop_exhausted`` is a structural signal that the brain
    has spent its budget without progress. Even if the viewport
    accidentally moved, the recovery should NOT advance — the
    rewriter / next retry needs the same step_index to operate on
    the same step. The audit specifically flagged this skip-on-
    brain-loop as wrong."""
    moved = _snap(viewport_stage=1, scroll_sig="a")
    pre = _snap(viewport_stage=0, scroll_sig="a")
    runner = _runner_with_snap(pre, moved)
    policy = StepRecoveryPolicy(parent=runner)

    plan = MicroPlan(domain="x", steps=[
        MicroIntent(intent="scroll", type="scroll"),
    ])
    step = plan.steps[0]
    result = StepResult(
        step_index=0, intent="scroll", success=False,
        failure_class="brain_loop_exhausted",
    )
    outcome = policy.handle_failure(
        step=step, step_result=result, plan=plan,
        step_index=0, step_retry_counts={}, max_retries=2,
        loop_counters={}, listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.step_index == 0  # kept for retry
    assert outcome.halt_reason == "scroll_brain_loop_keep_step"


def test_scroll_recovery_advances_when_snapshot_capture_raises(monkeypatch) -> None:
    """Snapshot capture is best-effort — a stub env that raises must
    not break the recovery path. Falls back to the legacy advance
    behaviour so non-MicroPlanRunner test environments keep working."""
    pre = _snap()
    runner = _runner_with_snap(pre, None)
    from mantis_agent.gym import step_snapshot as _snap_mod

    def _raise(_r):
        raise RuntimeError("env stub doesn't support snapshot capture")

    monkeypatch.setattr(_snap_mod, "capture", _raise)
    policy = StepRecoveryPolicy(parent=runner)

    plan = MicroPlan(domain="x", steps=[
        MicroIntent(intent="scroll", type="scroll"),
    ])
    step = plan.steps[0]
    result = StepResult(step_index=0, intent="scroll", success=False)
    outcome = policy.handle_failure(
        step=step, step_result=result, plan=plan,
        step_index=0, step_retry_counts={}, max_retries=2,
        loop_counters={}, listings_on_page=0,
    )
    # Falls back to advance (the legacy behaviour) when verification
    # is impossible — preserves test coverage.
    assert outcome.step_index == 1
