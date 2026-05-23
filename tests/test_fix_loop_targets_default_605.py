"""Tests for issue #605 — default unset loop_target to the click step.

Claude frequently omits ``loop_target`` on ``loop`` steps in its
decomposition output. The parse layer falls back to
``s.get("loop_target", -1) = -1`` (plan_decomposer.py:979). The
runner's loop handler then computes ``target = state.step_index``
(the loop step's own index), so the loop SELF-SPINS for
``max_loop_iterations`` (200) cycles before advancing — no actual
iteration of the click→extract body ever runs.

Production run ``20260523_001336_a88df3d0`` extracted exactly 1 lead
this way: the dedup fix from #604 let the first iteration succeed,
then the loop step self-spun and the plan ended with 0 additional
iterations.

These tests pin: ``_fix_loop_targets`` defaults any ``loop`` step
with ``loop_target < 0`` to the first extraction-section ``click``
step. They also pin that the existing close-target retarget behavior
is preserved (regression guard on the original Fix 3 behavior).
"""

from __future__ import annotations

from mantis_agent.plan_decomposer import MicroIntent, MicroPlan, PlanDecomposer


def _plan(*steps: MicroIntent) -> MicroPlan:
    plan = MicroPlan(steps=list(steps))
    for i, s in enumerate(plan.steps):
        s.index = i
    return plan


def test_unset_loop_target_defaults_to_click_idx():
    """A loop step with ``loop_target=-1`` (Claude omitted) must be
    retargeted to the first extraction click step."""
    plan = _plan(
        MicroIntent(intent="navigate", type="navigate", section="setup"),
        MicroIntent(intent="click listing", type="click", section="extraction"),
        MicroIntent(intent="extract_url", type="extract_url", section="extraction"),
        MicroIntent(intent="extract_data", type="extract_data", section="extraction"),
        MicroIntent(intent="loop", type="loop", section="extraction"),
    )
    # Claude omitted loop_target → parse-time default is -1.
    assert plan.steps[4].loop_target == -1

    PlanDecomposer._fix_loop_targets(plan)

    # Should now point at the click step (index 1).
    assert plan.steps[4].loop_target == 1


def test_loop_target_already_correct_unchanged():
    """A loop step that already points at the click step should be
    left alone — the fix is idempotent."""
    plan = _plan(
        MicroIntent(intent="click", type="click", section="extraction"),
        MicroIntent(intent="extract_data", type="extract_data", section="extraction"),
        MicroIntent(intent="loop", type="loop", section="extraction", loop_target=0),
    )
    PlanDecomposer._fix_loop_targets(plan)
    assert plan.steps[2].loop_target == 0


def test_close_wrong_target_retargets_to_click():
    """Pre-existing behavior: a loop_target that's 1-2 off from the
    click step gets snapped to the click step (typical decomposer
    error: pointing at extract_url instead of click)."""
    plan = _plan(
        MicroIntent(intent="click", type="click", section="extraction"),     # 0
        MicroIntent(intent="extract_url", type="extract_url", section="extraction"),  # 1
        MicroIntent(intent="extract_data", type="extract_data", section="extraction"),  # 2
        MicroIntent(intent="loop", type="loop", section="extraction", loop_target=1),  # 3
    )
    PlanDecomposer._fix_loop_targets(plan)
    # loop_target was 1 (extract_url); should snap to 0 (click).
    assert plan.steps[3].loop_target == 0


def test_far_wrong_target_left_alone():
    """Pre-existing behavior: a loop_target that's 3+ off from click
    is NOT retargeted — too risky to assume the decomposer's intent."""
    plan = _plan(
        MicroIntent(intent="click", type="click", section="extraction"),     # 0
        MicroIntent(intent="extract_url", type="extract_url", section="extraction"),  # 1
        MicroIntent(intent="extract_data", type="extract_data", section="extraction"),  # 2
        MicroIntent(intent="scroll", type="scroll", section="extraction"),    # 3
        MicroIntent(intent="navigate_back", type="navigate_back", section="extraction"),  # 4
        MicroIntent(intent="loop", type="loop", section="extraction", loop_target=4),  # 5
    )
    PlanDecomposer._fix_loop_targets(plan)
    # 4 - 0 = 4, more than abs ≤ 2 — leave alone.
    assert plan.steps[5].loop_target == 4


def test_no_click_in_extraction_section_does_nothing():
    """When the plan has no extraction-section click step, the fix
    skips entirely (no click_idx to default to)."""
    plan = _plan(
        MicroIntent(intent="navigate", type="navigate", section="setup"),
        MicroIntent(intent="extract_data", type="extract_data", section="extraction"),
        MicroIntent(intent="loop", type="loop", section="extraction"),
    )
    assert plan.steps[2].loop_target == -1
    PlanDecomposer._fix_loop_targets(plan)
    # No click step to anchor on → -1 preserved.
    assert plan.steps[2].loop_target == -1


def test_setup_section_click_does_not_anchor():
    """Only EXTRACTION-section clicks anchor the loop default. A
    click in setup (e.g., cookie banner dismiss) is the wrong target
    — the loop body wraps the listing-card click, not the setup."""
    plan = _plan(
        MicroIntent(intent="navigate", type="navigate", section="setup"),
        MicroIntent(intent="dismiss cookie banner", type="click", section="setup"),  # 1
        MicroIntent(intent="click listing", type="click", section="extraction"),    # 2
        MicroIntent(intent="extract_data", type="extract_data", section="extraction"),
        MicroIntent(intent="loop", type="loop", section="extraction"),
    )
    PlanDecomposer._fix_loop_targets(plan)
    # The fix should pick the extraction click (index 2), not the
    # setup click (index 1).
    assert plan.steps[4].loop_target == 2


def test_multiple_loops_all_get_defaulted():
    """Both the inner (per-listing) and outer (per-page) loop steps
    should get the same default if both are unset — they both want
    to re-enter at the click step after their respective control."""
    plan = _plan(
        MicroIntent(intent="navigate", type="navigate", section="setup"),
        MicroIntent(intent="click listing", type="click", section="extraction"),     # 1
        MicroIntent(intent="extract_data", type="extract_data", section="extraction"),
        MicroIntent(intent="navigate_back", type="navigate_back", section="extraction"),
        MicroIntent(intent="loop body", type="loop", section="extraction"),          # 4
        MicroIntent(intent="paginate", type="paginate", section="pagination"),
        MicroIntent(intent="loop page", type="loop", section="pagination"),          # 6
    )
    PlanDecomposer._fix_loop_targets(plan)
    assert plan.steps[4].loop_target == 1
    assert plan.steps[6].loop_target == 1


def test_non_loop_steps_with_negative_loop_target_untouched():
    """Defensive: non-loop steps that happen to have ``loop_target=-1``
    (the dataclass default) must not be mutated — only ``type=='loop'``
    steps consume the field."""
    plan = _plan(
        MicroIntent(intent="click", type="click", section="extraction"),
        MicroIntent(intent="extract_data", type="extract_data", section="extraction"),
        MicroIntent(intent="loop", type="loop", section="extraction"),
    )
    PlanDecomposer._fix_loop_targets(plan)
    # click + extract_data still have the dataclass default.
    assert plan.steps[0].loop_target == -1
    assert plan.steps[1].loop_target == -1
    # loop got the click_idx default.
    assert plan.steps[2].loop_target == 0
