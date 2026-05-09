"""Tests for the agentic handler-escalation loop (#224 follow-up).

Surfaced by the staff-crm post-PR-#231 rerun: the agentic-retry
feedback (#230) successfully made the LLM pick different
coordinates on each retry — but all three coordinates landed in
the same problematic region (left-edge status-pill column) because
the form handler's ``find_form_target`` is fundamentally a label-
text matcher. With the search prompt asking it to find a button
labeled "Qualified", every match is going to be the visible
"Qualified" status pill, never the row link.

The fix: when the same step has accumulated 2+ ``no_state_change``
failures, the default handler is locked on the wrong element class.
The runtime escalates to ``Holo3StepHandler`` for the next retry —
brain grounding over intent prose ("click the first lead row whose
status is qualified") instead of label text matching.

These tests pin:

- ``_maybe_set_handler_override`` decision logic (trigger threshold,
  kind specificity, no-brain guard)
- The dispatcher's escalation routing (``execute_step`` reads the
  override before normal type-based routing)
- Per-step-index isolation (failures on step 5 don't escalate step 7)
- Success-path clearing (override cleared after a successful step)

General-purpose contract: the trigger is purely the observed
failure pattern. No plan-specific terms, no domain hardcoding —
the same logic kicks in for any submit step that keeps clicking
elements that don't navigate.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.run_executor import RunExecutor, RunState
from mantis_agent.gym.step_snapshot import StepStateSnapshot
from mantis_agent.plan_decomposer import MicroIntent


def _make_runner(
    *,
    extractor: MagicMock | None = None,
    brain: MagicMock | None = None,
    history: list[dict] | None = None,
) -> MagicMock:
    runner = MagicMock()
    runner.extractor = extractor
    runner.brain = brain or MagicMock()
    runner._last_submit_pre_screenshot = None
    runner._last_submit_target = None
    runner._step_failure_history = {5: history or []}
    runner._step_handler_override = {}
    runner.costs = {"claude_extract": 0}
    runner._last_known_url = "https://crm.test/"
    runner._current_page = 1
    runner._viewport_stage = 0
    runner._scroll_state = {}
    runner._last_extracted = {}
    runner._extracted_titles = []
    runner._seen_urls = set()
    runner.env = MagicMock()
    runner.env.last_focused_input = None
    return runner


# ── _maybe_set_handler_override decision logic ───────────────────────────


def test_one_failure_does_not_escalate() -> None:
    """A single failure isn't enough signal to switch handlers — could
    be a render race or first-time scrolling miss. Wait for the
    pattern to stabilize over 2+ same-kind failures."""
    runner = _make_runner(history=[
        {"x": 39, "y": 137, "label": "Qualified", "kind": "no_state_change"},
    ])
    RunExecutor._maybe_set_handler_override(runner=runner, step_index=5)
    assert 5 not in runner._step_handler_override


def test_two_no_state_change_failures_escalate_to_holo3() -> None:
    """The canonical trigger: 2× ``no_state_change`` on the same step
    means the click is happening but the page never updates — the
    default text-match handler is finding wrong elements. Escalate
    to brain-grounded routing for the next retry."""
    runner = _make_runner(history=[
        {"x": 39, "y": 137, "label": "Qualified", "kind": "no_state_change"},
        {"x": 66, "y": 395, "label": "Qualified", "kind": "no_state_change"},
    ])
    RunExecutor._maybe_set_handler_override(runner=runner, step_index=5)
    assert runner._step_handler_override[5] == "holo3"


def test_three_no_state_change_failures_still_escalate() -> None:
    """Idempotent — re-calling after a third failure keeps the
    override (the dispatcher consumes it once per retry but the
    record stays so a later success can clear it cleanly)."""
    runner = _make_runner(history=[
        {"x": 1, "y": 1, "label": "X", "kind": "no_state_change"},
        {"x": 2, "y": 2, "label": "X", "kind": "no_state_change"},
        {"x": 3, "y": 3, "label": "X", "kind": "no_state_change"},
    ])
    RunExecutor._maybe_set_handler_override(runner=runner, step_index=5)
    assert runner._step_handler_override[5] == "holo3"


def test_form_target_not_found_does_not_escalate() -> None:
    """``form_target_not_found`` indicates the button isn't on the
    page (rendering issue, button truly missing) — Holo3 wouldn't
    help there. The escalation is specific to ``no_state_change``
    which signals "the click happened but the page didn't change."
    """
    runner = _make_runner(history=[
        {"x": None, "y": None, "label": "Save", "kind": "form_target_not_found"},
        {"x": None, "y": None, "label": "Save", "kind": "form_target_not_found"},
    ])
    RunExecutor._maybe_set_handler_override(runner=runner, step_index=5)
    assert 5 not in runner._step_handler_override


def test_mixed_kinds_count_only_no_state_change() -> None:
    """One ``no_state_change`` plus one ``form_target_not_found``
    isn't 2× of the same kind — don't escalate. The pattern needs
    to be stable for the trigger to fire."""
    runner = _make_runner(history=[
        {"x": 39, "y": 137, "label": "X", "kind": "no_state_change"},
        {"x": None, "y": None, "label": "X", "kind": "form_target_not_found"},
    ])
    RunExecutor._maybe_set_handler_override(runner=runner, step_index=5)
    assert 5 not in runner._step_handler_override


def test_no_brain_disables_escalation() -> None:
    """Holo3 brain is the escalation target — without it, escalation
    is meaningless. Don't set the override; let the default retry
    path play out instead."""
    runner = _make_runner(history=[
        {"x": 1, "y": 1, "label": "X", "kind": "no_state_change"},
        {"x": 2, "y": 2, "label": "X", "kind": "no_state_change"},
    ])
    runner.brain = None
    RunExecutor._maybe_set_handler_override(runner=runner, step_index=5)
    assert 5 not in runner._step_handler_override


def test_escalation_is_per_step_index() -> None:
    """Failures on step 5 must not escalate step 7. Each step's
    history is independent."""
    runner = _make_runner(history=[
        {"x": 1, "y": 1, "label": "X", "kind": "no_state_change"},
        {"x": 2, "y": 2, "label": "X", "kind": "no_state_change"},
    ])
    runner._step_failure_history = {5: runner._step_failure_history[5]}
    RunExecutor._maybe_set_handler_override(runner=runner, step_index=5)
    RunExecutor._maybe_set_handler_override(runner=runner, step_index=7)
    assert runner._step_handler_override.get(5) == "holo3"
    assert 7 not in runner._step_handler_override


def test_runner_without_failure_history_attr_is_safe() -> None:
    """Guard against runners constructed via test fixtures that
    don't initialise ``_step_failure_history`` / ``_step_handler_override``."""
    runner = MagicMock()
    runner.brain = MagicMock()
    # Deliberately strip the new fields.
    del runner._step_failure_history
    del runner._step_handler_override
    # Must not raise.
    RunExecutor._maybe_set_handler_override(runner=runner, step_index=5)


# ── _maybe_demote_form_no_change records + escalates ───────────────────


def test_demote_path_records_and_then_escalates() -> None:
    """End-to-end: two demoted submits in a row → second demotion
    triggers escalation. The first demotion records but doesn't
    escalate (only 1 record); the second demotion records (now
    2 records) and sets the override."""
    runner = _make_runner()
    runner._last_submit_target = {
        "x": 39, "y": 137, "label": "Qualified", "matched_label": "Qualified",
    }
    runner._step_failure_history = {}

    executor = RunExecutor(parent=runner)

    # First demotion attempt.
    state = RunState.fresh(run_key="t", session_name="t", plan_signature="sig")
    state.step_index = 5
    state.results.append(StepResult(
        step_index=5, intent="Click Qualified lead",
        success=True, data="submit:Qualified", duration=3.0, steps_used=1,
    ))
    pre = StepStateSnapshot(url="https://crm.test/")
    submit_step = MicroIntent(intent="Click Qualified lead", type="submit")

    executor._maybe_demote_form_no_change(state, submit_step, pre)
    # First demotion: 1 record, no override yet.
    assert len(runner._step_failure_history[5]) == 1
    assert 5 not in runner._step_handler_override

    # Second attempt — different click target.
    runner._last_submit_target = {
        "x": 66, "y": 395, "label": "Qualified", "matched_label": "Qualified",
    }
    state.results.append(StepResult(
        step_index=5, intent="Click Qualified lead",
        success=True, data="submit:Qualified@(66,395)", duration=3.0,
        steps_used=1,
    ))
    executor._maybe_demote_form_no_change(state, submit_step, pre)
    # Second demotion: 2 records, override now set.
    assert len(runner._step_failure_history[5]) == 2
    assert runner._step_handler_override[5] == "holo3"


# ── Dispatcher routing reads the override ──────────────────────────────


def test_dispatcher_routes_to_holo3_when_override_set() -> None:
    """``execute_step`` must consult ``_step_handler_override``
    before normal type-based routing. When set to ``"holo3"``,
    the step routes through ``Holo3StepHandler`` regardless of
    the original step.type."""
    from mantis_agent.gym import _runner_helpers

    runner = MagicMock()
    runner.brain = MagicMock()
    runner.extractor = MagicMock()
    runner._step_handler_override = {5: "holo3"}
    runner._handler_registry = MagicMock()

    holo3_called = []

    def _stub_holo3(r, step, idx):
        holo3_called.append((idx, step.type))
        return StepResult(
            step_index=idx, intent=step.intent, success=True,
            data="holo3-completed", duration=5.0, steps_used=3,
        )

    submit_step = MicroIntent(
        intent="Click Qualified lead", type="submit",
        budget=4,  # the typical submit budget — too tight for scroll-and-find
        params={"label": "Qualified"},
    )

    # Patch execute_holo3_step so the test doesn't try to spin up
    # a real GymRunner. Capture the synthesized step so we can
    # assert the escalation budget bump.
    captured_step: list = []

    def _stub_holo3_capture(r, step_arg, idx):
        captured_step.append(step_arg)
        return _stub_holo3(r, step_arg, idx)

    orig = _runner_helpers.execute_holo3_step
    _runner_helpers.execute_holo3_step = _stub_holo3_capture
    try:
        result = _runner_helpers.execute_step(runner, submit_step, 5)
    finally:
        _runner_helpers.execute_holo3_step = orig

    assert result.success is True
    assert result.data == "holo3-completed"
    # Holo3 was called preserving the original step.type — we don't
    # morph the step into a click, just route differently.
    assert holo3_called == [(5, "submit")]
    # The synthesised step that reached Holo3 has a bumped budget.
    # Original budget=4 is too tight for "scroll down then maybe
    # paginate to find off-screen target"; escalation bumps to >= 25.
    assert captured_step[0].type == "submit"
    assert captured_step[0].budget >= 25
    # Intent prose preserved at the start (no history → no augment).
    assert captured_step[0].intent.startswith("Click Qualified lead")
    # Default registry handlers were not touched.
    runner._handler_registry.get.assert_not_called()


def test_dispatcher_augments_intent_with_failure_history_when_present() -> None:
    """When the runner has accumulated failure records for this step,
    the dispatcher's escalation path must surface them in the
    Holo3-bound task prose. Holo3 sees screenshots directly, but
    giving it the failure trace means it doesn't re-try the same
    wrong elements and gets explicit licence to paginate when the
    visible viewport has no matching target."""
    from mantis_agent.gym import _runner_helpers

    runner = MagicMock()
    runner.brain = MagicMock()
    runner.extractor = MagicMock()
    runner._step_handler_override = {5: "holo3"}
    runner._step_failure_history = {
        5: [
            {"x": 39, "y": 137, "label": "Qualified",
             "matched_label": "Qualified",
             "kind": "no_state_change", "reason": ""},
            {"x": 66, "y": 395, "label": "Qualified",
             "matched_label": "Qualified",
             "kind": "no_state_change", "reason": ""},
        ],
    }
    runner._handler_registry = MagicMock()

    captured_step: list = []

    def _capture(r, step_arg, idx):
        captured_step.append(step_arg)
        return StepResult(
            step_index=idx, intent=step_arg.intent, success=True,
            data="holo3-completed",
        )

    submit_step = MicroIntent(
        intent="Click the first lead row whose Status is Qualified",
        type="submit", budget=4,
        params={"label": "Qualified"},
    )
    orig = _runner_helpers.execute_holo3_step
    _runner_helpers.execute_holo3_step = _capture
    try:
        _runner_helpers.execute_step(runner, submit_step, 5)
    finally:
        _runner_helpers.execute_holo3_step = orig

    augmented = captured_step[0].intent
    # Intent prose retains the original task as the leading clause.
    assert augmented.startswith(
        "Click the first lead row whose Status is Qualified"
    )
    # Failure context appended.
    assert "previous attempts" in augmented.lower()
    assert "(39, 137)" in augmented
    assert "(66, 395)" in augmented
    # Pagination guidance present so Holo3 explores beyond the
    # current viewport when the target genuinely isn't on screen.
    assert "pagination" in augmented.lower() or "later page" in augmented.lower()
    assert "scroll" in augmented.lower()


def test_dispatcher_skips_escalation_when_no_override() -> None:
    """No override set → normal type-based routing applies. This is
    the regression test for the unmodified happy path."""
    from mantis_agent.gym import _runner_helpers

    runner = MagicMock()
    runner.brain = MagicMock()
    runner.extractor = MagicMock()
    runner._step_handler_override = {}  # empty — no escalation

    holo3_called = []
    _runner_helpers_holo3 = _runner_helpers.execute_holo3_step
    _runner_helpers.execute_holo3_step = lambda *a: holo3_called.append(a)
    try:
        # Use a click step to drive normal listings routing.
        click_step = MicroIntent(
            intent="Click next listing", type="click",
            section="extraction",
        )
        # Stub the registry handler the click path hits.
        registry_handler = MagicMock()
        registry_handler.execute.return_value = StepResult(
            step_index=2, intent="Click next listing", success=True,
        )
        runner._handler_registry = MagicMock()
        runner._handler_registry.get.return_value = registry_handler

        # Stub ensure_results_filters so we don't need the full plan/state.
        orig_ensure = _runner_helpers.ensure_results_filters
        _runner_helpers.ensure_results_filters = lambda r, i: True
        try:
            _runner_helpers.execute_step(runner, click_step, 2)
        finally:
            _runner_helpers.ensure_results_filters = orig_ensure
    finally:
        _runner_helpers.execute_holo3_step = _runner_helpers_holo3

    # Holo3 was NOT called — normal click routing applied.
    assert holo3_called == []


def test_dispatcher_skips_escalation_when_no_brain() -> None:
    """Override set but brain is None — fall through to normal
    routing rather than crash the run."""
    from mantis_agent.gym import _runner_helpers

    runner = MagicMock()
    runner.brain = None
    runner.extractor = MagicMock()
    runner._step_handler_override = {5: "holo3"}

    # Stub a click step + registry handler.
    click_step = MicroIntent(
        intent="Click X", type="click", section="extraction",
    )
    registry_handler = MagicMock()
    registry_handler.execute.return_value = StepResult(
        step_index=5, intent="Click X", success=True,
    )
    runner._handler_registry = MagicMock()
    runner._handler_registry.get.return_value = registry_handler

    orig_ensure = _runner_helpers.ensure_results_filters
    _runner_helpers.ensure_results_filters = lambda r, i: True
    holo3_called = []
    orig_holo3 = _runner_helpers.execute_holo3_step
    _runner_helpers.execute_holo3_step = lambda *a: holo3_called.append(a)
    try:
        _runner_helpers.execute_step(runner, click_step, 5)
    finally:
        _runner_helpers.ensure_results_filters = orig_ensure
        _runner_helpers.execute_holo3_step = orig_holo3

    # Holo3 was NOT called — no-brain guard fired.
    assert holo3_called == []
