"""RunExecutor + RunState unit tests — Phase 3 of EPIC #161.

The 300-LOC while-loop body that used to live on
``MicroPlanRunner.run`` is now ``RunExecutor.execute``. These tests
pin the orchestration logic with a fake parent runner — no Xvfb,
no real GymRunner, no live brain.

Coverage focus areas:

- Init / loop preamble: cancel event, pause, budget cap, time cap
- Step orchestration: effective-step build, loop step dispatch
- Outcome routing: success path, failure path (delegates to recovery
  policy), DUPLICATE handling, no-state-change form demote
- Finalization: completed / halted / pause-status accounting
- RunState defaults + factory

The recovery policy itself is unit-tested separately in
``test_step_recovery_policy.py``; here we just verify the executor
delegates correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.gym.checkpoint import RunCheckpoint, StepResult
from mantis_agent.gym.listings_scanner import ListingsScanner
from mantis_agent.gym.run_executor import RunExecutor, RunState
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan


def _plan(*types: str) -> MicroPlan:
    plan = MicroPlan(domain="test")
    for i, t in enumerate(types):
        plan.steps.append(MicroIntent(intent=f"step-{i}-{t}", type=t))
    return plan


def _state(checkpoint: RunCheckpoint | None = None) -> RunState:
    return RunState(
        checkpoint=checkpoint or RunCheckpoint(
            run_key="test", plan_signature="sig", session_name="sess",
        ),
    )


def _runner_stub(monkeypatch=None) -> MagicMock:
    """Build a MagicMock runner with the attributes RunExecutor reads."""
    runner = MagicMock()
    runner._final_status = "running"
    runner.plan_signature = "sig"
    runner.run_key = "test"
    runner.session_name = "sess"
    runner.resume_state = False
    runner.checkpoint_path = "/tmp/c.json"
    runner.max_cost = 100.0
    runner.max_time = 3600.0
    runner.max_retries = 2
    runner.costs = {"gpu_steps": 0, "claude_extract": 0, "claude_grounding": 0, "proxy_mb": 0}
    runner._run_start = 0.0
    runner._cost_totals.return_value = (0.0, 0.0, 0.0, 0.0)
    runner._is_cancelled.return_value = False
    runner.tool_channel.is_paused.return_value = False
    runner._results_base_url = ""
    runner._extract_url_from_intent.return_value = ""
    runner._derive_filter_tokens.return_value = ()
    runner._compute_plan_signature.return_value = "sig"
    # Phase 4: use a real ListingsScanner so scroll_directive_for / is_duplicate
    # / mark_seen / on_page_change return real values rather than MagicMock
    # placeholders.
    runner.scanner = ListingsScanner()
    return runner


# ── RunState ────────────────────────────────────────────────────────


def test_run_state_fresh_factory():
    state = RunState.fresh(run_key="r", session_name="s", plan_signature="p")
    assert state.step_index == 0
    assert state.results == []
    assert state.loop_counters == {}
    assert state.listings_on_page == 0
    assert state.step_retry_counts == {}
    assert state.max_loop_iterations == 200
    assert state.checkpoint.run_key == "r"
    assert state.checkpoint.session_name == "s"
    assert state.checkpoint.plan_signature == "p"


def test_run_state_results_are_per_instance():
    a = RunState.fresh(run_key="x", session_name="x", plan_signature="x")
    b = RunState.fresh(run_key="y", session_name="y", plan_signature="y")
    a.results.append(StepResult(step_index=0, intent="x", success=True))
    assert b.results == []


# ── Loop preamble: cancel ──────────────────────────────────────────


def test_cancel_event_breaks_with_cancelled_status():
    runner = _runner_stub()
    runner._is_cancelled.return_value = True
    plan = _plan("navigate")
    state = _state()

    RunExecutor(runner).execute(plan, state)

    assert runner._final_status == "cancelled"
    runner._persist_checkpoint.assert_called()
    final_persist = runner._persist_checkpoint.call_args
    assert final_persist.kwargs["status"] == "cancelled"
    assert final_persist.kwargs["halt_reason"] == "cancel_event"


def test_pause_breaks_with_paused_status():
    runner = _runner_stub()
    runner.tool_channel.is_paused.return_value = True
    plan = _plan("navigate")
    state = _state()

    RunExecutor(runner).execute(plan, state)

    assert runner._final_status == "paused"


def test_budget_cap_breaks_with_halted_status():
    runner = _runner_stub()
    runner._cost_totals.return_value = (50.0, 50.0, 50.0, 999.0)  # over cap
    runner.max_cost = 25.0
    plan = _plan("navigate")
    state = _state()

    RunExecutor(runner).execute(plan, state)

    persist_calls = runner._persist_checkpoint.call_args_list
    halt_call = next(
        c for c in persist_calls if c.kwargs.get("halt_reason") == "budget_cap"
    )
    assert halt_call.kwargs["status"] == "halted"


def test_time_cap_breaks_with_halted_status():
    runner = _runner_stub()
    import time as _time
    runner._run_start = _time.time() - 10000  # 10000s ago
    runner.max_time = 60  # 1 minute
    plan = _plan("navigate")
    state = _state()

    RunExecutor(runner).execute(plan, state)

    persist_calls = runner._persist_checkpoint.call_args_list
    halt_call = next(
        c for c in persist_calls if c.kwargs.get("halt_reason") == "time_cap"
    )
    assert halt_call.kwargs["status"] == "halted"


# ── Loop step ──────────────────────────────────────────────────────


def test_loop_step_jumps_to_target_when_under_count():
    runner = _runner_stub()
    plan = _plan("navigate", "extract_data", "loop")
    plan.steps[2].loop_target = 1
    plan.steps[2].loop_count = 3
    state = _state()
    state.step_index = 2  # start at the loop step

    # Stop after the loop step's first iteration jumps us back
    runner._execute_step.return_value = StepResult(
        step_index=1, intent="extract_data", success=True,
    )
    runner._capture_screenshot_bytes.return_value = None

    # Execute one tick of the loop
    executor = RunExecutor(runner)
    # Manually drive _handle_loop_step
    executor._handle_loop_step(plan, plan.steps[2], state)

    assert state.loop_counters[2] == 1
    assert state.step_index == 1  # jumped to target


def test_loop_step_advances_when_max_count_reached():
    runner = _runner_stub()
    plan = _plan("loop")
    plan.steps[0].loop_count = 1
    state = _state()
    state.step_index = 0
    state.loop_counters[0] = 1  # already at max

    RunExecutor(runner)._handle_loop_step(plan, plan.steps[0], state)

    assert state.loop_counters[0] == 2  # incremented past max
    assert state.step_index == 1  # advanced past loop


# ── Effective step build ───────────────────────────────────────────


def test_build_effective_step_no_listings_uses_original_intent():
    runner = _runner_stub()
    state = _state()
    state.listings_on_page = 0
    step = MicroIntent(intent="Click the next title", type="click", section="extraction")

    eff = RunExecutor(runner)._build_effective_step(step, state)
    assert eff.intent == "Click the next title"
    assert eff.type == "click"


def test_build_effective_step_with_listings_injects_scroll_directive():
    runner = _runner_stub()
    state = _state()
    state.listings_on_page = 3
    step = MicroIntent(intent="Click the next title", type="click", section="extraction")

    eff = RunExecutor(runner)._build_effective_step(step, state)
    assert "Scroll down past the first 3 listings" in eff.intent
    assert "Then click the next listing title" in eff.intent
    assert eff.type == "click"


def test_build_effective_step_preserves_params_and_hints():
    runner = _runner_stub()
    state = _state()
    step = MicroIntent(
        intent="Submit", type="submit",
        params={"label": "Login"},
        hints={"layout": "single"},
    )
    eff = RunExecutor(runner)._build_effective_step(step, state)
    assert eff.params == {"label": "Login"}
    assert eff.hints == {"layout": "single"}


# ── _first_step_of_type ────────────────────────────────────────────


def test_first_step_of_type_finds_match():
    plan = _plan("navigate", "click", "extract_data", "loop")
    assert RunExecutor._first_step_of_type(plan, 0, "loop") == 3
    assert RunExecutor._first_step_of_type(plan, 0, "click") == 1


def test_first_step_of_type_returns_none_when_no_match():
    plan = _plan("navigate", "click")
    assert RunExecutor._first_step_of_type(plan, 0, "paginate") is None


# ── DUPLICATE outcome ──────────────────────────────────────────────


def test_duplicate_jumps_to_next_loop_and_increments_listings_on_page():
    runner = _runner_stub()
    plan = _plan("click", "extract_url", "loop")
    state = _state()
    state.step_index = 1  # extract_url just returned DUPLICATE
    state.listings_on_page = 5

    RunExecutor(runner)._handle_duplicate(plan, state)

    assert state.step_index == 2  # jumped to loop
    assert state.listings_on_page == 6  # incremented
    runner._return_to_results_page.assert_called_once()
    persist = runner._persist_checkpoint.call_args
    assert persist.kwargs["halt_reason"] == "duplicate_listing"


def test_duplicate_advances_when_no_loop_step():
    runner = _runner_stub()
    plan = _plan("click", "extract_url")
    state = _state()
    state.step_index = 1

    RunExecutor(runner)._handle_duplicate(plan, state)

    assert state.step_index == 2  # advanced past last step


# ── Finalize ────────────────────────────────────────────────────────


def test_finalize_marks_completed_when_step_index_at_end():
    runner = _runner_stub()
    runner._final_status = "running"
    plan = _plan("navigate", "extract_data")
    state = _state()
    state.step_index = 2  # past the last step

    RunExecutor(runner)._finalize(plan, state)

    assert runner._final_status == "completed"
    persist = runner._persist_checkpoint.call_args
    assert persist.kwargs["status"] == "completed"


def test_finalize_marks_halted_when_status_was_running_but_not_at_end():
    runner = _runner_stub()
    runner._final_status = "running"  # never set to terminal during loop
    plan = _plan("navigate", "extract_data", "extract_data")
    state = _state()
    state.step_index = 1  # mid-plan

    RunExecutor(runner)._finalize(plan, state)

    assert runner._final_status == "halted"
    persist = runner._persist_checkpoint.call_args
    assert persist.kwargs["status"] == "halted"
    assert persist.kwargs["halt_reason"] == "stopped"


def test_finalize_preserves_terminal_status_when_already_set():
    """If the loop set _final_status to cancelled / paused, finalize keeps it."""
    runner = _runner_stub()
    runner._final_status = "cancelled"
    plan = _plan("navigate")
    state = _state()
    state.step_index = 0  # never advanced

    RunExecutor(runner)._finalize(plan, state)

    # Should not be overwritten to halted
    assert runner._final_status == "cancelled"


def test_finalize_stashes_last_run_progress_on_runner():
    runner = _runner_stub()
    runner._final_status = "running"
    plan = _plan("navigate")
    state = _state()
    state.step_index = 1
    state.loop_counters = {3: 5}
    state.listings_on_page = 7

    RunExecutor(runner)._finalize(plan, state)

    assert runner._last_run_step_index == 1
    assert runner._last_loop_counters == {3: 5}
    assert runner._last_listings_on_page == 7


# ── Failure delegation ────────────────────────────────────────────


def test_handle_failure_delegates_to_recovery_policy_and_halts_on_outcome():
    from mantis_agent.gym.step_recovery import RecoveryOutcome

    runner = _runner_stub()
    runner._recovery_policy.handle_failure.return_value = RecoveryOutcome(
        halt=True, step_index=0, halt_reason="navigate_failed",
    )
    plan = _plan("navigate")
    state = _state()
    step = plan.steps[0]
    result = StepResult(step_index=0, intent="x", success=False)

    cont = RunExecutor(runner)._handle_failure(plan, state, step, result)
    assert cont is False
    runner._recovery_policy.handle_failure.assert_called_once()
    persist = runner._persist_checkpoint.call_args
    assert persist.kwargs["status"] == "halted"


def test_handle_failure_continues_when_outcome_is_not_halt():
    from mantis_agent.gym.step_recovery import RecoveryOutcome

    runner = _runner_stub()
    runner._recovery_policy.handle_failure.return_value = RecoveryOutcome(
        halt=False, step_index=2, halt_reason="page_exhausted",
    )
    plan = _plan("click", "extract_data", "paginate")
    state = _state()
    state.step_index = 0
    step = plan.steps[0]
    result = StepResult(step_index=0, intent="x", success=False, data="page_exhausted")

    cont = RunExecutor(runner)._handle_failure(plan, state, step, result)
    assert cont is True
    assert state.step_index == 2  # outcome's new index applied
    persist = runner._persist_checkpoint.call_args
    assert persist.kwargs["status"] == "running"
    assert persist.kwargs["halt_reason"] == "page_exhausted"


# ── _stamp_failure_context: preserve handler-stamped class ─────────


def test_stamp_failure_context_preserves_handler_stamped_class():
    """Regression: ``_stamp_failure_context`` runs after every failed
    step. Handlers (Holo3StepHandler for brain_loop_exhausted, click
    handler for wrong_target / no_state_change, the demotion path
    for no_state_change) stamp ``failure_class`` directly with the
    canonical class. The classifier here is a FALLBACK only — it
    must NOT clobber an already-stamped class. Otherwise the
    self-healing wiring (epic #377) is neutered: the rewriter keys
    off failure_class and the classifier's "unknown" overwrite
    means the signal is lost."""
    from mantis_agent.gym.run_executor import _stamp_failure_context

    sr = StepResult(
        step_index=0, intent="x", success=False,
        data="click_no_nav:no_change:modal already open",
        failure_class="no_state_change",  # handler-stamped
    )
    _stamp_failure_context(sr, env=MagicMock())
    assert sr.failure_class == "no_state_change"  # NOT clobbered to "unknown"


def test_stamp_failure_context_preserves_brain_loop_exhausted():
    from mantis_agent.gym.run_executor import _stamp_failure_context

    sr = StepResult(
        step_index=0, intent="x", success=False,
        data="",  # Holo3StepHandler doesn't write data on budget burn
        failure_class="brain_loop_exhausted",
    )
    _stamp_failure_context(sr, env=MagicMock())
    assert sr.failure_class == "brain_loop_exhausted"


def test_stamp_failure_context_falls_through_when_no_class_stamped():
    """When the handler didn't stamp anything, the classifier
    fallback kicks in over ``data`` + ``page_title``."""
    from mantis_agent.gym.run_executor import _stamp_failure_context

    sr = StepResult(
        step_index=0, intent="x", success=False,
        data="gate:FAIL:Error 403 forbidden",
    )
    assert sr.failure_class == ""  # baseline
    _stamp_failure_context(sr, env=MagicMock())
    assert sr.failure_class == "cf_challenge"  # classifier fallback


# ── Phase B: IntentRewriter wire-in ────────────────────────────────


def test_handle_failure_calls_rewriter_for_brain_loop_exhausted(monkeypatch):
    """When the just-failed step's class is in
    ``REWRITE_TRIGGERING_CLASSES`` and the budget allows, the executor
    calls ``intent_rewriter.propose_rewrite`` and stashes the returned
    new intent on the runner."""
    from unittest.mock import patch

    from mantis_agent.gym.step_recovery import RecoveryOutcome

    runner = _runner_stub()
    runner._recovery_policy.handle_failure.return_value = RecoveryOutcome(
        halt=False, step_index=0, halt_reason="",
    )
    # Real dicts so the rewriter wire-in mutates real state (rather
    # than MagicMock auto-created attributes).
    runner._step_intent_overrides = {}
    runner._step_rewrite_attempts = {}

    plan = _plan("scroll")
    state = _state()
    step = plan.steps[0]
    step.intent = "Scroll to reveal title, date, location, host details"
    result = StepResult(
        step_index=0, intent=step.intent, success=False,
        data="brain_loop_exhausted", failure_class="brain_loop_exhausted",
    )

    with patch(
        "mantis_agent.gym.intent_rewriter.propose_rewrite",
        return_value="Scroll down by one viewport",
    ) as mock_rewrite:
        cont = RunExecutor(runner)._handle_failure(plan, state, step, result)

    assert cont is True
    mock_rewrite.assert_called_once()
    assert runner._step_intent_overrides[0] == "Scroll down by one viewport"
    assert runner._step_rewrite_attempts[0] == 1


def test_handle_failure_skips_rewriter_for_unsupported_class(monkeypatch):
    """Failure classes outside the trigger set (selector_miss,
    cf_challenge, …) must NOT invoke the rewriter."""
    from unittest.mock import patch

    from mantis_agent.gym.step_recovery import RecoveryOutcome

    runner = _runner_stub()
    runner._recovery_policy.handle_failure.return_value = RecoveryOutcome(
        halt=False, step_index=0, halt_reason="",
    )
    runner._step_intent_overrides = {}
    runner._step_rewrite_attempts = {}

    plan = _plan("click")
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=False,
        data="cf_challenge:Error 403", failure_class="cf_challenge",
    )

    with patch("mantis_agent.gym.intent_rewriter.propose_rewrite") as mock_rewrite:
        RunExecutor(runner)._handle_failure(plan, state, plan.steps[0], result)

    mock_rewrite.assert_not_called()
    assert runner._step_intent_overrides == {}


def test_handle_failure_respects_per_step_rewrite_budget(monkeypatch):
    """One rewrite per step per run by default. A second failure on
    the same step does NOT invoke the rewriter again."""
    from unittest.mock import patch

    from mantis_agent.gym.step_recovery import RecoveryOutcome

    runner = _runner_stub()
    runner._recovery_policy.handle_failure.return_value = RecoveryOutcome(
        halt=False, step_index=0, halt_reason="",
    )
    runner._step_intent_overrides = {0: "earlier rewrite"}
    runner._step_rewrite_attempts = {0: 1}  # budget spent

    plan = _plan("click")
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="brain_loop_exhausted",
    )

    with patch("mantis_agent.gym.intent_rewriter.propose_rewrite") as mock_rewrite:
        RunExecutor(runner)._handle_failure(plan, state, plan.steps[0], result)

    mock_rewrite.assert_not_called()


def test_handle_failure_skips_rewriter_when_halting():
    """If recovery_policy halts, the rewriter must NOT fire — there's
    no retry coming."""
    from unittest.mock import patch

    from mantis_agent.gym.step_recovery import RecoveryOutcome

    runner = _runner_stub()
    runner._recovery_policy.handle_failure.return_value = RecoveryOutcome(
        halt=True, step_index=0, halt_reason="required_failed",
    )
    runner._step_intent_overrides = {}
    runner._step_rewrite_attempts = {}

    plan = _plan("click")
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="brain_loop_exhausted",
    )

    with patch("mantis_agent.gym.intent_rewriter.propose_rewrite") as mock_rewrite:
        cont = RunExecutor(runner)._handle_failure(plan, state, plan.steps[0], result)

    assert cont is False
    mock_rewrite.assert_not_called()


def test_handle_failure_keeps_original_intent_when_rewriter_returns_none(monkeypatch):
    """A None response from the rewriter (KEEP / API failure / empty)
    must NOT leave a stale override in place."""
    from unittest.mock import patch

    from mantis_agent.gym.step_recovery import RecoveryOutcome

    runner = _runner_stub()
    runner._recovery_policy.handle_failure.return_value = RecoveryOutcome(
        halt=False, step_index=0, halt_reason="",
    )
    runner._step_intent_overrides = {}
    runner._step_rewrite_attempts = {}

    plan = _plan("click")
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="wrong_target",
    )

    with patch(
        "mantis_agent.gym.intent_rewriter.propose_rewrite", return_value=None,
    ):
        RunExecutor(runner)._handle_failure(plan, state, plan.steps[0], result)

    # No override stashed; budget NOT consumed (a no-op rewrite
    # shouldn't burn the per-step budget).
    assert 0 not in runner._step_intent_overrides
    assert runner._step_rewrite_attempts.get(0, 0) == 0


def test_build_effective_step_applies_rewriter_override():
    """When ``_step_intent_overrides[step_index]`` carries a string,
    ``_build_effective_step`` uses it as the new intent."""
    runner = _runner_stub()
    runner._step_intent_overrides = {0: "Scroll down by one viewport"}

    plan = _plan("scroll")
    state = _state()
    state.step_index = 0
    plan.steps[0].intent = "Scroll to reveal title, date, location, host details"

    effective = RunExecutor(runner)._build_effective_step(plan.steps[0], state)
    assert effective.intent == "Scroll down by one viewport"


def test_build_effective_step_override_wins_over_scanner_directive():
    """Rewriter override beats the listings scanner's scroll directive
    — the rewrite was proposed in light of a specific failure."""
    runner = _runner_stub()
    runner._step_intent_overrides = {0: "REWRITE WINS"}
    runner.scanner.scroll_directive_for = MagicMock(return_value="SCANNER WOULD INJECT THIS")

    plan = _plan("click")
    state = _state()
    state.step_index = 0
    state.listings_on_page = 5

    effective = RunExecutor(runner)._build_effective_step(plan.steps[0], state)
    assert effective.intent == "REWRITE WINS"


# ── Form-shape no-state-change demote ─────────────────────────────


def test_no_state_change_demotes_submit_success_to_failure(monkeypatch):
    """Submit-shape success with no observable change → demoted to fail."""
    from mantis_agent.gym import step_snapshot as _snap

    runner = _runner_stub()
    state = _state()
    state.results.append(StepResult(step_index=0, intent="x", success=True, data="ok"))

    pre_snapshot = MagicMock()
    post_snapshot = MagicMock()
    monkeypatch.setattr(_snap, "capture", lambda r: post_snapshot)
    delta = MagicMock(has_changes=False)
    monkeypatch.setattr(_snap, "diff", lambda a, b: delta)

    eff_step = MicroIntent(intent="x", type="submit")
    RunExecutor(runner)._maybe_demote_form_no_change(state, eff_step, pre_snapshot)

    assert state.results[0].success is False
    assert ":no_state_change" in state.results[0].data


def test_no_state_change_skipped_for_non_form_steps(monkeypatch):
    runner = _runner_stub()
    state = _state()
    state.results.append(StepResult(step_index=0, intent="x", success=True, data="ok"))

    eff_step = MicroIntent(intent="x", type="click")  # not submit/select_option
    RunExecutor(runner)._maybe_demote_form_no_change(state, eff_step, MagicMock())

    assert state.results[0].success is True  # unchanged
    assert state.results[0].data == "ok"


# ── Click-shape no-state-change demote (epic #377 Phase A) ────────


def test_no_state_change_demotes_click_success_to_failure(monkeypatch):
    """Click success with no observable URL / page / scroll change →
    demoted to fail with ``failure_class=no_state_change``. Self-healing
    primitive that mirrors the submit demotion for click steps."""
    from mantis_agent.gym import step_snapshot as _snap

    runner = _runner_stub()
    state = _state()
    state.results.append(StepResult(step_index=0, intent="x", success=True, data="ok"))

    monkeypatch.setattr(_snap, "capture", lambda r: MagicMock())
    monkeypatch.setattr(_snap, "diff", lambda a, b: MagicMock(has_changes=False))

    eff_step = MicroIntent(intent="x", type="click")
    RunExecutor(runner)._maybe_demote_click_no_change(state, eff_step, MagicMock())

    assert state.results[0].success is False
    assert state.results[0].failure_class == "no_state_change"
    assert ":no_state_change" in state.results[0].data


def test_no_state_change_demotes_navigate_back_success_to_failure(monkeypatch):
    """``navigate_back`` is the other action-causing step whose lack of
    URL change cleanly signals a missed action — same demotion path."""
    from mantis_agent.gym import step_snapshot as _snap

    runner = _runner_stub()
    state = _state()
    state.results.append(StepResult(step_index=0, intent="x", success=True))

    monkeypatch.setattr(_snap, "capture", lambda r: MagicMock())
    monkeypatch.setattr(_snap, "diff", lambda a, b: MagicMock(has_changes=False))

    eff_step = MicroIntent(intent="x", type="navigate_back")
    RunExecutor(runner)._maybe_demote_click_no_change(state, eff_step, MagicMock())

    assert state.results[0].success is False
    assert state.results[0].failure_class == "no_state_change"


def test_click_demote_skipped_when_state_changed(monkeypatch):
    """Click success with a real state change → NOT demoted."""
    from mantis_agent.gym import step_snapshot as _snap

    runner = _runner_stub()
    state = _state()
    state.results.append(StepResult(step_index=0, intent="x", success=True, data="ok"))

    monkeypatch.setattr(_snap, "capture", lambda r: MagicMock())
    monkeypatch.setattr(_snap, "diff", lambda a, b: MagicMock(has_changes=True))

    eff_step = MicroIntent(intent="x", type="click")
    RunExecutor(runner)._maybe_demote_click_no_change(state, eff_step, MagicMock())

    assert state.results[0].success is True
    assert "no_state_change" not in (state.results[0].data or "")


def test_click_demote_skipped_for_form_step_types():
    """Form / extract / loop types are NOT in the click-demote allowlist
    (the form demotion handles submit; extract has no action to demote)."""
    runner = _runner_stub()
    state = _state()
    state.results.append(StepResult(step_index=0, intent="x", success=True))

    for step_type in ("submit", "fill_field", "select_option",
                      "extract_data", "extract_url", "scroll", "loop"):
        eff_step = MicroIntent(intent="x", type=step_type)
        RunExecutor(runner)._maybe_demote_click_no_change(state, eff_step, MagicMock())
        assert state.results[0].success is True, (
            f"step type {step_type!r} should not be demoted by the click-demote helper"
        )


def test_click_demote_uses_env_url_when_available(monkeypatch):
    """Issue #381: when the env exposes ``current_url`` directly, the
    demotion uses the browser's ground-truth URL (not
    ``runner._last_known_url`` which the handler can self-mutate).

    Pre-URL was /discover, post-URL still /discover → demote even if
    the snapshot delta shows handler-touched state changes (scroll,
    listings count). The lu.ma cluster A symptom this fixes."""
    from mantis_agent.gym import step_snapshot as _snap

    runner = _runner_stub()
    runner.env = MagicMock(current_url="https://luma.com/discover")
    runner._pre_step_env_url = "https://luma.com/discover"
    state = _state()
    state.results.append(StepResult(step_index=0, intent="x", success=True, data="ok"))

    pre_snapshot = MagicMock(
        viewport_stage=0, current_page=1, focused_input_signature="empty",
    )
    post_snapshot = MagicMock(
        viewport_stage=0, current_page=1, focused_input_signature="empty",
    )
    monkeypatch.setattr(_snap, "capture", lambda r: post_snapshot)

    eff_step = MicroIntent(intent="x", type="click")
    RunExecutor(runner)._maybe_demote_click_no_change(state, eff_step, pre_snapshot)

    assert state.results[0].success is False
    assert state.results[0].failure_class == "no_state_change"


def test_click_demote_skipped_when_env_url_actually_changed(monkeypatch):
    """Real navigation → don't demote, regardless of what
    ``runner._last_known_url`` says."""
    from mantis_agent.gym import step_snapshot as _snap

    runner = _runner_stub()
    runner.env = MagicMock(current_url="https://luma.com/event/abc")
    runner._pre_step_env_url = "https://luma.com/discover"
    state = _state()
    state.results.append(StepResult(step_index=0, intent="x", success=True))

    monkeypatch.setattr(_snap, "capture", lambda r: MagicMock())

    eff_step = MicroIntent(intent="x", type="click")
    RunExecutor(runner)._maybe_demote_click_no_change(state, eff_step, MagicMock())

    assert state.results[0].success is True


def test_click_demote_keeps_success_when_non_handler_state_changed(monkeypatch):
    """SPA-modal escape hatch: URL didn't change BUT a non-handler-
    touched snapshot field did (e.g. focused_input_signature). Treat
    as a legitimate intra-page action and keep success."""
    from mantis_agent.gym import step_snapshot as _snap

    runner = _runner_stub()
    runner.env = MagicMock(current_url="https://luma.com/discover")
    runner._pre_step_env_url = "https://luma.com/discover"
    state = _state()
    state.results.append(StepResult(step_index=0, intent="x", success=True))

    pre_snapshot = MagicMock(
        viewport_stage=0, current_page=1, focused_input_signature="empty",
    )
    post_snapshot = MagicMock(
        viewport_stage=0, current_page=1,
        focused_input_signature="search_input_focused",  # non-handler change
    )
    monkeypatch.setattr(_snap, "capture", lambda r: post_snapshot)

    eff_step = MicroIntent(intent="x", type="click")
    RunExecutor(runner)._maybe_demote_click_no_change(state, eff_step, pre_snapshot)

    assert state.results[0].success is True  # NOT demoted


def test_click_demote_falls_back_to_snapshot_when_env_url_unavailable(monkeypatch):
    """Legacy adapters / test stubs that don't expose
    ``env.current_url`` fall back to the original snapshot-diff
    behavior — primitive doesn't lose coverage."""
    from mantis_agent.gym import step_snapshot as _snap

    runner = _runner_stub()
    runner.env = MagicMock(spec=[])  # NO current_url
    runner._pre_step_env_url = ""
    state = _state()
    state.results.append(StepResult(step_index=0, intent="x", success=True))

    monkeypatch.setattr(_snap, "capture", lambda r: MagicMock())
    monkeypatch.setattr(_snap, "diff", lambda a, b: MagicMock(has_changes=False))

    eff_step = MicroIntent(intent="x", type="click")
    RunExecutor(runner)._maybe_demote_click_no_change(state, eff_step, MagicMock())

    assert state.results[0].success is False
    assert state.results[0].failure_class == "no_state_change"


def test_click_demote_respects_skip_envelope():
    """A step with ``skip=True`` (recipe rejection / advance signal)
    must NOT be demoted — skip is an intentional halt path, not a
    missed action."""
    runner = _runner_stub()
    state = _state()
    state.results.append(StepResult(
        step_index=0, intent="x", success=True,
        skip=True, skip_reason="dealer",
    ))

    eff_step = MicroIntent(intent="x", type="click")
    RunExecutor(runner)._maybe_demote_click_no_change(state, eff_step, MagicMock())

    assert state.results[0].success is True  # unchanged
    assert state.results[0].skip is True
