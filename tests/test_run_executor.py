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
    state = _state()
    state.listings_on_page = 0
    step = MicroIntent(intent="Click the next title", type="click", section="extraction")

    eff = RunExecutor._build_effective_step(step, state)
    assert eff.intent == "Click the next title"
    assert eff.type == "click"


def test_build_effective_step_with_listings_injects_scroll_directive():
    state = _state()
    state.listings_on_page = 3
    step = MicroIntent(intent="Click the next title", type="click", section="extraction")

    eff = RunExecutor._build_effective_step(step, state)
    assert "Scroll down past the first 3 listings" in eff.intent
    assert "Then click the next listing title" in eff.intent
    assert eff.type == "click"


def test_build_effective_step_preserves_params_and_hints():
    state = _state()
    step = MicroIntent(
        intent="Submit", type="submit",
        params={"label": "Login"},
        hints={"layout": "single"},
    )
    eff = RunExecutor._build_effective_step(step, state)
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
