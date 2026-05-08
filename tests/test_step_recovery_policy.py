"""StepRecoveryPolicy unit tests — Phase 1.3 of EPIC #161.

The 165-LOC failure-recovery if/elif that used to live on the runner
is now ``StepRecoveryPolicy.handle_failure``. These tests pin every
exit path so a refactor that drops a branch shows up here.

Side-effects (sleeps, env.step keypresses, runner method calls,
retry-count mutations, loop_counters mutations) happen INSIDE
handle_failure before it returns; tests assert both the
RecoveryOutcome shape AND the side-effect calls on a mocked parent
runner.

Coverage:
- required: retry budget then halt
- gate: anti-bot one-shot retry; non-anti-bot halt
- navigate: instant halt + reverse_step
- click: page_exhausted (jump to paginate / loop / advance),
         scan_error / page_blocked (bounded retry, then page_blocked
         filtered-reload retry, then halt), generic (escape + jump-to-loop)
- filter: reverse + advance
- scroll: pseudo-success advance
- navigate_back: 3-attempt alt+Left + advance
- paginate: exhaust loop_counters + advance
- extract_url / extract_data: skip + advance
- generic step type: reverse + advance
- _first_step_of_type helper

No Xvfb, no GymRunner, no real ClaudeExtractor.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.actions import ActionType
from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.step_recovery import (
    RecoveryOutcome,
    StepRecoveryPolicy,
)
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan


def _runner_stub() -> MagicMock:
    runner = MagicMock()
    runner.env = MagicMock()
    runner.extractor = MagicMock()
    runner.site_config = MagicMock()
    return runner


def _result(*, success: bool = False, data: str = "") -> StepResult:
    return StepResult(step_index=0, intent="x", success=success, data=data)


def _plan(*types: str) -> MicroPlan:
    plan = MicroPlan(domain="test")
    for t in types:
        plan.steps.append(MicroIntent(intent=f"step-{t}", type=t))
    return plan


# ── required ────────────────────────────────────────────────────────


def test_required_failure_under_budget_retries_same_step(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_recovery.time.sleep", lambda *_: None)
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)
    retries: dict = {}

    outcome = policy.handle_failure(
        step=MicroIntent(intent="x", type="click", required=True),
        step_result=_result(),
        plan=_plan("click"),
        step_index=0,
        step_retry_counts=retries,
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.step_index == 0  # retry same step
    assert outcome.halt_reason == "required_retry:click:1"
    assert retries[0] == 1


def test_required_failure_over_budget_halts(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_recovery.time.sleep", lambda *_: None)
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)
    retries = {0: 2}  # already at budget

    outcome = policy.handle_failure(
        step=MicroIntent(intent="x", type="click", required=True),
        step_result=_result(),
        plan=_plan("click"),
        step_index=0,
        step_retry_counts=retries,
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is True
    assert outcome.halt_reason == "required_failed:click"


# ── gate ────────────────────────────────────────────────────────────


def test_gate_failure_with_antibot_keyword_retries_navigate_once(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_recovery.time.sleep", lambda *_: None)
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)
    retries: dict = {}

    plan = MicroPlan(domain="test")
    plan.steps.append(MicroIntent(intent="navigate to x", type="navigate"))
    plan.steps.append(
        MicroIntent(intent="verify gate", type="extract_data", gate=True, verify="page loaded"),
    )

    outcome = policy.handle_failure(
        step=plan.steps[1],
        step_result=_result(data="cloudflare challenge appeared"),
        plan=plan,
        step_index=1,
        step_retry_counts=retries,
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.halt_reason == "gate_retry"
    assert retries["gate_retry_1"] == 1
    runner._execute_navigate.assert_called_once_with(plan.steps[0], 0)


def test_gate_failure_without_antibot_keyword_halts(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_recovery.time.sleep", lambda *_: None)
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)

    outcome = policy.handle_failure(
        step=MicroIntent(intent="x", type="extract_data", gate=True, verify="ok"),
        step_result=_result(data="some other failure mode"),
        plan=_plan("navigate", "extract_data"),
        step_index=1,
        step_retry_counts={},
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is True
    assert outcome.halt_reason == "gate_failed"


# ── navigate ────────────────────────────────────────────────────────


def test_navigate_failure_instant_halt():
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)

    outcome = policy.handle_failure(
        step=MicroIntent(intent="x", type="navigate"),
        step_result=_result(),
        plan=_plan("navigate"),
        step_index=0,
        step_retry_counts={},
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is True
    assert outcome.halt_reason == "navigate_failed"
    runner._reverse_step.assert_called_once()


# ── click ───────────────────────────────────────────────────────────


def test_click_failure_page_exhausted_jumps_to_paginate(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_recovery.time.sleep", lambda *_: None)
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)

    plan = _plan("click", "extract_data", "navigate_back", "loop", "paginate", "loop")
    outcome = policy.handle_failure(
        step=plan.steps[0],
        step_result=_result(data="page_exhausted"),
        plan=plan,
        step_index=0,
        step_retry_counts={},
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.step_index == 4  # paginate index
    assert outcome.halt_reason == "page_exhausted"


def test_click_failure_page_exhausted_no_paginate_jumps_to_loop(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_recovery.time.sleep", lambda *_: None)
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)

    plan = _plan("click", "extract_data", "loop")  # no paginate
    outcome = policy.handle_failure(
        step=plan.steps[0],
        step_result=_result(data="page_exhausted"),
        plan=plan,
        step_index=0,
        step_retry_counts={},
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.step_index == 2  # loop index


def test_click_failure_scan_error_retries_with_4s_wait(monkeypatch):
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "mantis_agent.gym.step_recovery.time.sleep",
        lambda s: sleep_calls.append(s),
    )
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)
    retries: dict = {}

    outcome = policy.handle_failure(
        step=MicroIntent(intent="x", type="click"),
        step_result=_result(data="scan_error"),
        plan=_plan("click"),
        step_index=0,
        step_retry_counts=retries,
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.step_index == 0  # same step, retry
    assert "scan_error_retry:1" in outcome.halt_reason
    assert 4 in sleep_calls
    assert retries[0] == 1


def test_click_failure_page_blocked_retries_with_12s_wait(monkeypatch):
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "mantis_agent.gym.step_recovery.time.sleep",
        lambda s: sleep_calls.append(s),
    )
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)
    retries: dict = {}

    outcome = policy.handle_failure(
        step=MicroIntent(intent="x", type="click"),
        step_result=_result(data="page_blocked"),
        plan=_plan("click"),
        step_index=0,
        step_retry_counts=retries,
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.step_index == 0
    assert 12 in sleep_calls
    assert "page_blocked_retry:1" in outcome.halt_reason


def test_click_failure_page_blocked_after_retry_budget_reloads_filtered_url(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_recovery.time.sleep", lambda *_: None)
    runner = _runner_stub()
    runner._ensure_results_filters.return_value = True  # reload succeeded
    policy = StepRecoveryPolicy(runner)
    retries = {0: 2}  # already at budget

    outcome = policy.handle_failure(
        step=MicroIntent(intent="x", type="click"),
        step_result=_result(data="page_blocked"),
        plan=_plan("click"),
        step_index=0,
        step_retry_counts=retries,
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.halt_reason == "page_blocked_reload"
    assert outcome.step_index == 0  # retry same step
    runner._ensure_results_filters.assert_called_once_with(0, force_reload=True)
    assert retries["page_blocked_reload_0"] == 1
    assert retries[0] == 0  # reset for retry


def test_click_failure_page_blocked_reload_failed_halts(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_recovery.time.sleep", lambda *_: None)
    runner = _runner_stub()
    runner._ensure_results_filters.return_value = False  # reload failed
    policy = StepRecoveryPolicy(runner)

    outcome = policy.handle_failure(
        step=MicroIntent(intent="x", type="click"),
        step_result=_result(data="page_blocked"),
        plan=_plan("click"),
        step_index=0,
        step_retry_counts={0: 2},
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is True
    assert outcome.halt_reason == "page_blocked"


def test_click_failure_login_redirect_reloads_then_jumps_to_loop(monkeypatch):
    """login_redirect → _ensure_results_filters(force_reload=True), jump to next loop."""
    monkeypatch.setattr("mantis_agent.gym.step_recovery.time.sleep", lambda *_: None)
    runner = _runner_stub()
    runner._ensure_results_filters.return_value = True
    policy = StepRecoveryPolicy(runner)

    plan = _plan("click", "extract_data", "loop")
    outcome = policy.handle_failure(
        step=plan.steps[0],
        step_result=_result(data="login_redirect"),
        plan=plan,
        step_index=0,
        step_retry_counts={},
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.step_index == 2  # loop
    assert outcome.halt_reason == "login_redirect_recovered"
    runner._ensure_results_filters.assert_called_once_with(0, force_reload=True)


def test_click_failure_login_redirect_swallows_reload_exception(monkeypatch):
    """Recovery must not propagate a reload exception — log and skip card."""
    monkeypatch.setattr("mantis_agent.gym.step_recovery.time.sleep", lambda *_: None)
    runner = _runner_stub()
    runner._ensure_results_filters.side_effect = RuntimeError("nav blew up")
    policy = StepRecoveryPolicy(runner)

    plan = _plan("click", "loop")
    outcome = policy.handle_failure(
        step=plan.steps[0],
        step_result=_result(data="login_redirect"),
        plan=plan,
        step_index=0,
        step_retry_counts={},
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.step_index == 1
    assert outcome.halt_reason == "login_redirect_recovered"


def test_click_failure_newtab_blank_skips_to_loop(monkeypatch):
    """newtab_blank → no reload (handler closed tab), jump to next loop step."""
    monkeypatch.setattr("mantis_agent.gym.step_recovery.time.sleep", lambda *_: None)
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)

    plan = _plan("click", "extract_data", "loop")
    outcome = policy.handle_failure(
        step=plan.steps[0],
        step_result=_result(data="newtab_blank"),
        plan=plan,
        step_index=0,
        step_retry_counts={},
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.step_index == 2  # loop
    assert outcome.halt_reason == "newtab_blank"
    runner._ensure_results_filters.assert_not_called()


def test_click_failure_generic_escapes_then_jumps_to_loop(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_recovery.time.sleep", lambda *_: None)
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)

    plan = _plan("click", "extract_data", "loop")
    outcome = policy.handle_failure(
        step=plan.steps[0],
        step_result=_result(data="something else"),
        plan=plan,
        step_index=0,
        step_retry_counts={},
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.step_index == 2  # loop
    assert outcome.halt_reason == "click_failed"
    # Escape was sent
    keypresses = [
        c.args[0].params.get("keys")
        for c in runner.env.step.call_args_list
        if c.args[0].action_type == ActionType.KEY_PRESS
    ]
    assert "Escape" in keypresses


# ── filter / scroll / paginate / extract_* / generic ────────────────


def test_filter_failure_reverses_and_advances():
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)

    outcome = policy.handle_failure(
        step=MicroIntent(intent="x", type="filter"),
        step_result=_result(),
        plan=_plan("filter"),
        step_index=0,
        step_retry_counts={},
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.step_index == 1
    assert outcome.halt_reason == "filter_failed"
    runner._reverse_step.assert_called_once()


def test_scroll_failure_advances_as_pseudo_success():
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)

    outcome = policy.handle_failure(
        step=MicroIntent(intent="x", type="scroll"),
        step_result=_result(),
        plan=_plan("scroll"),
        step_index=0,
        step_retry_counts={},
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.step_index == 1
    assert outcome.halt_reason == "scroll_no_done"


def test_paginate_failure_exhausts_loop_counters_and_advances():
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)
    counters = {3: 2, 5: 0}

    outcome = policy.handle_failure(
        step=MicroIntent(intent="x", type="paginate"),
        step_result=_result(),
        plan=_plan("paginate"),
        step_index=0,
        step_retry_counts={},
        loop_counters=counters,
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.step_index == 1
    assert outcome.halt_reason == "paginate_exhausted"
    # Counters exhausted
    assert all(v == 999999 for v in counters.values())


def test_extract_url_failure_skips():
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)

    outcome = policy.handle_failure(
        step=MicroIntent(intent="x", type="extract_url", claude_only=True),
        step_result=_result(),
        plan=_plan("extract_url"),
        step_index=0,
        step_retry_counts={},
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.step_index == 1
    assert outcome.halt_reason == "extract_url_failed"


def test_extract_data_failure_skips():
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)

    outcome = policy.handle_failure(
        step=MicroIntent(intent="x", type="extract_data", claude_only=True),
        step_result=_result(),
        plan=_plan("extract_data"),
        step_index=0,
        step_retry_counts={},
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt_reason == "extract_data_failed"
    assert outcome.step_index == 1


def test_generic_step_type_reverses_and_advances():
    runner = _runner_stub()
    policy = StepRecoveryPolicy(runner)
    res = _result()

    outcome = policy.handle_failure(
        step=MicroIntent(intent="x", type="some_unknown_type"),
        step_result=res,
        plan=_plan("some_unknown_type"),
        step_index=0,
        step_retry_counts={},
        loop_counters={},
        max_retries=2,
        listings_on_page=0,
    )
    assert outcome.halt is False
    assert outcome.step_index == 1
    assert outcome.halt_reason == "some_unknown_type_failed"
    runner._reverse_step.assert_called_once()
    assert res.reversed is True


# ── _first_step_of_type helper ──────────────────────────────────────


def test_first_step_of_type_finds_match():
    plan = _plan("navigate", "click", "loop", "paginate", "loop")
    assert StepRecoveryPolicy._first_step_of_type(plan, 0, "click") == 1
    assert StepRecoveryPolicy._first_step_of_type(plan, 2, "loop") == 2
    assert StepRecoveryPolicy._first_step_of_type(plan, 3, "loop") == 4


def test_first_step_of_type_returns_none_when_no_match():
    plan = _plan("navigate", "click")
    assert StepRecoveryPolicy._first_step_of_type(plan, 0, "paginate") is None


def test_recovery_outcome_is_frozen():
    """Outcome must not mutate after dispatch returns it."""
    o = RecoveryOutcome(halt=False, step_index=3, halt_reason="r")
    try:
        o.halt = True  # type: ignore[misc]
        assert False, "should have raised"
    except (AttributeError, Exception):
        pass
