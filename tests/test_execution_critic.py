"""ExecutionCritic — epic #377 Phase C.

Tests the post-step observer + the one concrete v1 capability
(``navigate_back`` + ``brain_loop_exhausted`` → ``InsertStep`` for
direct navigate). Wire-in tests verify the runner applies the
directive and records the healing event."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.critic import (
    ExecutionCritic,
    InsertStep,
    apply_directive,
)
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan


def _state(step_index: int = 0):
    from mantis_agent.gym.checkpoint import RunCheckpoint
    from mantis_agent.gym.run_executor import RunState
    return RunState(
        checkpoint=RunCheckpoint(run_key="t", plan_signature="s", session_name="x"),
        step_index=step_index,
    )


def _runner(*, results_base_url: str = "https://example.com/discover"):
    runner = SimpleNamespace(
        _results_base_url=results_base_url,
        _healing_events=[],
    )
    return runner


# ── observe_step: gating ─────────────────────────────────────────────────


def test_observe_returns_none_on_success() -> None:
    """Successful steps don't trigger recovery directives."""
    runner = _runner()
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="x", type="navigate_back"))
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=True,
        failure_class="brain_loop_exhausted",  # would normally trigger
    )
    out = critic.observe_step(
        plan, state, plan.steps[0], result, recovery_continued=True,
    )
    assert out is None


def test_observe_returns_none_when_recovery_halted() -> None:
    """No point inserting steps ahead of a halted run."""
    runner = _runner()
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="x", type="navigate_back"))
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="brain_loop_exhausted",
    )
    out = critic.observe_step(
        plan, state, plan.steps[0], result, recovery_continued=False,
    )
    assert out is None


def test_observe_returns_none_for_non_navigate_back_steps() -> None:
    runner = _runner()
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="x", type="scroll"))
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="brain_loop_exhausted",
    )
    out = critic.observe_step(
        plan, state, plan.steps[0], result, recovery_continued=True,
    )
    assert out is None


def test_observe_returns_none_for_non_brain_loop_failure() -> None:
    runner = _runner()
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="x", type="navigate_back"))
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="selector_miss",  # different class
    )
    out = critic.observe_step(
        plan, state, plan.steps[0], result, recovery_continued=True,
    )
    assert out is None


def test_observe_returns_none_when_no_base_url() -> None:
    """Without a destination, the critic has nothing to propose."""
    runner = _runner(results_base_url="")
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="x", type="navigate_back"))
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="brain_loop_exhausted",
    )
    out = critic.observe_step(
        plan, state, plan.steps[0], result, recovery_continued=True,
    )
    assert out is None


# ── observe_step: navigate_back recovery directive ──────────────────────


def test_observe_emits_insert_step_for_navigate_back_loop_burn() -> None:
    runner = _runner(results_base_url="https://luma.com/discover")
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="luma")
    plan.steps.append(MicroIntent(intent="Press back", type="navigate_back"))
    state = _state()
    result = StepResult(
        step_index=0, intent="Press back", success=False,
        failure_class="brain_loop_exhausted",
    )
    out = critic.observe_step(
        plan, state, plan.steps[0], result, recovery_continued=True,
    )
    assert isinstance(out, InsertStep)
    assert out.step_type == "navigate"
    assert "https://luma.com/discover" in out.intent
    assert out.params == {"url": "https://luma.com/discover"}


# ── apply_directive: runner mutation ─────────────────────────────────────


def test_apply_directive_splices_step_into_plan() -> None:
    runner = _runner()
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="step-0", type="navigate_back"))
    plan.steps.append(MicroIntent(intent="step-1", type="click"))
    state = _state(step_index=1)  # critic emits before step 1 runs

    apply_directive(runner, plan, state, InsertStep(
        intent="Navigate to https://example.com",
        step_type="navigate",
        reason="navigate_back recovery",
        params={"url": "https://example.com"},
    ))
    assert len(plan.steps) == 3
    inserted = plan.steps[1]
    assert inserted.type == "navigate"
    assert inserted.intent == "Navigate to https://example.com"
    assert inserted.params == {"url": "https://example.com"}
    # Step that was at index 1 is now at index 2.
    assert plan.steps[2].intent == "step-1"


def test_apply_directive_records_healing_event() -> None:
    runner = _runner()
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="step-0", type="navigate_back"))
    state = _state(step_index=1)

    apply_directive(runner, plan, state, InsertStep(
        intent="Navigate to https://example.com",
        step_type="navigate",
        reason="r",
    ))
    assert len(runner._healing_events) == 1
    ev = runner._healing_events[0]
    assert ev["kind"] == "insert_step"
    assert ev["source"] == "critic"
    assert ev["inserted_type"] == "navigate"


def test_apply_directive_no_op_on_non_directive() -> None:
    """Defensive: passing a non-InsertStep value (None, dict, …)
    must not mutate the plan."""
    runner = _runner()
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="step-0", type="click"))
    state = _state()

    apply_directive(runner, plan, state, None)  # type: ignore[arg-type]
    apply_directive(runner, plan, state, {"intent": "x"})  # type: ignore[arg-type]
    assert len(plan.steps) == 1
    assert runner._healing_events == []


# ── End-to-end: critic fires inside the executor ────────────────────────


def test_critic_fires_inside_executor_after_navigate_back_failure(monkeypatch):
    """Integration: when ``_handle_failure`` returns continue=True
    and the step that failed is navigate_back with
    brain_loop_exhausted, the critic emits InsertStep, the runner
    applies it, and the inserted navigate step lands in plan.steps."""
    from mantis_agent.gym.run_executor import RunExecutor
    from mantis_agent.gym.step_recovery import RecoveryOutcome

    runner = MagicMock()
    runner._final_status = "running"
    runner.plan_signature = "sig"
    runner._results_base_url = "https://luma.com/discover"
    # Real list / dicts so the rewriter wire-in's defensive
    # attempts_used < max_attempts comparison works (MagicMock-typed
    # attributes would crash that arithmetic).
    runner._healing_events = []
    runner._step_intent_overrides = {}
    runner._step_rewrite_attempts = {}
    runner._recovery_policy.handle_failure.return_value = RecoveryOutcome(
        halt=False, step_index=1, halt_reason="",
    )
    runner._critic = ExecutionCritic(runner)

    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="Press back", type="navigate_back"))
    plan.steps.append(MicroIntent(intent="Click X", type="click"))

    state = _state()
    state.results.append(StepResult(
        step_index=0, intent="Press back", success=False,
        failure_class="brain_loop_exhausted",
    ))

    cont = RunExecutor(runner)._handle_failure(
        plan, state, plan.steps[0], state.results[0],
    )
    assert cont is True

    # Now invoke the critic hook directly with the same args the
    # executor uses.
    RunExecutor(runner)._consult_critic(
        plan, state, plan.steps[0], state.results[0], continued=cont,
    )

    # The plan now has the inserted navigate step at the new
    # state.step_index position.
    assert len(plan.steps) == 3
    inserted = plan.steps[state.step_index]
    assert inserted.type == "navigate"
    assert "luma.com/discover" in inserted.intent
    # And the healing log records it.
    kinds = [e["kind"] for e in runner._healing_events]
    assert "insert_step" in kinds
