"""ExecutionCritic — epic #377 Phase C.

Tests the post-step observer + the one concrete v1 capability
(``navigate_back`` + ``brain_loop_exhausted`` → ``InsertStep`` for
direct navigate). Wire-in tests verify the runner applies the
directive and records the healing event."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.critic import (
    ExecutionCritic,
    InsertStep,
    ReplaceStep,
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


# ── ReplaceStep mutation + healing event (item 8) ─────────────────────


def _runner_with_frontier_state(*, results_base_url: str = ""):
    """Minimal runner with the shared-recovery-budget trackers that
    the frontier capability reads. Mirrors the real
    ``MicroPlanRunner`` attributes without pulling the heavy class."""
    return SimpleNamespace(
        _results_base_url=results_base_url,
        _healing_events=[],
        _step_failure_history={},
        _recovery_attempts_per_step={},
        _total_recovery_attempts=0,
        _recovery_hints={},
        _critic_frontier_fired_steps=set(),
        env=None,
    )


def test_apply_replace_step_swaps_step_in_place() -> None:
    """ReplaceStep replaces the step at ``state.step_index`` without
    shifting subsequent steps. Length stays the same, the new step
    occupies the same slot, neighbours are untouched."""
    runner = _runner_with_frontier_state()
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="step-0", type="navigate"))
    plan.steps.append(MicroIntent(intent="orig", type="submit",
                                   params={"label": "Contacted"},
                                   required=True, budget=4))
    plan.steps.append(MicroIntent(intent="step-2", type="extract_data"))
    state = _state(step_index=1)

    apply_directive(runner, plan, state, ReplaceStep(
        intent="Navigate to /leads?status=Contacted",
        step_type="navigate",
        params={"url": "/leads?status=Contacted"},
        reason="vision miss cascade — direct nav bypasses sidebar click",
    ))

    assert len(plan.steps) == 3
    replaced = plan.steps[1]
    assert replaced.type == "navigate"
    assert replaced.intent == "Navigate to /leads?status=Contacted"
    assert replaced.params == {"url": "/leads?status=Contacted"}
    # Neighbours untouched.
    assert plan.steps[0].intent == "step-0"
    assert plan.steps[2].intent == "step-2"
    # The original step's required + budget are inherited (critic
    # changes the action, not the plan-author's gate settings).
    assert replaced.required is True
    assert replaced.budget == 4


def test_apply_replace_step_records_healing_event() -> None:
    runner = _runner_with_frontier_state()
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="orig", type="submit"))
    state = _state(step_index=0)

    apply_directive(runner, plan, state, ReplaceStep(
        intent="Navigate to /x", step_type="navigate",
        params={"url": "/x"}, reason="r",
    ))
    assert len(runner._healing_events) == 1
    ev = runner._healing_events[0]
    assert ev["kind"] == "replace_step"
    assert ev["original_type"] == "submit"
    assert ev["new_type"] == "navigate"
    assert ev["source"] == "critic"


def test_apply_replace_step_resets_step_failure_history() -> None:
    """The replaced step gets a clean slate — the old step's retry
    pressure (which led the critic to intervene) doesn't carry to
    the new step. Otherwise the new step would inherit the rewrite-
    triggering failure_class and the agentic recovery loop would
    immediately fire again."""
    runner = _runner_with_frontier_state()
    runner._step_failure_history[5] = [
        {"x": 1, "y": 1, "kind": "wrong_target"},
        {"x": 2, "y": 2, "kind": "wrong_target"},
    ]
    runner._recovery_hints[5] = ["old hint"]
    plan = MicroPlan(domain="t")
    for _ in range(6):
        plan.steps.append(MicroIntent(intent="x", type="submit"))
    state = _state(step_index=5)

    apply_directive(runner, plan, state, ReplaceStep(
        intent="Navigate", step_type="navigate", reason="r",
    ))
    assert 5 not in runner._step_failure_history
    assert 5 not in runner._recovery_hints


def test_apply_replace_step_out_of_range_is_no_op() -> None:
    runner = _runner_with_frontier_state()
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="x", type="submit"))
    state = _state(step_index=99)  # past the end

    apply_directive(runner, plan, state, ReplaceStep(
        intent="X", step_type="navigate", reason="r",
    ))
    assert len(plan.steps) == 1
    assert plan.steps[0].intent == "x"
    assert runner._healing_events == []


# ── Frontier capability — env var gate + threshold + budget ──────────


def test_frontier_disabled_by_default(monkeypatch) -> None:
    """``MANTIS_CRITIC_FRONTIER`` unset → frontier capability is a
    no-op even with the threshold met. Preserves the existing cost
    profile for deployments that haven't opted in."""
    monkeypatch.delenv("MANTIS_CRITIC_FRONTIER", raising=False)
    runner = _runner_with_frontier_state()
    runner._step_failure_history[0] = [
        {"x": 1, "y": 1, "kind": "wrong_target"},
        {"x": 2, "y": 2, "kind": "wrong_target"},
    ]
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="x", type="submit"))
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="wrong_target",
    )
    out = critic.observe_step(
        plan, state, plan.steps[0], result, recovery_continued=True,
    )
    assert out is None


def test_frontier_skipped_below_wrong_target_threshold(monkeypatch) -> None:
    """Threshold = 2 wrong_target failures. After only one prior
    failure the critic stays quiet — the cheap retry / rewriter path
    deserves a chance first."""
    monkeypatch.setenv("MANTIS_CRITIC_FRONTIER", "enabled")
    runner = _runner_with_frontier_state()
    runner._step_failure_history[0] = [
        {"x": 1, "y": 1, "kind": "wrong_target"},
    ]
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="x", type="submit"))
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="wrong_target",
    )
    out = critic.observe_step(
        plan, state, plan.steps[0], result, recovery_continued=True,
    )
    assert out is None


def test_frontier_fires_on_persistent_wrong_target(monkeypatch) -> None:
    """At ≥2 wrong_target failures, env-enabled + budget-available,
    the critic invokes ``analyse_failure_and_recover`` and maps the
    decision to a directive."""
    monkeypatch.setenv("MANTIS_CRITIC_FRONTIER", "enabled")
    runner = _runner_with_frontier_state()
    runner._step_failure_history[0] = [
        {"x": 1, "y": 1, "kind": "wrong_target"},
        {"x": 2, "y": 2, "kind": "wrong_target"},
    ]

    # Stub the Claude call.
    from mantis_agent import agentic_recovery
    from mantis_agent.agentic_recovery import RecoveryDecision

    captured: list = []

    def _fake_analyse(**kwargs):
        captured.append(kwargs)
        return RecoveryDecision(
            mode="edit_step",
            reasoning="vision miss cascade; direct navigate bypasses",
            edited_step={
                "intent": "Navigate to /leads?status=Contacted",
                "type": "navigate",
                "params": {"url": "/leads?status=Contacted"},
            },
        )

    monkeypatch.setattr(agentic_recovery, "analyse_failure_and_recover", _fake_analyse)
    # Also patch the import-as-needed inside critic.py — the critic
    # imports lazily so we stub the underlying function rather than
    # the import path.

    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(
        intent="Click Contacted in sidebar", type="submit",
        params={"label": "Contacted"},
    ))
    state = _state()
    result = StepResult(
        step_index=0, intent="Click Contacted", success=False,
        failure_class="wrong_target",
    )
    out = critic.observe_step(
        plan, state, plan.steps[0], result, recovery_continued=True,
    )
    assert isinstance(out, ReplaceStep)
    assert out.step_type == "navigate"
    assert out.params == {"url": "/leads?status=Contacted"}
    # Budget counters incremented.
    assert runner._recovery_attempts_per_step[0] == 1
    assert runner._total_recovery_attempts == 1
    # Step is marked as consulted so a second observe_step on the
    # same step is a no-op.
    assert 0 in runner._critic_frontier_fired_steps
    # Claude got called once.
    assert len(captured) == 1


def test_frontier_skipped_if_already_fired_on_step(monkeypatch) -> None:
    """The frontier consultation happens at most once per step —
    even if the next step also reports wrong_target, we don't re-
    call Claude on the same slot."""
    monkeypatch.setenv("MANTIS_CRITIC_FRONTIER", "enabled")
    runner = _runner_with_frontier_state()
    runner._step_failure_history[0] = [
        {"x": 1, "y": 1, "kind": "wrong_target"},
        {"x": 2, "y": 2, "kind": "wrong_target"},
    ]
    runner._critic_frontier_fired_steps.add(0)

    from mantis_agent import agentic_recovery
    monkeypatch.setattr(
        agentic_recovery, "analyse_failure_and_recover",
        lambda **_: pytest.fail("Claude must not be called"),
    )

    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="x", type="submit"))
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="wrong_target",
    )
    out = critic.observe_step(
        plan, state, plan.steps[0], result, recovery_continued=True,
    )
    assert out is None


def test_frontier_skipped_when_per_step_budget_exhausted(monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_CRITIC_FRONTIER", "enabled")
    runner = _runner_with_frontier_state()
    runner._step_failure_history[0] = [
        {"x": 1, "y": 1, "kind": "wrong_target"},
        {"x": 2, "y": 2, "kind": "wrong_target"},
    ]
    # Existing terminal-failure recovery already burned the per-step
    # budget — the critic must not double-spend.
    from mantis_agent.agentic_recovery import DEFAULT_MAX_RECOVERIES_PER_STEP
    runner._recovery_attempts_per_step[0] = DEFAULT_MAX_RECOVERIES_PER_STEP

    from mantis_agent import agentic_recovery
    monkeypatch.setattr(
        agentic_recovery, "analyse_failure_and_recover",
        lambda **_: pytest.fail("Claude must not be called"),
    )

    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="x", type="submit"))
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="wrong_target",
    )
    out = critic.observe_step(
        plan, state, plan.steps[0], result, recovery_continued=True,
    )
    assert out is None


def test_frontier_maps_add_hint_decision_to_recovery_hint(monkeypatch) -> None:
    """add_hint mode doesn't return a directive — it writes to the
    runner's recovery_hints map so the next retry's prompt picks it
    up. The critic STILL marks the step as consulted so we don't
    redo Claude work."""
    monkeypatch.setenv("MANTIS_CRITIC_FRONTIER", "enabled")
    runner = _runner_with_frontier_state()
    runner._step_failure_history[0] = [
        {"x": 1, "y": 1, "kind": "wrong_target"},
        {"x": 2, "y": 2, "kind": "wrong_target"},
    ]

    from mantis_agent import agentic_recovery
    from mantis_agent.agentic_recovery import RecoveryDecision
    monkeypatch.setattr(
        agentic_recovery, "analyse_failure_and_recover",
        lambda **_: RecoveryDecision(
            mode="add_hint",
            reasoning="text",
            hint="The Contacted link is the second item in the LEAD VIEWS sidebar.",
        ),
    )

    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="x", type="submit"))
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="wrong_target",
    )
    out = critic.observe_step(
        plan, state, plan.steps[0], result, recovery_continued=True,
    )
    assert out is None
    assert runner._recovery_hints[0] == [
        "The Contacted link is the second item in the LEAD VIEWS sidebar."
    ]
    assert 0 in runner._critic_frontier_fired_steps


def test_frontier_maps_halt_decision_to_no_directive(monkeypatch) -> None:
    """halt mode → no directive. The existing terminal-failure path
    handles the actual halt (this critic capability is mid-run; it
    doesn't short-circuit the recovery policy)."""
    monkeypatch.setenv("MANTIS_CRITIC_FRONTIER", "enabled")
    runner = _runner_with_frontier_state()
    runner._step_failure_history[0] = [
        {"x": 1, "y": 1, "kind": "wrong_target"},
        {"x": 2, "y": 2, "kind": "wrong_target"},
    ]

    from mantis_agent import agentic_recovery
    from mantis_agent.agentic_recovery import RecoveryDecision
    monkeypatch.setattr(
        agentic_recovery, "analyse_failure_and_recover",
        lambda **_: RecoveryDecision(mode="halt", reasoning="anti-bot block"),
    )

    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="x", type="submit"))
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="wrong_target",
    )
    out = critic.observe_step(
        plan, state, plan.steps[0], result, recovery_continued=True,
    )
    assert out is None


def test_frontier_does_not_fire_on_non_wrong_target_failure(monkeypatch) -> None:
    """no_state_change / brain_loop_exhausted route through other
    capabilities — the wrong_target-specific frontier doesn't pre-
    empt them."""
    monkeypatch.setenv("MANTIS_CRITIC_FRONTIER", "enabled")
    runner = _runner_with_frontier_state()
    runner._step_failure_history[0] = [
        {"x": 1, "y": 1, "kind": "no_state_change"},
        {"x": 2, "y": 2, "kind": "no_state_change"},
    ]

    from mantis_agent import agentic_recovery
    monkeypatch.setattr(
        agentic_recovery, "analyse_failure_and_recover",
        lambda **_: pytest.fail("Claude must not be called"),
    )

    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="x", type="submit"))
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="no_state_change",
    )
    out = critic.observe_step(
        plan, state, plan.steps[0], result, recovery_continued=True,
    )
    assert out is None


def test_frontier_maps_insert_steps_to_insert_directive(monkeypatch) -> None:
    """insert_steps mode emits the FIRST inserted step as an
    InsertStep directive. Multi-step inserts come from the terminal
    path's full splice — this MVP keeps the critic surface clean."""
    monkeypatch.setenv("MANTIS_CRITIC_FRONTIER", "enabled")
    runner = _runner_with_frontier_state()
    runner._step_failure_history[0] = [
        {"x": 1, "y": 1, "kind": "wrong_target"},
        {"x": 2, "y": 2, "kind": "wrong_target"},
    ]

    from mantis_agent import agentic_recovery
    from mantis_agent.agentic_recovery import RecoveryDecision
    monkeypatch.setattr(
        agentic_recovery, "analyse_failure_and_recover",
        lambda **_: RecoveryDecision(
            mode="insert_steps",
            reasoning="dismiss the modal first, then retry the original step",
            inserted_steps=[
                {"intent": "Press Escape to dismiss modal", "type": "key_press",
                 "params": {"keys": "Escape"}},
            ],
        ),
    )

    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="t")
    plan.steps.append(MicroIntent(intent="x", type="submit"))
    state = _state()
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="wrong_target",
    )
    out = critic.observe_step(
        plan, state, plan.steps[0], result, recovery_continued=True,
    )
    assert isinstance(out, InsertStep)
    assert out.step_type == "key_press"
    assert "Escape" in out.intent
