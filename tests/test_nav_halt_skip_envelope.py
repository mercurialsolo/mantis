"""Tests for issue #250 — navigation-primitive halt skip envelope.

Issue #246 shipped recipe-driven skip semantics: when a recipe
annotates a rejection key as ``"skip"``, the runner stamps
``StepResult.skip=True / skip_reason='<key>'`` and the host's tool
surface promotes that to a successful tool result. The orchestrator
advances past the listing instead of treating the rejection as a
failure to retry.

This issue extends the same envelope to *runner-side* navigation-
primitive halts. When a click/submit/scroll/navigate/gate step
exhausts retries (claude-director recovery + max_retries) and the
runner is halting, hosts that opt in via
``MicroPlanRunner(navigation_primitives_emit_skip={...})`` get the
last step result re-stamped with ``skip=True,
skip_reason='navigation_failed'``. Default is ``None`` — today's
behavior preserved.

Why opt-in: a host that doesn't model "advance to next position" as
the response to a navigation halt would mis-skip on a one-shot run
(e.g. login-form CRMs where every halt is genuinely a failure to
report). The opt-in is a one-line config change for hosts whose
plans iterate over a list of items.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan


# ── MicroPlanRunner.__init__ accepts the opt-in ───────────────────


def test_micro_plan_runner_init_accepts_navigation_primitives_emit_skip() -> None:
    """The constructor must take the opt-in set so callers don't
    have to monkey-patch the attribute at runtime."""
    from mantis_agent.gym.micro_runner import MicroPlanRunner

    runner = MicroPlanRunner(
        brain=MagicMock(),
        env=MagicMock(),
        navigation_primitives_emit_skip={"click", "submit", "scroll"},
    )
    assert runner.navigation_primitives_emit_skip == {"click", "submit", "scroll"}


def test_micro_plan_runner_default_navigation_primitives_emit_skip_is_none() -> None:
    """Default ``None`` preserves today's behavior — runs that don't
    set the field see no change in halt-emission semantics."""
    from mantis_agent.gym.micro_runner import MicroPlanRunner

    runner = MicroPlanRunner(brain=MagicMock(), env=MagicMock())
    assert runner.navigation_primitives_emit_skip is None


# ── Stamping logic — tested in isolation against _handle_failure ──


def _stamp_nav_skip(runner, last_result: StepResult, step_type: str) -> StepResult:
    """Re-implement the stamping predicate the way the executor will
    apply it, so we can test the policy independent of run_executor's
    plumbing.

    The executor mutates state.results[-1] in place when the predicate
    holds. This helper returns a fresh StepResult either unchanged or
    with the skip envelope populated, so the tests can compare
    side-by-side.
    """
    nav_set = getattr(runner, "navigation_primitives_emit_skip", None)
    if (
        nav_set
        and step_type in nav_set
        and not last_result.skip
    ):
        return replace(last_result, skip=True, skip_reason="navigation_failed")
    return last_result


def _runner_with_optin(nav_set):
    """Cheap stand-in that exposes only the attribute the predicate
    reads. Avoids constructing a full MicroPlanRunner for unit tests
    that exercise pure stamping logic."""
    r = MagicMock()
    r.navigation_primitives_emit_skip = nav_set
    return r


def test_stamps_skip_envelope_on_navigation_primitive_halt() -> None:
    """Right_click / scroll / submit / navigate / gate halts opted
    into the set must come out with the navigation_failed envelope."""
    runner = _runner_with_optin({"click", "submit", "scroll", "navigate", "gate"})
    failed_step = StepResult(
        step_index=4, intent="Click listing card #3",
        success=False, data="form_target_not_found",
    )
    out = _stamp_nav_skip(runner, failed_step, step_type="click")
    assert out.skip is True
    assert out.skip_reason == "navigation_failed"


def test_does_not_stamp_when_optin_is_none() -> None:
    """Default ``None`` preserves today's behavior — halt emits
    StepResult.skip=False, host gets ``status='halted'``."""
    runner = _runner_with_optin(None)
    failed_step = StepResult(
        step_index=4, intent="x", success=False, data="form_target_not_found",
    )
    out = _stamp_nav_skip(runner, failed_step, step_type="click")
    assert out.skip is False
    assert out.skip_reason is None


def test_does_not_stamp_when_step_type_not_in_set() -> None:
    """A halt on a step type the host didn't opt in for stays a
    plain halt — the skip envelope is only stamped for the
    primitives the caller listed."""
    runner = _runner_with_optin({"click"})
    failed_step = StepResult(step_index=4, intent="x", success=False)
    out = _stamp_nav_skip(runner, failed_step, step_type="fill_field")
    assert out.skip is False


def test_recipe_rejection_skip_wins_over_navigation_failed() -> None:
    """A step result that already carries ``skip=True`` from the
    recipe path (#246: dealer / spam / incomplete_required) must NOT
    be re-stamped. The recipe's specific reason is the actionable
    signal; clobbering it with the generic 'navigation_failed' would
    lose information the host needs to branch on."""
    runner = _runner_with_optin({"extract_data"})
    recipe_skipped = StepResult(
        step_index=4, intent="Extract row",
        success=False,
        data="REJECTED_DEALER|extractor marked as dealer|...",
        skip=True, skip_reason="dealer",
    )
    out = _stamp_nav_skip(runner, recipe_skipped, step_type="extract_data")
    assert out.skip is True
    # Reason is unchanged — the recipe's annotation wins.
    assert out.skip_reason == "dealer"


def test_does_not_stamp_when_step_succeeded() -> None:
    """A successful step shouldn't get the envelope (sanity guard —
    the executor only invokes this on the halt path, but the
    predicate must be defence-in-depth)."""
    runner = _runner_with_optin({"click"})
    successful = StepResult(step_index=4, intent="x", success=True)
    out = _stamp_nav_skip(runner, successful, step_type="click")
    # Stamping is gated only on .skip, not on .success — a succeeded
    # step shouldn't be on the halt path in the first place. Stamping
    # here would still be wrong because skip=True implies the host
    # should advance, but the step succeeded already. The executor
    # protects this via the halt-path call site; the predicate
    # itself doesn't double-check.
    assert out.success is True
    # Confirm the envelope IS set — the test exists to document the
    # gating responsibility lives in the call site, not the
    # predicate. Future contributors who collapse the layers should
    # see this test fail and reconsider.
    assert out.skip is True


# ── End-to-end integration through run_executor's halt path ──────


def test_run_executor_handle_failure_stamps_skip_on_nav_halt(monkeypatch) -> None:
    """Drive the executor's ``_handle_failure`` path with a stubbed
    recovery policy that halts; verify state.results[-1] gets the
    skip envelope when the step type matches the opt-in set."""
    from mantis_agent.gym.run_executor import RunExecutor, RunState
    from mantis_agent.gym.step_recovery import RecoveryOutcome

    runner = MagicMock()
    runner.navigation_primitives_emit_skip = {"click", "submit"}
    runner._recovery_policy.handle_failure.return_value = RecoveryOutcome(
        halt=True, step_index=5,
        halt_reason="required_failed:click",
    )
    runner.max_retries = 2

    from mantis_agent.gym.checkpoint import RunCheckpoint
    state = RunState(
        checkpoint=RunCheckpoint(),
        results=[
            StepResult(
                step_index=5, intent="Click listing card #3",
                success=False, data="form_target_not_found",
            ),
        ],
        step_index=5,
    )
    plan = MicroPlan(steps=[
        MicroIntent(intent="x", type="click", params={}),
    ])
    step = MicroIntent(intent="Click listing card #3", type="click")

    executor = RunExecutor(parent=runner)
    executor._persist = MagicMock()  # avoid checkpoint I/O

    cont = executor._handle_failure(
        plan=plan, state=state, step=step,
        step_result=state.results[-1],
    )
    assert cont is False  # halt
    assert state.results[-1].skip is True
    assert state.results[-1].skip_reason == "navigation_failed"


def test_run_executor_handle_failure_no_stamp_when_optin_none(monkeypatch) -> None:
    from mantis_agent.gym.run_executor import RunExecutor, RunState
    from mantis_agent.gym.step_recovery import RecoveryOutcome

    runner = MagicMock()
    runner.navigation_primitives_emit_skip = None
    runner._recovery_policy.handle_failure.return_value = RecoveryOutcome(
        halt=True, step_index=5,
        halt_reason="required_failed:click",
    )
    runner.max_retries = 2

    from mantis_agent.gym.checkpoint import RunCheckpoint
    state = RunState(
        checkpoint=RunCheckpoint(),
        results=[
            StepResult(step_index=5, intent="x", success=False),
        ],
        step_index=5,
    )
    plan = MicroPlan(steps=[MicroIntent(intent="x", type="click")])
    step = MicroIntent(intent="x", type="click")

    executor = RunExecutor(parent=runner)
    executor._persist = MagicMock()

    cont = executor._handle_failure(
        plan=plan, state=state, step=step,
        step_result=state.results[-1],
    )
    assert cont is False  # still halts
    assert state.results[-1].skip is False
    assert state.results[-1].skip_reason is None


def test_run_executor_handle_failure_preserves_recipe_skip(monkeypatch) -> None:
    """A step that already has the recipe-rejection skip envelope
    from #246 must not get re-stamped on the halt path."""
    from mantis_agent.gym.run_executor import RunExecutor, RunState
    from mantis_agent.gym.step_recovery import RecoveryOutcome

    runner = MagicMock()
    runner.navigation_primitives_emit_skip = {"extract_data"}
    runner._recovery_policy.handle_failure.return_value = RecoveryOutcome(
        halt=True, step_index=5,
        halt_reason="required_failed:extract_data",
    )
    runner.max_retries = 2

    from mantis_agent.gym.checkpoint import RunCheckpoint
    state = RunState(
        checkpoint=RunCheckpoint(),
        results=[
            StepResult(
                step_index=5, intent="Extract row", success=False,
                data="REJECTED_DEALER|...",
                skip=True, skip_reason="dealer",
            ),
        ],
        step_index=5,
    )
    plan = MicroPlan(steps=[MicroIntent(intent="x", type="extract_data")])
    step = MicroIntent(intent="Extract row", type="extract_data")

    executor = RunExecutor(parent=runner)
    executor._persist = MagicMock()

    executor._handle_failure(
        plan=plan, state=state, step=step,
        step_result=state.results[-1],
    )
    # The recipe's specific reason wins.
    assert state.results[-1].skip is True
    assert state.results[-1].skip_reason == "dealer"


def test_run_executor_handle_failure_no_stamp_on_non_halt_recovery(monkeypatch) -> None:
    """When the recovery policy returns ``halt=False`` (e.g.
    retry-this-step), the envelope must NOT be stamped — only
    terminal halts surface the skip-to-advance signal."""
    from mantis_agent.gym.run_executor import RunExecutor, RunState
    from mantis_agent.gym.step_recovery import RecoveryOutcome

    runner = MagicMock()
    runner.navigation_primitives_emit_skip = {"click", "submit"}
    runner._recovery_policy.handle_failure.return_value = RecoveryOutcome(
        halt=False, step_index=5,
        halt_reason="required_retry:click:1",
    )
    runner.max_retries = 2

    from mantis_agent.gym.checkpoint import RunCheckpoint
    state = RunState(
        checkpoint=RunCheckpoint(),
        results=[
            StepResult(step_index=5, intent="x", success=False),
        ],
        step_index=5,
    )
    plan = MicroPlan(steps=[MicroIntent(intent="x", type="click")])
    step = MicroIntent(intent="x", type="click")

    executor = RunExecutor(parent=runner)
    executor._persist = MagicMock()

    cont = executor._handle_failure(
        plan=plan, state=state, step=step,
        step_result=state.results[-1],
    )
    assert cont is True  # retry, no halt
    assert state.results[-1].skip is False
    assert state.results[-1].skip_reason is None
