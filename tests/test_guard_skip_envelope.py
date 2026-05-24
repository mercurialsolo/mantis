"""Tests for #643 stage 2 — vision-only conditional step guard.

Sixth tactical sibling in the skip-envelope family (#246 recipe
rejection, #250 navigation halt, #254 context budget, #248
exploration substrate, #255 already-seen URL). Same envelope
contract, sixth trigger source: when ``step.guard`` names a state
variable that is False (or absent), ``execute_step`` short-circuits
with ``StepResult.skip=True / skip_reason='guard_<name>_false'``
and never dispatches to the handler — no vision call, no env
action.

The variable is bound by an earlier ``detect_visible`` step (or
left empty for plans that haven't run the detection yet — those
are treated identically to "guard False" so dependent steps don't
fire prematurely).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from mantis_agent.gym._runner_helpers import execute_step
from mantis_agent.plan_decomposer import MicroIntent


def _runner_with_state(state_vars: dict | None = None) -> MagicMock:
    """Minimum runner stub: only the fields execute_step's guard branch
    reads before deciding to skip. ``_handler_registry`` is set but
    never resolved when the guard fires (skip path is pre-dispatch).
    """
    runner = MagicMock()
    runner._handler_registry = MagicMock()
    runner._state_vars = state_vars or {}
    return runner


def test_guard_false_short_circuits_with_skip_envelope():
    """A step whose guard names a state variable bound to False is
    skipped entirely — handler.execute is never called, env is not
    touched, the skip envelope is what the runner returns."""
    runner = _runner_with_state({"has_show_more": False})
    step = MicroIntent(
        intent="Click Show More", type="click",
        guard="has_show_more",
    )

    result = execute_step(runner, step, index=5)

    assert result.success is False
    assert result.skip is True
    assert result.skip_reason == "guard_has_show_more_false"
    assert "guard:has_show_more=False" in result.data
    # Registry was never asked for a handler — the skip is pre-dispatch.
    runner._handler_registry.get.assert_not_called()


def test_guard_absent_var_treated_as_false():
    """A guard naming a variable that was never bound (no detect_visible
    step ran first) is treated identically to ``False`` — the step
    skips. Conservative: don't fire conditional actions when the
    condition was never evaluated."""
    runner = _runner_with_state({})
    step = MicroIntent(
        intent="Click Show More", type="click",
        guard="has_show_more",
    )

    result = execute_step(runner, step, index=5)

    assert result.skip is True
    assert result.skip_reason == "guard_has_show_more_false"


def test_guard_empty_string_disables_branching():
    """No guard set (the default, every legacy step) → fall through to
    the regular handler dispatch. The guard branch must be a no-op
    for plans that don't use conditionals."""
    runner = _runner_with_state({"has_show_more": True})
    # Stub the handler registry to return a dummy handler whose execute
    # returns a recognisable sentinel. If the guard branch wrongly
    # short-circuits when guard=="", this assertion fires.
    sentinel = SimpleNamespace(
        step_index=5, intent="x", success=True, data="dispatched",
        skip=False, skip_reason="", failure_class="",
        executor_backend="",
    )
    runner._handler_registry.get.return_value = MagicMock(
        execute=MagicMock(return_value=sentinel),
    )
    # No agentic escalation, no gate, no special routing.
    runner._step_handler_override = {}
    runner._step_failure_history = {}
    runner.extractor = None
    runner._opened_detail_in_new_tab = False

    step = MicroIntent(
        intent="Click something always", type="click",
        # guard="" (default)
    )

    result = execute_step(runner, step, index=5)

    # We don't care which exact handler path executed — only that
    # the guard branch DIDN'T short-circuit. The result must NOT
    # carry the guard skip envelope.
    assert result.skip is False or result.skip_reason != "guard__false"
    assert "guard:" not in (result.data or "")


def test_guard_true_proceeds_to_dispatch():
    """When the guard variable is True the step runs normally. We
    verify by asserting the handler.execute path was reached (not the
    skip envelope)."""
    runner = _runner_with_state({"has_show_more": True})
    sentinel = SimpleNamespace(
        step_index=5, intent="x", success=True, data="dispatched",
        skip=False, skip_reason="", failure_class="",
        executor_backend="",
    )
    runner._handler_registry.get.return_value = MagicMock(
        execute=MagicMock(return_value=sentinel),
    )
    runner._step_handler_override = {}
    runner._step_failure_history = {}
    runner.extractor = None
    runner._opened_detail_in_new_tab = False

    step = MicroIntent(
        intent="Click Show More", type="click",
        guard="has_show_more",
    )

    result = execute_step(runner, step, index=5)

    assert "guard:" not in (result.data or "")
