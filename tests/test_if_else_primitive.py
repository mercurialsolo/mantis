"""Tests for the ``if_else`` branching primitive.

Composes with ``detect_visible`` (which writes a bool to
``runner._state_vars[out_var]``). The ``if_else`` step reads the same
variable via ``condition_var`` and jumps to ``then_target`` (truthy)
or ``else_target`` (falsy/missing).

The handler is dispatched from ``run_executor`` next to ``loop``, so
these tests exercise the dispatch + state-mutation directly via a
minimal stub plan + state, no Xvfb or live brain required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.run_executor import RunExecutor
from mantis_agent.plan_decomposer import MicroIntent


class _StubRunner:
    """Minimal MicroPlanRunner stand-in.

    The executor reads only ``_state_vars`` (for condition_var lookup)
    plus the persistence callbacks. Tests inject the var bag and stub
    out the persist hook.
    """

    def __init__(self, state_vars: dict | None = None) -> None:
        self._state_vars = dict(state_vars or {})


@dataclass
class _StubPlan:
    """Stand-in for MicroPlan with the .steps list the dispatcher reads."""

    steps: list = field(default_factory=list)


@dataclass
class _StubState:
    """Mirror RunState fields touched by _handle_if_else_step."""

    step_index: int = 0
    results: list = field(default_factory=list)


def _make_executor(state_vars: dict | None = None) -> RunExecutor:
    """RunExecutor with the bare minimum wiring to drive _handle_if_else_step."""
    runner = _StubRunner(state_vars)
    ex = RunExecutor.__new__(RunExecutor)
    ex.parent = runner
    ex._persist = lambda plan, state: None  # type: ignore[assignment]
    return ex


def _make_step(**kw) -> MicroIntent:
    base = {
        "intent": "branch",
        "type": "if_else",
    }
    base.update(kw)
    return MicroIntent(**base)


def _plan_with(steps_len: int) -> _StubPlan:
    return _StubPlan(steps=[object()] * steps_len)


# ── happy path ──────────────────────────────────────────────────────


def test_truthy_var_jumps_to_then_target():
    ex = _make_executor({"logged_in": True})
    state = _StubState(step_index=2)
    step = _make_step(condition_var="logged_in", then_target=5, else_target=9)
    ex._handle_if_else_step(_plan_with(10), step, state)
    assert state.step_index == 5


def test_falsy_var_jumps_to_else_target():
    ex = _make_executor({"logged_in": False})
    state = _StubState(step_index=2)
    step = _make_step(condition_var="logged_in", then_target=5, else_target=9)
    ex._handle_if_else_step(_plan_with(10), step, state)
    assert state.step_index == 9


def test_missing_var_is_falsy():
    """A missing key is falsy — jumps to else_target."""
    ex = _make_executor({})
    state = _StubState(step_index=2)
    step = _make_step(condition_var="nonexistent", then_target=5, else_target=9)
    ex._handle_if_else_step(_plan_with(10), step, state)
    assert state.step_index == 9


# ── synthetic step result ──────────────────────────────────────────


def test_emits_synthetic_step_result_with_decision_trace():
    """The branch records a StepResult so the run trace carries the
    decision (var, value, branch, target) — readable via grep."""
    ex = _make_executor({"x": True})
    state = _StubState(step_index=3)
    step = _make_step(condition_var="x", then_target=7, else_target=10)
    ex._handle_if_else_step(_plan_with(12), step, state)
    assert len(state.results) == 1
    r = state.results[0]
    assert isinstance(r, StepResult)
    assert r.success is True
    assert r.step_index == 3
    assert "x=True" in r.data
    assert "branch=then" in r.data
    assert "target=7" in r.data


def test_else_branch_result_records_correctly():
    ex = _make_executor({"x": False})
    state = _StubState(step_index=3)
    step = _make_step(condition_var="x", then_target=7, else_target=10)
    ex._handle_if_else_step(_plan_with(12), step, state)
    r = state.results[0]
    assert "x=False" in r.data
    assert "branch=else" in r.data
    assert "target=10" in r.data


# ── safety fall-through ────────────────────────────────────────────


def test_missing_condition_var_falls_through_to_next_step():
    """A mis-authored step with no condition_var should NOT silently
    teleport — fall through to step_index+1."""
    ex = _make_executor({"x": True})
    state = _StubState(step_index=4)
    step = _make_step(condition_var="", then_target=7, else_target=10)
    ex._handle_if_else_step(_plan_with(12), step, state)
    assert state.step_index == 5
    # And no synthetic result on the malformed branch.
    assert state.results == []


def test_target_out_of_range_falls_through():
    """Out-of-range target → safer to fall through than hang."""
    ex = _make_executor({"x": True})
    state = _StubState(step_index=4)
    step = _make_step(condition_var="x", then_target=100, else_target=200)
    ex._handle_if_else_step(_plan_with(10), step, state)
    assert state.step_index == 5
    assert state.results == []


def test_negative_target_falls_through():
    """Default -1 sentinel (no target wired) → fall through."""
    ex = _make_executor({"x": True})
    state = _StubState(step_index=4)
    step = _make_step(condition_var="x", then_target=-1, else_target=-1)
    ex._handle_if_else_step(_plan_with(10), step, state)
    assert state.step_index == 5


# ── value coercion ─────────────────────────────────────────────────


@pytest.mark.parametrize("value,expected_branch_target", [
    (True, 5),
    (1, 5),
    ("non-empty", 5),
    ([1], 5),
    (False, 9),
    (0, 9),
    ("", 9),
    ([], 9),
    (None, 9),
])
def test_python_truthiness_drives_branch(value, expected_branch_target):
    """Variable value is coerced via bool() — same as Python truthiness."""
    ex = _make_executor({"v": value})
    state = _StubState(step_index=1)
    step = _make_step(condition_var="v", then_target=5, else_target=9)
    ex._handle_if_else_step(_plan_with(12), step, state)
    assert state.step_index == expected_branch_target


# ── MicroIntent field plumbing ─────────────────────────────────────


def test_microintent_has_branching_fields():
    """The dataclass exposes condition_var / then_target / else_target."""
    step = MicroIntent(
        intent="x", type="if_else",
        condition_var="logged_in", then_target=3, else_target=7,
    )
    assert step.condition_var == "logged_in"
    assert step.then_target == 3
    assert step.else_target == 7


def test_microintent_defaults_for_non_branch_steps():
    """Non-branch steps default to empty / -1 so plan-authoring stays
    forward-compat — existing plans don't need to touch these fields."""
    step = MicroIntent(intent="click x", type="click")
    assert step.condition_var == ""
    assert step.then_target == -1
    assert step.else_target == -1
