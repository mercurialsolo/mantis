"""Tests for the recovery decision types (Phase 1.3 of EPIC #161).

Types-only commit — no dispatch logic exists yet, so these tests pin
the public shape so a follow-up that adds dispatch can't accidentally
rename a field or drop a value the runner is about to switch on.
"""

from __future__ import annotations

import pytest

from mantis_agent.gym.step_recovery import (
    DECISION_SUCCEED,
    RecoveryAction,
    RecoveryDecision,
)


def test_recovery_action_values_match_runner_branches():
    """Every legacy branch in run() needs a corresponding enum value."""
    expected = {
        "succeed", "retry", "skip", "reverse_and_skip",
        "jump_to_type", "reload_and_retry", "halt",
    }
    assert {a.value for a in RecoveryAction} == expected


def test_recovery_decision_is_frozen():
    """Decisions must not mutate after dispatch returns — runner owns side effects."""
    d = RecoveryDecision(action=RecoveryAction.RETRY, halt_reason="x")
    with pytest.raises((AttributeError, Exception)):
        d.halt_reason = "y"  # type: ignore[misc]


def test_recovery_decision_default_fields():
    """Defaults map to a no-op decision so callers only set the fields they need."""
    d = RecoveryDecision(action=RecoveryAction.SUCCEED)
    assert d.halt_reason == ""
    assert d.wait_seconds == 0.0
    assert d.jump_target_types == ()
    assert d.reset_loop_counters is False
    assert d.reset_loop_counters_value == 0
    assert d.halt_reason_message == ""
    assert d.log_message == ""
    assert d.log_level == "info"
    assert d.extras == {}


def test_recovery_decision_carries_jump_target():
    """JUMP_TO_TYPE must be able to encode the step types to fast-forward to."""
    d = RecoveryDecision(
        action=RecoveryAction.JUMP_TO_TYPE,
        jump_target_types=("paginate", "loop"),
        halt_reason="page_exhausted",
    )
    assert d.jump_target_types == ("paginate", "loop")
    assert d.halt_reason == "page_exhausted"


def test_recovery_decision_carries_retry_wait():
    """RETRY decisions should carry the sleep duration and halt_reason."""
    d = RecoveryDecision(
        action=RecoveryAction.RETRY,
        wait_seconds=12.0,
        halt_reason="page_blocked_retry:1",
    )
    assert d.wait_seconds == 12.0
    assert d.halt_reason == "page_blocked_retry:1"


def test_decision_succeed_singleton():
    """The pre-allocated singleton matches the value-equality decision."""
    assert DECISION_SUCCEED.action is RecoveryAction.SUCCEED
    assert DECISION_SUCCEED == RecoveryDecision(action=RecoveryAction.SUCCEED)
