"""Tests for runtime ``settle_ceiling_seconds`` field (#561).

Promotes the per-run cap on ``adaptive_settle.settle_after_action``
from "edit each caller's max_seconds" to a single runtime field.
A single submission can globally clamp every page-stabilisation wait
to e.g. 2.0s without touching the 20+ call sites scattered across
step handlers.

Most pages stabilise in 1-1.5s; the 2-3s tail past that is pure
wall-clock tax that adds up across a 100+ step plan.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mantis_agent.gym import adaptive_settle
from mantis_agent.gym.micro_runner import MicroPlanRunner
from mantis_agent.server_utils import build_micro_suite, merge_runtime


@pytest.fixture(autouse=True)
def _reset_ceiling():
    """Reset module-level ceiling between tests so they don't leak."""
    adaptive_settle.set_runtime_ceiling(None)
    yield
    adaptive_settle.set_runtime_ceiling(None)


# ── set_runtime_ceiling / get_runtime_ceiling ─────────────────────


def test_default_ceiling_is_none() -> None:
    assert adaptive_settle.get_runtime_ceiling() is None


def test_set_ceiling_stores_value() -> None:
    adaptive_settle.set_runtime_ceiling(2.0)
    assert adaptive_settle.get_runtime_ceiling() == 2.0


def test_set_ceiling_none_clears() -> None:
    adaptive_settle.set_runtime_ceiling(2.0)
    adaptive_settle.set_runtime_ceiling(None)
    assert adaptive_settle.get_runtime_ceiling() is None


def test_set_ceiling_zero_treated_as_none() -> None:
    adaptive_settle.set_runtime_ceiling(0)
    assert adaptive_settle.get_runtime_ceiling() is None


def test_set_ceiling_negative_treated_as_none() -> None:
    adaptive_settle.set_runtime_ceiling(-1.0)
    assert adaptive_settle.get_runtime_ceiling() is None


# ── _apply_ceiling arithmetic ──────────────────────────────────────


def test_apply_ceiling_no_clamp_when_unset() -> None:
    adaptive_settle.set_runtime_ceiling(None)
    assert adaptive_settle._apply_ceiling(5.0) == 5.0


def test_apply_ceiling_clamps_when_max_above_ceiling() -> None:
    adaptive_settle.set_runtime_ceiling(2.0)
    assert adaptive_settle._apply_ceiling(5.0) == 2.0


def test_apply_ceiling_preserves_when_max_below_ceiling() -> None:
    adaptive_settle.set_runtime_ceiling(2.0)
    assert adaptive_settle._apply_ceiling(1.0) == 1.0


def test_apply_ceiling_preserves_when_max_equals_ceiling() -> None:
    adaptive_settle.set_runtime_ceiling(2.0)
    assert adaptive_settle._apply_ceiling(2.0) == 2.0


# ── settle_after_action end-to-end ─────────────────────────────────


def test_settle_after_action_respects_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ceiling is set and the call site asks for more, the
    actual sleep is capped at the ceiling."""
    slept = []
    monkeypatch.setattr("mantis_agent.gym.adaptive_settle.time.sleep", lambda s: slept.append(s))

    # Plain object with no _screenshot / screenshot attrs → forces the
    # no-capture fixed-sleep branch where we can read elapsed directly.
    class _NoCaptureEnv:
        pass

    env = _NoCaptureEnv()

    adaptive_settle.set_runtime_ceiling(2.0)
    elapsed = adaptive_settle.settle_after_action(env, max_seconds=5.0)
    assert elapsed == 2.0
    assert slept == [2.0]


def test_settle_after_action_no_ceiling_uses_full_max(monkeypatch: pytest.MonkeyPatch) -> None:
    slept = []
    monkeypatch.setattr("mantis_agent.gym.adaptive_settle.time.sleep", lambda s: slept.append(s))

    class _NoCaptureEnv:
        pass

    env = _NoCaptureEnv()
    adaptive_settle.set_runtime_ceiling(None)
    elapsed = adaptive_settle.settle_after_action(env, max_seconds=5.0)
    assert elapsed == 5.0
    assert slept == [5.0]


# ── MicroPlanRunner ctor wires the ceiling ─────────────────────────


def _runner(**kwargs) -> MicroPlanRunner:
    return MicroPlanRunner(brain=MagicMock(), env=MagicMock(), **kwargs)


def test_runner_default_does_not_set_ceiling() -> None:
    _runner()
    assert adaptive_settle.get_runtime_ceiling() is None


def test_runner_explicit_ceiling_applies_globally() -> None:
    _runner(settle_ceiling_seconds=2.0)
    assert adaptive_settle.get_runtime_ceiling() == 2.0


def test_runner_stores_ceiling_on_self() -> None:
    runner = _runner(settle_ceiling_seconds=1.5)
    assert runner.settle_ceiling_seconds == 1.5


# ── build_micro_suite round-trip ───────────────────────────────────


def test_suite_omits_ceiling_when_caller_passes_none() -> None:
    suite = build_micro_suite([], "example.com")
    assert "_settle_ceiling_seconds" not in suite


def test_suite_persists_explicit_ceiling() -> None:
    suite = build_micro_suite([], "example.com", settle_ceiling_seconds=2.0)
    assert suite["_settle_ceiling_seconds"] == 2.0


# ── merge_runtime accepts settle_ceiling_seconds ───────────────────


def test_merge_runtime_threads_ceiling() -> None:
    merged = merge_runtime({"settle_ceiling_seconds": 2.0})
    assert merged["settle_ceiling_seconds"] == 2.0


def test_merge_runtime_override_wins_over_plan_default() -> None:
    merged = merge_runtime(
        {"settle_ceiling_seconds": 2.0},
        settle_ceiling_seconds=1.5,
    )
    assert merged["settle_ceiling_seconds"] == 1.5
