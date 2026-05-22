"""Tests for runtime ``max_recoveries_per_run`` / ``max_recoveries_per_step`` (#567).

Promotes the hardcoded ``DEFAULT_MAX_RECOVERIES_PER_*`` constants in
``agentic_recovery.py`` to per-submission runtime fields. A single
submission can raise caps for plans that legitimately need many
recoveries (long extraction loops, flaky pages) or lower caps for CI
runs that want tighter fail-fast — both without touching code or
redeploying.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from mantis_agent.agentic_recovery import (
    DEFAULT_MAX_RECOVERIES_PER_RUN,
    DEFAULT_MAX_RECOVERIES_PER_STEP,
    effective_max_recoveries,
)
from mantis_agent.gym.micro_runner import MicroPlanRunner
from mantis_agent.server_utils import build_micro_suite, merge_runtime


# ── effective_max_recoveries resolution ────────────────────────────


def test_no_overrides_returns_defaults() -> None:
    runner = SimpleNamespace()  # no attrs at all
    per_step, per_run = effective_max_recoveries(runner)
    assert per_step == DEFAULT_MAX_RECOVERIES_PER_STEP
    assert per_run == DEFAULT_MAX_RECOVERIES_PER_RUN


def test_explicit_overrides_win() -> None:
    runner = SimpleNamespace(
        max_recoveries_per_step=7, max_recoveries_per_run=42,
    )
    per_step, per_run = effective_max_recoveries(runner)
    assert per_step == 7
    assert per_run == 42


def test_none_override_falls_back_to_default() -> None:
    runner = SimpleNamespace(
        max_recoveries_per_step=None, max_recoveries_per_run=None,
    )
    per_step, per_run = effective_max_recoveries(runner)
    assert per_step == DEFAULT_MAX_RECOVERIES_PER_STEP
    assert per_run == DEFAULT_MAX_RECOVERIES_PER_RUN


def test_zero_override_falls_back_to_default() -> None:
    # 0 would brick the recovery layer entirely — treat as default.
    runner = SimpleNamespace(
        max_recoveries_per_step=0, max_recoveries_per_run=0,
    )
    per_step, per_run = effective_max_recoveries(runner)
    assert per_step == DEFAULT_MAX_RECOVERIES_PER_STEP
    assert per_run == DEFAULT_MAX_RECOVERIES_PER_RUN


def test_negative_override_falls_back_to_default() -> None:
    runner = SimpleNamespace(
        max_recoveries_per_step=-1, max_recoveries_per_run=-5,
    )
    per_step, per_run = effective_max_recoveries(runner)
    assert per_step == DEFAULT_MAX_RECOVERIES_PER_STEP
    assert per_run == DEFAULT_MAX_RECOVERIES_PER_RUN


def test_mixed_overrides() -> None:
    # Per-step set, per-run left default.
    runner = SimpleNamespace(
        max_recoveries_per_step=10, max_recoveries_per_run=None,
    )
    per_step, per_run = effective_max_recoveries(runner)
    assert per_step == 10
    assert per_run == DEFAULT_MAX_RECOVERIES_PER_RUN


def test_runner_without_attrs_does_not_raise() -> None:
    class _Bare:
        pass

    # No getattr raises, just returns defaults.
    per_step, per_run = effective_max_recoveries(_Bare())
    assert per_step == DEFAULT_MAX_RECOVERIES_PER_STEP
    assert per_run == DEFAULT_MAX_RECOVERIES_PER_RUN


# ── MicroPlanRunner storage ────────────────────────────────────────


def _runner(**kwargs) -> MicroPlanRunner:
    return MicroPlanRunner(brain=MagicMock(), env=MagicMock(), **kwargs)


def test_runner_default_recoveries_are_none() -> None:
    runner = _runner()
    assert runner.max_recoveries_per_run is None
    assert runner.max_recoveries_per_step is None


def test_runner_explicit_recoveries_stored() -> None:
    runner = _runner(max_recoveries_per_run=50, max_recoveries_per_step=5)
    assert runner.max_recoveries_per_run == 50
    assert runner.max_recoveries_per_step == 5


def test_runner_effective_max_recoveries_picks_up_overrides() -> None:
    runner = _runner(max_recoveries_per_run=50, max_recoveries_per_step=5)
    per_step, per_run = effective_max_recoveries(runner)
    assert per_step == 5
    assert per_run == 50


# ── build_micro_suite round-trip ───────────────────────────────────


def test_suite_omits_recovery_caps_when_caller_passes_none() -> None:
    suite = build_micro_suite([], "example.com")
    assert "_max_recoveries_per_run" not in suite
    assert "_max_recoveries_per_step" not in suite


def test_suite_persists_explicit_recovery_caps() -> None:
    suite = build_micro_suite(
        [], "example.com",
        max_recoveries_per_run=50, max_recoveries_per_step=5,
    )
    assert suite["_max_recoveries_per_run"] == 50
    assert suite["_max_recoveries_per_step"] == 5


def test_suite_persists_only_supplied_cap() -> None:
    # Per-run only.
    suite = build_micro_suite([], "example.com", max_recoveries_per_run=50)
    assert suite["_max_recoveries_per_run"] == 50
    assert "_max_recoveries_per_step" not in suite


# ── merge_runtime accepts the fields ───────────────────────────────


def test_merge_runtime_threads_recovery_caps() -> None:
    merged = merge_runtime({
        "max_recoveries_per_run": 50,
        "max_recoveries_per_step": 5,
    })
    assert merged["max_recoveries_per_run"] == 50
    assert merged["max_recoveries_per_step"] == 5


def test_merge_runtime_override_wins_over_plan_default() -> None:
    merged = merge_runtime(
        {"max_recoveries_per_run": 50},
        max_recoveries_per_run=25,
    )
    assert merged["max_recoveries_per_run"] == 25
