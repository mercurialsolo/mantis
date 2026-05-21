"""Tests for runtime brain-budget clamps (#560).

The decomposer LLM advertises generous upper bounds (scroll=10, click=8)
so it can reason about worst case. The runtime clamps each step's
``max_steps`` at Holo3 dispatch — capping wasted brain cycles without
forcing the decomposer to internalise per-deploy cost preferences.

Three behaviours covered:

1. ``effective_brain_budget`` picks the lower of (step.budget, cap).
2. ``MicroPlanRunner.brain_budgets`` defaults to
   ``DEFAULT_BRAIN_BUDGET_CAPS`` when constructor kwarg is ``None``,
   and honours an explicit ``{}`` as "no caps".
3. ``build_micro_suite(brain_budgets=...)`` round-trips through the
   suite dict so submitters can override per run.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.gym.micro_runner import MicroPlanRunner
from mantis_agent.gym.step_handlers.holo3 import (
    DEFAULT_BRAIN_BUDGET_CAPS,
    effective_brain_budget,
)
from mantis_agent.server_utils import build_micro_suite, merge_runtime


# ── effective_brain_budget ─────────────────────────────────────────


def test_clamp_returns_cap_when_step_budget_higher() -> None:
    assert effective_brain_budget("scroll", 10, {"scroll": 3}) == 3
    assert effective_brain_budget("click", 8, {"click": 4}) == 4


def test_clamp_returns_step_budget_when_cap_higher() -> None:
    assert effective_brain_budget("scroll", 2, {"scroll": 3}) == 2


def test_clamp_returns_step_budget_when_type_missing_from_caps() -> None:
    # paginate isn't in DEFAULT_BRAIN_BUDGET_CAPS — passthrough.
    assert effective_brain_budget("paginate", 10, {"scroll": 3}) == 10


def test_clamp_none_caps_means_no_cap() -> None:
    assert effective_brain_budget("scroll", 10, None) == 10


def test_clamp_empty_caps_means_no_cap() -> None:
    # Explicit ``{}`` is the escape hatch: honour every step's budget verbatim.
    assert effective_brain_budget("scroll", 10, {}) == 10


def test_clamp_ignores_invalid_cap_below_one() -> None:
    # A cap of 0 would brick the handler; fall back to step.budget.
    assert effective_brain_budget("scroll", 10, {"scroll": 0}) == 10
    assert effective_brain_budget("scroll", 10, {"scroll": -1}) == 10


# ── MicroPlanRunner storage ────────────────────────────────────────


def _runner(**kwargs) -> MicroPlanRunner:
    return MicroPlanRunner(brain=MagicMock(), env=MagicMock(), **kwargs)


def test_runner_defaults_to_DEFAULT_BRAIN_BUDGET_CAPS() -> None:
    runner = _runner()
    assert runner.brain_budgets == DEFAULT_BRAIN_BUDGET_CAPS


def test_runner_default_is_a_copy_not_shared_reference() -> None:
    # Mutating one runner's caps must not leak into the next default.
    runner_a = _runner()
    runner_a.brain_budgets["scroll"] = 99
    runner_b = _runner()
    assert runner_b.brain_budgets["scroll"] == DEFAULT_BRAIN_BUDGET_CAPS["scroll"]


def test_runner_explicit_empty_dict_disables_caps() -> None:
    runner = _runner(brain_budgets={})
    assert runner.brain_budgets == {}


def test_runner_explicit_override_replaces_defaults() -> None:
    runner = _runner(brain_budgets={"scroll": 5, "paginate": 6})
    assert runner.brain_budgets == {"scroll": 5, "paginate": 6}
    # Note: ``click`` default isn't reinstated — explicit override is
    # authoritative. Submitters who want partial overrides must merge
    # client-side.


# ── build_micro_suite round-trip ───────────────────────────────────


def test_suite_omits_brain_budgets_when_caller_passes_none() -> None:
    suite = build_micro_suite([], "example.com")
    assert "_brain_budgets" not in suite


def test_suite_persists_explicit_brain_budgets() -> None:
    suite = build_micro_suite([], "example.com", brain_budgets={"scroll": 2})
    assert suite["_brain_budgets"] == {"scroll": 2}


def test_suite_persists_explicit_empty_dict() -> None:
    # The escape-hatch "no caps" must survive the round-trip; the
    # runner reads ``{}`` (not None) and disables clamps.
    suite = build_micro_suite([], "example.com", brain_budgets={})
    assert suite["_brain_budgets"] == {}


# ── merge_runtime accepts brain_budgets ────────────────────────────


def test_merge_runtime_threads_brain_budgets() -> None:
    merged = merge_runtime({"brain_budgets": {"scroll": 2}})
    assert merged["brain_budgets"] == {"scroll": 2}


def test_merge_runtime_override_wins_over_plan_default() -> None:
    merged = merge_runtime(
        {"brain_budgets": {"scroll": 2}},
        brain_budgets={"scroll": 5},
    )
    assert merged["brain_budgets"] == {"scroll": 5}
