"""Tests for the exploration-mode runtime substrate (issue #248).

Three layers:

1. **Data shapes** — ``ExperimentEvent``, ``ExplorationBudget``,
   ``VariantOutcome`` carry the fields the refinement agent reads,
   defaults are sensible, and ``to_dict`` / ``from_dict`` round-trip
   cleanly through JSON.
2. **Derivation helpers** — ``rejection_histogram_from_steps`` and
   ``url_coverage_from_steps`` build the comparison signals a
   refinement agent uses to A/B variants without re-parsing
   ``StepResult.data`` strings on its end.
3. **Runner entrypoint** — ``MicroPlanRunner.run_with_exploration``
   returns one ``VariantOutcome`` per plan/recipe variant; budget is
   enforced; recipe swap on the extractor is reverted after each
   variant so legacy callers see no state leak.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.exploration import (
    EXPERIMENT_KINDS,
    ExperimentEvent,
    ExplorationBudget,
    VariantOutcome,
    rejection_histogram_from_steps,
    url_coverage_from_steps,
)
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan


# ── ExperimentEvent ─────────────────────────────────────────────────


def test_experiment_event_defaults() -> None:
    e = ExperimentEvent(
        kind="action_alternative_tried",
        intent="Click the third listing card",
        page_url="https://www.example.com/search",
    )
    assert e.attempted == {}
    assert e.outcome == {}
    assert e.cost == 0.0
    assert e.timestamp > 0  # default_factory=time.time


def test_experiment_event_round_trips_through_dict() -> None:
    e = ExperimentEvent(
        kind="recipe_rejection_observed",
        intent="Extract row",
        page_url="https://www.example.com/boat/1234",
        attempted={"selector": "tile_3"},
        outcome={"reason": "dealer", "summary": "..."},
        cost=0.04,
        timestamp=1715000000.0,
    )
    payload = e.to_dict()
    # Round-trips through JSON so a refinement agent can persist
    # outcomes to disk between sessions.
    payload2 = json.loads(json.dumps(payload))
    e2 = ExperimentEvent.from_dict(payload2)
    assert e == e2


def test_experiment_event_kinds_constant_enumerated() -> None:
    """The closed set of kinds documented in the issue. New kinds
    should be added here (and to the docstring) — refinement agents
    on older mantis versions still consume foreign kinds as opaque
    blobs but the canonical set is what we ship in the schema."""
    expected = {
        "action_alternative_tried",
        "recipe_rejection_observed",
        "dom_quirk_detected",
        "sub_goal_phrasing_variant",
        "navigation_drift",
        "extraction_field_coverage",
    }
    assert set(EXPERIMENT_KINDS) == expected


# ── ExplorationBudget ───────────────────────────────────────────────


def test_exploration_budget_has_cost_and_time_caps() -> None:
    b = ExplorationBudget()
    assert hasattr(b, "max_cost_usd")
    assert hasattr(b, "max_minutes")
    assert b.max_cost_usd > 0
    assert b.max_minutes > 0


def test_exploration_budget_accepts_overrides() -> None:
    b = ExplorationBudget(max_cost_usd=1.5, max_minutes=5.0)
    assert b.max_cost_usd == 1.5
    assert b.max_minutes == 5.0


# ── VariantOutcome ──────────────────────────────────────────────────


def test_variant_outcome_defaults_to_completed_with_empty_streams() -> None:
    v = VariantOutcome(variant_id="baseline")
    assert v.terminal_status == "completed"
    assert v.step_results == []
    assert v.experiments == []
    assert v.recipe_rejection_histogram == {}
    assert v.url_coverage == []
    assert v.cost_total == 0.0


def test_variant_outcome_round_trips_through_dict() -> None:
    """The whole bundle must survive JSON for offline analysis."""
    v = VariantOutcome(
        variant_id="recipe_v2_tile_loose",
        terminal_status="halted",
        step_results=[
            StepResult(
                step_index=0, intent="Navigate", success=True,
                data="", skip=False, skip_reason=None,
            ),
            StepResult(
                step_index=1, intent="Extract", success=False,
                data="REJECTED_DEALER|extractor marked as dealer|...",
                skip=True, skip_reason="dealer",
            ),
        ],
        experiments=[
            ExperimentEvent(
                kind="recipe_rejection_observed",
                intent="Extract",
                page_url="https://www.example.com/boat/1234",
                outcome={"reason": "dealer"},
                cost=0.04, timestamp=1715000000.0,
            ),
        ],
        per_intent_alternative_count={1: 2},
        recipe_rejection_histogram={"dealer": 1},
        url_coverage=["https://www.example.com/search", "https://www.example.com/boat/1234"],
        cost_total=0.42,
        wall_time_s=187.3,
    )
    payload = v.to_dict()
    rt_payload = json.loads(json.dumps(payload))
    v2 = VariantOutcome.from_dict(rt_payload)

    assert v2.variant_id == "recipe_v2_tile_loose"
    assert v2.terminal_status == "halted"
    assert len(v2.step_results) == 2
    # step_results round-trip as dicts (caller rehydrates via
    # StepResult.from_dict when typed access matters).
    assert v2.step_results[1]["skip_reason"] == "dealer"
    assert len(v2.experiments) == 1
    assert v2.experiments[0].kind == "recipe_rejection_observed"
    assert v2.per_intent_alternative_count == {1: 2}
    assert v2.recipe_rejection_histogram == {"dealer": 1}
    assert v2.url_coverage == [
        "https://www.example.com/search",
        "https://www.example.com/boat/1234",
    ]
    assert v2.cost_total == 0.42
    assert v2.wall_time_s == 187.3


# ── Histogram + coverage derivation ─────────────────────────────────


def test_rejection_histogram_prefers_skip_reason_envelope() -> None:
    """Issue #246's ``skip_reason`` is the recipe-author's canonical
    rejection key — when present, the histogram trusts it over
    parsing the ``data`` prefix."""
    steps = [
        StepResult(step_index=0, intent="x", success=False,
                   data="REJECTED_DEALER|...", skip=True, skip_reason="dealer"),
        StepResult(step_index=1, intent="y", success=False,
                   data="REJECTED_DEALER|...", skip=True, skip_reason="dealer"),
        StepResult(step_index=2, intent="z", success=False,
                   data="REJECTED_INCOMPLETE|missing year|...",
                   skip=False, skip_reason=None),
    ]
    hist = rejection_histogram_from_steps(steps)
    assert hist == {"dealer": 2, "incomplete": 1}


def test_rejection_histogram_falls_back_to_data_prefix() -> None:
    """Legacy recipes without ``rejection_intents`` produce no
    ``skip_reason``. The histogram still classifies by data-prefix
    parsing so the comparison signal stays meaningful for them."""
    steps = [
        StepResult(step_index=0, intent="x", success=False,
                   data="REJECTED_DEALER|reason|...", skip=False, skip_reason=None),
        StepResult(step_index=1, intent="y", success=True,
                   data="URL:https://example.com/boat/1", skip=False),
    ]
    hist = rejection_histogram_from_steps(steps)
    assert hist == {"dealer": 1}


def test_url_coverage_preserves_first_seen_order_dedup() -> None:
    """A refinement agent compares ``len(url_coverage)`` across
    variants to spot a variant that loops on tile #3 (thin
    coverage) versus one that paginates (broad coverage). Order
    must be stable so diffs are deterministic."""
    steps = [
        StepResult(step_index=0, intent="x", success=True,
                   data="URL:https://www.example.com/boat/1"),
        StepResult(step_index=1, intent="y", success=True,
                   data="URL:https://www.example.com/boat/2"),
        StepResult(step_index=2, intent="z", success=True,
                   data="URL:https://www.example.com/boat/1"),  # dup
        StepResult(step_index=3, intent="w", success=False,
                   data="REJECTED_DEALER|dealer|https://www.example.com/boat/3"),
    ]
    coverage = url_coverage_from_steps(steps)
    assert coverage == [
        "https://www.example.com/boat/1",
        "https://www.example.com/boat/2",
        "https://www.example.com/boat/3",
    ]


# ── Runner entrypoint: run_with_exploration ─────────────────────────


def _make_runner_stub() -> MagicMock:
    """A minimal MicroPlanRunner stub that the entrypoint can drive.

    We don't construct a real MicroPlanRunner — its __init__ depends
    on a real env + brain + grounding stack. Instead we wire the
    ``run_with_exploration`` method onto a MagicMock with the
    fields the entrypoint reads (``costs``, ``extractor``,
    ``cancel_event``, ``_run_start``).
    """
    from mantis_agent.gym.micro_runner import MicroPlanRunner

    runner = MagicMock(spec=MicroPlanRunner)
    runner.costs = {"gpu_steps": 0, "gpu_seconds": 0.0, "claude_extract": 0}
    runner.extractor = MagicMock()
    runner.extractor.schema = MagicMock()
    runner.cancel_event = None
    runner._run_start = time.time()
    runner.env = MagicMock()
    # Bind the unbound method so it operates on our mock.
    runner.run_with_exploration = (
        MicroPlanRunner.run_with_exploration.__get__(runner, type(runner))
    )
    return runner


def _plan(intent: str) -> MicroPlan:
    return MicroPlan(steps=[
        MicroIntent(intent=intent, type="navigate",
                    params={"url": "https://www.example.com"}),
    ])


def test_run_with_exploration_returns_one_outcome_per_plan_variant() -> None:
    """Two plan variants in → two VariantOutcomes out. Distinct
    ``variant_id``s so a refinement agent can address them by
    name."""
    runner = _make_runner_stub()

    # Stub the per-variant runner.run so each variant produces
    # different step_results.
    call_idx = {"i": 0}

    def fake_run(plan, resume=False):
        call_idx["i"] += 1
        return [
            StepResult(
                step_index=0, intent=plan.steps[0].intent, success=True,
                data=f"URL:https://www.example.com/variant{call_idx['i']}",
            ),
        ]

    runner.run = fake_run

    outcomes = runner.run_with_exploration(
        plan_variants=[_plan("variant A"), _plan("variant B")],
    )
    assert len(outcomes) == 2
    assert outcomes[0].variant_id != outcomes[1].variant_id
    assert outcomes[0].step_results[0].intent == "variant A"
    assert outcomes[1].step_results[0].intent == "variant B"


def test_run_with_exploration_swaps_recipe_per_variant() -> None:
    """When ``recipe_variants`` is provided, the runtime sets
    ``extractor.schema`` for the duration of the variant and
    restores the original afterwards so legacy callers see no
    state leak."""
    from mantis_agent.extraction import ExtractionSchema

    runner = _make_runner_stub()
    original_schema = ExtractionSchema(entity_name="original")
    runner.extractor.schema = original_schema

    schema_seen: list[Any] = []

    def fake_run(plan, resume=False):
        schema_seen.append(runner.extractor.schema)
        return [StepResult(step_index=0, intent="x", success=True)]

    runner.run = fake_run

    recipe_a = ExtractionSchema(entity_name="recipe_a")
    recipe_b = ExtractionSchema(entity_name="recipe_b")
    runner.run_with_exploration(
        plan_variants=[_plan("p")],
        recipe_variants=[recipe_a, recipe_b],
    )

    # 1 plan × 2 recipes = 2 variant runs; each sees its own recipe.
    assert [s.entity_name for s in schema_seen] == ["recipe_a", "recipe_b"]
    # The extractor's schema is restored to the original after.
    assert runner.extractor.schema is original_schema


def test_run_with_exploration_enforces_budget_max_minutes(monkeypatch) -> None:
    """A variant whose runtime exceeds ``max_minutes`` aborts with
    ``terminal_status='budget_exceeded'``. The runtime checks the
    budget before starting each variant — so the second variant in
    a list is the one that gets stamped budget_exceeded once the
    first variant has already burned the wall-clock cap."""
    runner = _make_runner_stub()

    # Pretend each variant takes 7 minutes (420s); the second
    # variant should be skipped if max_minutes is set to 5.
    start = [1_000.0]
    clock = [start[0]]
    monkeypatch.setattr("mantis_agent.gym.micro_runner.time.time", lambda: clock[0])
    monkeypatch.setattr("mantis_agent.gym.exploration.time.time", lambda: clock[0])

    def fake_run(plan, resume=False):
        clock[0] += 7 * 60  # 7 minutes per variant
        return [StepResult(step_index=0, intent=plan.steps[0].intent, success=True)]

    runner.run = fake_run

    outcomes = runner.run_with_exploration(
        plan_variants=[_plan("first"), _plan("second")],
        budget_per_variant=ExplorationBudget(max_minutes=5.0, max_cost_usd=100.0),
    )
    # First variant: ran to completion (it overshot the budget but
    # the runtime can't time-travel — it only checks before starting
    # the *next* variant).
    assert outcomes[0].terminal_status == "completed"
    # Second variant: budget exhausted before it started.
    assert outcomes[1].terminal_status == "budget_exceeded"
    assert outcomes[1].step_results == []


def test_run_with_exploration_no_recipe_variants_runs_each_plan_once() -> None:
    """Omitting ``recipe_variants`` runs each plan once with the
    extractor's existing schema (no swap)."""
    runner = _make_runner_stub()

    call_count = {"n": 0}

    def fake_run(plan, resume=False):
        call_count["n"] += 1
        return []

    runner.run = fake_run

    outcomes = runner.run_with_exploration(
        plan_variants=[_plan("a"), _plan("b"), _plan("c")],
    )
    assert call_count["n"] == 3
    assert len(outcomes) == 3


def test_run_with_exploration_builds_rejection_histogram_from_step_results() -> None:
    """Histogram derivation runs over step_results AFTER the variant
    completes — so the refinement agent gets it without parsing
    REJECTED_* prefixes itself."""
    runner = _make_runner_stub()

    def fake_run(plan, resume=False):
        return [
            StepResult(step_index=0, intent="x", success=False,
                       data="REJECTED_DEALER|reason|...",
                       skip=True, skip_reason="dealer"),
            StepResult(step_index=1, intent="y", success=False,
                       data="REJECTED_INCOMPLETE|missing year|...",
                       skip=False, skip_reason=None),
        ]

    runner.run = fake_run

    outcomes = runner.run_with_exploration(plan_variants=[_plan("p")])
    assert outcomes[0].recipe_rejection_histogram == {
        "dealer": 1, "incomplete": 1,
    }


def test_run_with_exploration_builds_url_coverage_from_step_results() -> None:
    runner = _make_runner_stub()

    def fake_run(plan, resume=False):
        return [
            StepResult(step_index=0, intent="x", success=True,
                       data="URL:https://www.example.com/a"),
            StepResult(step_index=1, intent="y", success=True,
                       data="URL:https://www.example.com/b"),
        ]

    runner.run = fake_run

    outcomes = runner.run_with_exploration(plan_variants=[_plan("p")])
    assert outcomes[0].url_coverage == [
        "https://www.example.com/a",
        "https://www.example.com/b",
    ]


def test_run_with_exploration_terminal_status_halted_on_required_failure() -> None:
    """When a step fails with ``required=True`` and recovery
    exhausts (signalled by the runner's existing halt path), the
    VariantOutcome carries ``terminal_status='halted'``. v1
    surfaces the runner's existing halt semantics — it doesn't
    add new halt logic."""
    runner = _make_runner_stub()

    def fake_run(plan, resume=False):
        # Simulate the runner halting by setting _final_status.
        runner._final_status = "halted"
        return [
            StepResult(step_index=0, intent="x", success=False,
                       data="some failure"),
        ]

    runner.run = fake_run

    outcomes = runner.run_with_exploration(plan_variants=[_plan("p")])
    assert outcomes[0].terminal_status == "halted"


def test_run_with_exploration_emits_zero_experiments_in_v1() -> None:
    """v1 ships the substrate. Concrete deviation strategies that
    emit ExperimentEvent records land in follow-up PRs. This test
    pins the v1 contract: outcomes carry an experiments list that
    is *queryable* (defaults to empty), not absent."""
    runner = _make_runner_stub()
    runner.run = lambda plan, resume=False: []

    outcomes = runner.run_with_exploration(plan_variants=[_plan("p")])
    assert outcomes[0].experiments == []
    assert isinstance(outcomes[0].experiments, list)
