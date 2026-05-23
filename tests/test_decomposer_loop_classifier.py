"""Tests for the plan-level loop classifier (#614).

The classifier walks each ``loop`` step in a ``MicroPlan`` back to its
``loop_target`` and tags the body with a parallelizability shape that
the fan-out runner (#616, #617) consults to decide whether the body is
safe to partition across workers.

These tests pin three things:

  1. The canonical extraction-loop shape (click → extract_url → scroll →
     extract_data → navigate_back → loop) is tagged
     ``parallelizable_url_collect``.
  2. A pagination-only loop (paginate → loop) is tagged
     ``parallelizable_pagination``.
  3. Form-driven / sequential bodies are tagged ``sequential`` so the
     fan-out path is gated off automatically.

The classifier is pure analysis — these tests assert ``plan.steps`` is
unchanged after the pass.
"""

from __future__ import annotations

import copy

from mantis_agent.plan_decomposer import (
    LoopGroup,
    MicroIntent,
    MicroPlan,
    PlanDecomposer,
)


def _extraction_loop_plan() -> MicroPlan:
    """The canonical url-collect extraction-loop shape from the
    decomposer prompt (``plan_decomposer.py:588-592``)."""
    plan = MicroPlan(source_plan="", domain="example.com")
    plan.steps = [
        MicroIntent(intent="Navigate", type="navigate", section="setup"),
        MicroIntent(
            intent="Click the next listing title",
            type="click",
            section="extraction",
        ),
        MicroIntent(intent="Read URL", type="extract_url", section="extraction"),
        MicroIntent(intent="Scroll", type="scroll", section="extraction"),
        MicroIntent(
            intent="Extract fields", type="extract_data", section="extraction"
        ),
        MicroIntent(intent="Go back", type="navigate_back", section="extraction"),
        MicroIntent(
            intent="Loop",
            type="loop",
            section="extraction",
            loop_target=1,
            loop_count=20,
        ),
    ]
    return plan


def _pagination_loop_plan() -> MicroPlan:
    plan = MicroPlan(source_plan="", domain="example.com")
    plan.steps = [
        MicroIntent(intent="Navigate", type="navigate", section="setup"),
        MicroIntent(intent="Paginate", type="paginate", section="pagination"),
        MicroIntent(
            intent="Loop",
            type="loop",
            section="pagination",
            loop_target=1,
            loop_count=10,
        ),
    ]
    return plan


def _form_plan() -> MicroPlan:
    plan = MicroPlan(source_plan="", domain="example.com")
    plan.steps = [
        MicroIntent(intent="Navigate", type="navigate", section="setup"),
        MicroIntent(
            intent="Fill email",
            type="fill_field",
            section="setup",
            params={"label": "Email", "value": "x@y.com"},
        ),
        MicroIntent(
            intent="Submit",
            type="submit",
            section="setup",
            params={"label": "Sign in"},
        ),
    ]
    return plan


# ── parallelizable_url_collect ─────────────────────────────────────────


def test_url_collect_body_tagged_parallelizable() -> None:
    plan = _extraction_loop_plan()
    PlanDecomposer._classify_loop_groups(plan)
    assert len(plan.loop_groups) == 1
    g = plan.loop_groups[0]
    assert g.shape == "parallelizable_url_collect"
    # body covers steps [click .. navigate_back] — the loop step itself
    # is excluded from the half-open range.
    assert g.body_range == (1, 6)
    assert g.loop_step_idx == 6


def test_classifier_is_pure_analysis() -> None:
    plan = _extraction_loop_plan()
    before = copy.deepcopy(plan.steps)
    PlanDecomposer._classify_loop_groups(plan)
    assert plan.steps == before


# ── parallelizable_pagination ──────────────────────────────────────────


def test_pagination_body_tagged_parallelizable() -> None:
    plan = _pagination_loop_plan()
    PlanDecomposer._classify_loop_groups(plan)
    assert len(plan.loop_groups) == 1
    assert plan.loop_groups[0].shape == "parallelizable_pagination"


# ── sequential / no-fanout ─────────────────────────────────────────────


def test_form_plan_emits_no_loop_groups() -> None:
    plan = _form_plan()
    PlanDecomposer._classify_loop_groups(plan)
    assert plan.loop_groups == []


def test_body_without_extract_url_is_sequential() -> None:
    """An extraction-shaped loop whose body skips ``extract_url`` (e.g.
    a workflow that only acts on each card without reading data) is NOT
    safe to partition — there's no URL primary key to map workers to."""
    plan = MicroPlan(source_plan="", domain="example.com")
    plan.steps = [
        MicroIntent(intent="Navigate", type="navigate", section="setup"),
        MicroIntent(
            intent="Click card", type="click", section="extraction"
        ),
        MicroIntent(intent="Scroll", type="scroll", section="extraction"),
        MicroIntent(intent="Back", type="navigate_back", section="extraction"),
        MicroIntent(
            intent="Loop",
            type="loop",
            section="extraction",
            loop_target=1,
            loop_count=5,
        ),
    ]
    PlanDecomposer._classify_loop_groups(plan)
    assert plan.loop_groups[0].shape == "sequential"


def test_body_without_extraction_section_is_sequential() -> None:
    """A loop whose entry click is in setup (e.g. a multi-step wizard
    that loops on a setup button until a state is reached) isn't
    a parallelizable extraction shape."""
    plan = MicroPlan(source_plan="", domain="example.com")
    plan.steps = [
        MicroIntent(
            intent="Click Next",
            type="click",
            section="setup",  # NOT extraction
        ),
        MicroIntent(intent="Read URL", type="extract_url", section="setup"),
        MicroIntent(
            intent="Loop",
            type="loop",
            section="setup",
            loop_target=0,
            loop_count=3,
        ),
    ]
    PlanDecomposer._classify_loop_groups(plan)
    assert plan.loop_groups[0].shape == "sequential"


# ── target normalization edge cases ───────────────────────────────────


def test_loop_target_negative_treated_as_self_index() -> None:
    """Mirrors :meth:`_handle_loop_step` semantics: when target < 0 the
    runner treats target as the loop step's own index (zero-length body
    after _fix_loop_targets misses a fix). Classifier should not crash
    and should emit an empty body tagged sequential."""
    plan = MicroPlan(source_plan="", domain="example.com")
    plan.steps = [
        MicroIntent(intent="Navigate", type="navigate", section="setup"),
        MicroIntent(
            intent="Loop",
            type="loop",
            section="extraction",
            loop_target=-1,  # unset / not normalized
            loop_count=5,
        ),
    ]
    PlanDecomposer._classify_loop_groups(plan)
    assert len(plan.loop_groups) == 1
    g = plan.loop_groups[0]
    assert g.shape == "sequential"
    assert g.body_range == (1, 1)


def test_multiple_loops_each_classified_independently() -> None:
    """A plan can have a pagination loop wrapping an extraction loop
    (the canonical marketplace_listings shape). Each gets its own
    ``LoopGroup`` entry with the right shape."""
    plan = MicroPlan(source_plan="", domain="example.com")
    plan.steps = [
        MicroIntent(intent="Navigate", type="navigate", section="setup"),
        MicroIntent(intent="Click", type="click", section="extraction"),
        MicroIntent(intent="Read URL", type="extract_url", section="extraction"),
        MicroIntent(intent="Extract", type="extract_data", section="extraction"),
        MicroIntent(intent="Back", type="navigate_back", section="extraction"),
        MicroIntent(
            intent="Loop body",
            type="loop",
            section="extraction",
            loop_target=1,
            loop_count=20,
        ),
        MicroIntent(intent="Paginate", type="paginate", section="pagination"),
        MicroIntent(
            intent="Loop pages",
            type="loop",
            section="pagination",
            loop_target=6,
            loop_count=10,
        ),
    ]
    PlanDecomposer._classify_loop_groups(plan)
    assert len(plan.loop_groups) == 2
    shapes = [g.shape for g in plan.loop_groups]
    assert shapes == ["parallelizable_url_collect", "parallelizable_pagination"]


# ── MicroPlan round-trip ───────────────────────────────────────────────


def test_loop_groups_round_trip_through_to_from_dict() -> None:
    plan = _extraction_loop_plan()
    PlanDecomposer._classify_loop_groups(plan)
    restored = MicroPlan.from_dict(plan.to_dict())
    assert restored.loop_groups == plan.loop_groups


def test_loop_groups_recomputed_on_legacy_payload() -> None:
    """A payload missing ``loop_groups`` (cached before #614 landed)
    triggers re-classification on load so the field is always populated
    when the runtime reads it."""
    plan = _extraction_loop_plan()
    payload = plan.to_dict()
    payload.pop("loop_groups")
    restored = MicroPlan.from_dict(payload)
    assert len(restored.loop_groups) == 1
    assert restored.loop_groups[0].shape == "parallelizable_url_collect"


# ── LoopGroup dataclass identity ───────────────────────────────────────


def test_loop_group_is_value_equal() -> None:
    a = LoopGroup(loop_step_idx=6, body_range=(1, 6), shape="parallelizable_url_collect")
    b = LoopGroup(loop_step_idx=6, body_range=(1, 6), shape="parallelizable_url_collect")
    assert a == b
