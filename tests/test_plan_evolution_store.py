"""Tests for plan_evolution_store — Phase 2 (#706).

Covers:
- record_rewrite_candidate idempotency
- apply_plan_overlay applies only `promoted` rewrites
- Promotion gate (3 consecutive successes → promoted)
- Demotion gate (2 consecutive failures while promoted → demoted)
- Cold transition (30 days idle → cold)
- Scope isolation (per workflow_id)
- Atomic writes (corrupt store → start fresh)
- finalize_run_outcomes routing
- CLI inspector commands
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mantis_agent.recipes import plan_evolution_store as store
from mantis_agent.recipes.plan_evolution_store import (
    DEMOTION_THRESHOLD,
    PROMOTION_THRESHOLD,
    PlanEvolution,
    StepRewrite,
    apply_plan_overlay,
    finalize_run_outcomes,
    list_plans,
    load_for_inspection,
    record_rewrite_candidate,
    record_run_outcome,
)


@pytest.fixture()
def temp_store(monkeypatch: pytest.MonkeyPatch, tmp_path) -> str:
    """Isolated store dir per test — env-overridable root keeps tests
    from polluting each other or hitting /data."""
    monkeypatch.setenv("MANTIS_PLAN_EVOLUTION_DIR", str(tmp_path))
    return str(tmp_path)


def _step_body(intent: str, url: str) -> dict:
    return {
        "intent": intent,
        "type": "navigate",
        "params": {"url": url},
    }


def _make_plan_with_steps(*urls: str) -> SimpleNamespace:
    """Build a duck-typed MicroPlan stand-in. The store only reads
    `.steps[i].intent / .type / .params` so a SimpleNamespace is
    enough — we don't need to import MicroPlan."""
    steps = []
    for url in urls:
        step = SimpleNamespace(
            intent=f"navigate to {url}",
            type="navigate",
            params={"url": url},
        )
        steps.append(step)
    return SimpleNamespace(steps=steps)


# ── record_rewrite_candidate ──────────────────────────────────────────


def test_record_candidate_creates_new_record(temp_store: str) -> None:
    rewrite = record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf1",
        step_index=0,
        original_step=_step_body("nav", "https://x.com/old"),
        rewritten_step=_step_body("nav", "https://x.com/new"),
        source="pattern_transform", confidence=0.6,
    )
    assert rewrite is not None
    assert rewrite.status == "candidate"
    assert rewrite.successful_runs == 0
    assert rewrite.consecutive_successes == 0
    # File written
    evo = load_for_inspection(plan_hash="plan-1", workflow_id="wf1")
    assert len(evo.rewrites) == 1
    assert evo.rewrites[0].rewritten["params"]["url"] == "https://x.com/new"


def test_record_candidate_idempotent_on_same_url(temp_store: str) -> None:
    """Recording the same rewrite twice updates the existing record."""
    for _ in range(3):
        record_rewrite_candidate(
            plan_hash="plan-1", workflow_id="wf1",
            step_index=0,
            original_step=_step_body("nav", "https://x.com/old"),
            rewritten_step=_step_body("nav", "https://x.com/new"),
            source="pattern_transform", confidence=0.6,
        )
    evo = load_for_inspection(plan_hash="plan-1", workflow_id="wf1")
    assert len(evo.rewrites) == 1


def test_record_candidate_keeps_max_confidence(temp_store: str) -> None:
    record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf1",
        step_index=0,
        original_step=_step_body("nav", "https://x.com/old"),
        rewritten_step=_step_body("nav", "https://x.com/new"),
        source="pattern_transform", confidence=0.55,
    )
    record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf1",
        step_index=0,
        original_step=_step_body("nav", "https://x.com/old"),
        rewritten_step=_step_body("nav", "https://x.com/new"),
        source="page_links", confidence=0.75,
    )
    evo = load_for_inspection(plan_hash="plan-1", workflow_id="wf1")
    assert evo.rewrites[0].confidence == 0.75
    assert evo.rewrites[0].source == "page_links"


def test_record_candidate_no_op_when_missing_ids(temp_store: str) -> None:
    """Empty plan_hash or workflow_id → no-op, returns None."""
    out = record_rewrite_candidate(
        plan_hash="", workflow_id="wf1",
        step_index=0,
        original_step={}, rewritten_step={},
        source="manual", confidence=0.5,
    )
    assert out is None
    assert list_plans(workflow_id="wf1") == []


# ── promotion gate ────────────────────────────────────────────────────


def test_promotion_after_3_successes(temp_store: str) -> None:
    rewrite = record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf1",
        step_index=0,
        original_step=_step_body("nav", "https://x.com/old"),
        rewritten_step=_step_body("nav", "https://x.com/new"),
        source="pattern_transform", confidence=0.6,
    )
    assert rewrite.status == "candidate"

    for i in range(PROMOTION_THRESHOLD):
        record_run_outcome(
            plan_hash="plan-1", workflow_id="wf1",
            applied_rewrites=[rewrite],
            outcome="success",
        )
        evo = load_for_inspection(plan_hash="plan-1", workflow_id="wf1")
        stored = evo.rewrites[0]
        if i < PROMOTION_THRESHOLD - 1:
            assert stored.status == "candidate", f"premature promotion at i={i}"
        else:
            assert stored.status == "promoted", "no promotion after threshold"
    assert stored.consecutive_successes == PROMOTION_THRESHOLD
    assert stored.successful_runs == PROMOTION_THRESHOLD


def test_failure_resets_consecutive_successes(temp_store: str) -> None:
    """A failure between successes resets the counter — promotion needs
    *consecutive* wins."""
    rewrite = record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf1",
        step_index=0,
        original_step=_step_body("nav", "https://x.com/old"),
        rewritten_step=_step_body("nav", "https://x.com/new"),
        source="pattern_transform", confidence=0.6,
    )
    record_run_outcome(plan_hash="plan-1", workflow_id="wf1",
                       applied_rewrites=[rewrite], outcome="success")
    record_run_outcome(plan_hash="plan-1", workflow_id="wf1",
                       applied_rewrites=[rewrite], outcome="success")
    # Fail
    record_run_outcome(plan_hash="plan-1", workflow_id="wf1",
                       applied_rewrites=[rewrite], outcome="failure")
    evo = load_for_inspection(plan_hash="plan-1", workflow_id="wf1")
    assert evo.rewrites[0].consecutive_successes == 0
    assert evo.rewrites[0].status == "candidate"  # still candidate

    # Two more successes — still candidate (need 3 fresh)
    record_run_outcome(plan_hash="plan-1", workflow_id="wf1",
                       applied_rewrites=[rewrite], outcome="success")
    record_run_outcome(plan_hash="plan-1", workflow_id="wf1",
                       applied_rewrites=[rewrite], outcome="success")
    evo = load_for_inspection(plan_hash="plan-1", workflow_id="wf1")
    assert evo.rewrites[0].status == "candidate"

    # Third in the new streak → promoted
    record_run_outcome(plan_hash="plan-1", workflow_id="wf1",
                       applied_rewrites=[rewrite], outcome="success")
    evo = load_for_inspection(plan_hash="plan-1", workflow_id="wf1")
    assert evo.rewrites[0].status == "promoted"


# ── demotion gate ─────────────────────────────────────────────────────


def test_demotion_after_2_failures_while_promoted(temp_store: str) -> None:
    rewrite = record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf1",
        step_index=0,
        original_step=_step_body("nav", "https://x.com/old"),
        rewritten_step=_step_body("nav", "https://x.com/new"),
        source="pattern_transform", confidence=0.6,
    )
    # Promote
    for _ in range(PROMOTION_THRESHOLD):
        record_run_outcome(plan_hash="plan-1", workflow_id="wf1",
                           applied_rewrites=[rewrite], outcome="success")
    evo = load_for_inspection(plan_hash="plan-1", workflow_id="wf1")
    assert evo.rewrites[0].status == "promoted"

    # 2 consecutive failures → demoted
    for _ in range(DEMOTION_THRESHOLD):
        record_run_outcome(plan_hash="plan-1", workflow_id="wf1",
                           applied_rewrites=[rewrite], outcome="failure")
    evo = load_for_inspection(plan_hash="plan-1", workflow_id="wf1")
    assert evo.rewrites[0].status == "demoted"
    assert evo.rewrites[0].demotion_reason == f"{DEMOTION_THRESHOLD}_consecutive_failures"


def test_promoted_success_resets_failure_streak(temp_store: str) -> None:
    rewrite = record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf1",
        step_index=0,
        original_step=_step_body("nav", "https://x.com/old"),
        rewritten_step=_step_body("nav", "https://x.com/new"),
        source="pattern_transform", confidence=0.6,
    )
    for _ in range(PROMOTION_THRESHOLD):
        record_run_outcome(plan_hash="plan-1", workflow_id="wf1",
                           applied_rewrites=[rewrite], outcome="success")

    record_run_outcome(plan_hash="plan-1", workflow_id="wf1",
                       applied_rewrites=[rewrite], outcome="failure")
    record_run_outcome(plan_hash="plan-1", workflow_id="wf1",
                       applied_rewrites=[rewrite], outcome="success")
    record_run_outcome(plan_hash="plan-1", workflow_id="wf1",
                       applied_rewrites=[rewrite], outcome="failure")
    evo = load_for_inspection(plan_hash="plan-1", workflow_id="wf1")
    # Should still be promoted — failure streak was broken
    assert evo.rewrites[0].status == "promoted"


# ── apply_plan_overlay ────────────────────────────────────────────────


def test_apply_overlay_applies_promoted_only(temp_store: str) -> None:
    # Candidate (left as candidate — never promoted)
    record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf1",
        step_index=0,
        original_step=_step_body("nav", "https://x.com/a"),
        rewritten_step=_step_body("nav", "https://x.com/A"),
        source="pattern_transform", confidence=0.6,
    )
    # Promoted (3 successes)
    r2 = record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf1",
        step_index=1,
        original_step=_step_body("nav", "https://x.com/b"),
        rewritten_step=_step_body("nav", "https://x.com/B"),
        source="pattern_transform", confidence=0.6,
    )
    for _ in range(PROMOTION_THRESHOLD):
        record_run_outcome(plan_hash="plan-1", workflow_id="wf1",
                           applied_rewrites=[r2], outcome="success")

    plan = _make_plan_with_steps("https://x.com/a", "https://x.com/b")
    new_plan, applied = apply_plan_overlay(
        plan, plan_hash="plan-1", workflow_id="wf1",
    )

    # Only the promoted rewrite (step 1) was applied
    assert new_plan.steps[0].params["url"] == "https://x.com/a"  # unchanged
    assert new_plan.steps[1].params["url"] == "https://x.com/B"  # rewritten
    assert len(applied) == 1
    assert applied[0].step_index == 1


def test_apply_overlay_no_op_without_workflow_or_hash(temp_store: str) -> None:
    plan = _make_plan_with_steps("https://x.com/a")
    out, applied = apply_plan_overlay(plan, plan_hash="", workflow_id="wf1")
    assert out is plan
    assert applied == []


def test_apply_overlay_no_op_when_no_store_file(temp_store: str) -> None:
    plan = _make_plan_with_steps("https://x.com/a")
    out, applied = apply_plan_overlay(
        plan, plan_hash="not-there", workflow_id="wf-missing",
    )
    assert applied == []
    assert out.steps[0].params["url"] == "https://x.com/a"


def test_apply_overlay_include_candidates_applies_candidate(temp_store: str) -> None:
    """#894 exploration: with include_candidates, a not-yet-promoted
    candidate is ALSO applied (so it can accumulate wins toward promotion);
    the default path still leaves it untouched."""
    record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf1", step_index=0,
        original_step=_step_body("nav", "https://x.com/a"),
        rewritten_step=_step_body("nav", "https://x.com/A"),
        source="pattern_transform", confidence=0.6,
    )
    # Default (promoted-only): candidate untouched.
    plan = _make_plan_with_steps("https://x.com/a")
    _, applied_default = apply_plan_overlay(
        plan, plan_hash="plan-1", workflow_id="wf1",
    )
    assert applied_default == []
    assert plan.steps[0].params["url"] == "https://x.com/a"

    # Exploration: candidate applied.
    plan2 = _make_plan_with_steps("https://x.com/a")
    new_plan, applied = apply_plan_overlay(
        plan2, plan_hash="plan-1", workflow_id="wf1", include_candidates=True,
    )
    assert len(applied) == 1
    assert applied[0].status == "candidate"
    assert new_plan.steps[0].params["url"] == "https://x.com/A"


def test_apply_overlay_handles_out_of_range_step_index(temp_store: str) -> None:
    """A stored rewrite for step 10 on a 3-step plan should be skipped
    silently (plan structure may have changed since the rewrite was
    learned)."""
    rewrite = record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf1",
        step_index=10,
        original_step=_step_body("nav", "https://x.com/a"),
        rewritten_step=_step_body("nav", "https://x.com/A"),
        source="pattern_transform", confidence=0.6,
    )
    for _ in range(PROMOTION_THRESHOLD):
        record_run_outcome(plan_hash="plan-1", workflow_id="wf1",
                           applied_rewrites=[rewrite], outcome="success")
    plan = _make_plan_with_steps("https://x.com/a", "https://x.com/b")
    _, applied = apply_plan_overlay(plan, plan_hash="plan-1", workflow_id="wf1")
    assert applied == []


# ── cold transition ──────────────────────────────────────────────────


def test_promoted_rewrite_goes_cold_after_30_days(temp_store: str) -> None:
    """Manually age a rewrite's last_seen past 30 days → next overlay
    application demotes to cold + skips."""
    rewrite = record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf1",
        step_index=0,
        original_step=_step_body("nav", "https://x.com/a"),
        rewritten_step=_step_body("nav", "https://x.com/A"),
        source="pattern_transform", confidence=0.6,
    )
    for _ in range(PROMOTION_THRESHOLD):
        record_run_outcome(plan_hash="plan-1", workflow_id="wf1",
                           applied_rewrites=[rewrite], outcome="success")

    # Age the last_seen timestamp manually by reading + writing.
    evo = load_for_inspection(plan_hash="plan-1", workflow_id="wf1")
    old_iso = (
        datetime.now(timezone.utc) - timedelta(days=45)
    ).isoformat(timespec="seconds")
    evo.rewrites[0].last_seen = old_iso
    store._save(evo)

    plan = _make_plan_with_steps("https://x.com/a")
    _, applied = apply_plan_overlay(plan, plan_hash="plan-1", workflow_id="wf1")
    assert applied == []  # cold rewrites aren't applied

    evo_after = load_for_inspection(plan_hash="plan-1", workflow_id="wf1")
    assert evo_after.rewrites[0].status == "cold"
    assert evo_after.rewrites[0].demotion_reason == "idle_30d"


# ── scope isolation ──────────────────────────────────────────────────


def test_per_workflow_isolation(temp_store: str) -> None:
    record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf-A",
        step_index=0,
        original_step=_step_body("nav", "https://x.com/a"),
        rewritten_step=_step_body("nav", "https://x.com/A"),
        source="pattern_transform", confidence=0.6,
    )
    record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf-B",
        step_index=0,
        original_step=_step_body("nav", "https://x.com/a"),
        rewritten_step=_step_body("nav", "https://x.com/B-different"),
        source="pattern_transform", confidence=0.6,
    )
    evo_a = load_for_inspection(plan_hash="plan-1", workflow_id="wf-A")
    evo_b = load_for_inspection(plan_hash="plan-1", workflow_id="wf-B")
    assert evo_a.rewrites[0].rewritten["params"]["url"] == "https://x.com/A"
    assert evo_b.rewrites[0].rewritten["params"]["url"] == "https://x.com/B-different"


# ── corrupt-store recovery ────────────────────────────────────────────


def test_corrupt_store_falls_back_to_empty(temp_store: str) -> None:
    """A malformed JSON file should be treated as empty rather than
    raising — better to lose history than crash the run."""
    path = store._file_path("workflow", "wf1", "plan-1")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("not json at all")
    evo = load_for_inspection(plan_hash="plan-1", workflow_id="wf1")
    assert evo.rewrites == []


# ── finalize_run_outcomes ────────────────────────────────────────────


def test_finalize_routes_success_and_failure_separately(
    temp_store: str,
) -> None:
    """Mixed step outcomes split into success / failure batches."""
    r1 = record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf1", step_index=0,
        original_step=_step_body("nav", "https://x.com/a"),
        rewritten_step=_step_body("nav", "https://x.com/A"),
        source="pattern_transform", confidence=0.6,
    )
    r2 = record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf1", step_index=1,
        original_step=_step_body("nav", "https://x.com/b"),
        rewritten_step=_step_body("nav", "https://x.com/B"),
        source="pattern_transform", confidence=0.6,
    )
    step_results = [
        SimpleNamespace(success=True),
        SimpleNamespace(success=False),
    ]
    finalize_run_outcomes(
        plan_hash="plan-1", workflow_id="wf1",
        applied_rewrites=[r1, r2], step_results=step_results,
    )
    evo = load_for_inspection(plan_hash="plan-1", workflow_id="wf1")
    by_idx = {r.step_index: r for r in evo.rewrites}
    assert by_idx[0].consecutive_successes == 1
    assert by_idx[0].consecutive_failures == 0
    assert by_idx[1].consecutive_successes == 0
    assert by_idx[1].consecutive_failures == 1


def test_finalize_skips_out_of_range_step(temp_store: str) -> None:
    r = record_rewrite_candidate(
        plan_hash="plan-1", workflow_id="wf1", step_index=99,
        original_step=_step_body("nav", "https://x.com/x"),
        rewritten_step=_step_body("nav", "https://x.com/X"),
        source="pattern_transform", confidence=0.6,
    )
    finalize_run_outcomes(
        plan_hash="plan-1", workflow_id="wf1",
        applied_rewrites=[r], step_results=[SimpleNamespace(success=True)],
    )
    evo = load_for_inspection(plan_hash="plan-1", workflow_id="wf1")
    assert evo.rewrites[0].consecutive_successes == 0


# ── list_plans ───────────────────────────────────────────────────────


def test_list_plans_returns_stored_hashes(temp_store: str) -> None:
    record_rewrite_candidate(
        plan_hash="plan-AAA", workflow_id="wf1", step_index=0,
        original_step={}, rewritten_step={},
        source="manual", confidence=0.5,
    )
    record_rewrite_candidate(
        plan_hash="plan-BBB", workflow_id="wf1", step_index=0,
        original_step={}, rewritten_step={},
        source="manual", confidence=0.5,
    )
    out = list_plans(workflow_id="wf1")
    assert "plan-AAA" in out
    assert "plan-BBB" in out


def test_list_plans_empty_for_missing_workflow(temp_store: str) -> None:
    assert list_plans(workflow_id="never-existed") == []


# ── round-trip ────────────────────────────────────────────────────────


def test_step_rewrite_dataclass_round_trip() -> None:
    r = StepRewrite(
        step_index=2,
        original={"intent": "x", "params": {"url": "u1"}},
        rewritten={"intent": "y", "params": {"url": "u2"}},
        source="page_links", confidence=0.7,
    )
    d = r.to_dict()
    r2 = StepRewrite.from_dict(d)
    assert r2.step_index == r.step_index
    assert r2.original == r.original
    assert r2.rewritten == r.rewritten
    assert r2.source == r.source
    assert r2.confidence == r.confidence


def test_plan_evolution_round_trip() -> None:
    evo = PlanEvolution(plan_hash="p", scope="workflow", scope_id="w")
    evo.rewrites.append(StepRewrite(
        step_index=0, original={}, rewritten={},
        source="manual", confidence=0.5,
    ))
    d = evo.to_dict()
    evo2 = PlanEvolution.from_dict(d)
    assert evo2.plan_hash == "p"
    assert len(evo2.rewrites) == 1


# ── _urls_match helper ───────────────────────────────────────────────


def test_urls_match_ignores_query_params() -> None:
    """Query-string diffs (tracker params) shouldn't break recognition."""
    assert store._urls_match(
        "https://x.com/path?utm=abc", "https://x.com/path?utm=xyz",
    )


def test_urls_match_distinguishes_path() -> None:
    assert not store._urls_match(
        "https://x.com/path1", "https://x.com/path2",
    )


def test_urls_match_handles_empty() -> None:
    assert not store._urls_match("", "https://x.com/")
    assert not store._urls_match("https://x.com/", "")
