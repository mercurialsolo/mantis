"""Tests for #183 — model-promotion scorecard.

Covers gate-direction semantics, tier transitions, missing-input
handling, and the artefact-extraction helpers that compose the
existing pipeline outputs (eval harness, shadow analytics, labeller).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


_TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(_TRAINING))

from promotion_scorecard import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    TIERS,
    aggregate_label_reasons,
    escalation_rate_from_shadow,
    evaluate,
    task_pass_rate_from_eval_report,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _all_passing_overrides() -> dict[str, float]:
    """Override block that meets every ``first_sft`` threshold cleanly."""
    return {
        "task_pass_rate": 0.6,
        "parser_validity": 0.99,
        "grounding_accuracy": 0.72,
        "forbidden_region_avoidance": 0.96,
        "loop_rate_max": 0.04,
        "gallery_recovery_rate": 0.86,
        "escalation_rate_max": 0.04,
        "done_completeness": 0.86,
        "cost_per_success_usd_max": 0.25,
    }


# ── Gate direction (min vs max) ────────────────────────────────────────


def test_min_direction_gate_passes_at_or_above_threshold():
    overrides = _all_passing_overrides()
    overrides["task_pass_rate"] = DEFAULT_THRESHOLDS["task_pass_rate"]["first_sft"]
    sc = evaluate(metric_overrides=overrides, tier="first_sft")
    gate = next(g for g in sc.gates if g.name == "task_pass_rate")
    assert gate.passed is True


def test_min_direction_gate_fails_below_threshold():
    overrides = _all_passing_overrides()
    overrides["task_pass_rate"] = 0.10  # well below first_sft 0.55
    sc = evaluate(metric_overrides=overrides, tier="first_sft")
    gate = next(g for g in sc.gates if g.name == "task_pass_rate")
    assert gate.passed is False
    assert sc.overall_passed is False
    assert "task_pass_rate" in sc.regressions()


def test_max_direction_gate_passes_at_or_below_threshold():
    overrides = _all_passing_overrides()
    overrides["loop_rate_max"] = DEFAULT_THRESHOLDS["loop_rate_max"]["first_sft"]
    sc = evaluate(metric_overrides=overrides, tier="first_sft")
    gate = next(g for g in sc.gates if g.name == "loop_rate_max")
    assert gate.passed is True


def test_max_direction_gate_fails_above_threshold():
    overrides = _all_passing_overrides()
    overrides["escalation_rate_max"] = 0.5  # way over 0.05
    sc = evaluate(metric_overrides=overrides, tier="first_sft")
    assert sc.overall_passed is False
    assert "escalation_rate_max" in sc.regressions()


# ── Tier transitions ───────────────────────────────────────────────────


def test_tier_base_passes_when_metrics_meet_loose_thresholds():
    overrides = {
        # base threshold for task_pass_rate is 0.40
        "task_pass_rate": 0.45,
        "parser_validity": 0.96,           # base 0.95
        "grounding_accuracy": 0.60,        # base 0.55
        "forbidden_region_avoidance": 0.92, # base 0.90
        "loop_rate_max": 0.08,             # base 0.10 — below cap = pass
        "gallery_recovery_rate": 0.72,     # base 0.70
        "escalation_rate_max": 0.08,       # base 0.10
        "done_completeness": 0.72,         # base 0.70
        "cost_per_success_usd_max": 0.45,  # base 0.50
    }
    sc = evaluate(metric_overrides=overrides, tier="base")
    assert sc.overall_passed is True


def test_same_metrics_fail_at_first_sft_tier():
    """A candidate that passes ``base`` doesn't necessarily pass
    ``first_sft`` — that's the whole point of the tiered gates."""
    overrides = {
        "task_pass_rate": 0.45,            # base 0.40 ✓ first_sft 0.55 ✗
        "parser_validity": 0.96,
        "grounding_accuracy": 0.60,
        "forbidden_region_avoidance": 0.92,
        "loop_rate_max": 0.08,
        "gallery_recovery_rate": 0.72,
        "escalation_rate_max": 0.08,
        "done_completeness": 0.72,
        "cost_per_success_usd_max": 0.45,
    }
    sc = evaluate(metric_overrides=overrides, tier="first_sft")
    assert sc.overall_passed is False
    failing = sc.regressions()
    assert "task_pass_rate" in failing


def test_unknown_tier_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown tier"):
        evaluate(tier="prod_only")


def test_tiers_are_ordered_base_first_sft_future():
    assert TIERS == ("base", "first_sft", "future")


# ── Missing-input behavior ─────────────────────────────────────────────


def test_missing_inputs_skip_gates_without_failing_overall():
    """When no eval report / shadow summary / labelled traces are
    supplied, the scorecard must NOT mass-fail every gate (that
    would block any candidate that hasn't run a full pipeline yet).
    Each missing-input gate gets ``input missing — gate skipped``
    and overall_passed stays ``True``."""
    sc = evaluate(tier="first_sft")  # no inputs at all
    assert sc.overall_passed is True
    skipped = [g for g in sc.gates if "input missing" in g.note]
    # Every gate skipped because nothing provided values.
    assert len(skipped) == len(sc.gates)


def test_partial_inputs_evaluate_what_they_can():
    """An eval report alone is enough to gate task_pass_rate +
    cost_per_success_usd_max; the rest are skipped, not failed."""
    eval_report = {
        "pass_rate": 0.6, "pass_count": 6, "task_count": 10,
        "results": [
            {"task_id": "t1", "passed": True, "outcome": {"cost": 0.10}},
            {"task_id": "t2", "passed": True, "outcome": {"cost": 0.15}},
            {"task_id": "t3", "passed": True, "outcome": {"cost": 0.05}},
            {"task_id": "t4", "passed": True, "outcome": {"cost": 0.05}},
            {"task_id": "t5", "passed": True, "outcome": {"cost": 0.10}},
            {"task_id": "t6", "passed": True, "outcome": {"cost": 0.10}},
            {"task_id": "t7", "passed": False, "outcome": {"cost": 0.20}},
        ],
    }
    sc = evaluate(eval_report=eval_report, tier="first_sft")
    by_name = {g.name: g for g in sc.gates}
    # task_pass_rate gate evaluated.
    assert by_name["task_pass_rate"].value == 0.6
    assert by_name["task_pass_rate"].passed is True
    # cost_per_success_usd_max derived from passing rows.
    assert by_name["cost_per_success_usd_max"].value < 0.20  # well under 0.30 cap
    assert by_name["cost_per_success_usd_max"].passed is True
    # Other gates skipped, not failed.
    assert "input missing" in by_name["parser_validity"].note


# ── Helpers extract from pipeline artefacts ───────────────────────────


def test_task_pass_rate_extractor_returns_none_for_non_eval_payload():
    assert task_pass_rate_from_eval_report({}) is None


def test_task_pass_rate_extractor_reads_pass_rate():
    assert task_pass_rate_from_eval_report({"pass_rate": 0.42}) == 0.42


def test_escalation_rate_from_shadow_reads_candidate_block():
    payload = {
        "variants": {
            "baseline":  {"escalation_rate": 0.10},
            "candidate": {"escalation_rate": 0.04},
        },
    }
    assert escalation_rate_from_shadow(payload) == 0.04


def test_escalation_rate_from_shadow_returns_none_when_variant_missing():
    assert escalation_rate_from_shadow({"variants": {}}) is None


def test_aggregate_label_reasons_walks_dir(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({
        "steps": [
            {"label_reason": "escalation"},
            {"label_reason": "gate_verify_pass"},
            {"label_reason": "escalation"},
        ],
    }))
    (tmp_path / "b.json").write_text(json.dumps({
        "steps": [
            {"label_reason": "failed_step"},
        ],
    }))
    counts, total = aggregate_label_reasons(tmp_path)
    assert counts["escalation"] == 2
    assert counts["gate_verify_pass"] == 1
    assert counts["failed_step"] == 1
    assert total == 4


def test_aggregate_label_reasons_handles_missing_dir(tmp_path):
    counts, total = aggregate_label_reasons(tmp_path / "does-not-exist")
    assert counts == {}
    assert total == 0


def test_aggregate_label_reasons_skips_broken_files(tmp_path):
    (tmp_path / "good.json").write_text(json.dumps({
        "steps": [{"label_reason": "escalation"}],
    }))
    (tmp_path / "broken.json").write_text("{not json")
    counts, total = aggregate_label_reasons(tmp_path)
    assert counts["escalation"] == 1
    assert total == 1


# ── Scorecard payload ──────────────────────────────────────────────────


def test_scorecard_to_dict_round_trip():
    sc = evaluate(metric_overrides=_all_passing_overrides(), tier="first_sft")
    payload = sc.to_dict()
    assert payload["tier"] == "first_sft"
    assert payload["overall_passed"] is True
    assert isinstance(payload["gates"], list)
    assert all("passed" in g for g in payload["gates"])


def test_metric_override_wins_over_artefact_extraction():
    """An explicit ``metric_overrides`` value beats whatever the
    artefact-derived value would have been. Lets operators inject
    ground-truth metrics from outside this script's reach."""
    eval_report = {"pass_rate": 0.10, "pass_count": 0, "results": []}
    sc = evaluate(
        eval_report=eval_report,
        metric_overrides={"task_pass_rate": 0.99},
        tier="first_sft",
    )
    gate = next(g for g in sc.gates if g.name == "task_pass_rate")
    assert gate.value == 0.99
    assert gate.passed is True
