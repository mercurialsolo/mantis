"""Tests for #155 step 5 — ShadowRouter + shadow_analytics aggregation.

Pin the router's determinism + percentage contract, the bucket
distribution, and the analytics aggregation math.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


_TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(_TRAINING))

from mantis_agent.gym.shadow_router import (  # noqa: E402
    VARIANT_BASELINE,
    VARIANT_CANDIDATE,
    ShadowRouter,
)
from shadow_analytics import (  # noqa: E402
    ShadowReport,
    VariantStats,
    aggregate,
)


# ── ShadowRouter ───────────────────────────────────────────────────────


def test_router_disabled_by_default_routes_baseline():
    router = ShadowRouter()
    assert router.route("any-key") == VARIANT_BASELINE


def test_router_at_zero_pct_always_baseline():
    router = ShadowRouter(candidate_pct=0.0)
    for key in ["a", "b", "c", "d", "12345", "tenant-acme"]:
        assert router.route(key) == VARIANT_BASELINE


def test_router_at_hundred_pct_always_candidate():
    router = ShadowRouter(candidate_pct=100.0)
    for key in ["a", "b", "c", "d", "12345", "tenant-acme"]:
        assert router.route(key) == VARIANT_CANDIDATE


def test_router_clamps_negative_pct():
    """Config typo ``candidate_pct=-5`` must NOT flip into 95% candidate
    by integer-wraparound — it must clamp to 0 and stay safe."""
    router = ShadowRouter(candidate_pct=-5.0)
    assert router.candidate_pct == 0.0
    assert router.route("any-key") == VARIANT_BASELINE


def test_router_clamps_pct_above_hundred():
    router = ShadowRouter(candidate_pct=200.0)
    assert router.candidate_pct == 100.0


def test_router_empty_key_routes_baseline():
    """Anonymous / no-key requests stay on the safer variant."""
    router = ShadowRouter(candidate_pct=50.0)
    assert router.route("") == VARIANT_BASELINE


def test_router_is_deterministic_for_same_key():
    """Same key → same variant, every call. Lets a re-run of a flaky
    task hit the same weights."""
    router = ShadowRouter(candidate_pct=25.0)
    a = router.route("tenant-acme:run-42")
    b = router.route("tenant-acme:run-42")
    c = router.route("tenant-acme:run-42")
    assert a == b == c


def test_router_distributes_across_keys_at_target_pct():
    """Across a large key population, the empirical candidate share
    should be within a few percentage points of the configured pct."""
    router = ShadowRouter(candidate_pct=20.0)
    candidate_count = sum(
        1 for i in range(2000)
        if router.route(f"k{i}") == VARIANT_CANDIDATE
    )
    empirical = candidate_count / 2000.0
    # Sha1 with 4-byte truncation is effectively uniform; tolerate ±5pp.
    assert 0.15 <= empirical <= 0.25


def test_router_salt_changes_assignment():
    """Same key + different salt should land different bucket positions
    so independent shadow rollouts don't collide."""
    a = ShadowRouter(candidate_pct=50.0, salt="rollout-A")
    b = ShadowRouter(candidate_pct=50.0, salt="rollout-B")
    # Across 100 keys, the salts should diverge on at least a few.
    differences = sum(
        1 for i in range(100) if a.route(f"k{i}") != b.route(f"k{i}")
    )
    assert differences > 0


# ── shadow_analytics.aggregate ─────────────────────────────────────────


def _trace(
    *,
    variant: str = "",
    steps: list[dict] | None = None,
) -> dict:
    return {
        "schema_version": 2,
        "run_id": "rid",
        "tenant_id": "t1",
        "variant": variant,
        "steps": steps or [],
    }


def _step(label: str, reason: str = "") -> dict:
    return {
        "step_index": 0,
        "label": label,
        "label_reason": reason,
    }


def test_aggregate_groups_runs_by_variant(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(_trace(
        variant="baseline",
        steps=[_step("positive", "gate_verify_pass"), _step("neutral")],
    )))
    (tmp_path / "b.json").write_text(json.dumps(_trace(
        variant="baseline",
        steps=[_step("negative", "failed_step")],
    )))
    (tmp_path / "c.json").write_text(json.dumps(_trace(
        variant="candidate",
        steps=[_step("positive", "gate_verify_pass")],
    )))
    report = aggregate(tmp_path)
    assert report.baseline().run_count == 2
    assert report.candidate().run_count == 1


def test_aggregate_counts_escalations_separately_from_failures(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps(_trace(
        variant="candidate",
        steps=[
            _step("negative", "escalation"),    # bumps escalation + failure
            _step("negative", "failed_step"),    # bumps failure only
            _step("positive", "gate_verify_pass"),
        ],
    )))
    report = aggregate(tmp_path)
    cand = report.candidate()
    assert cand.escalation_count == 1
    assert cand.failure_count == 2  # escalation + failed_step
    assert cand.step_count == 3


def test_aggregate_unassigned_variant_lands_in_bucket(tmp_path):
    """Legacy traces (no variant field) shouldn't be silently dropped —
    they go into ``__unassigned__`` so operators can see the gap."""
    (tmp_path / "legacy.json").write_text(json.dumps(_trace(
        variant="",
        steps=[_step("positive", "gate_verify_pass")],
    )))
    report = aggregate(tmp_path)
    assert "__unassigned__" in report.variants
    assert report.variants["__unassigned__"].run_count == 1


def test_aggregate_skips_broken_json(tmp_path):
    (tmp_path / "good.json").write_text(json.dumps(_trace(
        variant="baseline", steps=[_step("positive")],
    )))
    (tmp_path / "broken.json").write_text("{not json")
    report = aggregate(tmp_path)
    assert report.baseline().run_count == 1
    # Broken file didn't add a phantom variant.
    assert all(v != "broken" for v in report.variants)


# ── ShadowReport.comparison ────────────────────────────────────────────


def test_comparison_returns_empty_when_one_side_missing():
    report = ShadowReport()
    report.variants["baseline"] = VariantStats(
        variant="baseline", run_count=10, step_count=50,
        escalation_count=5, failure_count=10,
    )
    # No candidate
    assert report.comparison() == {}


def test_comparison_reports_lower_escalation_rate_for_candidate():
    report = ShadowReport()
    report.variants["baseline"] = VariantStats(
        variant="baseline", run_count=10, step_count=100,
        escalation_count=10, failure_count=20,  # 10% escalation
    )
    report.variants["candidate"] = VariantStats(
        variant="candidate", run_count=10, step_count=100,
        escalation_count=5, failure_count=10,   # 5% escalation
    )
    cmp_ = report.comparison()
    assert cmp_["escalation_rate_delta"] == pytest.approx(-0.05)
    assert cmp_["failure_rate_delta"] == pytest.approx(-0.10)
    assert cmp_["candidate_escalation_rate_lower"] is True


def test_comparison_flags_regression():
    report = ShadowReport()
    report.variants["baseline"] = VariantStats(
        variant="baseline", run_count=10, step_count=100,
        escalation_count=5, failure_count=10,   # 5% escalation
    )
    report.variants["candidate"] = VariantStats(
        variant="candidate", run_count=10, step_count=100,
        escalation_count=15, failure_count=20,  # 15% escalation — worse
    )
    cmp_ = report.comparison()
    assert cmp_["escalation_rate_delta"] == pytest.approx(0.10)
    assert cmp_["candidate_escalation_rate_lower"] is False


def test_variant_stats_zero_division_safety():
    """A variant with zero steps must report zero rates, not raise."""
    stats = VariantStats(variant="x", run_count=0, step_count=0)
    assert stats.escalation_rate == 0.0
    assert stats.failure_rate == 0.0
