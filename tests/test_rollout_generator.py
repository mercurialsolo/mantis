"""Rollout generator primitives (#894) — types + seed-sweep + failure-bias."""

from __future__ import annotations

import pytest

from mantis_agent.learning.rollout_generator import (
    FailureBiasedGenerator,
    RolloutGenerator,
    SeedSweepGenerator,
    TaskTemplate,
)


def _tmpl(tid: str, cluster: str = "c1") -> TaskTemplate:
    return TaskTemplate(
        template_id=tid, cluster=cluster, oracle_task_id=f"oracle_{tid}",
        plan_text=f"do {tid}",
    )


# ── TaskTemplate ──────────────────────────────────────────────────


def test_template_requires_plan_text_or_steps():
    with pytest.raises(ValueError):
        TaskTemplate(template_id="t", cluster="c", oracle_task_id="o")


def test_template_accepts_steps():
    t = TaskTemplate(
        template_id="t", cluster="c", oracle_task_id="o",
        plan_steps=[{"type": "navigate", "intent": "go"}],
    )
    assert t.plan_steps


# ── SeedSweepGenerator ────────────────────────────────────────────


def test_seed_sweep_cartesian_count():
    gen = SeedSweepGenerator(
        templates=[_tmpl("a"), _tmpl("b")], seeds=[1, 2, 3],
    )
    specs = list(gen.generate())
    assert len(specs) == 2 * 3  # templates × seeds × 1 sibling
    assert isinstance(gen, RolloutGenerator)  # satisfies the Protocol


def test_seed_sweep_siblings_share_group_id():
    gen = SeedSweepGenerator(
        templates=[_tmpl("a")], seeds=[7], siblings_per_instance=3,
    )
    specs = list(gen.generate())
    assert len(specs) == 3
    assert {s.group_id for s in specs} == {"a__seed7"}
    assert [s.sibling_index for s in specs] == [0, 1, 2]
    assert len({s.spec_id for s in specs}) == 3  # unique spec ids


def test_seed_sweep_rejects_zero_siblings():
    with pytest.raises(ValueError):
        SeedSweepGenerator(templates=[_tmpl("a")], seeds=[1], siblings_per_instance=0)


def test_seed_sweep_deterministic():
    mk = lambda: SeedSweepGenerator(templates=[_tmpl("a")], seeds=[1, 2])
    assert [s.spec_id for s in mk().generate()] == [s.spec_id for s in mk().generate()]


# ── FailureBiasedGenerator ────────────────────────────────────────


def test_failure_bias_allocates_proportional_to_failures():
    gen = FailureBiasedGenerator(
        templates_by_cluster={"hot": [_tmpl("h", "hot")], "cold": [_tmpl("c", "cold")]},
        cluster_failure_counts={"hot": 75, "cold": 25},
        seeds=[1, 2, 3],
        total_instances=100,
    )
    specs = list(gen.generate())
    assert len(specs) == 100
    by_cluster = {}
    for s in specs:
        by_cluster[s.template.cluster] = by_cluster.get(s.template.cluster, 0) + 1
    assert by_cluster["hot"] == 75 and by_cluster["cold"] == 25


def test_failure_bias_budget_sums_exactly_with_remainder():
    # 10 instances over 3 equal clusters → 4/3/3 (largest-remainder)
    gen = FailureBiasedGenerator(
        templates_by_cluster={c: [_tmpl(c, c)] for c in ("a", "b", "c")},
        cluster_failure_counts={"a": 1, "b": 1, "c": 1},
        seeds=[1], total_instances=10,
    )
    specs = list(gen.generate())
    assert len(specs) == 10


def test_failure_bias_skips_clusters_without_templates():
    gen = FailureBiasedGenerator(
        templates_by_cluster={"have": [_tmpl("h", "have")]},
        cluster_failure_counts={"have": 50, "missing": 50},  # missing has no template
        seeds=[1, 2], total_instances=20,
    )
    specs = list(gen.generate())
    assert len(specs) == 20  # full budget reallocated to the live cluster
    assert all(s.template.cluster == "have" for s in specs)


def test_failure_bias_empty_when_no_failures_or_budget():
    gen = FailureBiasedGenerator(
        templates_by_cluster={"a": [_tmpl("a", "a")]},
        cluster_failure_counts={"a": 0}, seeds=[1], total_instances=10,
    )
    assert list(gen.generate()) == []


def test_failure_bias_deterministic_with_rng_seed():
    mk = lambda: FailureBiasedGenerator(
        templates_by_cluster={"a": [_tmpl("a", "a")]},
        cluster_failure_counts={"a": 5}, seeds=[1, 2, 3, 4],
        total_instances=8, rng_seed=42,
    )
    assert [s.env_seed for s in mk().generate()] == [s.env_seed for s in mk().generate()]
