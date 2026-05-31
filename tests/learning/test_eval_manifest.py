"""Tests for the failure-cluster eval manifest loader + views.

The default manifest (``experiments/learning_allocator/eval/clusters.json``)
is loaded as-is so the shipped data stays valid; synthetic manifests exercise
the validation and the runnable/sealed filters.
"""

from __future__ import annotations

import json

import pytest

from mantis_agent.learning.eval import (
    CLUSTERS,
    EvalManifest,
    EvalTask,
    load_manifest,
)


# ── the shipped manifest ───────────────────────────────────────────────


def test_default_manifest_loads_and_validates() -> None:
    m = load_manifest()
    assert isinstance(m, EvalManifest)
    assert m.env == "mantis_boattrader"
    assert len(m.tasks) >= 1


def test_default_manifest_has_a_runnable_capability_task() -> None:
    m = load_manifest()
    runnable = m.runnable()
    assert runnable, "expected at least one runnable task in the shipped manifest"
    assert all(t.task_id for t in runnable)
    # BT01 is the wired oracle today.
    assert any(t.task_id == "BT01_lead_capture_filtered_search" for t in runnable)


def test_default_manifest_has_a_visible_sealed_pair() -> None:
    m = load_manifest()
    assert m.visible(), "expected a visible split"
    assert m.sealed(), "expected a sealed split (the overfitting check)"
    # visible and sealed should differ by seed for the same task_id.
    vis = {t.task_id: t.seed for t in m.visible() if t.task_id}
    seal = {t.task_id: t.seed for t in m.sealed() if t.task_id}
    shared = set(vis) & set(seal)
    assert shared, "expected a task present in both splits"
    assert all(vis[k] != seal[k] for k in shared), "splits must differ by seed"


def test_all_three_clusters_are_runnable_across_both_splits() -> None:
    m = load_manifest()
    # All three failure clusters now name a real oracle (BT01 capability,
    # BT02 knowledge, BT03 policy), so the allocator's per-cluster-winner
    # claim is fully measurable — not just hoped for.
    assert m.clusters_covered() == {"knowledge", "capability", "policy"}
    assert all(t.task_id for t in m.runnable())
    # Each cluster carries a visible/sealed (seed 42/7) overfitting pair.
    for cluster in ("knowledge", "capability", "policy"):
        splits = {t.split for t in m.by_cluster(cluster) if t.runnable}
        assert splits == {"visible", "sealed"}, cluster


# ── views ──────────────────────────────────────────────────────────────


def _manifest() -> EvalManifest:
    return EvalManifest(
        env="mantis_boattrader",
        tasks=[
            EvalTask(name="a", cluster="capability", split="visible", seed=42,
                     task_id="BT01", status="ready"),
            EvalTask(name="b", cluster="capability", split="sealed", seed=7,
                     task_id="BT01", status="ready"),
            EvalTask(name="c", cluster="knowledge", split="visible", seed=42,
                     status="needs_oracle"),
        ],
    )


def test_by_cluster_and_split_views() -> None:
    m = _manifest()
    assert len(m.by_cluster("capability")) == 2
    assert len(m.by_cluster("knowledge")) == 1
    assert len(m.visible()) == 2
    assert len(m.sealed()) == 1


def test_runnable_excludes_needs_oracle() -> None:
    m = _manifest()
    assert {t.name for t in m.runnable()} == {"a", "b"}
    assert m.clusters_covered() == {"capability"}


# ── validation ─────────────────────────────────────────────────────────


def test_load_rejects_unknown_cluster(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "env": "mantis_boattrader",
        "tasks": [{"name": "x", "cluster": "nonsense", "split": "visible",
                   "seed": 42, "status": "needs_oracle"}],
    }))
    with pytest.raises(ValueError, match="unknown cluster"):
        load_manifest(bad)


def test_load_rejects_ready_without_task_id(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "env": "mantis_boattrader",
        "tasks": [{"name": "x", "cluster": "capability", "split": "visible",
                   "seed": 42, "status": "ready"}],
    }))
    with pytest.raises(ValueError, match="no task_id"):
        load_manifest(bad)


def test_known_clusters_constant() -> None:
    assert CLUSTERS == ("knowledge", "capability", "policy")
