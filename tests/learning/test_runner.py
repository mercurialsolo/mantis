"""Tests for the Phase-2 experiment runner.

All offline: the loop is driven by an injected ``run_fn`` (canned
``run_result`` dicts) + :func:`offline_reward_fn` (ground truth baked into
the result), so nothing here boots an env or spends a cent. They cover the
no-op ``frozen`` rung, the offline reward channel, policy assembly, the
Table 1 / Fig 1 derivations + writers, and an end-to-end demo asserting the
allocator learns the per-cluster winner.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mantis_agent.learning.eval import EvalTask
from mantis_agent.learning.substrates.base import Durability, SubstrateContext

from experiments.learning_allocator.runner import (
    FROZEN,
    POLICY_SUBSTRATES,
    S0,
    S1,
    NoOpSubstrate,
    _OUTCOME_COLS,
    _outcome_row,
    build_policy_allocator,
    build_table1,
    fig1_series,
    offline_reward_fn,
    run_experiment,
    run_offline_demo,
    write_results,
)

# ── tasks ────────────────────────────────────────────────────────────────


def _ctx() -> SubstrateContext:
    return SubstrateContext(task_id="T", cluster="knowledge", budget_remaining=10.0)


def _tasks() -> list[EvalTask]:
    """Two clusters × visible/sealed — enough to exercise the split logic."""
    return [
        EvalTask(name="k_vis", cluster="knowledge", split="visible", seed=42,
                 task_id="K", status="ready"),
        EvalTask(name="k_seal", cluster="knowledge", split="sealed", seed=7,
                 task_id="K", status="ready"),
        EvalTask(name="c_vis", cluster="capability", split="visible", seed=42,
                 task_id="C", status="ready"),
        EvalTask(name="c_seal", cluster="capability", split="sealed", seed=7,
                 task_id="C", status="ready"),
    ]


# ── NoOpSubstrate (frozen) ───────────────────────────────────────────────


def test_noop_substrate_applies_nothing_at_zero_cost() -> None:
    sub = NoOpSubstrate()
    assert sub.name == FROZEN
    assert sub.cost_estimate(_ctx()) == 0.0
    res = sub.apply(_ctx())
    assert res.applied is False
    assert res.dollars_spent == 0.0
    assert res.durability == Durability.EPHEMERAL
    # observe is a no-op and must not raise
    assert sub.observe(_ctx(), res, 1.0) is None


# ── offline reward channel ───────────────────────────────────────────────


def test_offline_reward_reads_ground_truth_from_run_result() -> None:
    rr = offline_reward_fn(
        env_url="", admin_token="", task_id="K",
        run_result={
            "oracle_score": 0.9,
            "oracle_passed": True,
            "dynamic_verification_summary": {"verdict": "pass"},
            "costs": {"total": 0.5},
        },
        lam=0.1,
    )
    assert rr.oracle_score == 0.9
    assert rr.oracle_passed is True
    assert rr.proxy_verdict == "pass"
    assert rr.dollars == 0.5
    # reward = oracle − λ·dollars = 0.9 − 0.1·0.5 = 0.85
    assert rr.reward == pytest.approx(0.85)


def test_offline_reward_flags_false_pass() -> None:
    # proxy says pass, oracle says fail → false_pass for attribution noise.
    rr = offline_reward_fn(
        env_url="", admin_token="", task_id="K",
        run_result={
            "oracle_score": 0.0, "oracle_passed": False,
            "dynamic_verification_summary": {"verdict": "pass"},
            "costs": {"total": 0.2},
        },
    )
    assert rr.false_pass is True
    assert rr.false_fail is False


# ── policy assembly ──────────────────────────────────────────────────────


def test_build_policy_allocator_substrate_sets() -> None:
    assert [s.name for s in build_policy_allocator("frozen").substrates] == [FROZEN]
    assert [s.name for s in build_policy_allocator("S0_only").substrates] == [S0]
    assert [s.name for s in build_policy_allocator("S1_only").substrates] == [S1]
    assert {s.name for s in build_policy_allocator("allocator").substrates} == {S0, S1}


def test_build_policy_allocator_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        build_policy_allocator("nonsense")


def test_policy_substrates_cover_paper_baselines() -> None:
    assert set(POLICY_SUBSTRATES) == {"frozen", "S0_only", "S1_only", "allocator"}


# ── run_experiment ───────────────────────────────────────────────────────


def _const_run_fn(score: float, dollars: float, verdict: str = "pass"):
    def run_fn(task, plan, result):  # noqa: ARG001
        return {
            "oracle_score": score, "oracle_passed": score >= 0.5,
            "dynamic_verification_summary": {"verdict": verdict},
            "costs": {"total": dollars},
        }
    return run_fn


def test_run_experiment_runs_every_policy() -> None:
    result = run_experiment(
        tasks=_tasks(), run_fn=_const_run_fn(0.8, 0.1),
        reward_fn=offline_reward_fn, budget=100.0, rounds=1, epsilon=0.0,
    )
    assert set(result.reports) == set(POLICY_SUBSTRATES)
    # split metadata captured for every task
    assert result.split_of["k_seal"] == "sealed"
    assert result.cluster_of["c_vis"] == "capability"
    for rep in result.reports.values():
        assert rep.n_runs == len(_tasks())


def test_run_experiment_is_reproducible() -> None:
    kw = dict(tasks=_tasks(), run_fn=_const_run_fn(0.7, 0.2),
              reward_fn=offline_reward_fn, budget=100.0, rounds=2, seed=11)
    a = run_experiment(**kw)
    b = run_experiment(**kw)
    assert (build_table1(a)[0].score_per_dollar
            == build_table1(b)[0].score_per_dollar)


# ── Table 1 ──────────────────────────────────────────────────────────────


def test_build_table1_has_a_row_per_policy_plus_oracle() -> None:
    result = run_experiment(
        tasks=_tasks(), run_fn=_const_run_fn(0.8, 0.1),
        reward_fn=offline_reward_fn, budget=100.0, epsilon=0.0,
    )
    policies = {r.policy for r in build_table1(result)}
    assert policies == set(POLICY_SUBSTRATES) | {"oracle_allocator"}


def test_table1_visible_minus_sealed_gap() -> None:
    # Visible tasks score 1.0, sealed tasks score 0.0 → gap of 1.0 exactly.
    def split_run_fn(task, plan, result):  # noqa: ARG001
        score = 1.0 if task.split == "visible" else 0.0
        return {"oracle_score": score, "oracle_passed": score > 0.5,
                "dynamic_verification_summary": {"verdict": "pass"},
                "costs": {"total": 0.1}}

    result = run_experiment(
        tasks=_tasks(), run_fn=split_run_fn,
        reward_fn=offline_reward_fn, budget=100.0, epsilon=0.0,
    )
    frozen = next(r for r in build_table1(result) if r.policy == "frozen")
    assert frozen.visible_score == pytest.approx(1.0)
    assert frozen.sealed_score == pytest.approx(0.0)
    assert frozen.visible_minus_sealed == pytest.approx(1.0)


def test_table1_score_per_dollar() -> None:
    result = run_experiment(
        tasks=_tasks(), run_fn=_const_run_fn(0.8, 0.2),
        reward_fn=offline_reward_fn, budget=100.0, epsilon=0.0,
    )
    row = next(r for r in build_table1(result) if r.policy == "S0_only")
    # 4 runs × score 0.8 / (4 × $0.2) = 3.2 / 0.8 = 4.0
    assert row.score_per_dollar == pytest.approx(4.0)
    assert row.total_dollars == pytest.approx(0.8)


def test_oracle_allocator_is_upper_bound() -> None:
    # frozen always 0.2, S0 0.9 on knowledge / 0.1 on capability, S1 the
    # mirror. The oracle-allocator picks the per-task best, so its sealed
    # score must be >= every fixed policy's sealed score.
    def het_run_fn(task, plan, result):  # noqa: ARG001
        table = {
            (FROZEN, "knowledge"): 0.2, (FROZEN, "capability"): 0.2,
            (S0, "knowledge"): 0.9, (S0, "capability"): 0.1,
            (S1, "knowledge"): 0.1, (S1, "capability"): 0.9,
        }
        score = table.get((result.substrate, task.cluster), 0.0)
        return {"oracle_score": score, "oracle_passed": score >= 0.5,
                "dynamic_verification_summary": {"verdict": "pass"},
                "costs": {"total": 0.1}}

    result = run_experiment(
        tasks=_tasks(), run_fn=het_run_fn,
        reward_fn=offline_reward_fn, budget=100.0, epsilon=0.0,
    )
    rows = {r.policy: r for r in build_table1(result)}
    oracle = rows["oracle_allocator"].sealed_score
    for fixed in ("frozen", "S0_only", "S1_only"):
        assert oracle >= rows[fixed].sealed_score - 1e-9


# ── Fig 1 ────────────────────────────────────────────────────────────────


def test_fig1_series_cumulative_dollars_monotonic() -> None:
    result = run_experiment(
        tasks=_tasks(), run_fn=_const_run_fn(0.6, 0.15),
        reward_fn=offline_reward_fn, budget=100.0, epsilon=0.0,
    )
    series = fig1_series(result.reports["allocator"])
    assert len(series) == len(_tasks())
    cum = [p.cum_dollars for p in series]
    assert cum == sorted(cum)  # non-decreasing
    assert series[-1].cum_dollars == pytest.approx(0.15 * len(_tasks()))
    # constant score → running mean stays at 0.6
    assert series[-1].running_mean_oracle == pytest.approx(0.6)


# ── streaming: on_outcome callback ───────────────────────────────────────


def test_run_experiment_streams_each_outcome_with_policy() -> None:
    # run_experiment forwards every outcome to on_outcome as (policy, task,
    # outcome) the moment it lands — the seam live_runner writes rows from.
    seen: list[tuple[str, str, str | None]] = []
    run_experiment(
        tasks=_tasks(), run_fn=_const_run_fn(0.8, 0.1),
        reward_fn=offline_reward_fn, budget=100.0, rounds=1, epsilon=0.0,
        policies=("frozen", "S0_only"),
        on_outcome=lambda policy, task, o: seen.append((policy, task.name, o.substrate)),
    )
    # One emission per (policy, task).
    assert len(seen) == 2 * len(_tasks())
    assert {p for p, _, _ in seen} == {"frozen", "S0_only"}
    # The policy is bound per-iteration, not captured late: frozen's rows must
    # all read the no-op substrate, never S0's.
    assert {s for p, _, s in seen if p == "frozen"} == {FROZEN}


def test_streamed_rows_match_final_results_tsv(tmp_path: Path) -> None:
    # The rows streamed during the run must be byte-identical to the rows the
    # end-of-run batch writer emits — one row formatter, no divergence.
    streamed: list[list[str]] = []

    def on_outcome(policy, task, o):
        streamed.append([str(c) for c in _outcome_row(policy, o, task.split, task.seed)])

    result = run_experiment(
        tasks=_tasks(), run_fn=_const_run_fn(0.8, 0.1),
        reward_fn=offline_reward_fn, budget=100.0, epsilon=0.0,
        policies=("frozen", "S0_only"),
        on_outcome=on_outcome,
    )
    paths = write_results(result, tmp_path)
    with paths["results"].open() as fh:
        next(fh)  # banner
        rows = list(csv.reader(fh, delimiter="\t"))
    assert rows[0] == _OUTCOME_COLS
    assert streamed == rows[1:]


# ── writers ──────────────────────────────────────────────────────────────


def test_write_results_emits_stamped_parseable_tsv(tmp_path: Path) -> None:
    result = run_experiment(
        tasks=_tasks(), run_fn=_const_run_fn(0.8, 0.1),
        reward_fn=offline_reward_fn, budget=100.0, epsilon=0.0,
    )
    paths = write_results(result, tmp_path)
    assert set(paths) == {"results", "table1", "fig1"}

    for p in paths.values():
        lines = p.read_text().splitlines()
        # First line is the synthetic-source banner so demo output can never
        # be mistaken for real data.
        assert lines[0].startswith("# SOURCE=offline-demo")

    # table1.tsv parses and carries every policy row + the oracle bound.
    with paths["table1"].open() as fh:
        next(fh)  # banner
        rows = list(csv.DictReader(fh, delimiter="\t"))
    policies = {r["policy"] for r in rows}
    assert policies == set(POLICY_SUBSTRATES) | {"oracle_allocator"}


# ── end-to-end demo ──────────────────────────────────────────────────────


def test_offline_demo_allocator_learns_per_cluster_winner() -> None:
    # The synthetic demo encodes knowledge→S0, capability→S1, policy→S1.
    # After enough rounds the allocator's value table must rank the expected
    # winner above the loser in each cluster — the heterogeneity the whole
    # experiment hinges on.
    result = run_offline_demo(rounds=8, budget=500.0, epsilon=0.1, seed=3)
    vt = result.reports["allocator"].value_table

    def val(cluster: str, sub: str) -> float:
        return vt.get((cluster, sub), (float("-inf"), 0))[0]

    assert val("knowledge", S0) > val("knowledge", S1)
    assert val("capability", S1) > val("capability", S0)
    assert val("policy", S1) > val("policy", S0)


def test_offline_demo_allocator_beats_frozen_on_sealed() -> None:
    result = run_offline_demo(rounds=8, budget=500.0, epsilon=0.1, seed=3)
    rows = {r.policy: r for r in build_table1(result)}
    assert rows["allocator"].sealed_score > rows["frozen"].sealed_score


def test_run_offline_demo_covers_all_runnable_tasks() -> None:
    result = run_offline_demo(rounds=1, budget=500.0)
    # Seven runnable tasks: 3 clusters × 2 splits, plus the BT03 gated-reveal
    # policy variant (visible split only).
    assert result.reports["frozen"].n_runs == 7
    assert set(result.cluster_of.values()) == {"knowledge", "capability", "policy"}
