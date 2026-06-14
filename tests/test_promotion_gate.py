"""Champion/challenger promotion gate (#894)."""

from __future__ import annotations

from mantis_agent.learning.promotion_gate import (
    ArmResult,
    GateVerdict,
    PromotionGate,
)


def _arm(label, scores, costs=None):
    return ArmResult(label=label, scores=dict(scores), costs=dict(costs or {}))


def test_promotes_clear_improvement():
    champ = _arm("champ", {f"t{i}": 0.5 for i in range(10)})
    chal = _arm("chal", {f"t{i}": 0.8 for i in range(10)})
    v = PromotionGate().evaluate(champ, chal)
    assert v.promote is True
    assert v.mean_delta > 0.29 and v.win_rate == 1.0
    assert v.prob_improvement == 1.0  # zero-variance positive delta


def test_rejects_no_improvement():
    champ = _arm("champ", {f"t{i}": 0.6 for i in range(10)})
    chal = _arm("chal", {f"t{i}": 0.6 for i in range(10)})
    v = PromotionGate().evaluate(champ, chal)
    assert v.promote is False
    assert "mean_delta" in v.reason


def test_rejects_on_regression_even_if_mean_up():
    # Net positive on average, but one task regresses hard → hard gate blocks.
    champ = _arm("champ", {"a": 0.5, "b": 0.5, "c": 0.9})
    chal = _arm("chal", {"a": 0.9, "b": 0.9, "c": 0.3})  # c drops 0.6
    v = PromotionGate(min_prob_improvement=0.0, min_mean_delta=0.0).evaluate(champ, chal)
    assert v.promote is False
    assert "c" in v.regressions
    assert "regression" in v.reason


def test_regression_tolerance_allows_small_dip():
    champ = _arm("champ", {"a": 0.5, "b": 0.5, "c": 0.6})
    chal = _arm("chal", {"a": 0.9, "b": 0.9, "c": 0.57})  # c dips 0.03 < tol 0.05
    v = PromotionGate(min_prob_improvement=0.0).evaluate(champ, chal)
    assert v.regressions == []
    assert v.promote is True


def test_cost_gate_blocks_expensive_challenger():
    champ = _arm("champ", {"a": 0.5, "b": 0.5}, {"a": 0.10, "b": 0.10})
    chal = _arm("chal", {"a": 0.9, "b": 0.9}, {"a": 0.40, "b": 0.40})  # 4x cost
    v = PromotionGate(min_prob_improvement=0.0, max_cost_ratio=1.5).evaluate(champ, chal)
    assert v.promote is False
    assert "cost" in v.reason
    assert v.challenger_cost == 0.80 and v.champion_cost == 0.20


def test_only_paired_tasks_compared():
    champ = _arm("champ", {"a": 0.5, "b": 0.5, "only_champ": 0.1})
    chal = _arm("chal", {"a": 0.9, "b": 0.9, "only_chal": 0.1})
    v = PromotionGate(min_prob_improvement=0.0).evaluate(champ, chal)
    assert v.n_compared == 2  # a, b only


def test_no_overlap_is_safe_reject():
    v = PromotionGate().evaluate(_arm("c", {"a": 1.0}), _arm("x", {"b": 1.0}))
    assert v.promote is False and v.n_compared == 0
    assert "no overlapping" in v.reason


def test_significance_blocks_noisy_tie():
    # Tiny mean improvement swamped by noise → bootstrap p low → reject.
    champ = _arm("champ", {f"t{i}": float(i % 2) for i in range(20)})
    chal = _arm("chal", {f"t{i}": float((i + 1) % 2) for i in range(20)})
    v = PromotionGate(min_mean_delta=0.0, max_regressions=99).evaluate(champ, chal)
    # Half up, half down, mean ~0 → not significant.
    assert v.prob_improvement < 0.95
    assert v.promote is False


def test_verdict_is_deterministic():
    champ = _arm("champ", {f"t{i}": 0.4 + (i % 3) * 0.1 for i in range(15)})
    chal = _arm("chal", {f"t{i}": 0.55 + (i % 3) * 0.1 for i in range(15)})
    a = PromotionGate(rng_seed=7).evaluate(champ, chal)
    b = PromotionGate(rng_seed=7).evaluate(champ, chal)
    assert a == b
    assert isinstance(a, GateVerdict)
