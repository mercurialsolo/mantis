"""Tests for the dual reward channels (oracle vs proxy + cost).

The pure combinators are tested directly; the live oracle call inside
``reward_from_run`` is exercised with ``grade_run`` monkeypatched, so no env
boots.
"""

from __future__ import annotations

import pytest

from mantis_agent.gym.grading import GradingResult
from mantis_agent.learning import reward as R


# ── proxy_score ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "verdict,expected",
    [
        ("pass", 1.0),
        ("PASS", 1.0),
        ("partial", 0.5),
        ("fail", 0.0),
        ("unknown", 0.0),
        (None, 0.0),
        ("garbage", 0.0),
    ],
)
def test_proxy_score_mapping(verdict, expected) -> None:
    assert R.proxy_score(verdict) == expected


# ── compute_reward arithmetic ──────────────────────────────────────────


def test_reward_is_score_minus_lambda_cost() -> None:
    rec = R.compute_reward(
        task_id="t",
        oracle_score=0.8,
        oracle_passed=True,
        proxy_verdict="pass",
        dollars=2.0,
        lam=0.1,
    )
    # 0.8 − 0.1*2.0 = 0.6
    assert rec.reward == 0.6
    assert rec.oracle_score == 0.8
    assert rec.proxy_score == 1.0
    assert rec.dollars == 2.0


def test_default_lambda_used_when_unspecified() -> None:
    rec = R.compute_reward(
        task_id="t",
        oracle_score=1.0,
        oracle_passed=True,
        proxy_verdict="pass",
        dollars=1.0,
    )
    assert rec.reward == round(1.0 - R.DEFAULT_LAMBDA * 1.0, 4)


# ── attribution-noise labels (the Fig P signal) ────────────────────────


def test_false_pass_when_proxy_optimistic() -> None:
    # proxy says pass, oracle says the task failed → false pass
    rec = R.compute_reward(
        task_id="t",
        oracle_score=0.0,
        oracle_passed=False,
        proxy_verdict="pass",
        dollars=0.5,
    )
    assert rec.false_pass is True
    assert rec.false_fail is False


def test_false_fail_when_proxy_pessimistic() -> None:
    # proxy says fail, oracle says it passed → false fail
    rec = R.compute_reward(
        task_id="t",
        oracle_score=1.0,
        oracle_passed=True,
        proxy_verdict="fail",
        dollars=0.5,
    )
    assert rec.false_fail is True
    assert rec.false_pass is False


def test_agreement_sets_no_noise_flags() -> None:
    rec = R.compute_reward(
        task_id="t",
        oracle_score=1.0,
        oracle_passed=True,
        proxy_verdict="pass",
        dollars=0.0,
    )
    assert rec.false_pass is False
    assert rec.false_fail is False


def test_partial_verdict_is_not_a_clean_pass_or_fail() -> None:
    # 'partial' is the verifier hedging — it should not trip either label.
    rec = R.compute_reward(
        task_id="t",
        oracle_score=0.0,
        oracle_passed=False,
        proxy_verdict="partial",
        dollars=0.0,
    )
    assert rec.false_pass is False
    assert rec.false_fail is False
    assert rec.proxy_score == 0.5


# ── channel readers ────────────────────────────────────────────────────


def test_proxy_channel_reads_verdict() -> None:
    run = {"dynamic_verification_summary": {"verdict": "Pass"}}
    assert R.proxy_channel(run) == "pass"


def test_proxy_channel_missing_summary_is_unknown() -> None:
    assert R.proxy_channel({}) == "unknown"
    assert R.proxy_channel({"dynamic_verification_summary": None}) == "unknown"
    assert R.proxy_channel({"dynamic_verification_summary": {}}) == "unknown"


def test_cost_channel_reads_total() -> None:
    assert R.cost_channel({"costs": {"total": 1.23}}) == 1.23


def test_cost_channel_absent_is_zero() -> None:
    assert R.cost_channel({}) == 0.0
    assert R.cost_channel({"costs": None}) == 0.0
    assert R.cost_channel({"costs": {"total": None}}) == 0.0


# ── reward_from_run (oracle mocked) ────────────────────────────────────


def test_reward_from_run_ties_channels_together(monkeypatch) -> None:
    def fake_grade(env_url, admin_token, task_id, **kw):
        return GradingResult(
            task_id=task_id, passed=True, score=0.9, reasons=["ok"], diff={"hits": 3}
        )

    monkeypatch.setattr(R, "grade_run", fake_grade)

    run_result = {
        "dynamic_verification_summary": {"verdict": "pass"},
        "costs": {"total": 1.0},
    }
    rec = R.reward_from_run(
        env_url="https://env.example",
        admin_token="tok",
        task_id="BT01_lead_capture_filtered_search",
        run_result=run_result,
        lam=0.1,
    )
    assert rec.oracle_score == 0.9
    assert rec.oracle_passed is True
    assert rec.proxy_verdict == "pass"
    assert rec.dollars == 1.0
    assert rec.reward == round(0.9 - 0.1 * 1.0, 4)
    assert rec.extras["oracle_diff"] == {"hits": 3}


def test_reward_from_run_surfaces_oracle_error(monkeypatch) -> None:
    def fake_grade(env_url, admin_token, task_id, **kw):
        return GradingResult(task_id=task_id, error="oracle network error")

    monkeypatch.setattr(R, "grade_run", fake_grade)

    rec = R.reward_from_run(
        env_url="https://env.example",
        admin_token="tok",
        task_id="BT01_lead_capture_filtered_search",
        run_result={"costs": {"total": 0.2}},
    )
    assert rec.oracle_error == "oracle network error"
    assert rec.oracle_score == 0.0
    assert rec.proxy_verdict == "unknown"
