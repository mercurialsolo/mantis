"""Dual reward channels — the Mantis advantage over the paper's testbed.

The Learning Allocator paper has a noisy reward (ASR-attributed success) but
*no ground truth to measure the noise against*. Mantis ships both:

* an expensive **accurate oracle** — the sim-env mutation-log grader, reached
  via ``GET /__env__/oracle?task_id=<id>`` (``gym.grading.grade_run``). It reads
  server DB state, not the agent transcript, so it is the ground truth.
* a cheap **noisy proxy** — the LLM verifier (``DynamicPlanVerifier``), whose
  verdict rides on a completed run's ``dynamic_verification_summary``.

Pairing the two lets us do two things the paper cannot:

1. **Quantify partial observability** — log oracle verdict vs proxy verdict per
   task → false-pass / false-fail rates (the Fig P / PRM training data).
2. **Score the reward** the allocator optimises: ``R = oracle_score − λ·cost``,
   cost in dollars from the run's cost meter.

This module is deliberately split into *pure* combinators (testable without a
network) and thin *channel readers* that pull each signal from its source.
``reward_from_run`` ties them together for a completed run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..gym.grading import GradingResult, grade_run

# Cost coefficient λ in ``R = oracle_score − λ·cost``. A tunable knob, not a
# law: at λ=0.1 a dollar of spend is worth 0.1 of oracle score (scores live in
# 0..1), so a ~$1 run trades against a 0.1 score gain. Sweeping λ is what the
# budget-frontier figure (Fig 4) does; this is the default working point.
DEFAULT_LAMBDA = 0.1

# Map the proxy's categorical verdict onto the oracle's 0..1 scale so the two
# channels are comparable. ``partial`` is the verifier hedging — half credit.
_PROXY_SCORE: dict[str, float] = {
    "pass": 1.0,
    "partial": 0.5,
    "fail": 0.0,
    "unknown": 0.0,
}


def proxy_score(verdict: str | None) -> float:
    """Map a proxy verdict (``pass``/``partial``/``fail``) to a 0..1 score."""
    return _PROXY_SCORE.get((verdict or "unknown").strip().lower(), 0.0)


@dataclass
class RewardRecord:
    """One task's dual-channel reward + the paired labels for noise analysis.

    ``reward`` is the *absolute* working reward ``oracle_score − λ·dollars``.
    The allocator turns this into the paper's ``Δscore − λ·cost`` by
    subtracting a frozen-baseline record's ``oracle_score`` — kept out of here
    so this stays a pure per-run measurement.
    """

    task_id: str
    oracle_score: float
    oracle_passed: bool
    proxy_verdict: str
    proxy_score: float
    dollars: float
    reward: float
    # Attribution-noise labels (proxy vs ground truth) — feed Fig P / the PRM.
    false_pass: bool  # proxy said pass, oracle says the task failed
    false_fail: bool  # proxy said fail, oracle says the task passed
    oracle_error: str | None = None
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def compute_reward(
    *,
    task_id: str,
    oracle_score: float,
    oracle_passed: bool,
    proxy_verdict: str | None,
    dollars: float,
    lam: float = DEFAULT_LAMBDA,
    oracle_error: str | None = None,
    extras: dict[str, Any] | None = None,
) -> RewardRecord:
    """Combine the two channels + cost into a :class:`RewardRecord` (pure).

    No network, no run-result parsing — just the arithmetic and the
    paired-label bookkeeping. The channel readers below feed this.
    """
    verdict = (proxy_verdict or "unknown").strip().lower()
    p_score = proxy_score(verdict)
    reward = oracle_score - lam * dollars

    false_pass = verdict == "pass" and not oracle_passed
    false_fail = verdict == "fail" and oracle_passed

    return RewardRecord(
        task_id=task_id,
        oracle_score=round(float(oracle_score), 4),
        oracle_passed=bool(oracle_passed),
        proxy_verdict=verdict,
        proxy_score=p_score,
        dollars=round(float(dollars), 4),
        reward=round(reward, 4),
        false_pass=false_pass,
        false_fail=false_fail,
        oracle_error=oracle_error,
        extras=extras or {},
    )


# ── Channel readers ────────────────────────────────────────────────────────


def oracle_channel(env_url: str, admin_token: str, task_id: str) -> GradingResult:
    """Ground-truth channel: grade the env's DB state via the oracle endpoint.

    Thin wrapper over :func:`gym.grading.grade_run` — kept here so callers in
    the learning package have one import surface and so the live oracle call
    is trivially swappable in tests.
    """
    return grade_run(env_url, admin_token, task_id)


def proxy_channel(run_result: dict[str, Any]) -> str:
    """Noisy channel: pull the LLM-verifier verdict off a completed run.

    Reads ``run_result["dynamic_verification_summary"]["verdict"]``. Returns
    ``"unknown"`` when the run carried no verifier summary (e.g. verification
    disabled) rather than guessing.
    """
    summary = run_result.get("dynamic_verification_summary") or {}
    if not isinstance(summary, dict):
        return "unknown"
    verdict = summary.get("verdict")
    return str(verdict).strip().lower() if verdict else "unknown"


def cost_channel(run_result: dict[str, Any]) -> float:
    """Cost channel: total dollars for the run from its cost meter.

    Reads ``run_result["costs"]["total"]``. Returns ``0.0`` when absent.
    """
    costs = run_result.get("costs") or {}
    if not isinstance(costs, dict):
        return 0.0
    try:
        return float(costs.get("total") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def reward_from_run(
    *,
    env_url: str,
    admin_token: str,
    task_id: str,
    run_result: dict[str, Any],
    lam: float = DEFAULT_LAMBDA,
) -> RewardRecord:
    """End-to-end per-run reward: grade the env, read proxy + cost off the run.

    The oracle is queried live (DB ground truth); the proxy verdict and dollar
    cost are read from the already-returned ``run_result`` dict (the
    ``build_micro_result`` payload from the Modal CUA server).
    """
    graded = oracle_channel(env_url, admin_token, task_id)
    return compute_reward(
        task_id=task_id,
        oracle_score=graded.score,
        oracle_passed=graded.passed,
        proxy_verdict=proxy_channel(run_result),
        dollars=cost_channel(run_result),
        lam=lam,
        oracle_error=graded.error,
        extras={"oracle_reasons": graded.reasons, "oracle_diff": graded.diff},
    )


__all__ = [
    "DEFAULT_LAMBDA",
    "RewardRecord",
    "proxy_score",
    "compute_reward",
    "oracle_channel",
    "proxy_channel",
    "cost_channel",
    "reward_from_run",
]
