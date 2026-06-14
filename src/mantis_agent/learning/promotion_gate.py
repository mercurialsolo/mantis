"""Champion/challenger promotion gate (#894) — the safety keystone.

Before a learned change (a new substrate config, a promoted plan rewrite, a new
brain checkpoint) is shipped, it must beat the deployed *champion* on a FROZEN
benchmark — by a margin, with significance, without regressing sealed tasks, and
within a cost budget. This lets any loop close autonomously without shipping
regressions.

Pure + dependency-free (pure-Python paired bootstrap — no scipy/numpy) and
deterministic (RNG seeded), so a verdict is reproducible and auditable. It scores
two arms that were run over the SAME frozen task set (paired comparison); the
caller supplies per-task scores (e.g. oracle ``score`` or ``R = score − λ·cost``
from ``learning/reward.py``) — the gate never executes anything.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ArmResult:
    """Per-task outcomes for one arm over the frozen benchmark.

    ``scores`` maps ``task_id → reward`` (higher is better — oracle score or
    ``R``). ``costs`` maps ``task_id → dollars`` (optional; used for the cost
    gate). Only task_ids present in BOTH arms are compared (paired).
    """

    label: str
    scores: dict[str, float]
    costs: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GateVerdict:
    promote: bool
    reason: str
    n_compared: int
    mean_delta: float          # mean(challenger − champion) over paired tasks
    win_rate: float            # fraction of tasks where challenger >= champion
    prob_improvement: float    # bootstrap P(mean challenger > mean champion)
    regressions: list[str]     # tasks where challenger dropped > regression_tol
    champion_cost: float
    challenger_cost: float


@dataclass
class PromotionGate:
    """Decide whether a challenger may replace the champion.

    All gates must pass to promote:
      * ``mean_delta >= min_mean_delta`` — beats the champion on average;
      * ``prob_improvement >= min_prob_improvement`` — significance via paired
        bootstrap (default 0.95);
      * ``win_rate >= min_win_rate`` — broadly better, not one outlier;
      * ``len(regressions) <= max_regressions`` — no (or few) sealed-task drops;
      * challenger cost ``<= champion cost × max_cost_ratio`` (when costs given).
    """

    min_mean_delta: float = 0.02
    min_prob_improvement: float = 0.95
    min_win_rate: float = 0.5
    regression_tol: float = 0.05
    max_regressions: int = 0
    max_cost_ratio: float = 1.5
    bootstrap_samples: int = 2000
    rng_seed: int = 0

    def evaluate(self, champion: ArmResult, challenger: ArmResult) -> GateVerdict:
        common = sorted(set(champion.scores) & set(challenger.scores))
        if not common:
            return GateVerdict(
                promote=False, reason="no overlapping tasks to compare",
                n_compared=0, mean_delta=0.0, win_rate=0.0,
                prob_improvement=0.0, regressions=[],
                champion_cost=0.0, challenger_cost=0.0,
            )

        deltas = [challenger.scores[t] - champion.scores[t] for t in common]
        n = len(deltas)
        mean_delta = sum(deltas) / n
        win_rate = sum(1 for d in deltas if d >= 0) / n
        regressions = [
            t for t, d in zip(common, deltas) if d < -self.regression_tol
        ]
        prob_improvement = _bootstrap_prob_positive(
            deltas, self.bootstrap_samples, self.rng_seed,
        )
        champ_cost = sum(champion.costs.get(t, 0.0) for t in common)
        chal_cost = sum(challenger.costs.get(t, 0.0) for t in common)

        checks: list[tuple[bool, str]] = [
            (mean_delta >= self.min_mean_delta,
             f"mean_delta {mean_delta:+.4f} < {self.min_mean_delta}"),
            (prob_improvement >= self.min_prob_improvement,
             f"prob_improvement {prob_improvement:.3f} < {self.min_prob_improvement}"),
            (win_rate >= self.min_win_rate,
             f"win_rate {win_rate:.3f} < {self.min_win_rate}"),
            (len(regressions) <= self.max_regressions,
             f"{len(regressions)} regressions > {self.max_regressions} "
             f"(tol {self.regression_tol})"),
        ]
        if champ_cost > 0:
            checks.append((
                chal_cost <= champ_cost * self.max_cost_ratio,
                f"cost {chal_cost:.4f} > champion {champ_cost:.4f} "
                f"× {self.max_cost_ratio}",
            ))

        failures = [msg for ok, msg in checks if not ok]
        promote = not failures
        reason = (
            f"promote: challenger beats champion (Δ={mean_delta:+.4f}, "
            f"p={prob_improvement:.3f}, win={win_rate:.2f}, n={n})"
            if promote else "reject: " + "; ".join(failures)
        )
        return GateVerdict(
            promote=promote, reason=reason, n_compared=n,
            mean_delta=mean_delta, win_rate=win_rate,
            prob_improvement=prob_improvement, regressions=regressions,
            champion_cost=champ_cost, challenger_cost=chal_cost,
        )


def _bootstrap_prob_positive(deltas: list[float], samples: int, seed: int) -> float:
    """Paired bootstrap: fraction of resamples whose mean delta > 0 — the
    probability the challenger truly improves over the champion. Deterministic
    given ``seed``. With zero variance (all deltas equal) returns 1.0 if the
    delta is positive, else 0.0."""
    n = len(deltas)
    if n == 0:
        return 0.0
    if all(d == deltas[0] for d in deltas):
        return 1.0 if deltas[0] > 0 else 0.0
    rng = random.Random(seed)
    wins = 0
    for _ in range(samples):
        resample_sum = 0.0
        for _ in range(n):
            resample_sum += deltas[rng.randrange(n)]
        if resample_sum > 0:  # mean > 0 ⇔ sum > 0
            wins += 1
    return wins / samples
