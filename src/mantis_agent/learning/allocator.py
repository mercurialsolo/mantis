"""The myopic contextual-bandit allocator (PLAN §5).

The decision layer that makes "continual improvement" a *budget-constrained
choice* rather than a fixed pipeline. On each task it scores every
affordable rung of the substrate ladder (S0→S3) with one currency —
``R = Δscore − λ·cost`` — picks one, applies it, then folds the realised
reward back into a per-``(cluster, substrate)`` value estimate.

It is deliberately **myopic** (the paper's framing): a one-step contextual
bandit, not a planner. The "context" is the task's failure cluster
(``knowledge`` | ``capability`` | ``policy``); the "arms" are the
substrates. Two properties the experiment leans on fall out of this design:

* **Beats any fixed substrate** — because the value table is per-cluster,
  the allocator can learn S0 wins ``knowledge`` while S2 wins
  ``capability``, which no single fixed rung can match (Table 1 / Fig 1).
* **Tracks non-stationarity** — :attr:`timeline` records every decision in
  order, so the drift of the winning rung as the agent matures is directly
  plottable (Fig 2).

Exploration is ε-greedy with an *explore-unknown-first* warm-up: until a
``(cluster, substrate)`` cell has been tried once, it is preferred over
exploitation, so every affordable rung gets at least one real pull before
the bandit settles. Ties (equal value, or the all-zero warm-up) break
toward the cheaper, more reversible rung — the durability ordering the
ladder is built on.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from .reward import DEFAULT_LAMBDA
from .substrates.base import (
    Durability,
    LearningSubstrate,
    SubstrateContext,
    SubstrateResult,
)

# Cheaper / more reversible first. Used to break value ties (and to order
# the warm-up over untried arms) toward the bottom of the ladder.
_DURABILITY_RANK: dict[Durability, int] = {
    Durability.EPHEMERAL: 0,
    Durability.SESSION: 1,
    Durability.POLICY: 2,
    Durability.WEIGHTS: 3,
}


@dataclass
class ValueEstimate:
    """Running mean reward for one ``(cluster, substrate)`` cell."""

    count: int = 0
    mean: float = 0.0

    def update(self, reward: float) -> None:
        self.count += 1
        # Incremental mean — no need to retain the sample history.
        self.mean += (reward - self.mean) / self.count


@dataclass
class AllocationRecord:
    """One decision the allocator made — the unit of :attr:`timeline`."""

    index: int
    cluster: str
    substrate: str
    reason: str  # "explore" | "exploit" | "explore-unknown"
    value_estimate: float
    cost_estimate: float


@dataclass
class SubstrateChoice:
    """The rung the allocator picked, plus why — returned by :meth:`choose`."""

    substrate: LearningSubstrate
    value_estimate: float
    cost_estimate: float
    reason: str

    @property
    def name(self) -> str:
        return self.substrate.name


class MyopicAllocator:
    """ε-greedy contextual bandit over the substrate ladder.

    Construct with the rungs to choose among; the allocator keeps a value
    table keyed on ``(cluster, substrate.name)`` and updated through
    :meth:`observe`. ``rng`` is injectable so tests are deterministic.
    """

    def __init__(
        self,
        substrates: list[LearningSubstrate],
        *,
        lam: float = DEFAULT_LAMBDA,
        epsilon: float = 0.1,
        rng: random.Random | None = None,
    ) -> None:
        if not substrates:
            raise ValueError("MyopicAllocator needs at least one substrate")
        self.substrates = list(substrates)
        self._by_name = {s.name: s for s in self.substrates}
        if len(self._by_name) != len(self.substrates):
            raise ValueError("substrate names must be unique")
        self.lam = float(lam)
        self.epsilon = float(epsilon)
        self._rng = rng or random.Random()
        self._values: dict[tuple[str, str], ValueEstimate] = {}
        self._timeline: list[AllocationRecord] = []
        self._n = 0

    # ── value table ─────────────────────────────────────────────────────

    @staticmethod
    def _key(cluster: str | None, name: str) -> tuple[str, str]:
        return (cluster or "", name)

    def value_of(self, cluster: str | None, name: str) -> float:
        v = self._values.get(self._key(cluster, name))
        return v.mean if v else 0.0

    def count_of(self, cluster: str | None, name: str) -> int:
        v = self._values.get(self._key(cluster, name))
        return v.count if v else 0

    # ── choice ──────────────────────────────────────────────────────────

    def _affordable(
        self, context: SubstrateContext,
    ) -> list[tuple[LearningSubstrate, float]]:
        """Rungs whose cost estimate fits the remaining budget."""
        out: list[tuple[LearningSubstrate, float]] = []
        for s in self.substrates:
            cost = float(s.cost_estimate(context))
            if cost <= context.budget_remaining + 1e-9:
                out.append((s, cost))
        return out

    def choose(self, context: SubstrateContext) -> SubstrateChoice | None:
        """Pick a rung for ``context``. ``None`` when nothing is affordable.

        ε-greedy with explore-unknown-first warm-up. Records the decision
        on :attr:`timeline` either way.
        """
        affordable = self._affordable(context)
        if not affordable:
            return None
        cluster = context.cluster or ""

        if self._rng.random() < self.epsilon:
            substrate, cost = self._rng.choice(affordable)
            reason = "explore"
        else:
            untried = [
                (s, c) for (s, c) in affordable
                if self.count_of(cluster, s.name) == 0
            ]
            pool = untried or affordable
            reason = "explore-unknown" if untried else "exploit"
            substrate, cost = min(pool, key=lambda item: self._rank(cluster, item))

        value = self.value_of(cluster, substrate.name)
        self._n += 1
        self._timeline.append(
            AllocationRecord(
                index=self._n,
                cluster=cluster,
                substrate=substrate.name,
                reason=reason,
                value_estimate=value,
                cost_estimate=cost,
            )
        )
        return SubstrateChoice(substrate, value, cost, reason)

    def _rank(
        self, cluster: str, item: tuple[LearningSubstrate, float],
    ) -> tuple[float, int, float, str]:
        """Sort key for the greedy pick — ``min`` over this picks the
        winner. Highest value first (negated), then the cheaper / more
        reversible rung on a tie. For untried arms every value is 0.0, so
        the warm-up falls through to the durability + cost ordering."""
        substrate, cost = item
        value = self.value_of(cluster, substrate.name)
        durability = getattr(substrate, "durability", Durability.EPHEMERAL)
        return (-value, _DURABILITY_RANK.get(durability, 0), cost, substrate.name)

    def allocate(
        self, context: SubstrateContext,
    ) -> tuple[SubstrateChoice, SubstrateResult] | None:
        """Choose a rung *and* apply it — the orchestrator's one-call path.

        Returns the ``(choice, result)`` pair, or ``None`` when nothing was
        affordable. The caller runs the plan, scores the reward channels,
        then hands the reward back via :meth:`observe`.
        """
        choice = self.choose(context)
        if choice is None:
            return None
        result = choice.substrate.apply(context)
        return choice, result

    # ── update ──────────────────────────────────────────────────────────

    def observe(
        self,
        context: SubstrateContext,
        substrate_name: str,
        reward: float,
        *,
        result: SubstrateResult | None = None,
    ) -> None:
        """Fold a realised reward into the value table and the rung itself.

        ``reward`` is the combined-currency ``R = Δscore − λ·cost`` the
        reward channels produced (see :mod:`mantis_agent.learning.reward`).
        """
        key = self._key(context.cluster, substrate_name)
        self._values.setdefault(key, ValueEstimate()).update(float(reward))
        substrate = self._by_name.get(substrate_name)
        if substrate is not None and result is not None:
            substrate.observe(context, result, float(reward))

    # ── reporting (Table 1 / Fig 1 / Fig 2) ─────────────────────────────

    def value_table(self) -> dict[tuple[str, str], tuple[float, int]]:
        """``(cluster, substrate) -> (mean_reward, n)`` — the Table 1 grid."""
        return {k: (v.mean, v.count) for k, v in self._values.items()}

    def selection_counts(self, cluster: str | None = None) -> Counter:
        """How often each substrate was chosen (optionally within a cluster)."""
        counts: Counter = Counter()
        for record in self._timeline:
            if cluster is None or record.cluster == cluster:
                counts[record.substrate] += 1
        return counts

    def selection_distribution(self, cluster: str | None = None) -> dict[str, float]:
        """:meth:`selection_counts` normalised to fractions — Fig 1's bars."""
        counts = self.selection_counts(cluster)
        total = sum(counts.values())
        if not total:
            return {}
        return {name: n / total for name, n in counts.items()}

    @property
    def timeline(self) -> list[AllocationRecord]:
        """Every decision in order — the raw signal for Fig 2."""
        return list(self._timeline)


__all__ = [
    "ValueEstimate",
    "AllocationRecord",
    "SubstrateChoice",
    "MyopicAllocator",
]
