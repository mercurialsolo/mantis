"""Tests for the myopic contextual-bandit allocator.

The allocator is the piece that makes the paper's central claim testable —
that a budget-aware bandit beats any *fixed* substrate by learning a
*per-cluster* winner. The tests pin the mechanics the experiment depends
on: deterministic warm-up coverage, per-cluster value isolation, budget
filtering, durability tie-breaks, and the selection-distribution / timeline
reporting that Fig 1 + Fig 2 read.

Fake substrates stand in for the real rungs so no env boots; the bandit
logic is what's under test, not the adapters (those have their own suites).
"""

from __future__ import annotations

import random

import pytest

from mantis_agent.learning.allocator import (
    MyopicAllocator,
    ValueEstimate,
)
from mantis_agent.learning.substrates.base import (
    Durability,
    SubstrateContext,
    SubstrateResult,
)


class FakeSubstrate:
    """Minimal LearningSubstrate with controllable cost + durability."""

    def __init__(
        self,
        name: str,
        *,
        cost: float = 0.0,
        durability: Durability = Durability.EPHEMERAL,
    ) -> None:
        self.name = name
        self._cost = cost
        self.durability = durability
        self.observed: list[float] = []

    def cost_estimate(self, context: SubstrateContext) -> float:  # noqa: ARG002
        return self._cost

    def apply(self, context: SubstrateContext) -> SubstrateResult:  # noqa: ARG002
        return SubstrateResult(
            substrate=self.name, applied=True,
            dollars_spent=self._cost, durability=self.durability,
        )

    def observe(self, context, result, reward: float) -> None:  # noqa: ARG002
        self.observed.append(reward)


def _ctx(cluster: str = "capability", *, budget: float = 100.0) -> SubstrateContext:
    return SubstrateContext(task_id="t", cluster=cluster, budget_remaining=budget)


def _greedy(substrates, **kw) -> MyopicAllocator:
    # epsilon=0 ⇒ deterministic; no random exploration branch.
    return MyopicAllocator(substrates, epsilon=0.0, **kw)


# ── construction guards ────────────────────────────────────────────────


def test_requires_at_least_one_substrate() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MyopicAllocator([])


def test_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        MyopicAllocator([FakeSubstrate("S0"), FakeSubstrate("S0")])


# ── warm-up: every affordable arm tried once before exploitation ───────


def test_warmup_tries_each_arm_once_in_ladder_order() -> None:
    s0 = FakeSubstrate("S0", durability=Durability.EPHEMERAL)
    s1 = FakeSubstrate("S1", durability=Durability.SESSION)
    alloc = _greedy([s1, s0])  # deliberately out of order

    picks = []
    for _ in range(2):
        choice = alloc.choose(_ctx())
        picks.append(choice.name)
        alloc.observe(_ctx(), choice.name, 0.0)

    # Untried arms are explored cheapest/most-reversible first ⇒ S0 then S1.
    assert picks == ["S0", "S1"]
    assert all(p == "explore-unknown" for p in
               [r.reason for r in alloc.timeline])


# ── exploitation: highest per-cluster value wins ───────────────────────


def test_exploit_picks_highest_value_arm() -> None:
    s0, s1 = FakeSubstrate("S0"), FakeSubstrate("S1", durability=Durability.SESSION)
    alloc = _greedy([s0, s1])
    # Prime both as tried, with S1 the better arm in this cluster.
    alloc.observe(_ctx(), "S0", 0.1)
    alloc.observe(_ctx(), "S1", 0.9)

    choice = alloc.choose(_ctx())

    assert choice.name == "S1"
    assert choice.reason == "exploit"
    assert choice.value_estimate == pytest.approx(0.9)


def test_value_is_per_cluster_not_global() -> None:
    # The contextual claim: the same arm can win one cluster and lose another.
    s0, s1 = FakeSubstrate("S0"), FakeSubstrate("S1", durability=Durability.SESSION)
    alloc = _greedy([s0, s1])
    # knowledge: S0 strong, S1 weak.
    alloc.observe(_ctx("knowledge"), "S0", 1.0)
    alloc.observe(_ctx("knowledge"), "S1", 0.0)
    # capability: reversed.
    alloc.observe(_ctx("capability"), "S0", 0.0)
    alloc.observe(_ctx("capability"), "S1", 1.0)

    assert alloc.choose(_ctx("knowledge")).name == "S0"
    assert alloc.choose(_ctx("capability")).name == "S1"


# ── budget gating ──────────────────────────────────────────────────────


def test_budget_filters_unaffordable_arms() -> None:
    cheap = FakeSubstrate("S0", cost=0.0)
    pricey = FakeSubstrate("S3", cost=3.0, durability=Durability.WEIGHTS)
    alloc = _greedy([cheap, pricey])

    # Budget below the pricey rung ⇒ it's never an option.
    choice = alloc.choose(_ctx(budget=0.5))
    assert choice.name == "S0"


def test_choose_returns_none_when_nothing_affordable() -> None:
    pricey = FakeSubstrate("S3", cost=3.0, durability=Durability.WEIGHTS)
    alloc = _greedy([pricey])
    assert alloc.choose(_ctx(budget=0.0)) is None


def test_free_arm_affordable_at_zero_budget() -> None:
    free = FakeSubstrate("S0", cost=0.0)
    alloc = _greedy([free])
    assert alloc.choose(_ctx(budget=0.0)).name == "S0"


# ── tie-breaking ───────────────────────────────────────────────────────


def test_ties_break_toward_lower_durability() -> None:
    # Two free arms, both untried (value 0) ⇒ prefer the more reversible.
    ephem = FakeSubstrate("S0", durability=Durability.EPHEMERAL)
    weighty = FakeSubstrate("S3", durability=Durability.WEIGHTS)
    alloc = _greedy([weighty, ephem])
    assert alloc.choose(_ctx()).name == "S0"


# ── observe / value table ──────────────────────────────────────────────


def test_observe_updates_running_mean() -> None:
    alloc = _greedy([FakeSubstrate("S0")])
    for r in (1.0, 0.0, 1.0):
        alloc.observe(_ctx(), "S0", r)
    assert alloc.value_of("capability", "S0") == pytest.approx(2 / 3)
    assert alloc.count_of("capability", "S0") == 3


def test_observe_forwards_reward_to_substrate() -> None:
    s0 = FakeSubstrate("S0")
    alloc = _greedy([s0])
    result = s0.apply(_ctx())
    alloc.observe(_ctx(), "S0", 0.7, result=result)
    assert s0.observed == [0.7]


def test_observe_without_result_skips_substrate_hook() -> None:
    s0 = FakeSubstrate("S0")
    alloc = _greedy([s0])
    alloc.observe(_ctx(), "S0", 0.7)  # no result ⇒ table updates, hook skipped
    assert s0.observed == []
    assert alloc.value_of("capability", "S0") == pytest.approx(0.7)


def test_value_table_reports_mean_and_count() -> None:
    alloc = _greedy([FakeSubstrate("S0")])
    alloc.observe(_ctx("knowledge"), "S0", 0.5)
    alloc.observe(_ctx("knowledge"), "S0", 0.5)
    table = alloc.value_table()
    assert table[("knowledge", "S0")] == (pytest.approx(0.5), 2)


# ── allocate (choose + apply) ──────────────────────────────────────────


def test_allocate_chooses_and_applies() -> None:
    s0 = FakeSubstrate("S0")
    alloc = _greedy([s0])
    choice, result = alloc.allocate(_ctx())
    assert choice.name == "S0"
    assert result.applied is True
    assert result.substrate == "S0"


def test_allocate_returns_none_when_nothing_affordable() -> None:
    alloc = _greedy([FakeSubstrate("S3", cost=9.0)])
    assert alloc.allocate(_ctx(budget=0.0)) is None


# ── reporting: Fig 1 (distribution) + Fig 2 (timeline) ─────────────────


def test_selection_distribution_normalises() -> None:
    s0, s1 = FakeSubstrate("S0"), FakeSubstrate("S1", durability=Durability.SESSION)
    alloc = _greedy([s0, s1])
    # Warm-up then settle: 1×S0, 1×S1, then exploit S0 (give it the edge).
    for name, reward in (("S0", 1.0), ("S1", 0.0)):
        c = alloc.choose(_ctx())
        alloc.observe(_ctx(), c.name, reward)
    for _ in range(2):
        c = alloc.choose(_ctx())
        alloc.observe(_ctx(), c.name, 1.0)

    dist = alloc.selection_distribution()
    assert sum(dist.values()) == pytest.approx(1.0)
    # 4 picks total, S0 chosen 3× (warm-up + 2 exploits), S1 once.
    assert dist["S0"] == pytest.approx(0.75)
    assert dist["S1"] == pytest.approx(0.25)


def test_selection_distribution_is_per_cluster() -> None:
    alloc = _greedy([FakeSubstrate("S0")])
    alloc.choose(_ctx("knowledge"))
    alloc.choose(_ctx("capability"))
    assert alloc.selection_distribution("knowledge") == {"S0": 1.0}
    assert alloc.selection_counts("knowledge")["S0"] == 1


def test_timeline_records_every_decision() -> None:
    alloc = _greedy([FakeSubstrate("S0"), FakeSubstrate("S1",
                     durability=Durability.SESSION)])
    alloc.choose(_ctx())
    alloc.choose(_ctx("knowledge"))
    tl = alloc.timeline
    assert len(tl) == 2
    assert [r.index for r in tl] == [1, 2]
    assert tl[0].cluster == "capability"
    assert tl[1].cluster == "knowledge"


def test_empty_distribution_before_any_choice() -> None:
    alloc = _greedy([FakeSubstrate("S0")])
    assert alloc.selection_distribution() == {}


# ── exploration branch ─────────────────────────────────────────────────


def test_epsilon_one_always_explores() -> None:
    s0, s1 = FakeSubstrate("S0"), FakeSubstrate("S1", durability=Durability.SESSION)
    alloc = MyopicAllocator([s0, s1], epsilon=1.0, rng=random.Random(0))
    choice = alloc.choose(_ctx())
    assert choice.reason == "explore"
    assert choice.name in {"S0", "S1"}


# ── ValueEstimate unit ─────────────────────────────────────────────────


def test_value_estimate_incremental_mean() -> None:
    v = ValueEstimate()
    for r in (1.0, 0.0, 1.0, 0.0):
        v.update(r)
    assert v.count == 4
    assert v.mean == pytest.approx(0.5)
