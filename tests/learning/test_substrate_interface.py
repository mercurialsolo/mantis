"""Contract tests for the LearningSubstrate interface.

These pin the shape the allocator depends on: a substrate is anything that
carries a ``name`` and round-trips ``cost_estimate → apply → observe``. We
verify the contract with a recording fake rather than a live rung, so the
interface can be exercised before any real substrate exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mantis_agent.learning import (
    Durability,
    LearningSubstrate,
    SubstrateContext,
    SubstrateResult,
)


@dataclass
class RecordingSubstrate:
    """Minimal in-memory rung that records what the allocator did to it.

    Implements the :class:`LearningSubstrate` protocol structurally (no
    inheritance) — which is exactly how the real adapters are written.
    """

    name: str = "mock"
    unit_cost: float = 0.05
    durability: Durability = Durability.EPHEMERAL
    observed: list[tuple[str, float]] = field(default_factory=list)

    def cost_estimate(self, context: SubstrateContext) -> float:
        return self.unit_cost

    def apply(self, context: SubstrateContext) -> SubstrateResult:
        # Decline cheaply when the budget can't cover the spend.
        if context.budget_remaining < self.unit_cost:
            return SubstrateResult(
                substrate=self.name, applied=False, notes="budget too tight"
            )
        return SubstrateResult(
            substrate=self.name,
            applied=True,
            dollars_spent=self.unit_cost,
            durability=self.durability,
            delta_artifacts={"hint": f"for {context.task_id}"},
        )

    def observe(
        self, context: SubstrateContext, result: SubstrateResult, reward: float
    ) -> None:
        self.observed.append((context.task_id, reward))


def _ctx(**kw: Any) -> SubstrateContext:
    base: dict[str, Any] = {"task_id": "bt01-0", "budget_remaining": 1.0}
    base.update(kw)
    return SubstrateContext(**base)


def test_recording_substrate_satisfies_protocol() -> None:
    sub = RecordingSubstrate()
    # runtime_checkable Protocol — structural typing, no base class needed.
    assert isinstance(sub, LearningSubstrate)


def test_cost_estimate_is_cheap_scalar() -> None:
    sub = RecordingSubstrate(unit_cost=0.05)
    cost = sub.cost_estimate(_ctx())
    assert isinstance(cost, float)
    assert cost == 0.05


def test_apply_produces_a_result_with_spend_and_artifacts() -> None:
    sub = RecordingSubstrate(unit_cost=0.05)
    result = sub.apply(_ctx(task_id="bt01-7"))
    assert result.applied is True
    assert result.substrate == "mock"
    assert result.dollars_spent == 0.05
    assert result.delta_artifacts == {"hint": "for bt01-7"}


def test_apply_declines_when_budget_too_tight() -> None:
    sub = RecordingSubstrate(unit_cost=0.50)
    result = sub.apply(_ctx(budget_remaining=0.10))
    assert result.applied is False
    assert result.dollars_spent == 0.0  # declining costs nothing
    assert "budget" in result.notes


def test_observe_folds_reward_back_in() -> None:
    sub = RecordingSubstrate()
    ctx = _ctx(task_id="bt01-3")
    result = sub.apply(ctx)
    sub.observe(ctx, result, reward=0.42)
    assert sub.observed == [("bt01-3", 0.42)]


def test_substrate_result_defaults_are_conservative() -> None:
    # A bare result is un-applied-cost-free-ephemeral: the safe default a
    # rung returns when it has nothing to contribute.
    r = SubstrateResult(substrate="s0", applied=False)
    assert r.dollars_spent == 0.0
    assert r.durability is Durability.EPHEMERAL
    assert r.delta_artifacts == {}
    assert r.notes == ""


def test_substrate_context_defaults() -> None:
    ctx = SubstrateContext(task_id="t")
    assert ctx.cluster is None
    assert ctx.failure_signal == {}
    assert ctx.budget_remaining == 0.0
    assert ctx.extras == {}


def test_durability_orders_cheapest_first() -> None:
    order = list(Durability)
    assert order == [
        Durability.EPHEMERAL,
        Durability.SESSION,
        Durability.POLICY,
        Durability.WEIGHTS,
    ]
