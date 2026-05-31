"""The uniform action space the Learning Allocator chooses over.

Continual improvement, in the Learning Allocator framing, is a
budget-constrained choice over a *ladder* of learning mechanisms ordered
by cost and durability:

    S0 retrieval / context   — cheap, reversible, lasts one task
    S1 exemplar replay       — cheap, lasts a session
    S2 skill / macro synth    — moderate, becomes reusable policy
    S3 weight consolidation  — expensive, baked into the model

The allocator must score every rung with the *same* currency
(``R = Δscore − λ·cost``) before paying, then actually pay for the one it
picks. That only works if every rung presents the same four-method shape.
This module defines that shape — :class:`LearningSubstrate` — plus the
two value types that cross the boundary (:class:`SubstrateContext` in,
:class:`SubstrateResult` out) and the :class:`Durability` axis that orders
the ladder. The concrete rungs live in sibling modules
(``retrieval.py`` … ``consolidation.py``) and wrap mechanisms that already
exist in the repo; they are added per the P2–P4 issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Durability(str, Enum):
    """How long a substrate's effect persists — the ladder's vertical axis.

    Ordered cheapest/most-reversible first. The allocator uses durability
    both to break cost ties (prefer the reversible rung when scores match)
    and to justify the non-stationarity claim: as the agent matures the
    optimal rung drifts *down* this list (ephemeral hint → durable weights).
    """

    EPHEMERAL = "ephemeral"   # S0: applies to the current task only
    SESSION = "session"       # S1: persists within a run / session
    POLICY = "policy"         # S2: a reusable macro across tasks
    WEIGHTS = "weights"       # S3: consolidated into model parameters


@dataclass
class SubstrateContext:
    """Read-only snapshot the allocator hands a substrate when it picks it.

    A substrate reads this to decide *what* to retrieve / replay / synthesize
    / distill. It is deliberately small and free of live handles — the
    cost-estimate path must stay cheap enough to call for every candidate
    rung on every task.
    """

    task_id: str
    cluster: str | None = None
    """Failure cluster this task belongs to: ``knowledge`` | ``capability``
    | ``policy``. Drives which rung is expected to win (see PLAN §3)."""

    failure_signal: dict[str, Any] = field(default_factory=dict)
    """Coarse signal about the recurring failure (e.g. a ``failure_class``
    tag, the offending step index, a layout-drift hint)."""

    budget_remaining: float = 0.0
    """Dollars left in the hard budget ``B``. A substrate may decline (return
    an un-applied result) when its cost would blow the remaining budget."""

    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubstrateResult:
    """The outcome of applying one substrate to one task.

    ``delta_artifacts`` is whatever the substrate produced — a hint string,
    an exemplar trace id, a synthesized playbook, a checkpoint path. The
    allocator treats it opaquely; the run executor knows how to consume it.
    ``applied=False`` means the substrate chose not to act (nothing to
    retrieve, budget too tight) — distinct from acting and failing, and it
    should incur ~no cost.
    """

    substrate: str
    applied: bool
    dollars_spent: float = 0.0
    durability: Durability = Durability.EPHEMERAL
    delta_artifacts: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@runtime_checkable
class LearningSubstrate(Protocol):
    """One rung of the cost/durability-ordered ladder S0→S3.

    The allocator needs exactly four things from a rung: a stable
    :attr:`name` to key its value estimates on, a *cheap* :meth:`cost_estimate`
    so the bandit can score ``R = Δscore − λ·cost`` before paying,
    :meth:`apply` to actually spend compute and produce a delta, and
    :meth:`observe` to fold the realised reward back into whatever internal
    state the rung keeps. Keeping the surface this small is what lets a
    free hint cache and an expensive distillation run be compared head to
    head by the same bandit.
    """

    name: str

    def cost_estimate(self, context: SubstrateContext) -> float:
        """Expected dollar cost of applying this substrate to ``context``.

        Must be cheap — no network or model calls. The allocator invokes it
        for every candidate rung on every task to rank them before paying.
        """
        ...

    def apply(self, context: SubstrateContext) -> SubstrateResult:
        """Spend compute and produce a delta artifact — the expensive call.

        Invoked only for the rung the allocator chose. Implementations should
        return ``applied=False`` (cheaply) rather than raise when there is
        nothing to do for this ``context``.
        """
        ...

    def observe(
        self, context: SubstrateContext, result: SubstrateResult, reward: float
    ) -> None:
        """Fold the realised reward back in — the online-update hook.

        Called after the run executor consumes ``result`` and the reward
        channels score it. Stateless rungs may no-op; rungs that maintain
        their own caches or value estimates update them here.
        """
        ...
