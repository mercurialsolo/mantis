"""S0 — retrieval / context substrate (the cheapest rung).

The bottom of the ladder: stamp grounding hints the agent already learned
onto the plan before it runs, so the brain re-uses a known visual anchor
instead of re-discovering it from pixels. It wraps the trajectory hint
memory (``gym/hint_memory.py``) — specifically :func:`apply_hint_overlay`,
which had no caller until now.

Why this is S0 (PLAN §3, the ``knowledge`` cluster's expected winner):

* **Cheapest** — overlaying a stored anchor is a dict lookup. No model
  call, no network, so :meth:`cost_estimate` is ``0.0``.
* **Most reversible** — the effect is one ``preferred_target_description``
  hint on the current plan and nothing else; it evaporates when the run
  ends. That is :class:`Durability.EPHEMERAL`.

CUA purity (``feedback_cua_no_dom_access.md``): the hints this rung
replays are visual anchors Holo3 *saw* (``elv_text`` strings, screen
offsets), never DOM selectors — that invariant is enforced upstream in
``hint_memory`` and inherited here unchanged.
"""

from __future__ import annotations

from typing import Any

from ...gym.hint_memory import HintStore, NullHintStore, apply_hint_overlay
from .base import Durability, SubstrateContext, SubstrateResult


class RetrievalSubstrate:
    """S0 — overlay stored grounding anchors onto the plan about to run.

    The substrate carries the :class:`HintStore` backend (dependency-
    injected so tests use an in-memory store and prod uses the disk /
    Modal-dict one). Per-task specifics arrive through
    ``context.extras``:

    * ``plan`` — the live plan object whose steps get the overlay. Absent
      in a pure cost-probe; present when the allocator actually applies
      this rung. We never read it in :meth:`cost_estimate`.
    * ``plan_signature`` — the 12-char plan hash that keys the store.
    * ``start_url`` — seeds the URL-pattern axis of the hint key.
    """

    durability = Durability.EPHEMERAL

    def __init__(
        self, store: HintStore | None = None, *, name: str = "S0_retrieval",
    ) -> None:
        # NullHintStore default keeps the rung safe to construct with no
        # backend wired — apply() then simply finds nothing to overlay.
        self.store: HintStore = store or NullHintStore()
        self.name = name

    def cost_estimate(self, context: SubstrateContext) -> float:  # noqa: ARG002
        """Always free — a hint overlay is a store read, no model call."""
        return 0.0

    def apply(self, context: SubstrateContext) -> SubstrateResult:
        plan: Any = context.extras.get("plan")
        plan_signature = str(context.extras.get("plan_signature", "") or "")
        start_url = str(context.extras.get("start_url", "") or "")

        if plan is None or not plan_signature:
            return SubstrateResult(
                substrate=self.name,
                applied=False,
                dollars_spent=0.0,
                durability=self.durability,
                notes="no plan / plan_signature in context.extras — nothing to overlay",
            )

        n = apply_hint_overlay(
            plan, store=self.store, plan_signature=plan_signature, start_url=start_url,
        )
        return SubstrateResult(
            substrate=self.name,
            applied=n > 0,
            dollars_spent=0.0,
            durability=self.durability,
            delta_artifacts={"hints_applied": n, "plan_signature": plan_signature},
            notes=(
                f"overlaid {n} grounding hint(s)"
                if n
                else "no matching anchors in store"
            ),
        )

    def observe(
        self, context: SubstrateContext, result: SubstrateResult, reward: float,
    ) -> None:
        """No-op — hint *recording* is the run executor's job (it writes
        anchors back to the store on step success, see
        ``hint_memory.record_hint_if_eligible``). The allocator's reward
        signal doesn't feed this rung's internal state."""
        return None


__all__ = ["RetrievalSubstrate"]
