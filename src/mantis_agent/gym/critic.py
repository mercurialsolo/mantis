"""Run-time observer that proposes corrections after each step.

Phase C of epic #377. Skeleton + one concrete capability:

* **Skeleton** (v1): :class:`ExecutionCritic` runs after every step
  via :meth:`observe_step`. Reads ``failure_class`` + step context +
  recovery-policy outcome and decides whether to emit a directive.
  Future phases plug additional capabilities (Claude-vision obstacle
  detection, brain-ladder policy, intent-rewriter unification) into
  the same hook.

* **One concrete capability**: when a ``navigate_back`` step fails
  with ``failure_class=brain_loop_exhausted`` (the brain spent its
  budget trying to drive the browser back, typically on SPAs whose
  history stack is broken), emit
  :class:`InsertStep` with type=``navigate`` to the runner's
  ``_results_base_url``. The runner inserts the step ahead of the
  next iteration, restoring the agent to the expected page without
  relying on the browser's back button.

Why generic: every signal the critic reads is framework-level
(``step.type``, ``failure_class``, ``runner._results_base_url``).
Zero plan / URL / domain content reads in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ..plan_decomposer import MicroIntent

if TYPE_CHECKING:
    from .checkpoint import StepResult
    from .micro_runner import MicroPlanRunner
    from .run_executor import RunState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InsertStep:
    """Critic directive: splice a new step into the plan at the
    current ``state.step_index`` (shifting subsequent steps right).
    The runner records the insertion as a healing event and the
    inserted step runs next.
    """

    intent: str
    step_type: str
    reason: str
    params: dict[str, Any] | None = None


class ExecutionCritic:
    """Observer that proposes corrections after each step.

    v1 capabilities:
    - Detect ``navigate_back`` + ``brain_loop_exhausted`` → emit
      ``InsertStep`` for a direct ``navigate`` to the results base URL.

    Future capabilities to plug in via the same ``observe_step`` hook:
    - Claude-vision modal / banner detection → ``InsertStep`` for
      ``dismiss_overlay``.
    - Cross-attempt URL drift → directive to the IntentRewriter with
      drift context.
    - Brain-ladder policy: when N failures across all steps have
      escalated to Holo3 without progress, force fallback to Claude.

    The critic does NOT replace ``_recovery_policy`` or the existing
    rewriters. It runs AFTER ``_handle_failure`` and adds capabilities
    that the recovery policy doesn't cover.
    """

    def __init__(self, runner: "MicroPlanRunner") -> None:
        self.runner = runner

    def observe_step(
        self,
        plan: Any,  # MicroPlan — typed via TYPE_CHECKING to avoid import cycle
        state: "RunState",
        step: MicroIntent,
        step_result: "StepResult",
        *,
        recovery_continued: bool,
    ) -> Optional[InsertStep]:
        """Called after every step dispatch + recovery handling.

        ``recovery_continued`` is True when the recovery policy
        decided retry/advance (i.e. ``_handle_failure`` returned
        True), False on halt. The critic only emits directives when
        the run is continuing — there's no point inserting a step
        ahead of a halted run.

        Returns an :class:`InsertStep` directive or ``None``. The
        runner consumes the directive and mutates ``plan.steps``;
        the critic's job is the decision, not the mutation.
        """
        if not recovery_continued:
            return None
        if step_result.success:
            return None

        return self._maybe_recover_navigate_back(step, step_result)

    # ── Capabilities ────────────────────────────────────────────────────

    def _maybe_recover_navigate_back(
        self,
        step: MicroIntent,
        step_result: "StepResult",
    ) -> Optional[InsertStep]:
        """When ``navigate_back`` exhausts the brain budget, the
        browser is stuck on the wrong URL. Insert a direct
        ``navigate`` to the results base URL — agents don't need
        the back button if we can name the destination.

        Generic: the trigger is ``step.type == "navigate_back"`` AND
        ``failure_class == "brain_loop_exhausted"`` AND the runner
        has a non-empty ``_results_base_url``. Any plan with a
        navigate_back step on any site benefits.
        """
        if step.type != "navigate_back":
            return None
        if (step_result.failure_class or "") != "brain_loop_exhausted":
            return None
        base_url = getattr(self.runner, "_results_base_url", "") or ""
        if not base_url:
            # No base URL recorded — without a destination, we can't
            # propose an alternative. Fall through to whatever the
            # recovery policy decided.
            return None
        return InsertStep(
            intent=f"Navigate to {base_url}",
            step_type="navigate",
            reason=(
                "navigate_back hit brain_loop_exhausted — replacing "
                "back-button recovery with a direct navigate to the "
                "results base URL"
            ),
            params={"url": base_url},
        )


def apply_directive(
    runner: "MicroPlanRunner",
    plan: Any,
    state: "RunState",
    directive: InsertStep,
) -> None:
    """Mutate ``plan.steps`` per a critic-emitted ``InsertStep``.

    Splices the inserted step at ``state.step_index`` so it runs on
    the next loop iteration. Records the action as a healing event
    on the runner for audit.

    Stays separate from :class:`ExecutionCritic` so the critic stays
    pure (returns directives, doesn't mutate state) — the runner
    owns plan mutation.
    """
    if not isinstance(directive, InsertStep):
        return
    new_step = MicroIntent(
        intent=directive.intent,
        type=directive.step_type,
        params=dict(directive.params or {}),
        budget=3,
        required=False,
        section="recovery",
    )
    insert_at = max(0, min(state.step_index, len(plan.steps)))
    plan.steps.insert(insert_at, new_step)

    from . import healing_events
    healing_events.record_insert_step(
        runner,
        after_step_index=insert_at - 1,
        inserted_intent=directive.intent,
        inserted_type=directive.step_type,
        reason=directive.reason,
    )
    logger.warning(
        "  [critic] inserting recovery step %d: %s (%s)",
        insert_at, directive.intent[:80], directive.reason[:80],
    )


__all__ = ["ExecutionCritic", "InsertStep", "apply_directive"]
