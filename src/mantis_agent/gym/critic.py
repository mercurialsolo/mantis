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
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from ..plan_decomposer import MicroIntent

if TYPE_CHECKING:
    from .checkpoint import StepResult
    from .micro_runner import MicroPlanRunner
    from .run_executor import RunState

logger = logging.getLogger(__name__)


# How many ``wrong_target`` failures on the same step before the
# frontier-model capability fires. Set to 2 so the first failure
# still goes through the existing recovery loop (cheap rewriter,
# label-match retries) — only the persistent miss escalates to a
# Claude call.
_FRONTIER_WRONG_TARGET_THRESHOLD: int = 2


def _frontier_enabled() -> bool:
    """``MANTIS_CRITIC_FRONTIER=enabled`` flips the opt-in.

    Default off so deployments that don't want the extra Claude
    spend behave exactly as before. The lone existing rule
    (``navigate_back`` + ``brain_loop_exhausted``) keeps working
    regardless of this gate.
    """
    return os.environ.get("MANTIS_CRITIC_FRONTIER", "").strip().lower() == "enabled"


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


@dataclass(frozen=True)
class ReplaceStep:
    """Critic directive: replace the step at the current
    ``state.step_index`` IN PLACE — does NOT shift subsequent steps.
    The new step occupies the same plan slot and runs on the next
    loop iteration.

    Used when the frontier-model observer (#435 item 8) decides the
    current step's structure is wrong rather than its execution — a
    sidebar-link click cascade should become a direct ``navigate``
    to the filtered URL, for example. The runner records the
    replacement as a healing event and resets the per-step retry
    history so the new step starts with a clean slate.
    """

    intent: str
    step_type: str
    reason: str
    params: dict[str, Any] | None = None
    hints: dict[str, Any] | None = None


# Union of every directive type ``observe_step`` may return — used
# in apply_directive's dispatch and for type hints on the public
# surface.
Directive = Union[InsertStep, ReplaceStep]


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
    ) -> Optional[Directive]:
        """Called after every step dispatch + recovery handling.

        ``recovery_continued`` is True when the recovery policy
        decided retry/advance (i.e. ``_handle_failure`` returned
        True), False on halt. The critic only emits directives when
        the run is continuing — there's no point mutating the plan
        ahead of a halted run.

        Returns one of:

        * :class:`InsertStep` — splice a NEW step at the current
          index (existing capability: navigate_back + brain loop).
        * :class:`ReplaceStep` — replace the step in place (new in
          #435 item 8: frontier-model observer on persistent
          ``wrong_target`` failures).
        * ``None`` — let the recovery policy continue as-is.

        Capabilities are checked cheapest-first. Rule-based
        capabilities run unconditionally; the frontier-model
        capability is opt-in via ``MANTIS_CRITIC_FRONTIER``.
        """
        if not recovery_continued:
            return None
        if step_result.success:
            return None

        # Rule-based: navigate_back loop exhaustion → direct nav.
        rule_directive = self._maybe_recover_navigate_back(step, step_result)
        if rule_directive is not None:
            return rule_directive

        # Frontier-model: persistent wrong_target → ask Claude.
        return self._maybe_frontier_recover_wrong_target(
            plan, state, step, step_result,
        )

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


    # ── Frontier-model capability (#435 item 8) ─────────────────────────

    def _maybe_frontier_recover_wrong_target(
        self,
        plan: Any,
        state: "RunState",
        step: MicroIntent,
        step_result: "StepResult",
    ) -> Optional[Directive]:
        """Ask the frontier model (Claude via :mod:`agentic_recovery`)
        for a mid-run plan change when a ``wrong_target`` failure
        pattern persists.

        Trigger conditions (all generic — no plan / URL / domain
        content reads in):

        * ``MANTIS_CRITIC_FRONTIER=enabled`` (opt-in).
        * ``step_result.failure_class == "wrong_target"``.
        * At least :data:`_FRONTIER_WRONG_TARGET_THRESHOLD`
          ``wrong_target`` records in ``_step_failure_history[step_index]``
          (skip the first miss — that's still in the cheap retry zone).
        * The frontier model hasn't already been consulted for this
          step (tracked on ``runner._critic_frontier_fired_steps``).
        * The shared recovery budget (per-step + per-run, from
          :mod:`agentic_recovery`) isn't exhausted.

        Mode mapping:

        * ``add_hint`` — append to ``_recovery_hints[step_index]`` and
          return ``None`` (no directive — the next retry's prompt
          carries the hint and we don't disturb the plan).
        * ``edit_step`` — :class:`ReplaceStep` with the model's
          ``intent`` / ``type`` / ``params``. Missing fields preserve
          the original step's values.
        * ``insert_steps`` — first inserted step becomes an
          :class:`InsertStep` directive (MVP simplification — multi-
          step inserts come from the terminal-failure path that
          already supports splicing).
        * ``halt`` — return ``None`` so the recovery loop's terminal
          path handles it.

        On any fallback (no API key, API error, decision schema
        violation, exception) the method returns ``None`` and the
        existing recovery flow continues unchanged.
        """
        if not _frontier_enabled():
            return None
        failure_class = str(getattr(step_result, "failure_class", "") or "")
        if failure_class != "wrong_target":
            return None

        runner = self.runner
        step_index = int(state.step_index)
        history = (
            runner._step_failure_history.get(step_index, [])
            if hasattr(runner, "_step_failure_history") else []
        )
        wrong_target_count = sum(
            1 for r in history
            if isinstance(r, dict) and r.get("kind") == "wrong_target"
        )
        if wrong_target_count < _FRONTIER_WRONG_TARGET_THRESHOLD:
            return None

        # Already consulted Claude for this step? Don't double-spend.
        fired = getattr(runner, "_critic_frontier_fired_steps", None)
        if not isinstance(fired, set):
            fired = set()
            runner._critic_frontier_fired_steps = fired
        if step_index in fired:
            return None

        # Reuse the existing recovery budget pool so the critic's
        # frontier call and step_recovery's terminal call share one
        # pot — keeps the total Claude spend bounded by the same
        # per-step / per-run caps.
        per_step_dict = getattr(runner, "_recovery_attempts_per_step", None)
        total_attempts = getattr(runner, "_total_recovery_attempts", None)
        if not isinstance(per_step_dict, dict) or not isinstance(total_attempts, int):
            logger.debug(
                "  [critic-frontier] runner missing budget trackers — skip"
            )
            return None
        try:
            from ..agentic_recovery import (
                DEFAULT_MAX_RECOVERIES_PER_RUN,
                DEFAULT_MAX_RECOVERIES_PER_STEP,
            )
        except Exception as exc:  # noqa: BLE001 — never break runs
            logger.debug("  [critic-frontier] agentic_recovery import failed: %s", exc)
            return None
        if per_step_dict.get(step_index, 0) >= DEFAULT_MAX_RECOVERIES_PER_STEP:
            return None
        if total_attempts >= DEFAULT_MAX_RECOVERIES_PER_RUN:
            return None

        # Capture the post-failure screenshot for Claude's analysis.
        # Same shape ``step_recovery._try_agentic_recovery`` uses.
        env = getattr(runner, "env", None)
        screenshot = None
        if env is not None and hasattr(env, "screenshot"):
            try:
                screenshot = env.screenshot()
            except Exception as exc:  # noqa: BLE001
                logger.debug("  [critic-frontier] env.screenshot failed: %s", exc)

        plan_context = [
            f"step {i}: type={s.type}, intent={s.intent[:60]}"
            for i, s in enumerate(getattr(plan, "steps", []) or [])
        ]
        failure_data = (
            f"failure_class={failure_class}; "
            f"prior_wrong_target={wrong_target_count}; "
            f"data={(step_result.data or '')[:160]}"
        )

        try:
            from ..agentic_recovery import analyse_failure_and_recover
            decision = analyse_failure_and_recover(
                step=step,
                failure_data=failure_data,
                screenshot=screenshot,
                plan_context=plan_context,
                attempts=wrong_target_count,
            )
        except Exception as exc:  # noqa: BLE001 — never break runs
            logger.warning("  [critic-frontier] Claude call raised: %s", exc)
            return None

        # Mark fired BEFORE applying — even if Claude returned halt /
        # invalid, we don't want to retry the consultation on the same
        # step (the policy budget exists for that).
        fired.add(step_index)
        # Increment the shared budget counters so the terminal path
        # sees this consultation count.
        per_step_dict[step_index] = per_step_dict.get(step_index, 0) + 1
        runner._total_recovery_attempts = total_attempts + 1

        if decision is None:
            logger.info(
                "  [critic-frontier] step %d: Claude returned no decision "
                "(API/key/parse fallback) — recovery continues unchanged",
                step_index,
            )
            return None
        if decision.mode == "halt":
            return None
        if decision.mode == "add_hint" and decision.hint:
            from . import recovery_hints
            recovery_hints.add_hint(runner, step_index, decision.hint)
            logger.warning(
                "  [critic-frontier] step %d: add_hint — %s",
                step_index, decision.hint[:120],
            )
            return None
        if decision.mode == "edit_step":
            edited = decision.edited_step or {}
            new_intent = (edited.get("intent") or step.intent or "").strip()
            new_type = (edited.get("type") or step.type or "").strip()
            new_params = dict(edited.get("params") or step.params or {})
            if not new_intent or not new_type:
                return None
            logger.warning(
                "  [critic-frontier] step %d: edit_step → ReplaceStep "
                "(type=%s, intent=%s)",
                step_index, new_type, new_intent[:80],
            )
            return ReplaceStep(
                intent=new_intent,
                step_type=new_type,
                params=new_params,
                hints=dict(getattr(step, "hints", {}) or {}),
                reason=(
                    f"frontier critic replaced step on persistent "
                    f"wrong_target ({wrong_target_count} prior): "
                    f"{decision.reasoning[:120]}"
                ),
            )
        if decision.mode == "insert_steps":
            steps = decision.inserted_steps or []
            if not steps:
                return None
            first = steps[0]
            intent = str(first.get("intent") or "").strip()
            step_type = str(first.get("type") or "").strip()
            if not intent or not step_type:
                return None
            logger.warning(
                "  [critic-frontier] step %d: insert_steps[0] → InsertStep "
                "(type=%s, intent=%s)",
                step_index, step_type, intent[:80],
            )
            return InsertStep(
                intent=intent,
                step_type=step_type,
                params=dict(first.get("params") or {}),
                reason=(
                    f"frontier critic insert on persistent wrong_target "
                    f"({wrong_target_count} prior): "
                    f"{decision.reasoning[:120]}"
                ),
            )
        return None


def apply_directive(
    runner: "MicroPlanRunner",
    plan: Any,
    state: "RunState",
    directive: Directive,
) -> None:
    """Mutate ``plan.steps`` per a critic-emitted directive.

    Dispatches on directive type:

    * :class:`InsertStep` — splice a new step at ``state.step_index``
      so it runs on the next loop iteration.
    * :class:`ReplaceStep` — replace the step at ``state.step_index``
      in place (no shift). Resets ``_step_failure_history`` for the
      slot so the new step doesn't inherit the old one's retry
      pressure.

    Stays separate from :class:`ExecutionCritic` so the critic stays
    pure (returns directives, doesn't mutate state) — the runner
    owns plan mutation.
    """
    if isinstance(directive, InsertStep):
        _apply_insert_step(runner, plan, state, directive)
        return
    if isinstance(directive, ReplaceStep):
        _apply_replace_step(runner, plan, state, directive)
        return


def _apply_insert_step(
    runner: "MicroPlanRunner",
    plan: Any,
    state: "RunState",
    directive: InsertStep,
) -> None:
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


def _apply_replace_step(
    runner: "MicroPlanRunner",
    plan: Any,
    state: "RunState",
    directive: ReplaceStep,
) -> None:
    idx = int(state.step_index)
    if idx < 0 or idx >= len(plan.steps):
        return
    original = plan.steps[idx]
    original_type = str(getattr(original, "type", "") or "")
    new_step = MicroIntent(
        intent=directive.intent,
        type=directive.step_type,
        params=dict(directive.params or {}),
        hints=dict(directive.hints or {}),
        budget=int(getattr(original, "budget", 3) or 3),
        required=bool(getattr(original, "required", False)),
        section=str(getattr(original, "section", "") or ""),
        gate=bool(getattr(original, "gate", False)),
    )
    plan.steps[idx] = new_step

    # Reset the per-step retry history so the replacement runs with a
    # clean slate. Inheriting the original step's failure pattern
    # would defeat the point of replacing the step.
    if hasattr(runner, "_step_failure_history"):
        try:
            runner._step_failure_history.pop(idx, None)
        except Exception:  # noqa: BLE001
            pass
    # Same for the recovery hint accumulator — the new step likely
    # has a different label / shape and the old hints don't apply.
    if hasattr(runner, "_recovery_hints"):
        try:
            runner._recovery_hints.pop(idx, None)
        except Exception:  # noqa: BLE001
            pass

    from . import healing_events
    healing_events.record_replace_step(
        runner,
        step_index=idx,
        original_type=original_type,
        new_intent=directive.intent,
        new_type=directive.step_type,
        reason=directive.reason,
    )
    logger.warning(
        "  [critic] replacing step %d in place: %s → %s (%s)",
        idx, original_type or "?",
        directive.step_type, directive.reason[:120],
    )


__all__ = [
    "ExecutionCritic",
    "InsertStep",
    "ReplaceStep",
    "Directive",
    "apply_directive",
]
