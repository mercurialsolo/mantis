"""Step-recovery decision types — RecoveryAction enum + RecoveryDecision.

Phase 1.3 of EPIC #161 (refactor MicroPlanRunner into composable
modules). Defines the *types* that the eventual ``StepRecoveryPolicy``
will return so the runner's main loop can switch on a structured
decision instead of the 230-line nested ``if/elif`` block currently in
``MicroPlanRunner.run()``.

This commit is **types-only**: no dispatch logic, no behavior change.
The runner still owns the failure-handling code path. A follow-up phase
will:

1. Extract the decision logic into ``StepRecoveryPolicy.dispatch(step,
   result, ctx) -> RecoveryDecision`` — a pure function over (step
   metadata, step result data, retry-state context).
2. Replace the runner's ``if step.required: ... elif step.gate: ...
   elif step.type == "navigate": ...`` cascade with a switch on
   ``RecoveryDecision.action`` that applies side effects (sleep,
   ``_reverse_step``, ``persist``, ``self.env.step(Escape)``, etc.).

Why split the type definitions from the dispatch:

- Once these types exist, callers (Prometheus emitters, trace
  recorders, future plan-authoring product validators in #155) can
  consume the structured decision without depending on the runner.
- A standalone enum is cheap to test and review; the dispatch swap is
  the risky bisectable step that wants its own PR.

Decision-action vocabulary picked to map 1:1 onto the existing recovery
branches in ``run()`` so the eventual lift is mechanical:

================  ================================================
RecoveryAction    Existing ``run()`` branch it captures
================  ================================================
SUCCEED           Step succeeded — advance step_index
RETRY             ``step.required``, ``page_blocked``, ``scan_error``,
                  ``cloudflare`` gate-retry
SKIP              Filter, scroll-no-done, extract_*-failed,
                  click-failed escape-out paths
REVERSE_AND_SKIP  Generic-failure fallback (``_reverse_step`` + skip)
JUMP_TO_TYPE      ``page_exhausted`` (jump to paginate or loop),
                  click-failed (jump to loop), DUPLICATE handling
RELOAD_AND_RETRY  ``page_blocked`` after retry budget — filtered URL
                  reload + retry
HALT              ``required`` retries exhausted, gate failure,
                  navigate failure, page_blocked after reload
================  ================================================
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..actions import Action, ActionType

logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """Coarse decision the runner should take after a step result.

    Every enum value maps to one branch of the legacy nested if/elif in
    ``MicroPlanRunner.run()``. Side-effect parameters (wait time, halt
    reason, jump target step type) ride on :class:`RecoveryDecision`.
    """

    SUCCEED = "succeed"
    RETRY = "retry"
    SKIP = "skip"
    REVERSE_AND_SKIP = "reverse_and_skip"
    JUMP_TO_TYPE = "jump_to_type"
    RELOAD_AND_RETRY = "reload_and_retry"
    HALT = "halt"


@dataclass(frozen=True)
class RecoveryDecision:
    """Structured outcome of ``StepRecoveryPolicy.dispatch``.

    The runner applies this by reading ``action`` and the matching
    parameter fields. ``frozen=True`` so a decision can't mutate after
    the policy returns it — the runner is the only side-effect agent.

    Field semantics (only the fields relevant to ``action`` carry
    meaning; the rest stay at their defaults):

    - ``halt_reason``: passed to ``persist(step_index, status=...,
      halt_reason=halt_reason)``. Always set on RETRY / SKIP /
      REVERSE_AND_SKIP / RELOAD_AND_RETRY / HALT so the on-disk
      checkpoint records *why*.
    - ``wait_seconds``: ``time.sleep(wait_seconds)`` before retrying;
      0.0 = no sleep. Used by RETRY / RELOAD_AND_RETRY / JUMP_TO_TYPE.
    - ``jump_target_types``: tuple of step types to fast-forward to,
      checked in order against ``plan.steps[step_index+1:]``. Empty
      means "advance one step". Used by JUMP_TO_TYPE.
    - ``reset_loop_counters``: when True, set every entry in
      ``loop_counters`` to ``reset_loop_counters_value`` (used by
      paginate-failed → set all to 999_999 to exhaust the outer loop;
      and paginate-success → reset inner loops to 0).
    - ``halt_reason_message``: optional human-readable message for
      ``print(...)`` (used today on the HALT branches before ``break``).
    - ``log_message``: free-text log line to emit before applying the
      decision. Empty = no extra log.
    """

    action: RecoveryAction
    halt_reason: str = ""
    wait_seconds: float = 0.0
    jump_target_types: tuple[str, ...] = ()
    reset_loop_counters: bool = False
    reset_loop_counters_value: int = 0
    halt_reason_message: str = ""
    log_message: str = ""
    log_level: str = "info"  # one of: debug, info, warning, error

    # Carrier for branch-specific extras the policy needs to communicate
    # without growing the dataclass — e.g., ``page_blocked`` retries that
    # need a sentinel key, or anti-bot retry that needs to know which
    # navigate step to re-run. Kept as a free-form dict to avoid locking
    # the shape before the dispatch lift lands.
    extras: dict = field(default_factory=dict)


# Sentinel decisions for the simple no-arg cases — micro-optimizes
# allocation churn on the hot path. The dispatch can return these
# directly when no parameters are needed; tests assert identity.
DECISION_SUCCEED = RecoveryDecision(action=RecoveryAction.SUCCEED)


def _track_plan_rewrite_candidate(
    runner: Any, *, step_index: int, edited: dict,
    decision: Any, original_step: Any,
) -> None:
    """Stash a URL-rewrite candidate on the runner for terminal accounting.

    Plan-evolution Phase 2 (#706): consumed by
    ``MicroPlanRunner.run`` → ``finalize_run_outcomes`` at run terminal
    to apply the promotion / demotion gates. Errors here are debug-only
    — store accounting must never block recovery.
    """
    try:
        from ..recipes.plan_evolution_store import StepRewrite
    except Exception:  # noqa: BLE001
        return
    # Extract source from the decision's reasoning ("rewrite_url:<source>: ...")
    source = "pattern_transform"
    reasoning = str(getattr(decision, "reasoning", "") or "")
    if reasoning.startswith("rewrite_url:"):
        head = reasoning.split(" ", 1)[0]  # "rewrite_url:<source>"
        source = head.split(":", 1)[1] if ":" in head else source
    confidence = 0.6
    if "(confidence=" in reasoning:
        try:
            tail = reasoning.split("(confidence=", 1)[1]
            confidence = float(tail.split(";", 1)[0].strip())
        except (ValueError, IndexError):
            pass
    rewrite = StepRewrite(
        step_index=step_index,
        original={
            "intent": getattr(original_step, "intent", "") or "",
            "type": getattr(original_step, "type", "") or "",
            "params": dict(getattr(original_step, "params", {}) or {}),
        },
        rewritten=dict(edited),
        source=source if source in (
            "pattern_transform", "page_links", "web_search",
            "brain_proposal", "manual",
        ) else "pattern_transform",
        confidence=confidence,
        scope="workflow",
        status="candidate",
    )
    if not hasattr(runner, "_applied_plan_rewrites"):
        runner._applied_plan_rewrites = []
    runner._applied_plan_rewrites.append(rewrite)


# ── Phase 1.3 dispatch lift — runtime policy ────────────────────────────────


@dataclass(frozen=True)
class RecoveryOutcome:
    """What the runner's recovery loop should do after a failed step.

    Returned by :meth:`StepRecoveryPolicy.handle_failure`. The runner
    applies it via:

        if outcome.halt:
            persist(outcome.step_index, status="halted", halt_reason=outcome.halt_reason)
            break
        persist(outcome.step_index, status="running", halt_reason=outcome.halt_reason)
        step_index = outcome.step_index
        # fall through to next loop iteration

    Side effects (sleeps, env.step keypresses, _reverse_step calls,
    sub-step executions, listings_on_page mutations) happened INSIDE
    ``handle_failure`` before the outcome was constructed — the
    runner's loop just persists + advances.
    """

    halt: bool
    step_index: int
    halt_reason: str
    listings_on_page: int = 0  # echoed back when paginate-fail mutates it
    listings_on_page_changed: bool = False  # whether to apply the field above


class StepRecoveryPolicy:
    """Encapsulates the failure-recovery decision tree for ``MicroPlanRunner.run``.

    Phase 1.3 of EPIC #161. Lifts the 165-LOC failure-recovery if/elif
    that lived under ``run()``'s ``else`` branch (``run()`` lines
    ~887–1053 pre-cleanup) into a single ``handle_failure`` method.

    Behaviour is identical to the in-runner version: same retry budgets,
    same halt conditions, same jump-to-type logic, same anti-bot
    re-navigate path, same page_blocked filtered-reload retry, same
    Escape-then-jump-to-loop on click failure.

    The class follows the BrowserState / CheckpointManager / ListingsScanner
    pattern from #115 — a thin shim with a parent back-reference that
    reads/writes runner state and calls runner methods (``_reverse_step``,
    ``_execute_navigate``, ``_ensure_results_filters``).

    Phase 3 (PlanExecutor extraction) consumes this class via dependency
    injection rather than parent back-reference; that change is one
    follow-up PR away.
    """

    def __init__(self, parent: Any) -> None:  # parent: MicroPlanRunner
        self.parent = parent

    def handle_failure(
        self,
        *,
        step: Any,           # MicroIntent — current step
        step_result: Any,    # StepResult — what _execute_step returned
        plan: Any,           # MicroPlan — for jump-to-type searches
        step_index: int,
        step_retry_counts: dict[int | str, int],  # mutated in place
        loop_counters: dict[int, int],  # mutated in place
        max_retries: int,
        listings_on_page: int,
    ) -> RecoveryOutcome:
        """Decide and apply recovery for a failed step. Returns the outcome.

        Side effects performed inline (not exposed via the outcome):

        - ``time.sleep`` between retries (3s required, 12s page_blocked,
          4s scan_error, 15s anti-bot)
        - ``env.step(KEY_PRESS Escape)`` on click failure before jumping
        - ``env.step(KEY_PRESS alt+Left)`` retries on navigate_back failure
        - ``parent._reverse_step`` on navigate / filter / generic failure
        - ``parent._execute_navigate`` for the anti-bot gate retry
        - ``parent._ensure_results_filters(force_reload=True)`` for the
          page_blocked retry
        - mutations on ``step_retry_counts`` (own and sentinel keys)
        - mutations on ``loop_counters`` (paginate failure → 999_999)
        """
        runner = self.parent
        logger_ = logging.getLogger("mantis_agent.gym.micro_runner")

        # ── required: retry budget then halt ────────────────────────────
        if step.required:
            attempt = step_retry_counts.get(step_index, 0) + 1

            # Deterministic CDP scroll fallback — applies to required
            # scroll steps that have already burned at least one full
            # Holo3 brain budget on a brain_loop_exhausted failure.
            # The scroll IS what the plan asked for; dispatching it via
            # CDP is the same action by a different mechanism after
            # the visual loop has demonstrably failed. Permitted by
            # feedback_cua_no_dom_access.md (CDP allowed for dispatching
            # vision-derived actions, not for deriving targets).
            #
            # Without this short-circuit, the required path burns
            # max_retries × Holo3-budget-25 steps with zero observable
            # scroll progress, then escalates to agentic_recovery which
            # correctly chooses halt. This wastes ~3 minutes and
            # ~$0.50 per stuck-scroll incident. Live repro:
            # BoatTrader urlnav-cdpscroll runs 1779134490 / 1779134923.
            if (
                step.type == "scroll"
                and str(getattr(step_result, "failure_class", "") or "") == "brain_loop_exhausted"
                and attempt >= 2
            ):
                env = getattr(runner, "env", None)
                cdp_eval = getattr(env, "cdp_evaluate", None) if env is not None else None
                if callable(cdp_eval):
                    # Read the actual browser scroll position before and
                    # after the dispatch. step_snapshot.capture() reads
                    # runner-internal counters (_viewport_stage,
                    # _scroll_state) that aren't updated by side-effect
                    # CDP calls, so we'd always see "unchanged" using
                    # the snapshot path. Reading window.scrollY via CDP
                    # is post-action verification of an action we
                    # already dispatched — distinct from "DOM-derived
                    # grounding" which feedback_cua_no_dom_access.md
                    # forbids. We're not deriving a target; we're
                    # checking whether our own scroll took effect.
                    def _read_scroll_y() -> float:
                        try:
                            v = cdp_eval(
                                "(window.scrollY || document.documentElement.scrollTop || 0)"
                            )
                            return float(v) if v is not None else 0.0
                        except Exception:
                            return -1.0  # sentinel — can't read

                    def _read_dist_from_bottom() -> float:
                        # px between the current scroll position and the
                        # document bottom. 0 (≤ tol) ⇒ already at the
                        # bottom; large ⇒ real content still below the
                        # fold. Same provenance carve-out as _read_scroll_y:
                        # this is post-action verification of layout
                        # metrics to decide whether the plan's
                        # "scroll to bottom" intent is *already satisfied*,
                        # not DOM-derived grounding of a click target
                        # (feedback_cua_no_dom_access.md). -1.0 = can't read.
                        try:
                            v = cdp_eval(
                                "(function(){"
                                "var se=document.scrollingElement"
                                "||document.documentElement;"
                                "var y=window.scrollY||se.scrollTop||0;"
                                "var ih=window.innerHeight||0;"
                                "var sh=se.scrollHeight"
                                "||document.body.scrollHeight||0;"
                                "return Math.max(0, sh-(y+ih));})()"
                            )
                            return float(v) if v is not None else -1.0
                        except Exception:
                            return -1.0  # sentinel — can't read

                    pre_y = _read_scroll_y()
                    # Multi-prong scroll dispatch — covers three
                    # browser scroll mechanisms in one CDP call so we
                    # don't need separate readbacks per attempt:
                    # (a) window.scrollBy — works when <body> is the
                    #     scrolling root.
                    # (b) document.scrollingElement.scrollBy — works
                    #     when the page sets a different scrolling
                    #     root (e.g. <html> with overflow on body).
                    # (c) PageDown KeyboardEvent on document — routes
                    #     through the browser's keyboard-driven scroll
                    #     handler, which inner-scroll containers
                    #     ("results panel" SPA pattern) typically
                    #     subscribe to.
                    # Still action-only — we're not deriving any
                    # target from the DOM; we're just dispatching the
                    # plan-requested scroll through every standard
                    # mechanism.
                    scroll_js = (
                        "(function(){"
                        "  var h = window.innerHeight;"
                        "  window.scrollBy(0, h);"
                        "  if (document.scrollingElement) "
                        "    document.scrollingElement.scrollBy(0, h);"
                        "  document.dispatchEvent(new KeyboardEvent("
                        "    'keydown',"
                        "    {key:'PageDown', code:'PageDown', "
                        "     keyCode:34, which:34, bubbles:true}"
                        "  ));"
                        "})()"
                    )
                    try:
                        cdp_eval(scroll_js)
                    except Exception as exc:  # noqa: BLE001
                        logger_.warning(
                            f"  [{step_index}] scroll CDP fallback dispatch failed: {exc}"
                        )
                    else:
                        post_y = _read_scroll_y()
                        # Treat ≥ 50px as a meaningful scroll. Below
                        # that and the page either was already at the
                        # bottom or has overflow:hidden / a sub-element
                        # scroller capturing the event.
                        moved = (
                            pre_y >= 0 and post_y >= 0 and (post_y - pre_y) >= 50
                        )
                        if moved:
                            logger_.warning(
                                f"  [{step_index}] scroll brain_loop_exhausted "
                                f"x{attempt} — CDP fallback dispatched "
                                f"window.scrollBy(0, innerHeight); scrollY "
                                f"{pre_y:.0f} → {post_y:.0f}, advancing"
                            )
                            return RecoveryOutcome(
                                halt=False, step_index=step_index + 1,
                                halt_reason="scroll_cdp_fallback",
                            )
                        # Δ<50px is ambiguous: either (a) we're already at
                        # the document bottom — the plan's "scroll to
                        # bottom" intent is *satisfied*, there's simply
                        # nowhere further to go, so advancing is correct;
                        # or (b) an overflow:hidden / sub-element scroller
                        # ate the event and real content still sits below
                        # the fold — a genuine failure that should keep
                        # burning the retry budget. Disambiguate via the
                        # document's own scroll metrics: scrollY+innerHeight
                        # within tol of scrollHeight ⇒ case (a). Without
                        # this, a required scroll-to-bottom step on a page
                        # shorter than the brain expects (e.g. BoatTrader
                        # by-owner detail pages) halts the whole run one
                        # step short of the reveal it was scrolling toward.
                        AT_BOTTOM_TOL = 96.0
                        dist = _read_dist_from_bottom()
                        if 0.0 <= dist <= AT_BOTTOM_TOL:
                            logger_.warning(
                                f"  [{step_index}] scroll brain_loop_exhausted "
                                f"x{attempt} — CDP fallback fired, scrollY "
                                f"{pre_y:.0f} → {post_y:.0f} (Δ<50px) but "
                                f"document bottom is {dist:.0f}px away "
                                f"(≤{AT_BOTTOM_TOL:.0f}px tol); scroll-to-bottom "
                                f"intent satisfied, advancing"
                            )
                            return RecoveryOutcome(
                                halt=False, step_index=step_index + 1,
                                halt_reason="scroll_reached_bottom",
                            )
                        logger_.warning(
                            f"  [{step_index}] scroll brain_loop_exhausted "
                            f"x{attempt} — CDP fallback fired but scrollY "
                            f"{pre_y:.0f} → {post_y:.0f} (Δ<50px) and document "
                            f"bottom is {dist:.0f}px away (sub-element scroller "
                            f"or content below fold); continuing retry budget"
                        )
                        # Fall through to the normal retry path below.

            # NB: an earlier ``extract_scroll_to_top_fallback`` deterministic
            # branch lived here. It was removed in favor of letting
            # agentic_recovery (which now receives scrollY + viewport_h via
            # the page_context block and an explicit overscroll pattern in
            # the prompt) decide whether the right recovery is "scroll up"
            # or "URL drift restore" or "halt". The LLM sees the same
            # signals plus the screenshot and can make a more contextual
            # call than the threshold check we had hardcoded.

            if attempt <= max_retries:
                step_retry_counts[step_index] = attempt
                logger_.warning(
                    f"  [{step_index}] REQUIRED step failed — retry {attempt}/{max_retries}"
                )
                time.sleep(3)
                return RecoveryOutcome(
                    halt=False, step_index=step_index,
                    halt_reason=f"required_retry:{step.type}:{attempt}",
                )
            # Before halting, give the agentic recovery loop a chance.
            # Claude analyses the failure (step + last screenshot +
            # failure data) and either:
            #   - returns a hint to augment the next retry's prompt
            #   - returns an edited version of the step (changed type,
            #     params, label, aliases)
            #   - returns helper steps to splice in BEFORE this step
            #   - returns halt = surface the legacy halt path
            # Bounded by per-step + per-run budgets so we don't loop
            # indefinitely. Falls through to the legacy halt path on
            # any failure of the recovery call itself.
            recovery_outcome = self._try_agentic_recovery(
                step=step,
                step_result=step_result,
                step_index=step_index,
                plan=plan,
                step_retry_counts=step_retry_counts,
                attempts=attempt,
            )
            if recovery_outcome is not None:
                return recovery_outcome

            logger_.error(
                f"  [{step_index}] REQUIRED step failed after {max_retries} retries — HALTING"
            )
            print(f"  HALT: Required step '{step.intent[:50]}' failed. Cannot proceed.")
            return RecoveryOutcome(
                halt=True, step_index=step_index,
                halt_reason=f"required_failed:{step.type}",
            )

        # ── gate: anti-bot one-shot retry, then soft-advance ────────────
        if step.gate:
            gate_data = step_result.data or ""
            gate_retry_key = f"gate_retry_{step_index}"
            antibot = (
                "cloudflare" in gate_data.lower()
                or "blocked" in gate_data.lower()
                or "security" in gate_data.lower()
                or "something went wrong" in gate_data.lower()
                or "request fail" in gate_data.lower()
            )
            if antibot and not step_retry_counts.get(gate_retry_key):
                step_retry_counts[gate_retry_key] = 1
                print("  [gate] Anti-bot detected — waiting 15s and retrying from navigate")
                time.sleep(15)
                nav_step = plan.steps[0] if plan.steps[0].type == "navigate" else None
                if nav_step:
                    runner._execute_navigate(nav_step, 0)
                    time.sleep(5)
                return RecoveryOutcome(
                    halt=False, step_index=step_index, halt_reason="gate_retry",
                )
            # A gate that reaches here is always required=False: a
            # *required* gate is handled by the required branch above
            # (retry-budget then halt) and never falls through. So this is
            # a soft checkpoint, not a setup precondition — e.g. the
            # decomposer auto-promotes any extract_data step whose intent
            # reads like verification ("verify"/"confirm" the confirmation
            # banner) to gate=True without marking it required (issue #244),
            # so a post-submit banner check on a boat the agent legitimately
            # declined (no submit → no banner) lands here. Halting the whole
            # run on that is wrong: advance past the checkpoint so the
            # surrounding loop can try the next listing.
            logger_.warning(
                f"  [{step_index}] soft gate failed (not required) — advancing: {step.verify[:60]}"
            )
            return RecoveryOutcome(
                halt=False, step_index=step_index + 1,
                halt_reason="gate_failed_soft",
            )

        # ── per-type recovery ───────────────────────────────────────────
        if step.type == "navigate":
            logger_.error(f"  [{step_index}] NAVIGATE FAILED — cannot proceed")
            runner._reverse_step(step)
            return RecoveryOutcome(
                halt=True, step_index=step_index, halt_reason="navigate_failed",
            )

        if step.type == "click":
            return self._handle_click_failure(
                step=step, step_result=step_result, plan=plan,
                step_index=step_index, step_retry_counts=step_retry_counts,
                max_retries=max_retries,
            )

        if step.type == "filter":
            logger_.warning(f"  [{step_index}] FILTER FAILED — skipping")
            runner._reverse_step(step)
            return RecoveryOutcome(
                halt=False, step_index=step_index + 1, halt_reason="filter_failed",
            )

        if step.type == "scroll":
            # #audit item 3: don't blindly advance on every scroll
            # failure. The old code assumed "scroll always moves the
            # page so missing-done() is benign" — but that's false
            # when the page is already at the bottom OR when the
            # scroll never registered (e.g. scroll bubbled into an
            # open dropdown that swallowed the wheel event).
            #
            # New behaviour:
            #
            # * ``failure_class == "brain_loop_exhausted"`` → keep
            #   the same step so the rewriter / next retry can
            #   actually affect it. Advancing past would discard the
            #   rewrite opportunity.
            # * Otherwise: compare pre/post snapshot ``viewport_stage``
            #   + ``scroll_signature``. If something changed →
            #   advance (legacy success path). If nothing changed
            #   → keep the same step (true failure that needs retry
            #   or different verb).
            failure_class = str(
                getattr(step_result, "failure_class", "") or ""
            )
            if failure_class == "brain_loop_exhausted":
                # First failure: keep the step so intent_rewriter can
                # convert the goal-shaped scroll intent into something
                # mechanical ("Press Page Down by viewport height") and
                # Holo3 gets a second pass with the cleaner phrasing.
                #
                # Second+ failure: Holo3 demonstrably can't make
                # observable scroll progress on this page (sticky
                # header swallowing wheel events, overlay capturing
                # focus, page already at the bottom, …). Dispatch a
                # deterministic scroll via CDP and check whether the
                # viewport actually moves. The scroll is a vision-
                # derived action (the plan asked for a scroll) executed
                # via a CDP keyboard event — within the CUA contract
                # per feedback_cua_no_dom_access.md (CDP allowed for
                # dispatching vision-derived actions; we're not reading
                # DOM state to derive a target).
                #
                # We track failure count under a scroll-specific key
                # in step_retry_counts (the ``scroll_brain_loop:<idx>``
                # prefix) so we don't share the count with the
                # ``required`` budget on the same step (different
                # semantics — required halts at max_retries, scroll
                # has its own retry/CDP escalation policy). Live repro
                # of the third-Holo3-budget-burn pattern this
                # addresses: BoatTrader run 20260518 (urlnav-pp-nosort)
                # — three consecutive brain_loop_exhausted on identical
                # scroll intents with no viewport delta and no critic
                # action.
                scroll_key = f"scroll_brain_loop:{step_index}"
                prior_failures = step_retry_counts.get(scroll_key, 0)
                env = getattr(runner, "env", None)
                cdp_eval = getattr(env, "cdp_evaluate", None) if env is not None else None
                if prior_failures >= 1 and callable(cdp_eval):
                    # Verify CDP scroll via actual ``window.scrollY``
                    # readback rather than the runner's internal
                    # ``viewport_stage`` counter. That counter is only
                    # updated by Holo3-dispatched scroll actions
                    # (set_scroll_state callbacks fire on brain ops);
                    # CDP scrolls don't update it. Pre-fix: CDP scroll
                    # fires, page moves, viewport_stage unchanged →
                    # "viewport unchanged" → keep step → infinite
                    # loop. Live repro: iter 8 of boattrader plan-passes
                    # loop wedged for 22 min at $0.77 on a detail-page
                    # scroll. Plus advance-after-3 ceiling so we never
                    # loop more than the brain budget × 3.
                    def _read_scroll_y_fallback() -> float:
                        try:
                            v = cdp_eval(
                                "(window.scrollY || document.documentElement.scrollTop || 0)"
                            )
                            return float(v) if v is not None else 0.0
                        except Exception:
                            return -1.0

                    pre_y = _read_scroll_y_fallback()
                    try:
                        cdp_eval("window.scrollBy(0, window.innerHeight)")
                    except Exception as exc:  # noqa: BLE001
                        logger_.warning(
                            f"  [{step_index}] scroll CDP fallback "
                            f"dispatch failed: {exc} — keeping step for retry"
                        )
                    else:
                        post_y = _read_scroll_y_fallback()
                        moved = (
                            pre_y >= 0 and post_y >= 0
                            and abs(post_y - pre_y) >= 50
                        )
                        if moved:
                            logger_.warning(
                                f"  [{step_index}] scroll brain_loop_exhausted "
                                f"x{prior_failures+1} — CDP scrollBy dispatched; "
                                f"scrollY {pre_y:.0f} → {post_y:.0f}; advancing"
                            )
                            return RecoveryOutcome(
                                halt=False, step_index=step_index + 1,
                                halt_reason="scroll_cdp_fallback",
                            )
                        # No movement OR can't verify (no CDP eval).
                        # If we've burned more than 3 cycles, advance
                        # anyway — the page is genuinely unscrollable
                        # (sticky overlay, page already at bottom,
                        # SPA with custom scroll container). Looping
                        # produces no information; the next step has
                        # a chance to recognize the state.
                        if prior_failures >= 2:
                            logger_.warning(
                                f"  [{step_index}] scroll brain_loop_exhausted "
                                f"x{prior_failures+1} — CDP fired but scrollY "
                                f"{pre_y:.0f} → {post_y:.0f} (no movement); "
                                f"exceeded 3 cycles, scrolling to top then "
                                f"advancing so downstream extract sees the "
                                f"page content, not the footer"
                            )
                            # The next step (typically extract_data) needs
                            # the page's primary content in the viewport.
                            # By the time scroll plateaus at the bottom,
                            # the viewport is showing the footer — extract
                            # screenshots that and finds nothing useful.
                            # Reset to top so the next step starts from a
                            # known-good viewport position.
                            try:
                                cdp_eval("window.scrollTo(0, 0)")
                            except Exception as exc:  # noqa: BLE001
                                logger_.debug(
                                    f"  [{step_index}] post-scroll reset-to-top "
                                    f"failed (non-fatal): {exc}"
                                )
                            return RecoveryOutcome(
                                halt=False, step_index=step_index + 1,
                                halt_reason="scroll_no_movement_advance",
                            )
                        logger_.warning(
                            f"  [{step_index}] scroll brain_loop_exhausted "
                            f"x{prior_failures+1} — CDP fired but scrollY "
                            f"{pre_y:.0f} → {post_y:.0f}; keeping step for "
                            f"one more retry"
                        )
                # First failure (or no CDP env): legacy keep-step path
                # so intent_rewriter / next retry can affect the step.
                # Bump the scroll-specific counter so the CDP fallback
                # gate (``prior_failures >= 1``) fires on the next pass.
                step_retry_counts[scroll_key] = prior_failures + 1
                logger_.warning(
                    f"  [{step_index}] scroll brain_loop_exhausted "
                    f"(scroll_retry={prior_failures + 1}) — "
                    f"keeping step for retry (CDP fallback armed for next pass)"
                )
                return RecoveryOutcome(
                    halt=False, step_index=step_index,
                    halt_reason="scroll_brain_loop_keep_step",
                )

            from . import step_snapshot as _snap
            pre_snap = getattr(runner, "_pre_step_snapshot", None)
            scrolled = False
            if pre_snap is not None:
                try:
                    post_snap = _snap.capture(runner)
                    scrolled = (
                        pre_snap.viewport_stage != post_snap.viewport_stage
                        or pre_snap.scroll_signature != post_snap.scroll_signature
                    )
                except Exception as exc:  # noqa: BLE001
                    logger_.debug(
                        "scroll-recovery snapshot capture failed: %s", exc,
                    )
                    # Fall through to legacy advance path when we
                    # can't verify either way — safer than introducing
                    # a new halt path on env-stub test stubs.
                    scrolled = True

            if scrolled:
                logger_.info(
                    f"  [{step_index}] scroll completed "
                    f"(no done() but viewport advanced)"
                )
                return RecoveryOutcome(
                    halt=False, step_index=step_index + 1,
                    halt_reason="scroll_no_done",
                )
            logger_.warning(
                f"  [{step_index}] scroll failed (no viewport delta) — "
                f"keeping step for retry"
            )
            return RecoveryOutcome(
                halt=False, step_index=step_index,
                halt_reason="scroll_no_delta",
            )

        if step.type == "navigate_back":
            logger_.warning(f"  [{step_index}] BACK FAILED — retrying CDP-back then Alt+Left")
            # #583: prefer CDP history.back() — more reliable than the
            # xdotool keyboard shortcut on SPA pushState sites. Falls
            # back to Alt+Left if CDP unavailable or didn't navigate.
            cdp_back = getattr(runner.env, "cdp_history_back", None)
            for back_attempt in range(3):
                try:
                    if callable(cdp_back) and cdp_back():
                        time.sleep(0.5)
                    else:
                        runner.env.step(Action(
                            action_type=ActionType.KEY_PRESS,
                            params={"keys": "alt+Left"},
                        ))
                        time.sleep(3)
                except Exception:
                    pass
                if runner.extractor:
                    screenshot = runner.env.screenshot()
                    check = runner.extractor.extract(screenshot)
                    url = check.url if check else ""
                    if url:
                        runner._last_known_url = url
                    if (
                        url
                        and runner.site_config.is_results_page(url)
                        and not runner.site_config.is_detail_page(url)
                    ):
                        logger_.info(
                            f"  [back] Verified on results page after "
                            f"{back_attempt + 1} attempts"
                        )
                        break
            return RecoveryOutcome(
                halt=False, step_index=step_index + 1,
                halt_reason="navigate_back_recovered",
            )

        if step.type == "paginate":
            logger_.warning(
                f"  [{step_index}] PAGINATE FAILED — no more pages, ending"
            )
            # Exhaust the outer loop so it doesn't restart on the same page
            for k in list(loop_counters.keys()):
                loop_counters[k] = 999999
            return RecoveryOutcome(
                halt=False, step_index=step_index + 1,
                halt_reason="paginate_exhausted",
            )

        if step.type in ("extract_url", "extract_data"):
            return RecoveryOutcome(
                halt=False, step_index=step_index + 1,
                halt_reason=f"{step.type}_failed",
            )

        # ── generic fallback: reverse + skip ────────────────────────────
        runner._reverse_step(step)
        step_result.reversed = True
        logger_.warning(f"  [{step_index}] FAILED + reversed — skipping")
        return RecoveryOutcome(
            halt=False, step_index=step_index + 1,
            halt_reason=f"{step.type}_failed",
        )

    # ── click-failure sub-policy ────────────────────────────────────────

    def _handle_click_failure(
        self,
        *,
        step: Any,
        step_result: Any,
        plan: Any,
        step_index: int,
        step_retry_counts: dict[int | str, int],
        max_retries: int,
    ) -> RecoveryOutcome:
        """Click-step failure has the most branches: page_exhausted /
        scan_error / page_blocked / unknown.

        Extracted for readability — the wrapping ``handle_failure`` would
        otherwise be a 200-line method.
        """
        runner = self.parent
        logger_ = logging.getLogger("mantis_agent.gym.micro_runner")

        # page_exhausted → jump to paginate, then loop, then advance
        if step_result.data == "page_exhausted":
            logger_.info(f"  [{step_index}] PAGE EXHAUSTED — jumping to paginate")
            jumped = self._first_step_of_type(plan, step_index + 1, "paginate")
            if jumped is None:
                jumped = self._first_step_of_type(plan, step_index + 1, "loop")
            new_index = jumped if jumped is not None else step_index + 1
            return RecoveryOutcome(
                halt=False, step_index=new_index, halt_reason="page_exhausted",
            )

        # login_redirect → click hijacked into auth wall. Reload the
        # filtered results URL to roll back, mark the card tried (handler
        # already appended to _extracted_titles), and skip to the next
        # loop iteration so the next card is attempted on a clean page.
        if step_result.data == "login_redirect":
            logger_.warning(
                f"  [{step_index}] LOGIN REDIRECT — rolling back to results URL"
            )
            try:
                runner._ensure_results_filters(step_index, force_reload=True)
            except Exception as exc:  # noqa: BLE001 — rollback must not raise
                logger_.warning(
                    f"  [{step_index}] rollback reload failed: {exc}"
                )
            jumped = self._first_step_of_type(plan, step_index + 1, "loop")
            new_index = jumped if jumped is not None else step_index + 1
            return RecoveryOutcome(
                halt=False, step_index=new_index,
                halt_reason="login_redirect_recovered",
            )

        # newtab_blank → middle-click opened chrome://newtab/. The handler
        # already closed the empty tab; treat as a generic skip-this-card
        # so the next loop iteration runs on the original results tab.
        if step_result.data == "newtab_blank":
            logger_.warning(
                f"  [{step_index}] BLANK NEW TAB — card has no navigable link, skipping"
            )
            jumped = self._first_step_of_type(plan, step_index + 1, "loop")
            new_index = jumped if jumped is not None else step_index + 1
            return RecoveryOutcome(
                halt=False, step_index=new_index, halt_reason="newtab_blank",
            )

        # scan_error / page_blocked → bounded retry, then page_blocked reload
        if step_result.data in ("scan_error", "page_blocked"):
            attempt = step_retry_counts.get(step_index, 0) + 1
            if attempt <= max_retries:
                step_retry_counts[step_index] = attempt
                wait_s = 12 if step_result.data == "page_blocked" else 4
                logger_.warning(
                    f"  [{step_index}] {step_result.data.upper()} — "
                    f"waiting {wait_s}s and retrying ({attempt}/{max_retries})"
                )
                time.sleep(wait_s)
                return RecoveryOutcome(
                    halt=False, step_index=step_index,
                    halt_reason=f"{step_result.data}_retry:{attempt}",
                )
            logger_.warning(
                f"  [{step_index}] {step_result.data.upper()} — retry budget exhausted"
            )
            if step_result.data == "page_blocked":
                reload_key = f"page_blocked_reload_{step_index}"
                reload_attempt = step_retry_counts.get(reload_key, 0) + 1
                if reload_attempt <= 1 and runner._ensure_results_filters(
                    step_index, force_reload=True,
                ):
                    step_retry_counts[reload_key] = reload_attempt
                    step_retry_counts[step_index] = 0
                    logger_.warning(
                        f"  [{step_index}] PAGE_BLOCKED — reloaded filtered URL, "
                        "retrying click"
                    )
                    return RecoveryOutcome(
                        halt=False, step_index=step_index,
                        halt_reason="page_blocked_reload",
                    )
                logger_.error(
                    f"  [{step_index}] PAGE_BLOCKED after filtered reload — halting"
                )
                print("  HALT: Filtered results page is blocked/erroring.")
                return RecoveryOutcome(
                    halt=True, step_index=step_index, halt_reason="page_blocked",
                )

        # Generic click failure: Escape + jump to loop
        logger_.warning(f"  [{step_index}] CLICK FAILED — skipping to next")
        try:
            runner.env.step(Action(
                action_type=ActionType.KEY_PRESS, params={"keys": "Escape"},
            ))
            time.sleep(0.5)
        except Exception:
            pass
        jumped = self._first_step_of_type(plan, step_index + 1, "loop")
        new_index = jumped if jumped is not None else step_index + 1
        return RecoveryOutcome(
            halt=False, step_index=new_index, halt_reason="click_failed",
        )

    @staticmethod
    def _first_step_of_type(plan: Any, start: int, step_type: str) -> int | None:
        """Find the first step at or after ``start`` whose type matches.

        Returns the index, or ``None`` if no such step exists.
        Encapsulates the ``for j in range(...): if plan.steps[j].type == ...``
        pattern that appears 4 times in the legacy if/elif.
        """
        for j in range(start, len(plan.steps)):
            if plan.steps[j].type == step_type:
                return j
        return None

    # ── Agentic failure-recovery loop ───────────────────────────────────

    # #435 follow-up: typical edit forms have 5-10 inputs. 12 Tabs
    # covers all of them and a couple of "exit form" tabs into nav
    # without dragging recovery latency past ~1.5s. Tunable via
    # ``MANTIS_RECOVERY_TAB_BLUR_COUNT`` for forms with more inputs.
    _TAB_BLUR_DEFAULT_COUNT: int = 12

    def _tab_blur_traversal(self, runner: Any, step_index: int) -> None:
        """Send Tab × N to trigger blur-driven validation rendering.

        Why: a ``no_state_change`` failure on a submit step is often
        a silent client-side validation rejection — the form's submit
        handler short-circuits because some field is invalid, but no
        red error is RENDERED until focus leaves the bad field. The
        post-failure screenshot is visually clean, so the recovery
        analyser (Claude) sees no validation indicators and picks
        ``halt`` even when ``insert_steps`` is the right answer.

        Walking Tab across the form forces every input through a blur
        event, which is what triggers the ``:invalid`` /
        ``aria-invalid`` styles in React / Vue / vanilla HTML forms.
        The recovery screenshot is then captured AFTER traversal, so
        whatever the form was hiding is now visible.

        Best-effort: any exception inside the traversal is swallowed
        and we fall through to the normal recovery path with whatever
        screenshot the env can produce. Side-effect-free in the sense
        that Tab key events don't mutate field values or submit the
        form — worst case the cursor lands somewhere else, which
        recovery would have to handle anyway.
        """
        try:
            count = int(os.environ.get(
                "MANTIS_RECOVERY_TAB_BLUR_COUNT",
                self._TAB_BLUR_DEFAULT_COUNT,
            ))
        except (TypeError, ValueError):
            count = self._TAB_BLUR_DEFAULT_COUNT
        if count <= 0:
            return
        env = getattr(runner, "env", None)
        if env is None:
            return
        logger.info(
            "  [%d] recovery: tab-blur traversal × %d to force "
            "validation rendering before screenshot",
            step_index, count,
        )
        for _ in range(count):
            try:
                env.step(Action(
                    action_type=ActionType.KEY_PRESS,
                    params={"keys": "Tab"},
                ))
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.debug("recovery: tab-blur step raised: %s", exc)
                return
            # Tiny settle so each blur event can fire its
            # validation handler before the next Tab. 50ms × 12 =
            # 0.6s total — acceptable recovery overhead.
            time.sleep(0.05)
        # One final settle so the LAST blur's validation render has
        # time to paint before we screenshot.
        time.sleep(0.3)

    def _try_agentic_recovery(
        self,
        *,
        step: Any,
        step_result: Any,
        step_index: int,
        plan: Any,
        step_retry_counts: dict[int | str, int],
        attempts: int,
    ) -> RecoveryOutcome | None:
        """Last-resort: ask Claude to analyse + adapt before HALTING.

        Invoked when a REQUIRED step has exhausted its retry budget
        (and, if the wrong-handler escalation triggered, that has
        also failed). Returns a :class:`RecoveryOutcome` describing
        the next action to take, or ``None`` to fall through to the
        legacy halt path.

        Budget enforcement (per-step max 2 + per-run max 5) lives
        here so a pathological step / page can't spend the whole
        run on recovery alone. Budgets are tracked on the runner so
        they survive across step retries within the same plan.

        Modes applied:

        - ``add_hint`` — append the hint to the step's
          ``_recovery_hints[step_index]`` list. The form / click
          handlers read this on next attempt and surface it in the
          search prompt. Returns a ``RecoveryOutcome`` that re-runs
          the same step.
        - ``edit_step`` — mutate ``plan.steps[step_index]`` with
          the changed fields (intent / type / params). Returns a
          ``RecoveryOutcome`` that re-runs the (now edited) step.
        - ``insert_steps`` — splice helper steps before the failed
          step via :func:`agentic_recovery.splice_inserted_steps`.
          Returns a ``RecoveryOutcome`` that jumps to the first
          inserted step.
        - ``halt`` — return ``None`` so the caller surfaces the
          legacy halt path.
        """
        runner = self.parent

        # Budget guards. Defensively check that the runner exposes
        # the recovery-state fields with the right shape — legacy
        # runners (or test stubs that use MagicMock) won't, and we
        # should fall through to the legacy halt path rather than
        # crash with type errors.
        per_step_dict = getattr(runner, "_recovery_attempts_per_step", None)
        total_attempts = getattr(runner, "_total_recovery_attempts", None)
        if not isinstance(per_step_dict, dict) or not isinstance(total_attempts, int):
            # Issue #431: surface every silent-skip path so a halt that
            # bypassed Claude-backed recovery is debuggable from logs
            # alone (this branch fires on legacy / non-MicroPlanRunner
            # call sites that don't initialize the budget trackers).
            logger.warning(
                "  [%d] recovery_skipped: runner missing budget trackers "
                "(per_step=%s, total=%s) — falling through to legacy halt",
                step_index, type(per_step_dict).__name__,
                type(total_attempts).__name__,
            )
            return None
        # #567: per-run override via runtime fields wins over the
        # module DEFAULT_* fallback. Same shape as #560/#561/#571.
        from ..agentic_recovery import effective_max_recoveries
        max_per_step, max_per_run = effective_max_recoveries(self.parent)
        per_step = per_step_dict.get(step_index, 0)
        per_run = total_attempts
        if per_step >= max_per_step:
            logger.warning(
                "  [%d] recovery_skipped: per-step budget exhausted (%d/%d)",
                step_index, per_step, max_per_step,
            )
            return None
        if per_run >= max_per_run:
            logger.warning(
                "  [%d] recovery_skipped: per-run budget exhausted (%d/%d)",
                step_index, per_run, max_per_run,
            )
            return None

        # #435 follow-up: before capturing the recovery screenshot on
        # a ``no_state_change`` submit, walk Tab across the form to
        # force blur-driven validation rendering. Most React forms
        # render :invalid / aria-invalid styles only after focus
        # leaves a bad field, so the post-submit-fail screenshot is
        # often visually CLEAN even when the form's submit handler
        # short-circuited on a validation error (canonical staff-crm
        # pattern: ``Estimated Deal Value: 461927.81`` silently
        # rejected by an integer-only rule). Tab traversal forces
        # the rendering, so Claude actually sees what's invalid and
        # picks ``insert_steps`` instead of ``halt``. Bounded to 12
        # Tabs (covers most edit forms); opt-out via
        # ``MANTIS_RECOVERY_TAB_BLUR=disabled``.
        failure_class = str(getattr(step_result, "failure_class", "") or "")
        step_type = str(getattr(step, "type", "") or "")
        if (
            failure_class == "no_state_change"
            and step_type == "submit"
            and os.environ.get("MANTIS_RECOVERY_TAB_BLUR", "") != "disabled"
        ):
            self._tab_blur_traversal(runner, step_index)

        # Capture screenshot for the analysis. Best-effort — Claude
        # can reason without it.
        screenshot = None
        try:
            screenshot = runner._safe_screenshot()
        except Exception:
            screenshot = None

        # Plan context = intent strings of successful steps before
        # this one. Helps Claude understand workflow position.
        plan_context: list[str] = []
        try:
            for i, prev in enumerate(plan.steps[:step_index]):
                plan_context.append(getattr(prev, "intent", "") or "")
        except Exception:  # noqa: BLE001
            pass

        from ..agentic_recovery import (
            analyse_failure_and_recover,
            splice_inserted_steps,
        )
        # #432: Haiku-level vision frequently misses field-level
        # validation errors (red borders / aria-invalid rings), so on
        # ``no_state_change`` submit failures — the canonical
        # form-validation-blocked-submit shape — escalate to Opus.
        # Other failure classes stay on Haiku (the cheap analysis
        # task it was designed for). Per-step + per-run budgets still
        # cap the spend regardless of model choice.
        failure_class = str(getattr(step_result, "failure_class", "") or "")
        recovery_model = (
            "claude-opus-4-7"
            if failure_class == "no_state_change"
            and getattr(step, "type", "") == "submit"
            else "claude-haiku-4-5-20251001"
        )
        # H8: pass the accumulated hints on this step so Claude can
        # detect the "same hint twice, still failing" pattern and bias
        # toward structural recovery instead of yet another add_hint.
        prior_hints = []
        hint_map = getattr(runner, "_recovery_hints", None)
        if isinstance(hint_map, dict):
            stored = hint_map.get(step_index, [])
            if isinstance(stored, list):
                prior_hints = [str(h) for h in stored if str(h).strip()]
        # #523 PR B-5 — wrap the recovery call in a ``step_recovery``
        # modelio context. The raw requests.post inside
        # analyse_failure_and_recover (-> _call_recovery_tool) checks
        # this contextvar at the boundary and emits one
        # modelio/<step>-step_recovery-<seq>.json record on success.
        # No-op when augur is None or inactive.
        # Page-state hints help the LLM disambiguate URL drift
        # (current page is not the one the plan anchored to) from
        # overscroll (viewport is past the fold) — two common
        # extract-failure shapes that used to be handled by
        # deterministic if/else fallbacks. The LLM sees the
        # screenshot AND the live values, then picks the right
        # ``insert_steps`` recovery without anyone hardcoding the
        # decision tree.
        page_ctx: dict = {}
        env = getattr(runner, "env", None)
        anchor_url = str(getattr(runner, "_last_known_url", "") or "")
        if anchor_url:
            page_ctx["anchor_url"] = anchor_url
        if env is not None:
            try:
                cur = str(getattr(env, "current_url", "") or "")
                if cur:
                    page_ctx["current_url"] = cur
            except Exception:  # noqa: BLE001
                pass
            cdp_eval = getattr(env, "cdp_evaluate", None)
            if callable(cdp_eval):
                try:
                    sy = cdp_eval(
                        "(window.scrollY || document.documentElement.scrollTop || 0)"
                    )
                    if sy is not None:
                        page_ctx["scroll_y"] = float(sy)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    vh = cdp_eval("(window.innerHeight || 0)")
                    if vh:
                        page_ctx["viewport_h"] = float(vh)
                except Exception:  # noqa: BLE001
                    pass

        from ..observability.modelio import publish_modelio_context
        with publish_modelio_context(
            getattr(runner, "_augur", None),
            layer="step_recovery", step_index=step_index,
        ):
            decision = analyse_failure_and_recover(
                step=step,
                failure_data=str(getattr(step_result, "data", "") or ""),
                screenshot=screenshot,
                plan_context=plan_context,
                attempts=attempts,
                model=recovery_model,
                prior_hints=prior_hints,
                page_context=page_ctx,
                # Plan-evolution Phase 1 (#705): page_links source needs
                # CDP access to read links off the currently-loaded page.
                # The local impl + RemoteComputerImpl both expose
                # ``cdp_evaluate``; the source degrades silently when
                # env is None or lacks the method.
                env=getattr(runner, "env", None),
                # Plan-evolution Phase 2 (#706): persistence hooks for
                # the rewrite_url candidate the recovery may produce.
                # Empty values disable persistence (legacy callers,
                # tests, ad-hoc runs); production sets all three.
                plan_hash=str(getattr(runner, "_plan_hash", "") or ""),
                workflow_id=str(getattr(runner, "_workflow_id", "") or ""),
                step_index=step_index,
            )
        if decision is None:
            # #431: even though analyse_failure_and_recover already logs
            # its own warnings for the api-key / api-error / no-tool-use
            # paths, the caller has no signal that recovery was actually
            # attempted vs short-circuited by a budget gate. Make the
            # "attempted but returned no decision" outcome explicit.
            logger.warning(
                "  [%d] recovery_skipped: analyse_failure_and_recover "
                "returned None (Anthropic call failed, malformed "
                "response, or no api_key — see preceding log line)",
                step_index,
            )
            return None

        # Increment budgets *before* applying so a recovery that
        # itself triggers another failure doesn't infinite-loop.
        per_step_dict[step_index] = per_step + 1
        runner._total_recovery_attempts = per_run + 1

        # Surface budget consumption explicitly so the next investigator
        # can attribute who burned what — critic-frontier and
        # step_recovery share one pool. Without this line the only
        # signal was "skipped — per-step budget exhausted (N/N)" with
        # no way to know who consumed the N.
        logger.warning(
            "  [%d] step_recovery: consumed recovery budget "
            "(per_step=%d/%d, per_run=%d/%d) — decision.mode=%s",
            step_index, per_step + 1, max_per_step,
            per_run + 1, max_per_run, decision.mode,
        )
        logger.warning(
            "  [%d] agentic recovery: mode=%s — %s",
            step_index, decision.mode, decision.reasoning[:200],
        )
        # #634 follow-up: surface recovery rationale to the Augur
        # Reasoning-trace tab. ``decision.reasoning`` is the Claude
        # explanation for why this recovery mode was chosen — already
        # logged at WARN, also worth structured surfacing alongside
        # brain + verifier reasoning.
        augur = getattr(runner, "_augur", None)
        if augur is not None and decision.reasoning:
            try:
                augur.record_reasoning(
                    step_index=step_index,
                    format="recovery",
                    content=f"mode={decision.mode}: {decision.reasoning}",
                )
            except Exception:  # noqa: BLE001 — never block recovery
                pass

        if decision.mode == "halt":
            return None  # fall through to legacy halt

        if decision.mode == "add_hint":
            if not decision.hint:
                return None
            if not hasattr(runner, "_recovery_hints"):
                runner._recovery_hints = {}
            runner._recovery_hints.setdefault(step_index, []).append(
                decision.hint
            )
            # Reset retry count so the step gets fresh attempts with
            # the hint. Keep the recovery budget bumped above so we
            # don't loop forever.
            step_retry_counts[step_index] = 0
            return RecoveryOutcome(
                halt=False, step_index=step_index,
                halt_reason=f"recovery_hint:{decision.hint[:60]}",
            )

        if decision.mode == "edit_step":
            edited = decision.edited_step or {}
            if not edited:
                return None
            # Mutate the step in place. Only the fields the LLM
            # supplied; missing fields preserve original values.
            original_intent = step.intent
            if "intent" in edited and edited["intent"]:
                step.intent = str(edited["intent"])
            if "type" in edited and edited["type"]:
                step.type = str(edited["type"])
            if "params" in edited and isinstance(edited["params"], dict):
                # Merge — recipe-side params extend the existing dict.
                merged = dict(getattr(step, "params", {}) or {})
                merged.update(edited["params"])
                step.params = merged
            step_retry_counts[step_index] = 0
            # Record so the operator can audit what the framework did
            # (epic #377 Phase C accounting). The older agentic_recovery
            # path is a sibling to the Phase B intent_rewriter; both
            # land here so result.json shows a unified rewrite log.
            if step.intent != original_intent:
                from . import healing_events
                healing_events.record_rewrite(
                    self.parent,
                    step_index=step_index,
                    from_intent=original_intent,
                    to_intent=step.intent,
                    source="agentic_recovery",
                    failure_class=getattr(step_result, "failure_class", "") or "",
                )

            # Plan-evolution Phase 2 (#706): if this edit_step came from
            # a URL-rewrite recovery (bad_url failure → rewrite_url),
            # track the candidate on the runner so finalize_run_outcomes
            # can record per-rewrite success / failure at run terminal.
            # We recognise URL-rewrite edits by their reasoning prefix —
            # `rewrite_url:<source>` — set by agentic_recovery.
            if str(decision.reasoning or "").startswith("rewrite_url:"):
                _track_plan_rewrite_candidate(
                    runner, step_index=step_index, edited=edited,
                    decision=decision, original_step=step,
                )

            return RecoveryOutcome(
                halt=False, step_index=step_index,
                halt_reason=f"recovery_edit:{step.type}",
            )

        if decision.mode == "insert_steps":
            if not decision.inserted_steps:
                return None
            # #435 cascade cap: if THIS step is itself an inserted-by-
            # recovery step (its index is in ``runner._recovery_inserted_steps``),
            # and we're about to insert MORE steps because the inserted
            # one failed, that's a cascade. Allow ≤2 cascade depth
            # before halting cleanly — otherwise insert_steps loops
            # blow the per-run recovery budget on the same root cause.
            # Observed on the tab-blur Modal run: 5/5 recoveries spent
            # on sequential inserted fill_field failures, $0.92 burned.
            recovery_inserted = getattr(runner, "_recovery_inserted_steps", None)
            cascade_depth = getattr(runner, "_recovery_cascade_depth", None)
            if (
                isinstance(recovery_inserted, set)
                and isinstance(cascade_depth, dict)
                and step_index in recovery_inserted
            ):
                depth = cascade_depth.get(step_index, 0)
                if depth >= 2:
                    logger.warning(
                        "  [%d] recovery_skipped: inserted-step cascade "
                        "depth %d exceeded — halting cleanly instead of "
                        "looping insert_steps on the same root cause",
                        step_index, depth,
                    )
                    return None
                cascade_depth[step_index] = depth + 1
            # Build MicroIntent objects from the dict-shaped specs.
            # #435 task #2: the inserted helper steps inherit the
            # parent step's ``hints`` (specifically ``region``) so
            # ``find_form_target`` benefits from the same region
            # cropping that the parent submit had. Without this, the
            # inserted ``fill_field`` faces the full screen and hits
            # the same disambiguation pressure the parent did.
            from ..plan_decomposer import MicroIntent
            parent_hints = dict(getattr(step, "hints", {}) or {})
            inherited_hints: dict = {}
            # Only carry hints that are useful on the inserted step.
            # ``region`` is the canonical one (scoping the grounding);
            # ``visual`` and ``position`` are submit-step prose hints
            # that don't translate. Operators can opt fields in/out
            # by editing this set.
            for k in ("region",):
                if k in parent_hints:
                    inherited_hints[k] = parent_hints[k]
            new_steps = [
                MicroIntent(
                    intent=spec["intent"],
                    type=spec["type"],
                    params=spec.get("params") or {},
                    section=getattr(step, "section", "") or "",
                    required=False,  # helper steps are best-effort
                    hints=dict(inherited_hints),
                )
                for spec in decision.inserted_steps
            ]
            plan.steps = splice_inserted_steps(
                plan.steps, step_index, new_steps,
            )
            # #435 cascade cap: track the indices of inserted steps
            # so a future failure on those steps can be detected as
            # a cascade. The splice puts them at indices
            # ``step_index .. step_index + len(new_steps) - 1``;
            # the original failed step is now at
            # ``step_index + len(new_steps)``. Don't rely on the
            # attribute existing — initialize defensively.
            if not hasattr(runner, "_recovery_inserted_steps"):
                runner._recovery_inserted_steps = set()
            if not hasattr(runner, "_recovery_cascade_depth"):
                runner._recovery_cascade_depth = {}
            for offset in range(len(new_steps)):
                runner._recovery_inserted_steps.add(step_index + offset)
            # Reset retry count so the (now-deferred) failed step
            # gets fresh attempts after the helper sub-flow.
            step_retry_counts[step_index] = 0
            # Jump to the first inserted step.
            return RecoveryOutcome(
                halt=False, step_index=step_index,
                halt_reason=f"recovery_insert:{len(new_steps)}_steps",
            )

        # Unknown mode — defensive fallthrough.
        return None
