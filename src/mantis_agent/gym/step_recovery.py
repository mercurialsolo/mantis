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
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..actions import Action, ActionType


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
            logger_.error(
                f"  [{step_index}] REQUIRED step failed after {max_retries} retries — HALTING"
            )
            print(f"  HALT: Required step '{step.intent[:50]}' failed. Cannot proceed.")
            return RecoveryOutcome(
                halt=True, step_index=step_index,
                halt_reason=f"required_failed:{step.type}",
            )

        # ── gate: anti-bot one-shot retry then halt ─────────────────────
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
            logger_.error(
                f"  [{step_index}] GATE FAILED: {step.verify[:60]} — HALTING"
            )
            print(f"  HALT: Gate verification '{step.verify[:50]}' failed. Setup incomplete.")
            return RecoveryOutcome(
                halt=True, step_index=step_index, halt_reason="gate_failed",
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
            # Scroll "failure" usually means the model didn't call done()
            # but the page DID scroll — treat as success.
            logger_.info(
                f"  [{step_index}] Scroll completed (no done() but page changed)"
            )
            return RecoveryOutcome(
                halt=False, step_index=step_index + 1, halt_reason="scroll_no_done",
            )

        if step.type == "navigate_back":
            logger_.warning(f"  [{step_index}] BACK FAILED — retrying Alt+Left")
            for back_attempt in range(3):
                try:
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
