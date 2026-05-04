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

from dataclasses import dataclass, field
from enum import Enum


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
