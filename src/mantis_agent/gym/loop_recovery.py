"""Action-class transition policy for stuck loops (#302).

The :class:`LoopDetector` flags three loop shapes — byte-equal repeats,
coordinate-drift, and frozen-state. Today's runner responds with a
contextual nudge in the next prompt ("you've clicked here three times,
try typing instead"). That works some of the time, but the benchmark
reports show many runs where the brain reads the nudge and emits the
same click class again.

This module returns a *forced action* — a different action class the
runner can substitute for the brain's stuck output. The existing
form-controller / force-submit / claude-director path handles the
"plan value matches the focused field" case; this policy is the
fallback when those substitutions didn't fire.

Initial rules (narrow on purpose — the soft-loop signal is noisy and
forcing the wrong class is worse than another nudge cycle):

* **focused-click loop with no plan value** → force ``key_press(Tab)``
  to move to a different field. Often the brain re-clicks because the
  field's label doesn't match any extracted plan value; Tab gives the
  loop a way out.
* **submit-shaped click loop with frame frozen** → force
  ``key_press(Return)``. Some sites' submit buttons absorb clicks
  without state change (overlay / disabled); the Enter key triggers
  the form-level submit handler.

Both rules require ``LoopDetector.is_any_loop`` to fire AND the
existing substitution chain to have passed through unchanged. The
runner consults this policy ONCE per soft-loop signal so a single
forced recovery doesn't itself become a new loop trigger.

Ablation toggle ``MANTIS_LOOP_RECOVERY=disabled`` short-circuits the
policy back to ``LoopRecoveryDecision(forced_action=None)`` — runner
falls through to its legacy nudge path.

Surfaces:

* ``TrajectoryStep.loop_recovery_reason`` — set to the stable reason
  code when this policy substituted the brain's action.
* ``RunResult.loop_recoveries_by_reason`` — aggregate counter per code.
* ``/v1/cua`` response gains ``loop_recoveries_by_reason`` for ablation
  data points.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from ..actions import Action, ActionType

logger = logging.getLogger(__name__)


_ENV_TOGGLE: str = "MANTIS_LOOP_RECOVERY"


# Stable reason codes — safe to log, surface on the API, and key reward
# analyses by. New codes append; never rename or repurpose.
REASON_TYPE_PENDING_VALUE: str = "type_pending_value"
REASON_TAB_TO_NEXT_FIELD: str = "tab_to_next_field"
REASON_PRESS_RETURN_FOR_SUBMIT: str = "press_return_for_submit"

REASON_CODES: tuple[str, ...] = (
    REASON_TYPE_PENDING_VALUE,
    REASON_TAB_TO_NEXT_FIELD,
    REASON_PRESS_RETURN_FOR_SUBMIT,
)


def is_enabled() -> bool:
    """``MANTIS_LOOP_RECOVERY=disabled`` short-circuits the policy."""
    return os.environ.get(_ENV_TOGGLE, "enabled").lower() != "disabled"


@dataclass
class LoopRecoveryDecision:
    """Outcome of one policy evaluation.

    * ``forced_action`` — the action the runner should dispatch instead
      of the brain's emitted action. ``None`` when no rule applied.
    * ``reason`` — stable code in :data:`REASON_CODES`; empty when no
      forced action.
    * ``detail`` — human-readable detail for logging and the trajectory.
    """

    forced_action: Action | None
    reason: str = ""
    detail: str = ""

    def __bool__(self) -> bool:
        return self.forced_action is not None


_SUBMIT_KEYWORDS: tuple[str, ...] = (
    "submit", "sign in", "sign-in", "signin", "log in", "log-in", "login",
    "register", "save", "confirm", "continue",
)


def decide_recovery(
    *,
    action: Action,
    action_history: list[Action],
    focused_input: dict | None,
    pending_form_values: list[dict],
    recent_frame_hashes: list[str],
    task: str,
    soft_loop_window: int,
) -> LoopRecoveryDecision:
    """Decide whether to substitute the brain's action for a recovery.

    Called only when :class:`LoopDetector` has flagged a soft loop AND
    the existing force-fill / force-submit substitutions didn't apply.

    Returns :class:`LoopRecoveryDecision` with ``forced_action=None``
    when no rule matches — the caller falls through to its legacy nudge
    path.
    """
    if not is_enabled():
        return LoopRecoveryDecision(None)

    # Rule 1: stuck CLICK on a focused input.
    #
    # The form-controller substitution would have fired first if the
    # focused field's label matched a pending plan value; reaching this
    # branch means the brain is clicking a field for which we have no
    # value (or the controller is exhausted). Force a key-press to
    # break the click loop.
    if action.action_type == ActionType.CLICK and focused_input is not None:
        # Prefer typing if we still have an unconsumed value AND the
        # controller's label-match logic didn't pick it (e.g., the
        # focused field's label is too vague to match).
        if pending_form_values:
            entry = pending_form_values[0]
            value = str(entry.get("value") or "")
            label = str(entry.get("label") or "")
            if value:
                return LoopRecoveryDecision(
                    forced_action=Action(
                        ActionType.TYPE, {"text": value},
                        reasoning=(
                            f"loop-recovery: typing pending plan value "
                            f"for {label!r} into focused field"
                        ),
                    ),
                    reason=REASON_TYPE_PENDING_VALUE,
                    detail=(
                        f"focused={focused_input.get('name') or focused_input.get('placeholder') or ''}; "
                        f"val_label={label}"
                    ),
                )
        # Submit-shaped click loop while a form input is focused: the
        # brain is hunting for a submit button it can't land. Pressing
        # Return submits the focused field's form via the keyboard (HTML
        # implicit submission) — Tab would only move focus off the very
        # field we want to submit from. Gated on a frozen frame so we
        # fire only when the click genuinely isn't working, mirroring
        # Rule 2's submit path for the unfocused case.
        if _is_submit_shaped(action, task) and _frame_window_stable(
            recent_frame_hashes, soft_loop_window
        ):
            return LoopRecoveryDecision(
                forced_action=Action(
                    ActionType.KEY_PRESS, {"keys": "Return"},
                    reasoning=(
                        "loop-recovery: pressing Return — submit-shaped "
                        "click loop with a form input focused"
                    ),
                ),
                reason=REASON_PRESS_RETURN_FOR_SUBMIT,
                detail=(
                    "submit-shaped focused click; focused="
                    + (
                        focused_input.get("name")
                        or focused_input.get("placeholder")
                        or "unknown"
                    )
                ),
            )
        # No value to type — Tab to the next field. Common pattern:
        # brain re-clicks the same field because it can't find the
        # right action; Tab gives the loop a way out.
        return LoopRecoveryDecision(
            forced_action=Action(
                ActionType.KEY_PRESS, {"keys": "Tab"},
                reasoning="loop-recovery: Tab to next field — stuck on focused-input click",
            ),
            reason=REASON_TAB_TO_NEXT_FIELD,
            detail=f"focused={focused_input.get('name') or 'unknown'}",
        )

    # Rule 2: submit-shaped CLICK loop with frame frozen.
    #
    # No focused input AND the action looks submit-shaped (reasoning
    # text or run-level task mentions submit/sign-in/etc.) AND the
    # frame hasn't changed across the recent window → press Return so
    # the form's submit handler fires instead of relying on the
    # button-click handler that's clearly broken.
    if (
        action.action_type == ActionType.CLICK
        and focused_input is None
        and _frame_window_stable(recent_frame_hashes, soft_loop_window)
        and _is_submit_shaped(action, task)
    ):
        return LoopRecoveryDecision(
            forced_action=Action(
                ActionType.KEY_PRESS, {"keys": "Return"},
                reasoning=(
                    "loop-recovery: pressing Return — submit-shaped click "
                    "loop with frozen frame"
                ),
            ),
            reason=REASON_PRESS_RETURN_FOR_SUBMIT,
            detail="submit-shaped click stuck on stable frame",
        )

    return LoopRecoveryDecision(None)


def _frame_window_stable(hashes: list[str], window: int) -> bool:
    """Last ``window`` frame hashes are all identical AND non-empty."""
    if len(hashes) < window:
        return False
    tail = hashes[-window:]
    if not all(tail):
        return False
    return all(h == tail[0] for h in tail)


def _is_submit_shaped(action: Action, task: str) -> bool:
    haystacks = [
        (action.reasoning or "").lower(),
        (task or "").lower(),
    ]
    return any(
        kw in haystack
        for haystack in haystacks
        for kw in _SUBMIT_KEYWORDS
    )
