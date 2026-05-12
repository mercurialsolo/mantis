"""Deterministic Done-acceptance gate (#303).

Brains — Holo3 in particular — sometimes emit ``done(success=True)`` before
the actual workflow is complete:

* **Run 009 / 010**: ``done(success=True, summary='')`` after a string of
  waits, without ever engaging the workflow.
* **Run 023**: a fabricated summary claiming a downstream outcome
  ("Updated lead industry to Space Exploration") after only the click loop
  in front of the login form.
* **Per-step done-condition confusion**: model treats a step-local
  "Done when:" clause as the whole-task completion.

This gate runs **before** the optional model-based ``verify_done``. It uses
only signals the runner already has — summary text, plan progress, recent
actions / frames, force-fill state — so there's no vision call, no API
spend, no token cost.

Decision API::

    decision = check_done_acceptance(
        summary=action.params.get("summary", ""),
        plan=plan,
        plan_step_idx=plan_step_idx,
        recent_actions=action_history[-5:],
        recent_frame_hashes=[t.frame_hash for t in trajectory[-5:]],
        recent_urls=[t.observed_state.get("url", "") for t in trajectory[-5:]],
        pending_form_labels=remaining_force_fill_labels,
        required_summary_fields=plan_output_fields,
    )
    if not decision.accept:
        # Reject; record decision.reason in TrajectoryStep.done_rejected_reason.
        ...

A rejection is **not** the end of the run — the caller substitutes the
``done`` with a no-op ``wait`` so the brain gets another chance, mirroring
the existing model-based verifier path.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..actions import Action, ActionType


# Rejection reason codes — stable strings safe to log, surface on the API,
# and key reward analyses by. New codes append; never rename or repurpose.
REJECT_EMPTY_SUMMARY: str = "empty_summary"
REJECT_PLAN_STEPS_INCOMPLETE: str = "plan_steps_incomplete"
REJECT_PENDING_FORM_VALUES: str = "pending_form_values"
REJECT_SUMMARY_MISSING_FIELDS: str = "summary_missing_required_fields"
REJECT_NO_DELTA_AFTER_WAITS: str = "no_observed_delta_after_waits"
REJECT_NO_PROGRESS_IN_WINDOW: str = "no_progress_in_window"

REJECT_CODES: tuple[str, ...] = (
    REJECT_EMPTY_SUMMARY,
    REJECT_PLAN_STEPS_INCOMPLETE,
    REJECT_PENDING_FORM_VALUES,
    REJECT_SUMMARY_MISSING_FIELDS,
    REJECT_NO_DELTA_AFTER_WAITS,
    REJECT_NO_PROGRESS_IN_WINDOW,
)


@dataclass
class DoneAcceptanceDecision:
    """Outcome of one gate evaluation. Truthy when accepted."""

    accept: bool
    reason: str = ""
    detail: str = ""

    def __bool__(self) -> bool:
        return self.accept


def _summary_is_empty(summary: str) -> bool:
    """A summary that's empty or only whitespace describes nothing observable.

    Mantis brains routinely emit ``done(success=True, summary='')`` after
    a string of waits — the model gives up but signals success. Treat the
    empty summary as a hard reject.
    """
    return not (summary and summary.strip())


def _all_recent_are_wait(actions: list[Action], window: int) -> bool:
    if len(actions) < window:
        return False
    return all(a.action_type == ActionType.WAIT for a in actions[-window:])


def _hashes_stable(hashes: list[str], window: int) -> bool:
    """Last ``window`` frame hashes are all identical (and non-empty)."""
    if len(hashes) < window:
        return False
    tail = hashes[-window:]
    if not tail[-1]:
        return False
    return all(h == tail[-1] for h in tail)


def _urls_stable(urls: list[str], window: int) -> bool:
    if len(urls) < window:
        return False
    tail = urls[-window:]
    return all(u == tail[-1] for u in tail)


def check_done_acceptance(
    *,
    summary: str,
    plan: object | None = None,
    plan_step_idx: int = 0,
    recent_actions: list[Action] | None = None,
    recent_frame_hashes: list[str] | None = None,
    recent_urls: list[str] | None = None,
    pending_form_labels: list[str] | None = None,
    required_summary_fields: list[str] | None = None,
    wait_window: int = 3,
    progress_window: int = 5,
) -> DoneAcceptanceDecision:
    """Apply deterministic predicates to a candidate ``done(success=True)``.

    First-rejection-wins. Returns ``DoneAcceptanceDecision(accept=True)``
    when no predicate fires; the caller may then run the optional
    model-based ``verify_done`` for a second opinion.

    Predicates, in order:

    1. ``empty_summary`` — summary is empty / whitespace.
    2. ``plan_steps_incomplete`` — when a structured ``Plan`` is present,
       ``plan_step_idx`` must be at the last step.
    3. ``pending_form_values`` — credentials the runner extracted from
       the plan but never typed (run 023 pattern).
    4. ``summary_missing_required_fields`` — plan declares output-schema
       fields the summary doesn't mention.
    5. ``no_observed_delta_after_waits`` — last ``wait_window`` actions
       are all ``WAIT`` and the frame hash hasn't changed.
    6. ``no_progress_in_window`` — last ``progress_window`` steps show
       neither URL change nor frame change.
    """
    if _summary_is_empty(summary):
        return DoneAcceptanceDecision(
            False, REJECT_EMPTY_SUMMARY,
            "done(success=true) emitted with empty summary",
        )

    if plan is not None:
        steps = list(getattr(plan, "steps", None) or [])
        if steps and plan_step_idx < len(steps) - 1:
            remaining = len(steps) - plan_step_idx - 1
            return DoneAcceptanceDecision(
                False, REJECT_PLAN_STEPS_INCOMPLETE,
                f"plan has {remaining} step(s) remaining "
                f"(at idx {plan_step_idx} of {len(steps)})",
            )

    if pending_form_labels:
        labels = [str(lbl) for lbl in pending_form_labels if lbl]
        if labels:
            head = ", ".join(labels[:3])
            more = "" if len(labels) <= 3 else f" (+{len(labels) - 3} more)"
            return DoneAcceptanceDecision(
                False, REJECT_PENDING_FORM_VALUES,
                f"unconsumed form values: {head}{more}",
            )

    if required_summary_fields:
        lower_summary = summary.lower()
        missing = [
            f for f in required_summary_fields
            if f and f.lower() not in lower_summary
        ]
        if missing:
            head = ", ".join(missing[:5])
            more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
            return DoneAcceptanceDecision(
                False, REJECT_SUMMARY_MISSING_FIELDS,
                f"summary missing fields: {head}{more}",
            )

    actions = recent_actions or []
    hashes = recent_frame_hashes or []
    urls = recent_urls or []

    if (
        _all_recent_are_wait(actions, wait_window)
        and _hashes_stable(hashes, wait_window)
    ):
        return DoneAcceptanceDecision(
            False, REJECT_NO_DELTA_AFTER_WAITS,
            f"last {wait_window} actions are WAIT with stable frame",
        )

    if (
        len(actions) >= progress_window
        and _urls_stable(urls, progress_window)
        and _hashes_stable(hashes, progress_window)
    ):
        return DoneAcceptanceDecision(
            False, REJECT_NO_PROGRESS_IN_WINDOW,
            f"no URL or frame change in last {progress_window} steps",
        )

    return DoneAcceptanceDecision(True)
