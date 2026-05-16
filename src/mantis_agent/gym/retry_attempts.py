"""Shared retry-attempt rendering — items 2 + 7 of roadmap #435.

When a plan step fails (no_state_change / wrong_target / brain_loop_
exhausted), the runner records the failed (x, y) coordinates + label
into ``MicroPlanRunner._step_failure_history[step_index]``. On retry,
brain adapters render these structured records in the prompt so the
model can refute the same coordinates / patterns.

This module centralises:

1. The window cap — how many prior attempts to surface (older entries
   add noise without signal, and the brain's context is the scarcer
   resource than runner memory).
2. The line format — what one attempt looks like in prose:
   ``clicked (x, y) targeting "<matched>" → <outcome>``.
3. The block render — full ``Recent attempts on this sub-goal:``
   block ready to splice into a brain prompt.

Holo3, Claude, and Fara adapters all call ``render_attempts_block``
so the rendered shape is identical regardless of which brain reads
the prompt. Without a shared formatter the three adapters drifted
on outcome verbs / structure / window cap (the original
``holo3.py``-private helper from #440 item C).
"""

from __future__ import annotations

# How many prior-failure records to surface to the brain. Older
# attempts add noise without signal; the brain's prompt is the
# scarcer resource than runner memory.
_PRIOR_ATTEMPTS_WINDOW: int = 3


# Failure-class → human-readable outcome verb. Keyed on the ``kind``
# the runner stamps when demoting a step. Unknown classes fall back
# to ``failed (<kind>)`` so a refactor that adds a new class still
# renders something legible.
_OUTCOME_VERBS: dict[str, str] = {
    "no_state_change": "no observable state change (click registered but page didn't react)",
    "wrong_target": "wrong target (click changed state but not toward the success criterion)",
    "brain_loop_exhausted": "brain ran out of moves before reaching the goal",
    "selector_miss": "the chosen coordinates didn't hit an interactive element",
}


def format_prior_attempt(record: dict) -> str:
    """One human-readable line per prior failed attempt.

    Format: ``clicked (x, y) targeting "<matched_label>" → <outcome>``.

    The outcome verb is keyed off ``record["kind"]`` (the failure
    class the runner stamps when demoting). Falls back to a generic
    ``failed (<kind>)`` when the kind is unknown.

    Defensive on every field — a partial record (e.g. no
    ``matched_label``) still produces a sensible line rather than
    a stringified ``None``.
    """
    x = record.get("x", "?")
    y = record.get("y", "?")
    matched = str(record.get("matched_label") or record.get("label") or "").strip()
    kind = str(record.get("kind") or "").strip()
    reason = str(record.get("reason") or "").strip()
    outcome = _OUTCOME_VERBS.get(kind, f"failed ({kind or 'unknown'})")
    target_clause = f' targeting "{matched}"' if matched else ""
    detail = f"; {reason[:120]}" if reason else ""
    return f"clicked ({x}, {y}){target_clause} → {outcome}{detail}"


def render_attempts_block(
    attempts: list[dict] | None,
    *,
    window: int = _PRIOR_ATTEMPTS_WINDOW,
    header: str = "Recent attempts on this sub-goal (do NOT repeat these coordinates / patterns)",
) -> str:
    """Render an outcome-tagged ``Recent attempts...`` block, or ``""``
    when there's nothing to surface.

    Caller appends the returned block (when non-empty) into the
    executor prompt. Empty input — None, empty list, all-malformed
    entries — returns the empty string so the caller's "if it's non-
    empty, append" splice stays one-line.

    Window cap: only the most-recent ``window`` (default 3) records
    are rendered. The brain doesn't benefit from seeing every miss
    in a long retry chain — only the recent pattern.
    """
    if not attempts:
        return ""
    recent = [r for r in attempts if isinstance(r, dict)][-max(1, int(window)):]
    if not recent:
        return ""
    lines = [f"  {i + 1}. {format_prior_attempt(r)}" for i, r in enumerate(recent)]
    return f"{header}:\n" + "\n".join(lines)


__all__ = [
    "format_prior_attempt",
    "render_attempts_block",
]
