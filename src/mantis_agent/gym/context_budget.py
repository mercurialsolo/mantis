"""Per-context sub-goal budget (issue #254).

Fourth tactical sibling in the recipe-orchestrator integration
series — #246 (recipe-rejection skip), #250 (navigation-primitive
halt skip), #248 (exploration substrate), and now this. Same
underlying mechanism, third trigger source: the runner tracks how
many sub-goals (one per :meth:`MicroPlanRunner.run` invocation)
have executed against a given URL anchor. When the counter would
exceed a recipe-author bound, the next :meth:`run` short-circuits
with a synthetic ``StepResult`` carrying ``skip=True /
skip_reason='listing_budget_exceeded'``. Host's existing skip-
handling path promotes that to ``status: skipped`` and the
orchestrator advances past the over-budgeted context.

Why this had to live runner-side rather than as a host-side
counter:

- LLM orchestrators don't obey plan-text counting rules. Three
  separate rule-text patches against staffai's BoatTrader plan
  ("RETRY POLICY", "FORBIDDEN tile-read list", "AT MOST 3
  sub-goals per listing") have all been overridden in live runs.
  The orchestrator treats them as preferences.
- Tool-result envelopes that *change downstream state* are the
  only mechanism that reliably steers orchestrator behavior. The
  #246 / #250 envelopes work end-to-end in production today.
- Mantis sits at the intersection of intent stream + page state +
  skip envelope emission. Anywhere else in the stack (the recipe
  doesn't know URLs, the host orchestrator doesn't know its own
  sub-goal count) is missing one of those signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# Closed enumeration — typed for clarity; validated at construction
# time so a typo'd config doesn't silently fall through to
# ``emit_skip``.
OnExceededMode = Literal["emit_skip", "halt", "log_only"]


_VALID_MODES: tuple[str, ...] = ("emit_skip", "halt", "log_only")


@dataclass
class ContextBudget:
    """Per-context sub-goal bound carried on :class:`MicroPlanRunner`.

    ``max_sub_goals_per_url`` — max ``run()`` invocations the runner
    will execute against a given URL anchor before short-circuiting.
    ``None`` (default) disables the per-URL bound.

    ``max_sub_goals_per_iteration`` — max ``run()`` invocations
    across all URLs in the runner's lifetime. Trip this and every
    subsequent run skips. ``None`` (default) disables.

    ``on_exceeded`` — what happens when a bound trips:
      * ``"emit_skip"`` (default, production): return a synthetic
        ``StepResult(skip=True, skip_reason='listing_budget_exceeded')``
        without invoking the executor. Matches the #246 / #250
        envelope pattern.
      * ``"halt"``: return an empty results list and mark
        ``_final_status='halted'``. Host's halt path applies as
        usual; no skip-semantic surfaced.
      * ``"log_only"``: log a WARNING but execute normally. Useful
        for shadow-mode evaluation — see what bounds we'd hit
        without changing production behavior.
    """

    max_sub_goals_per_url: int | None = None
    max_sub_goals_per_iteration: int | None = None
    on_exceeded: OnExceededMode = "emit_skip"

    def __post_init__(self) -> None:
        if self.on_exceeded not in _VALID_MODES:
            raise ValueError(
                f"on_exceeded must be one of {_VALID_MODES}, "
                f"got {self.on_exceeded!r}"
            )


__all__ = ["ContextBudget", "OnExceededMode"]
