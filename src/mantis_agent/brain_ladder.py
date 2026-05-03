"""Brain fallback ladder (#124).

Wraps a primary + fallback brain and exposes the ``Brain`` protocol.
The runner / harness calls :meth:`force_fallback` after N retries on the
same step; the next :meth:`think` call routes to the fallback brain
instead of the primary, with cost attribution attached.

This is the *mechanism*; the *policy* (when to escalate) lives where the
context is — in the runner or test harness, where action history,
retry counters, and step state all live. Keeping the ladder dumb means:

* tests can drive escalation deterministically without simulating runner state
* operators can wire any escalation policy without changing this module

Usage::

    from mantis_agent.brain_ladder import BrainLadder
    from mantis_agent.brain_protocol import resolve_brain

    ladder = BrainLadder(
        primary=resolve_brain("holo3"),
        fallback=resolve_brain("claude"),
    )
    runner = MicroPlanRunner(brain=ladder, ...)

    # Inside the runner's retry loop, after detecting a stuck step:
    if step_retry_counts[step_index] >= 2:
        ladder.force_fallback()

    # Optionally reset between steps:
    ladder.reset()

The wrapped :class:`InferenceResult` gets a ``brain_used`` attribute
(``"primary"`` or ``"fallback"``) so the cost meter can attribute the
call. Brains that don't carry that field on their result class get one
attached via ``setattr`` — non-invasive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

    from .actions import Action
    from .brain_protocol import Brain

logger = logging.getLogger(__name__)


# Public attribute name set on every InferenceResult that goes through
# the ladder. Lets the cost meter / dashboards count primary vs fallback
# calls without needing to subclass each brain's InferenceResult.
BRAIN_USED_ATTR: str = "brain_used"


@dataclass
class LadderAttempt:
    """One think() call through the ladder. Carries which brain handled
    it so callers can dashboard the escalation distribution."""

    brain_used: str  # "primary" | "fallback"
    duration: float


class BrainLadder:
    """Ladder of two brains. Routes to primary by default; the caller
    flips a flag to escalate to the fallback for one or more think()
    calls.

    The wrapper is stateful (the ``_use_fallback`` flag survives across
    think() calls) but the underlying brains are stateless w.r.t. each
    other — escalating doesn't reset the fallback brain or share any
    context between them.
    """

    def __init__(
        self,
        primary: "Brain",
        fallback: "Brain",
        *,
        sticky_fallback: bool = False,
    ) -> None:
        """Args:
            primary: The default brain.
            fallback: Used when ``force_fallback()`` was called since the
                last :meth:`reset`.
            sticky_fallback: If True, once :meth:`force_fallback` is
                called the ladder stays on the fallback brain until
                :meth:`reset`. If False (default), the fallback is used
                for exactly one think() call and the ladder reverts to
                the primary on the next call. The runner's policy
                decides which behavior is right per workflow — sticky
                makes sense after a hard escalation; one-shot makes
                sense for surgical interventions.
        """
        self.primary = primary
        self.fallback = fallback
        self.sticky_fallback = sticky_fallback
        self._use_fallback = False
        self.attempts: list[LadderAttempt] = []

    # ── Brain protocol ─────────────────────────────────────────────────

    def load(self) -> None:
        """Load both brains. Each ``load()`` is documented as idempotent
        on the protocol so calling them eagerly is safe — operators get
        fast-fail config errors from either side at startup, not after
        the first escalation."""
        self.primary.load()
        try:
            self.fallback.load()
        except Exception as exc:  # noqa: BLE001 — fallback can be lazy
            logger.warning(
                "BrainLadder fallback failed to load (will retry on first use): %s",
                exc,
            )

    def think(
        self,
        frames: "list[Image.Image]",
        task: str,
        action_history: "list[Action] | None" = None,
        screen_size: tuple[int, int] = (1920, 1080),
    ) -> Any:
        import time as _time

        if self._use_fallback:
            brain = self.fallback
            brain_used = "fallback"
        else:
            brain = self.primary
            brain_used = "primary"

        t0 = _time.monotonic()
        try:
            result = brain.think(
                frames=frames,
                task=task,
                action_history=action_history,
                screen_size=screen_size,
            )
        finally:
            self.attempts.append(
                LadderAttempt(
                    brain_used=brain_used,
                    duration=_time.monotonic() - t0,
                )
            )
            # One-shot fallback: after this think(), revert to primary.
            # Sticky fallback: stay on fallback until reset().
            if self._use_fallback and not self.sticky_fallback:
                self._use_fallback = False

        # Annotate the result so cost-attribution sees which brain ran.
        # ``setattr`` is non-invasive — works on dataclasses, dicts,
        # SimpleNamespace, anything attribute-settable. If the result is
        # immutable (frozen dataclass), the attribute set will raise; we
        # swallow that since attribution is observability, not contract.
        try:
            setattr(result, BRAIN_USED_ATTR, brain_used)
        except (AttributeError, TypeError):
            logger.debug(
                "BrainLadder could not annotate result with brain_used "
                "(immutable type?); cost attribution will rely on .attempts",
            )
        return result

    # ── Escalation control ─────────────────────────────────────────────

    def force_fallback(self) -> None:
        """Route the *next* think() call to the fallback brain.

        Idempotent — calling twice has the same effect as once. Effect
        clears either after one think() call (default) or on
        :meth:`reset` (when ``sticky_fallback=True``).
        """
        self._use_fallback = True

    def reset(self) -> None:
        """Drop the fallback flag (and forget the attempt log if you
        want a fresh per-step view — call :meth:`reset_attempts` for
        that)."""
        self._use_fallback = False

    def reset_attempts(self) -> None:
        """Clear the attempt log. Independent from :meth:`reset` so
        callers can keep the fallback flag while wiping observability
        state, or vice versa."""
        self.attempts.clear()

    # ── Diagnostics ────────────────────────────────────────────────────

    @property
    def is_using_fallback(self) -> bool:
        return self._use_fallback

    def primary_calls(self) -> int:
        return sum(1 for a in self.attempts if a.brain_used == "primary")

    def fallback_calls(self) -> int:
        return sum(1 for a in self.attempts if a.brain_used == "fallback")


__all__ = ["BrainLadder", "LadderAttempt", "BRAIN_USED_ATTR"]
