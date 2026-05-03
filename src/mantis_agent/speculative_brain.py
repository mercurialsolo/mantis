"""Drop-in Brain wrapper that overlaps think() with the action settle window (#118, step 2).

Each call to ``SpeculativeBrain.think(frames, ...)`` follows this protocol:

  1. If there's a pending speculation from the previous call AND the
     ``frames[-1]`` we just received matches the frame the speculation
     started with (per :func:`speculation.frames_close_enough`), the
     speculative result is consumed — saves the entire synchronous
     ``think()`` round-trip.
  2. Otherwise, fall through to a synchronous ``inner_brain.think()``.
  3. Either way, kick off a *new* speculation using the just-received
     frames so the *next* call has something to consume.

The bet is that, for many steps, the screen between two consecutive
``think()`` calls stays equivalent — either the action was a visual
no-op (e.g. typing into a focused field), or the page had already
settled by the time the previous call returned. When that bet wins,
the runner skips a serial blocking round-trip on the brain.

This is **opt-in** — wrap a brain to enable it, leave it bare for the
old serial behavior. The wrapper satisfies the :class:`Brain` protocol
so the runner doesn't change. Cost-attribution dashboards can read
``result.brain_used`` (set to ``"speculative"`` or ``"synchronous"``)
to track hit rates.

Why this is safe:
- Validation uses the same dHash as the loop detector. With strict
  default tolerance (Hamming distance 0), only pixel-equivalent frames
  pass — speculative results never drive an action when the screen
  visibly changed.
- Any speculation exception is treated as an invalidation — the
  synchronous path always takes over.
- A pending speculation is canceled on invalidation so workers free up.

A future revision could loosen the validator (e.g. distance ≤ 4) when
the workflow tolerates animations; that's per-deployment tuning, not
a property of the wrapper.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from . import speculation

if TYPE_CHECKING:
    from PIL import Image

    from .actions import Action
    from .brain_protocol import Brain

logger = logging.getLogger(__name__)


# Public attribute set on every InferenceResult that comes through the
# wrapper. Lets cost-attribution see which path was taken without
# subclassing each brain's result type.
BRAIN_USED_ATTR: str = "brain_used"


class SpeculativeBrain:
    """Wraps a :class:`Brain` so consecutive ``think()`` calls can overlap.

    Args:
        inner: The underlying brain. Must expose the standard
            :class:`Brain` protocol.
        validator: Predicate ``(hash_at_start, hash_after_settle) -> bool``
            that decides whether a pending speculation is still valid.
            Defaults to :func:`speculation.frames_close_enough` with the
            strict ``max_hamming_distance=0`` tolerance.

    Attributes (read-only counters):
        hits — speculative results that validated and were consumed
        misses — pending speculations whose post-frame failed validation
        synchronous_starts — calls that fell through to sync think()
                             because no pending speculation existed
                             (typically just the first call after a reset)
    """

    def __init__(
        self,
        inner: "Brain",
        *,
        validator: Callable[[str, str], bool] = speculation.frames_close_enough,
    ) -> None:
        self.inner = inner
        self.validator = validator
        self._pending: speculation.Speculation | None = None
        self.hits: int = 0
        self.misses: int = 0
        self.synchronous_starts: int = 0

    # ── Brain protocol ─────────────────────────────────────────────────

    def load(self) -> None:
        """Defer entirely to the wrapped brain — no extra setup needed."""
        self.inner.load()

    def think(
        self,
        frames: "list[Image.Image]",
        task: str,
        action_history: "list[Action] | None" = None,
        screen_size: tuple[int, int] = (1920, 1080),
    ) -> Any:
        """Either consume a validated speculation or fall through to a
        synchronous ``inner.think()``. In both cases, kick off a new
        speculation against ``frames`` for the next call to consume.
        """
        result, brain_used = self._consume_pending(frames, task, action_history, screen_size)

        # Annotate result with attribution. Wrapped brains may return
        # frozen dataclasses or arbitrary types; setattr failures fall
        # through to the counters for observability.
        try:
            setattr(result, BRAIN_USED_ATTR, brain_used)
        except (AttributeError, TypeError):
            logger.debug(
                "SpeculativeBrain could not annotate result with brain_used "
                "(immutable type?); attribution will rely on counters",
            )

        # Kick off the next speculation. Even when the previous one was
        # consumed, we want a new one queued for the *next* call.
        self._pending = self._launch_next(frames, task, action_history, screen_size)
        return result

    # ── Internals ──────────────────────────────────────────────────────

    def _consume_pending(
        self,
        frames: "list[Image.Image]",
        task: str,
        action_history: "list[Action] | None",
        screen_size: tuple[int, int],
    ) -> tuple[Any, str]:
        """Try to consume a pending speculation. Returns ``(result, brain_used)``."""
        pending = self._pending
        self._pending = None  # we either consume it or discard it; either way clear

        if pending is None:
            # No prior speculation queued — first call after reset.
            self.synchronous_starts += 1
            return self._call_sync(frames, task, action_history, screen_size), "synchronous"

        if not frames:
            # Nothing to validate against — discard speculation and go sync.
            pending.cancel()
            self.misses += 1
            return self._call_sync(frames, task, action_history, screen_size), "synchronous"

        try:
            valid = pending.is_valid(frames[-1], validator=self.validator)
        except Exception as exc:  # noqa: BLE001 — observability never breaks runs
            logger.debug("speculation validation failed: %s — falling through to sync", exc)
            valid = False

        if not valid:
            pending.cancel()
            self.misses += 1
            return self._call_sync(frames, task, action_history, screen_size), "synchronous"

        # Validated — wait for the future and use its result. This
        # typically returns immediately because the speculation has been
        # running during the action's settle window.
        try:
            result = pending.result()
        except Exception as exc:  # noqa: BLE001 — synchronous fallback on any error
            logger.warning(
                "speculative think() raised — falling back to synchronous: %s", exc,
            )
            self.misses += 1
            return self._call_sync(frames, task, action_history, screen_size), "synchronous"

        self.hits += 1
        return result, "speculative"

    def _call_sync(
        self,
        frames: "list[Image.Image]",
        task: str,
        action_history: "list[Action] | None",
        screen_size: tuple[int, int],
    ) -> Any:
        return self.inner.think(
            frames=frames,
            task=task,
            action_history=action_history,
            screen_size=screen_size,
        )

    def _launch_next(
        self,
        frames: "list[Image.Image]",
        task: str,
        action_history: "list[Action] | None",
        screen_size: tuple[int, int],
    ) -> speculation.Speculation:
        """Submit the next speculation. Frames are passed by reference;
        callers should not mutate them while the speculation is in
        flight (the runner doesn't, so this is safe by convention)."""
        return speculation.start(
            self.inner,
            frames=frames,
            task=task,
            action_history=action_history,
            screen_size=screen_size,
        )

    # ── Lifecycle ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Drop any pending speculation and zero the counters. Useful at
        the start of a new task / episode so per-run hit rates are clean.
        Doesn't tear down the underlying brain."""
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        self.hits = 0
        self.misses = 0
        self.synchronous_starts = 0

    def hit_rate(self) -> float:
        total = self.hits + self.misses + self.synchronous_starts
        return self.hits / total if total else 0.0


__all__ = ["SpeculativeBrain", "BRAIN_USED_ATTR"]
