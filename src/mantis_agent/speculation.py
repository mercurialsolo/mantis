"""Speculative brain inference primitive (#118, step 1).

Currently, the runner's perception → reasoning → action cycle is fully
serial: action.execute() → settle_time → frames captured → brain.think().
For click/type-heavy paths the brain is idle during the entire settle
window, even though the streamer keeps producing frames it could already
reason about.

This module ships the **building block** for overlap: a :class:`Speculation`
wrapping a `brain.think()` call started before the action settles, with
a :meth:`is_valid` check the runner can use to decide between trusting
the speculative output and discarding it.

Validation strategy: a perceptual hash distance check between the frame
the speculation started with and the frame after settle. If they're
"close enough" (default: equal — the conservative choice) the
speculation is valid. The runner can tune the threshold per workflow.

Pure infrastructure — does not touch the runner. A follow-on PR wires
this into the GymRunner / MicroPlanRunner loops with explicit opt-in
config so a regression on the OSWorld 83% path is reviewable in
isolation.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from .loop_detector import phash_64

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


# Module-level thread pool. We use a single shared pool across all live
# Speculation instances to bound concurrency at the process level rather
# than per-runner. Default of 2 workers is enough for the agent's serial
# step pattern — speculative + actual think() never overlap by more
# than one cycle.
_DEFAULT_WORKERS: int = 2
_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    """Lazy-init the shared executor. Idempotent."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=_DEFAULT_WORKERS,
            thread_name_prefix="speculation",
        )
    return _executor


def shutdown_executor(wait: bool = True) -> None:
    """Stop the speculation pool — for clean process shutdown / tests."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=wait)
        _executor = None


# ── Hamming distance over the dHash bits ────────────────────────────────


def _hamming_distance(a: str, b: str) -> int:
    """Compare two phash_64 hex digests by bit difference. Both arguments
    must be the same length (17 hex chars in current dHash format). Empty
    or unequal-length inputs return a large number so callers always treat
    them as "different".
    """
    if not a or not b or len(a) != len(b):
        return 64  # treat as max-distance
    try:
        ai = int(a, 16)
        bi = int(b, 16)
    except ValueError:
        return 64
    return bin(ai ^ bi).count("1")


def frames_close_enough(
    hash_at_start: str,
    hash_after_settle: str,
    *,
    max_hamming_distance: int = 0,
) -> bool:
    """Default validation predicate.

    With ``max_hamming_distance=0`` (the conservative default) only
    pixel-equivalent frames pass — useful when the brain's reasoning is
    very position-sensitive. Loosen the bound (e.g. 4-8) when the
    workflow tolerates animations and minor rendering differences.
    """
    return _hamming_distance(hash_at_start, hash_after_settle) <= max_hamming_distance


# ── Speculation handle ─────────────────────────────────────────────────


@dataclass
class _SpeculationStats:
    """Per-speculation metrics for dashboards. Optional read by callers."""

    submitted_at: float
    frame_hash_at_start: str
    valid_on_check: bool | None = None
    distance_on_check: int | None = None


class Speculation:
    """Pending ``brain.think()`` call started before the action settled.

    Lifecycle:
      1. ``start(brain, frames, ...)`` — submit on a worker thread.
      2. The runner executes the action and waits for settle.
      3. After settle: ``is_valid(post_frame)`` decides if the speculative
         result is usable.
      4. ``result()`` blocks until the future completes (typically
         already done by the time we check).
      5. If invalidated: ``cancel()`` releases the worker.

    The class is intentionally NOT a context manager — the runner needs
    to make the validity decision asynchronously w.r.t. the speculative
    call's lifecycle, and a `with` block would defeat that.
    """

    def __init__(
        self,
        future: Future,
        frame_hash_at_start: str,
        submitted_at: float,
    ) -> None:
        self._future = future
        self._frame_hash_at_start = frame_hash_at_start
        self.stats = _SpeculationStats(
            submitted_at=submitted_at,
            frame_hash_at_start=frame_hash_at_start,
        )

    # ── Validation ─────────────────────────────────────────────────────

    def is_valid(
        self,
        post_settle_frame: "Image.Image",
        validator: Callable[[str, str], bool] = frames_close_enough,
    ) -> bool:
        """Compare the post-settle frame to the frame the speculation
        started with. Returns True when the validator says they're close
        enough that the speculative result is still meaningful.

        Records the distance + decision on ``self.stats`` for dashboards.
        """
        try:
            after_hash = phash_64(post_settle_frame)
        except Exception as exc:  # noqa: BLE001 — observability never breaks runs
            logger.debug("speculation post-frame hash failed: %s", exc)
            self.stats.valid_on_check = False
            return False
        ok = bool(validator(self._frame_hash_at_start, after_hash))
        self.stats.valid_on_check = ok
        self.stats.distance_on_check = _hamming_distance(
            self._frame_hash_at_start, after_hash
        )
        return ok

    # ── Result access ──────────────────────────────────────────────────

    def result(self, timeout: float | None = None) -> Any:
        """Block until the speculative think() call returns. ``timeout``
        is in seconds; ``None`` means wait forever.

        Re-raises any exception the underlying think() raised.
        """
        return self._future.result(timeout=timeout)

    def done(self) -> bool:
        return self._future.done()

    def cancel(self) -> bool:
        """Best-effort cancel. Returns whether cancellation succeeded —
        a future already executing on a worker can't be canceled. Safe
        to call regardless; returns False if the call already started.
        """
        return self._future.cancel()


# ── Entry point ────────────────────────────────────────────────────────


def start(
    brain: Any,
    frames: "list[Image.Image]",
    task: str,
    action_history: Any = None,
    screen_size: tuple[int, int] = (1920, 1080),
) -> Speculation:
    """Kick off a speculative ``brain.think()`` call.

    Computes the frame_hash up-front so :meth:`Speculation.is_valid`
    can compare against the post-settle frame without re-hashing the
    initial frame.

    Returns a :class:`Speculation` handle the runner can poll. The
    underlying call runs on the shared module thread pool; canceled
    speculations release the worker as soon as `cancel()` is called
    (only effective if the worker hasn't picked up the future yet).
    """
    import time

    if not frames:
        # No frames means the validator can't compare — degrade safely
        # by returning a Speculation whose hash is empty so any
        # post-frame check fails the validator.
        last_frame_hash = ""
    else:
        last_frame_hash = phash_64(frames[-1])

    submitted_at = time.monotonic()

    future = _get_executor().submit(
        brain.think,
        frames=frames,
        task=task,
        action_history=action_history,
        screen_size=screen_size,
    )
    return Speculation(
        future=future,
        frame_hash_at_start=last_frame_hash,
        submitted_at=submitted_at,
    )


__all__ = [
    "Speculation",
    "frames_close_enough",
    "shutdown_executor",
    "start",
]
