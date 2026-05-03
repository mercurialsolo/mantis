"""StreamerGymEnv — bridge ScreenStreamer + ActionExecutor to the GymEnvironment ABC (#119, step 1).

The repo currently runs two separate agent loops:
  * ``StreamingCUA`` (in ``mantis_agent.agent``) drives a host desktop via
    a continuous async :class:`~mantis_agent.streamer.ScreenStreamer` and
    fires actions through :class:`~mantis_agent.executor.ActionExecutor`.
  * ``GymRunner`` (in ``mantis_agent.gym.runner``) drives any
    :class:`~mantis_agent.gym.base.GymEnvironment` and contains the
    feature-rich logic (plan persistence, feedback strings, hybrid DOM
    execution, grounding, loop detection, world-model trajectory schema).

#119 wants one canonical loop. Instead of choosing between the two
observation models, this PR ships a thin **adapter** that makes the
streamer + executor pair satisfy the :class:`GymEnvironment` interface:

  reset(task)   → capture one frame (no async loop needed for sync stepping)
  step(action)  → executor.execute → settle → capture
  close()       → no-op (the underlying objects are stateless across runs)
  screen_size   → reported by the streamer's first capture

This is **purely additive** — no existing caller starts using it
automatically. The follow-on PR (#119 step 2) migrates ``StreamingCUA``
to delegate to ``GymRunner`` via this adapter, with an opt-in flag so
any OSWorld benchmark regression is reviewable in isolation.

Why use ``ScreenStreamer.capture_once`` instead of the continuous async
loop:
  * The :class:`GymEnvironment` contract is sync (``step`` returns the
    post-action observation directly). Async continuous capture buys
    nothing in that mode — and would force the adapter into an event
    loop the runner doesn't want.
  * The streamer's ``capture_once()`` already does the right thing: pulls
    one mss frame, applies the configured scale, returns immediately.
  * Callers that want async continuous capture (i.e. ``StreamingCUA``)
    keep using the streamer directly.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ..actions import Action
from .base import GymEnvironment, GymObservation, GymResult

if TYPE_CHECKING:
    from ..executor import ActionExecutor
    from ..streamer import ScreenStreamer

logger = logging.getLogger(__name__)


# Default screen-size used before the first capture. Matches the rest of
# the codebase (1920x1080 is the default everywhere a screen size is
# guessed). Replaced by the actual size after reset()/step() runs the
# first capture_once().
_DEFAULT_SCREEN_SIZE: tuple[int, int] = (1920, 1080)


class StreamerGymEnv(GymEnvironment):
    """:class:`GymEnvironment` that drives a host desktop via a
    :class:`ScreenStreamer` (capture) + :class:`ActionExecutor` (input).

    Args:
        streamer: A :class:`ScreenStreamer` for screenshot capture. If
            ``None``, one is created with default settings on first use.
            The continuous async loop is NOT started — only
            :meth:`capture_once` is called.
        executor: An :class:`ActionExecutor` for keyboard/mouse input.
            If ``None``, one is created with default settings on first use.
        settle_time: Seconds to sleep between executing an action and
            capturing the post-action frame. Matches the existing
            ``StreamingCUA.settle_time`` default of 0.5s.

    All three constructor args are optional and default to lazy
    construction so tests can pass mocks without dragging in the heavy
    ``[local-cua]`` extras.
    """

    def __init__(
        self,
        streamer: "ScreenStreamer | None" = None,
        executor: "ActionExecutor | None" = None,
        settle_time: float = 0.5,
    ) -> None:
        self._streamer = streamer
        self._executor = executor
        self.settle_time = settle_time
        # Set on the first capture_once(). Until then we report a
        # sensible default so callers reading screen_size() pre-reset
        # still get a valid tuple.
        self._screen_size: tuple[int, int] = _DEFAULT_SCREEN_SIZE
        self._closed = False

    # ── Lazy construction ─────────────────────────────────────────────

    def _get_streamer(self) -> "ScreenStreamer":
        if self._streamer is None:
            from ..streamer import ScreenStreamer
            self._streamer = ScreenStreamer()
        return self._streamer

    def _get_executor(self) -> "ActionExecutor":
        if self._executor is None:
            from ..executor import ActionExecutor
            self._executor = ActionExecutor()
        return self._executor

    # ── GymEnvironment protocol ────────────────────────────────────────

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        """Capture a single frame to represent the initial state. No
        device-level reset happens — the host desktop persists between
        runs and the caller is responsible for navigating to the right
        starting URL / app.

        ``kwargs`` is accepted for protocol compatibility (the runner
        passes ``task_id``, ``start_url``, ``seed``) but ignored — the
        host desktop has no concept of task IDs or seeded resets.
        """
        if self._closed:
            raise RuntimeError("StreamerGymEnv is closed; create a new instance")
        streamer = self._get_streamer()
        # capture_once also updates self._screen_size as a side effect
        # of mss.monitors[N] introspection.
        frame = streamer.capture_once()
        if streamer.screen_size != (0, 0):
            self._screen_size = streamer.screen_size
        # Sync the executor's clamp bounds with the actual screen.
        try:
            self._get_executor().screen_bounds = self._screen_size
        except Exception as exc:  # noqa: BLE001 — observability, not contract
            logger.debug("could not sync executor.screen_bounds: %s", exc)
        return GymObservation(screenshot=frame.image)

    def step(self, action: Action) -> GymResult:
        """Execute an action, wait for settle, capture and return.

        ``info`` is intentionally minimal in this adapter — the streamer
        has no concept of URL / focused_input the way a Playwright env
        does. Reward is always 0.0 (the runner injects per-task rewards
        via :class:`RewardFn`); ``done`` is always False (only the brain's
        ``ActionType.DONE`` terminates a host-desktop session).
        """
        if self._closed:
            raise RuntimeError("StreamerGymEnv is closed; create a new instance")
        execution = self._get_executor().execute(action)
        if not execution.success and execution.error:
            logger.warning("StreamerGymEnv action failed: %s", execution.error)
        if self.settle_time > 0:
            time.sleep(self.settle_time)
        frame = self._get_streamer().capture_once()
        info: dict[str, Any] = {
            "execution_success": execution.success,
            "execution_duration": execution.duration,
        }
        if execution.error:
            info["execution_error"] = execution.error
        return GymResult(
            observation=GymObservation(screenshot=frame.image),
            reward=0.0,
            done=False,
            info=info,
        )

    def close(self) -> None:
        """Mark the env closed. ScreenStreamer / ActionExecutor have no
        explicit teardown in their sync paths — they're idempotent objects.
        ``close()`` is here to satisfy the ABC and to fail-fast on use-
        after-close from buggy callers."""
        self._closed = True

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._screen_size

    # ── Convenience for #74 / observability ───────────────────────────

    def screenshot(self):
        """One-shot capture — used by the screenshot-cap observability
        path in :class:`MicroPlanRunner`. Equivalent to ``step()`` minus
        the action and settle, returning the PIL image directly.
        """
        if self._closed:
            raise RuntimeError("StreamerGymEnv is closed; create a new instance")
        return self._get_streamer().capture_once().image


__all__ = ["StreamerGymEnv"]
