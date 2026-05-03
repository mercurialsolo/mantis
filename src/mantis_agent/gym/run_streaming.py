"""Unified streaming-mode runner — the canonical replacement for StreamingCUA (#119, step 2).

This module wires :class:`StreamerGymEnv` + :class:`GymRunner` into one
convenience function, so a host-desktop CUA session can use all of
GymRunner's feature-rich logic (plan persistence, feedback strings,
hybrid DOM execution, grounding, loop detection, world-model
trajectory schema, plan-aware reverse) without callers having to wire
the two pieces themselves.

This is the recommended entrypoint for new streaming-mode callers.
The legacy :class:`mantis_agent.agent.StreamingCUA` keeps working
unchanged — its async event-emission contract is still useful for the
viewer and main.py paths. New code should prefer the unified runner;
StreamingCUA can be migrated incrementally caller-by-caller.

Usage::

    from mantis_agent.brain_protocol import resolve_brain
    from mantis_agent.gym.run_streaming import run_streaming

    brain = resolve_brain("holo3")
    brain.load()

    result = run_streaming(brain=brain, task="Find the cheapest flight to Tokyo")
    print(result.success, result.total_steps, result.total_time)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from .runner import GymRunner, RunResult
from .streamer_env import StreamerGymEnv

if TYPE_CHECKING:
    from ..brain_protocol import Brain
    from ..executor import ActionExecutor
    from ..streamer import ScreenStreamer

logger = logging.getLogger(__name__)


def run_streaming(
    brain: "Brain",
    task: str,
    *,
    task_id: str = "streaming",
    streamer: "ScreenStreamer | None" = None,
    executor: "ActionExecutor | None" = None,
    settle_time: float = 0.5,
    max_steps: int = 50,
    frames_per_inference: int = 5,
    soft_loop_window: int = 3,
    hard_loop_window: int = 8,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    grounding: Any = None,
    seed: int | None = None,
) -> RunResult:
    """Run a streaming-mode CUA task using the canonical :class:`GymRunner` loop.

    Equivalent to manually constructing a :class:`StreamerGymEnv` and a
    :class:`GymRunner`, then calling ``runner.run(task)``. Provides a one-line
    entrypoint for the most common case: drive the host desktop with a single
    brain, no plan / DOM executor / page discovery.

    Args:
        brain: Any object satisfying :class:`Brain`. Wrap with
            :class:`SpeculativeBrain` / :class:`BrainLadder` if desired.
        task: Natural-language task description.
        task_id: Stable identifier for the run (logs / trajectory keying).
        streamer: Optional :class:`ScreenStreamer` instance. Default constructs one.
        executor: Optional :class:`ActionExecutor` instance. Default constructs one.
        settle_time: Seconds between executing an action and capturing the
            post-action frame. 0.5s matches the legacy ``StreamingCUA`` default.
        max_steps: Cap before forced termination.
        frames_per_inference: Number of recent frames the brain receives.
        soft_loop_window / hard_loop_window: Loop detection thresholds.
        on_step: Optional callback fired with one dict per agent step
            (``{"type": "step"|"action"|"thinking"|...}``). For a viewer.
        grounding: Optional grounding model. None disables refinement.
        seed: Forwarded to ``env.reset(seed=...)``.

    Returns:
        :class:`RunResult` with trajectory, termination reason, total time,
        reward, and the world-model schema fields populated by the runner.
    """
    env = StreamerGymEnv(streamer=streamer, executor=executor, settle_time=settle_time)
    runner = GymRunner(
        brain=brain,
        env=env,
        max_steps=max_steps,
        frames_per_inference=frames_per_inference,
        soft_loop_window=soft_loop_window,
        hard_loop_window=hard_loop_window,
        grounding=grounding,
        on_step=on_step,
    )
    try:
        return runner.run(task=task, task_id=task_id, seed=seed)
    finally:
        env.close()


__all__ = ["run_streaming"]
