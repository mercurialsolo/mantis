"""GymRunner — drives any GymEnvironment with a Mantis brain.

This is the generic evaluation loop, analogous to StreamingCUA.run() but
decoupled from local screen capture (ScreenStreamer) and local execution
(ActionExecutor). Instead, it uses the GymEnvironment abstraction for
observation and action.

The runner handles:
- Frame history (rolling window of recent screenshots)
- Action history (for brain context + loop detection)
- Loop detection (identical actions → early termination)
- Trajectory recording for post-hoc analysis
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image

from ..actions import Action, ActionType
from .base import GymEnvironment, GymObservation

logger = logging.getLogger(__name__)


class Brain(Protocol):
    """Protocol for any Mantis-compatible brain (Gemma4Brain, LlamaCppBrain, etc.)."""

    def think(
        self,
        frames: list[Image.Image],
        task: str,
        action_history: list[Action] | None = None,
        screen_size: tuple[int, int] = (1920, 1080),
    ) -> Any:
        """Run inference and return a result with .action and .thinking attributes."""
        ...


@dataclass
class TrajectoryStep:
    """A single step in the agent's trajectory."""

    step: int
    action: Action
    thinking: str
    reward: float
    done: bool
    inference_time: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class RunResult:
    """Final result of a GymRunner evaluation."""

    task: str
    task_id: str
    success: bool
    total_reward: float
    total_steps: int
    total_time: float
    trajectory: list[TrajectoryStep]
    termination_reason: str  # "done", "max_steps", "loop", "env_done"


class GymRunner:
    """Drives a Mantis brain against any GymEnvironment.

    Args:
        brain: Any object implementing the Brain protocol (think method).
        env: A GymEnvironment instance.
        max_steps: Maximum steps before forced termination.
        frames_per_inference: Number of recent frames to feed the brain.
        loop_window: Number of recent actions to check for loops.
    """

    def __init__(
        self,
        brain: Brain,
        env: GymEnvironment,
        max_steps: int = 50,
        frames_per_inference: int = 5,
        loop_window: int = 5,
    ):
        self.brain = brain
        self.env = env
        self.max_steps = max_steps
        self.frames_per_inference = frames_per_inference
        self.loop_window = loop_window

    def run(self, task: str, task_id: str = "default", seed: int | None = None) -> RunResult:
        """Execute a task and return the evaluation result.

        Args:
            task: Natural language task description.
            task_id: Environment-specific task identifier.
            seed: Optional seed for reproducibility.
        """
        logger.info(f"Starting task: {task!r} (id={task_id})")
        t0 = time.time()

        frame_history: list[Image.Image] = []
        action_history: list[Action] = []
        trajectory: list[TrajectoryStep] = []
        total_reward = 0.0

        # Reset environment
        reset_kwargs: dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            reset_kwargs["seed"] = seed

        obs = self.env.reset(task, **reset_kwargs)
        frame_history.append(obs.screenshot)

        termination_reason = "max_steps"

        for step_num in range(1, self.max_steps + 1):
            logger.info(f"--- Step {step_num}/{self.max_steps} ---")

            # Feed recent frames to the brain
            recent_frames = frame_history[-self.frames_per_inference:]

            t_infer = time.time()
            result = self.brain.think(
                frames=recent_frames,
                task=task,
                action_history=action_history,
                screen_size=self.env.screen_size,
            )
            inference_time = time.time() - t_infer

            action = result.action
            thinking = getattr(result, "thinking", "")
            logger.info(f"Action: {action} ({inference_time:.2f}s)")

            # Check if agent signals done
            if action.action_type == ActionType.DONE:
                success = action.params.get("success", False)
                trajectory.append(TrajectoryStep(
                    step=step_num,
                    action=action,
                    thinking=thinking,
                    reward=0.0,
                    done=True,
                    inference_time=inference_time,
                ))
                termination_reason = "done"
                return RunResult(
                    task=task,
                    task_id=task_id,
                    success=success,
                    total_reward=total_reward,
                    total_steps=step_num,
                    total_time=time.time() - t0,
                    trajectory=trajectory,
                    termination_reason=termination_reason,
                )

            # Execute action in the environment
            gym_result = self.env.step(action)
            action_history.append(action)
            frame_history.append(gym_result.observation.screenshot)
            total_reward += gym_result.reward

            trajectory.append(TrajectoryStep(
                step=step_num,
                action=action,
                thinking=thinking,
                reward=gym_result.reward,
                done=gym_result.done,
                inference_time=inference_time,
            ))

            # Environment signaled completion
            if gym_result.done:
                termination_reason = "env_done"
                break

            # Loop detection
            if self._detect_loop(action_history):
                logger.warning("Action loop detected — stopping")
                termination_reason = "loop"
                break

            # Trim frame history to prevent unbounded memory growth
            if len(frame_history) > self.frames_per_inference * 3:
                frame_history = frame_history[-self.frames_per_inference * 2:]

        total_time = time.time() - t0
        logger.info(
            f"Task finished: {termination_reason}, "
            f"{len(trajectory)} steps, {total_time:.1f}s, reward={total_reward}"
        )

        return RunResult(
            task=task,
            task_id=task_id,
            success=termination_reason == "env_done" and total_reward > 0,
            total_reward=total_reward,
            total_steps=len(trajectory),
            total_time=total_time,
            trajectory=trajectory,
            termination_reason=termination_reason,
        )

    def _detect_loop(self, action_history: list[Action]) -> bool:
        """Check if the agent is repeating the same action."""
        if len(action_history) < self.loop_window:
            return False
        recent = action_history[-self.loop_window:]
        first = recent[0]
        return all(
            a.action_type == first.action_type and a.params == first.params
            for a in recent[1:]
        )
