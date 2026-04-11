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

    Includes two-tier loop detection:
    - Soft loop (3 repeats): Nudges the model with context about what it
      already did and what to do next. Form-aware: if an input field is
      focused, tells the model to type instead of clicking again.
    - Hard loop (8 repeats): Terminates the task.

    Args:
        brain: Any object implementing the Brain protocol (think method).
        env: A GymEnvironment instance.
        max_steps: Maximum steps before forced termination.
        frames_per_inference: Number of recent frames to feed the brain.
        soft_loop_window: Repeated actions before nudge.
        hard_loop_window: Repeated actions before termination.
    """

    def __init__(
        self,
        brain: Brain,
        env: GymEnvironment,
        max_steps: int = 50,
        frames_per_inference: int = 5,
        soft_loop_window: int = 3,
        hard_loop_window: int = 8,
    ):
        self.brain = brain
        self.env = env
        self.max_steps = max_steps
        self.frames_per_inference = frames_per_inference
        self.soft_loop_window = soft_loop_window
        self.hard_loop_window = hard_loop_window

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
        nudge: str = ""  # Appended to task when soft loop detected
        last_focused_input: dict | None = None

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

            # Build effective task: original + any nudge from loop detection
            effective_task = task + nudge if nudge else task

            t_infer = time.time()
            result = self.brain.think(
                frames=recent_frames,
                task=effective_task,
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

            # Track focused input for form-aware nudges
            focused_input = gym_result.info.get("focused_input")
            if focused_input is not None:
                last_focused_input = focused_input

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

            # Two-tier loop detection
            nudge = ""  # Reset each step

            if self._detect_repeat(action_history, self.hard_loop_window):
                logger.warning("Hard action loop detected — stopping")
                termination_reason = "loop"
                break

            if self._detect_repeat(action_history, self.soft_loop_window):
                nudge = self._build_nudge(action_history, last_focused_input)
                logger.info(f"Soft loop — nudge: {nudge[:100]}")

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

    def _detect_repeat(self, action_history: list[Action], window: int) -> bool:
        """Check if the last N actions are identical (type + params)."""
        if len(action_history) < window:
            return False
        recent = action_history[-window:]
        first = recent[0]
        return all(
            a.action_type == first.action_type and a.params == first.params
            for a in recent[1:]
        )

    @staticmethod
    def _build_nudge(action_history: list[Action], focused_input: dict | None) -> str:
        """Build a contextual nudge to break the model out of a loop.

        Form-aware: if an input field is focused, tells the model to type
        instead of clicking again. Otherwise, suggests a different approach.
        """
        last = action_history[-1]
        repeated_action = f"{last.action_type.value}({last.params})"

        # Form-aware nudge: input field is focused, model should type
        if focused_input and last.action_type == ActionType.CLICK:
            field_desc = (
                focused_input.get("placeholder")
                or focused_input.get("name")
                or focused_input.get("id")
                or focused_input.get("type")
                or "input"
            )
            is_empty = focused_input.get("empty", True)
            current_value = focused_input.get("value", "")

            if is_empty:
                return (
                    f"\n\nIMPORTANT: You have already clicked the '{field_desc}' field "
                    f"and it now has focus. Do NOT click it again. "
                    f"Your next action must be type_text() to enter the value. "
                    f"The field is empty and ready for input."
                )
            else:
                return (
                    f"\n\nIMPORTANT: The '{field_desc}' field is focused and already "
                    f"contains: \"{current_value}\". If you need to change it, "
                    f"first select all with key_press('ctrl+a'), then type_text() "
                    f"the new value. Do NOT click the same field again."
                )

        # Generic nudge: model is repeating the same action
        return (
            f"\n\nIMPORTANT: You have repeated the same action ({repeated_action}) "
            f"multiple times with no progress. This approach is not working. "
            f"Try a DIFFERENT action — for example: scroll to find the right element, "
            f"use keyboard navigation (Tab, Enter), type in a focused field, or "
            f"navigate to a different part of the page."
        )
