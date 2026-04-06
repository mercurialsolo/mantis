"""StreamingCUA — the main agent that fuses perception, reasoning, and action.

This is the key architectural difference from Agent-S and other CUA frameworks:

Traditional (decoupled):
    screenshot → grounding_model → reasoning_model → exec → screenshot → repeat
    (Each step is serial. ~3-5 seconds per cycle. Model never sees transitions.)

StreamingCUA (unified):
    continuous_frames → gemma4(perceive+reason+act) → execute → model sees results
    (One model, one pass. Frame buffer gives temporal context. Tight feedback loop.)

The agent loop:
1. ScreenStreamer continuously fills a frame buffer in the background
2. Each cycle, Gemma4Brain receives recent frames + task + action history
3. Brain outputs a structured action (click, type, scroll, etc.)
4. ActionExecutor performs the action immediately
5. Next cycle, the frame buffer naturally contains the action's consequences
6. Repeat until done or max steps reached

For OSWorld benchmark performance:
- Temporal context lets the model understand loading states and animations
- Single model avoids error propagation between grounding and reasoning
- Action history prevents loops (model knows what it already tried)
- Thinking mode enables multi-step planning for complex tasks
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .actions import Action, ActionType
from .brain import Gemma4Brain, InferenceResult
from .executor import ActionExecutor, ExecutionResult
from .streamer import ScreenStreamer

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of a single agent step (one perception-reasoning-action cycle)."""

    step: int
    inference: InferenceResult
    execution: ExecutionResult
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """Final result after the agent completes a task."""

    task: str
    success: bool
    summary: str
    steps: list[StepResult]
    total_time: float
    total_steps: int


class StreamingCUA:
    """The streaming computer use agent.

    Orchestrates the continuous perception-action loop where Gemma4 watches
    the screen and acts in real-time.

    Args:
        brain: The Gemma4Brain instance (handles inference).
        streamer: The ScreenStreamer instance (handles frame capture).
        executor: The ActionExecutor instance (handles action execution).
        max_steps: Maximum steps before forcing termination.
        frames_per_inference: Number of recent frames to feed the model each cycle.
        settle_time: Seconds to wait after an action before the next inference,
                    letting the screen settle (animations, renders, etc.).
    """

    def __init__(
        self,
        brain: Gemma4Brain,
        streamer: ScreenStreamer | None = None,
        executor: ActionExecutor | None = None,
        max_steps: int = 50,
        frames_per_inference: int = 5,
        settle_time: float = 0.5,
    ):
        self.brain = brain
        self.streamer = streamer or ScreenStreamer()
        self.executor = executor or ActionExecutor()
        self.max_steps = max_steps
        self.frames_per_inference = frames_per_inference
        self.settle_time = settle_time
        self._action_history: list[Action] = []
        self._steps: list[StepResult] = []

    async def run(self, task: str) -> AgentResult:
        """Execute a task using the streaming perception-action loop.

        Args:
            task: Natural language description of what to accomplish.

        Returns:
            AgentResult with success status, summary, and step history.
        """
        logger.info(f"Starting task: {task}")
        t0 = time.time()
        self._action_history.clear()
        self._steps.clear()

        # Start continuous screen capture
        await self.streamer.start()

        # Give the buffer a moment to fill with initial frames
        await asyncio.sleep(1.0 / self.streamer.fps * 3)

        # Update executor with actual screen bounds
        if self.streamer.screen_size != (0, 0):
            self.executor.screen_bounds = self.streamer.screen_size

        try:
            result = await self._run_loop(task)
        finally:
            await self.streamer.stop()

        total_time = time.time() - t0
        logger.info(
            f"Task {'completed' if result.success else 'failed'} "
            f"in {result.total_steps} steps ({total_time:.1f}s)"
        )
        return result

    async def _run_loop(self, task: str) -> AgentResult:
        """The core perception-action loop."""
        for step_num in range(1, self.max_steps + 1):
            logger.info(f"─── Step {step_num}/{self.max_steps} ───")

            # 1. PERCEIVE — grab recent frames from the rolling buffer
            frames = self.streamer.get_recent_frames(self.frames_per_inference)
            if not frames:
                # Buffer empty (shouldn't happen, but be safe)
                self.streamer.capture_once()
                frames = self.streamer.get_recent_frames(1)

            # 2. THINK — Gemma4 perceives + reasons + selects action in one pass
            loop = asyncio.get_running_loop()
            inference = await loop.run_in_executor(
                None,
                lambda: self.brain.think(
                    frames=frames,
                    task=task,
                    action_history=self._action_history,
                    screen_size=self.streamer.screen_size,
                ),
            )

            action = inference.action
            logger.info(f"Action: {action}")
            if inference.thinking:
                logger.debug(f"Thinking: {inference.thinking[:200]}...")

            # 3. Check for task completion
            if action.action_type == ActionType.DONE:
                success = action.params.get("success", False)
                summary = action.params.get("summary", "Task ended")
                self._steps.append(
                    StepResult(
                        step=step_num,
                        inference=inference,
                        execution=ExecutionResult(action=action, success=True),
                    )
                )
                return AgentResult(
                    task=task,
                    success=success,
                    summary=summary,
                    steps=self._steps,
                    total_time=time.time(),
                    total_steps=step_num,
                )

            # 4. ACT — execute the action on the computer
            execution = self.executor.execute(action)
            self._action_history.append(action)

            self._steps.append(
                StepResult(step=step_num, inference=inference, execution=execution)
            )

            if not execution.success:
                logger.warning(f"Action failed: {execution.error}")

            # 5. SETTLE — let the screen update before next inference
            # The streamer keeps capturing during this time, so the model
            # will see the action's consequences in the next cycle's frames
            await asyncio.sleep(self.settle_time)

            # Detect loops: if the last 5 actions are identical, we're stuck
            if self._detect_loop():
                logger.warning("Action loop detected — breaking out")
                return AgentResult(
                    task=task,
                    success=False,
                    summary="Agent detected a loop and stopped",
                    steps=self._steps,
                    total_time=time.time(),
                    total_steps=step_num,
                )

        # Max steps reached
        return AgentResult(
            task=task,
            success=False,
            summary=f"Max steps ({self.max_steps}) reached without completion",
            steps=self._steps,
            total_time=time.time(),
            total_steps=self.max_steps,
        )

    def _detect_loop(self, window: int = 5) -> bool:
        """Check if the agent is stuck repeating the same action."""
        if len(self._action_history) < window:
            return False
        recent = self._action_history[-window:]
        # All recent actions are the same type with the same params
        first = recent[0]
        return all(
            a.action_type == first.action_type and a.params == first.params
            for a in recent[1:]
        )
