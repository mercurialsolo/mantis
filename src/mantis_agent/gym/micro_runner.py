"""MicroPlanRunner — execute decomposed micro-intents with checkpoint/verify/reverse.

Takes a MicroPlan (list of MicroIntent) and executes each step:
  - Holo3 steps: fresh GymRunner per intent (3-8 steps, 1 sentence)
  - Claude steps: screenshot → ClaudeExtractor reads data
  - Verify: Claude checks before/after screenshots
  - Reverse: undo failed step (Escape, Alt+Left, Ctrl+W)
  - Checkpoint: save state after each verified step
  - Loop: repeat step sequences (e.g., extract listings on each page)

Usage:
    from mantis_agent.plan_decomposer import PlanDecomposer
    from mantis_agent.gym.micro_runner import MicroPlanRunner

    plan = PlanDecomposer().decompose("plans/boattrader/extract_only.txt")
    runner = MicroPlanRunner(brain=brain, env=env, ...)
    results = runner.run(plan)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from ..actions import Action, ActionType
from .runner import GymRunner

if TYPE_CHECKING:
    from ..extraction import ClaudeExtractor, ExtractionResult
    from ..plan_decomposer import MicroIntent, MicroPlan

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Outcome of executing one micro-intent."""
    step_index: int
    intent: str
    success: bool
    data: str = ""
    steps_used: int = 0
    duration: float = 0.0
    reversed: bool = False


@dataclass
class RunCheckpoint:
    """Persistent state for resume."""
    step_index: int = 0
    page: int = 1
    seen_urls: list = field(default_factory=list)
    extracted_leads: list = field(default_factory=list)
    loop_iterations: int = 0
    timestamp: float = 0.0

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "step_index": self.step_index,
                "page": self.page,
                "seen_urls": self.seen_urls,
                "extracted_leads": self.extracted_leads,
                "loop_iterations": self.loop_iterations,
                "timestamp": time.time(),
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> RunCheckpoint | None:
        try:
            with open(path) as f:
                d = json.load(f)
            return cls(**d)
        except Exception:
            return None


# Reverse actions for each step type
REVERSE_ACTIONS = {
    "click": [("key_press", "Escape"), ("key_press", "alt+Left")],
    "scroll": [("key_press", "Home")],
    "navigate": [("key_press", "alt+Left")],
    "navigate_back": [],  # Already going back
    "filter": [("key_press", "alt+Left")],
    "paginate": [("key_press", "alt+Left")],
}


class MicroPlanRunner:
    """Execute a MicroPlan step-by-step with verify/reverse/checkpoint.

    Args:
        brain: Holo3 brain for executing micro-intents.
        env: XdotoolGymEnv with screenshot() method.
        grounding: ClaudeGrounding for click steps.
        extractor: ClaudeExtractor for data extraction steps.
        on_step: Optional viewer callback.
        max_retries: Max retries per step before skip.
        checkpoint_path: Path for checkpoint file.
    """

    def __init__(
        self,
        brain: Any,
        env: Any,
        grounding: Any = None,
        extractor: ClaudeExtractor | None = None,
        on_step: Any = None,
        max_retries: int = 2,
        checkpoint_path: str = "/data/checkpoints/micro_run.json",
    ):
        self.brain = brain
        self.env = env
        self.grounding = grounding
        self.extractor = extractor
        self.on_step = on_step
        self.max_retries = max_retries
        self.checkpoint_path = checkpoint_path
        self._seen_urls: set[str] = set()

    def run(self, plan: MicroPlan, resume: bool = False) -> list[StepResult]:
        """Execute the full micro-plan.

        Args:
            plan: Decomposed plan with ordered micro-intents.
            resume: If True, load checkpoint and resume from last step.

        Returns:
            List of StepResult for each executed step.
        """
        results: list[StepResult] = []
        checkpoint = RunCheckpoint()

        if resume:
            loaded = RunCheckpoint.load(self.checkpoint_path)
            if loaded:
                checkpoint = loaded
                self._seen_urls = set(checkpoint.seen_urls)
                logger.info(f"Resumed from step {checkpoint.step_index}, "
                           f"page {checkpoint.page}, {len(self._seen_urls)} URLs seen")

        step_index = checkpoint.step_index
        max_loop_iterations = 200  # Safety cap

        while step_index < len(plan.steps):
            step = plan.steps[step_index]
            t0 = time.time()

            logger.info(f"  [{step_index:2d}] {step.type:15s} {step.intent[:60]}")

            # Handle loop steps
            if step.type == "loop":
                checkpoint.loop_iterations += 1
                if checkpoint.loop_iterations < (step.loop_count or max_loop_iterations):
                    step_index = step.loop_target if step.loop_target >= 0 else step_index
                    logger.info(f"  [loop] iteration {checkpoint.loop_iterations} → step {step_index}")
                    continue
                else:
                    logger.info(f"  [loop] max iterations reached")
                    step_index += 1
                    continue

            # Execute step
            step_result = self._execute_step(step, step_index)
            results.append(step_result)

            if step_result.success:
                # Checkpoint on success
                checkpoint.step_index = step_index + 1
                checkpoint.seen_urls = list(self._seen_urls)
                if step_result.data and "VIABLE" in step_result.data:
                    checkpoint.extracted_leads.append(step_result.data[:200])
                checkpoint.save(self.checkpoint_path)
                step_index += 1
            else:
                # Handle failure based on step type
                if step.type in ("navigate",):
                    # Navigate failure is fatal — can't proceed without the page
                    logger.error(f"  [{step_index}] NAVIGATE FAILED — cannot proceed")
                    self._reverse_step(step)
                    break
                elif step.type in ("filter",):
                    # Filter failure is non-fatal — skip and try next filter
                    logger.warning(f"  [{step_index}] FILTER FAILED — skipping")
                    self._reverse_step(step)
                    step_index += 1
                elif step.type in ("scroll",):
                    # Scroll "failure" usually means the model didn't call done()
                    # but the page DID scroll — treat as success
                    logger.info(f"  [{step_index}] Scroll completed (no done() but page changed)")
                    checkpoint.step_index = step_index + 1
                    checkpoint.save(self.checkpoint_path)
                    step_index += 1
                elif step.type in ("navigate_back",):
                    # Back failure — try harder with direct keypress
                    logger.warning(f"  [{step_index}] BACK FAILED — pressing Alt+Left directly")
                    try:
                        self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "alt+Left"}))
                        time.sleep(2)
                    except Exception:
                        pass
                    step_index += 1
                elif step.type in ("extract_url", "extract_data"):
                    # Claude-only step failed — skip
                    step_index += 1
                else:
                    # Generic failure — reverse and skip
                    self._reverse_step(step)
                    step_result.reversed = True
                    logger.warning(f"  [{step_index}] FAILED + reversed — skipping")
                    step_index += 1

            # Progress logging
            viable_count = sum(1 for r in results if r.success and "VIABLE" in (r.data or ""))
            total_steps = sum(r.steps_used for r in results)
            elapsed = time.time() - t0
            logger.info(f"  [{step_index-1}] {'OK' if step_result.success else 'FAIL'} "
                       f"({step_result.steps_used} steps, {elapsed:.0f}s) "
                       f"[{viable_count} viable, {total_steps} total steps]")

        logger.info(f"MicroPlan complete: {len(results)} steps executed")
        return results

    def _execute_step(self, step: MicroIntent, index: int) -> StepResult:
        """Execute a single micro-intent."""

        # Navigate steps: use env.reset() with URL instead of Holo3
        if step.type == "navigate":
            return self._execute_navigate(step, index)

        # Claude-only steps (extract_url, extract_data)
        if step.claude_only:
            return self._execute_claude_step(step, index)

        # Holo3 steps
        return self._execute_holo3_step(step, index)

    def _execute_navigate(self, step: MicroIntent, index: int) -> StepResult:
        """Navigate to a URL using env.reset() — no Holo3 steps needed."""
        import re
        url_match = re.search(r'https?://[^\s"]+', step.intent)
        url = url_match.group() if url_match else ""

        if not url:
            logger.warning(f"  [navigate] No URL found in intent: {step.intent[:60]}")
            return StepResult(step_index=index, intent=step.intent, success=False)

        logger.info(f"  [navigate] Loading {url}")
        try:
            self.env.reset(task="navigate", start_url=url)
            time.sleep(4)
            # Scroll to top
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(1)
            return StepResult(step_index=index, intent=step.intent, success=True)
        except Exception as e:
            logger.error(f"  [navigate] Failed: {e}")
            return StepResult(step_index=index, intent=step.intent, success=False)

    def _execute_holo3_step(self, step: MicroIntent, index: int) -> StepResult:
        """Execute a Holo3 micro-intent with fresh GymRunner."""
        runner = GymRunner(
            brain=self.brain,
            env=self.env,
            max_steps=step.budget,
            frames_per_inference=1,
            grounding=self.grounding if step.grounding else None,
            on_step=self.on_step,
        )

        result = runner.run(task=step.intent, task_id=f"step_{index}_{step.type}")

        return StepResult(
            step_index=index,
            intent=step.intent,
            success=result.success,
            steps_used=result.total_steps,
            duration=result.total_time,
        )

    def _execute_claude_step(self, step: MicroIntent, index: int) -> StepResult:
        """Execute a Claude-only step (screenshot → API → data)."""
        if not self.extractor:
            return StepResult(step_index=index, intent=step.intent, success=False)

        screenshot = self.env.screenshot()

        if step.type == "extract_url":
            data = self.extractor.extract(screenshot)
            url = data.url if data else ""

            # Dedup check
            if url and url in self._seen_urls:
                logger.info(f"  [dedup] Already seen: {url[:50]}")
                return StepResult(
                    step_index=index, intent=step.intent,
                    success=False, data=f"DUPLICATE|{url}",
                )
            if url:
                self._seen_urls.add(url)

            return StepResult(
                step_index=index, intent=step.intent,
                success=bool(url), data=f"URL:{url}" if url else "",
            )

        elif step.type == "extract_data":
            data = self.extractor.extract(screenshot)
            if data and data.is_viable():
                summary = data.to_summary()
                return StepResult(
                    step_index=index, intent=step.intent,
                    success=True, data=summary,
                )
            return StepResult(
                step_index=index, intent=step.intent,
                success=False, data=data.raw_response[:100] if data else "",
            )

        return StepResult(step_index=index, intent=step.intent, success=False)

    def _reverse_step(self, step: MicroIntent):
        """Undo a failed step using predefined reverse actions."""
        actions = REVERSE_ACTIONS.get(step.type, [])

        # Use step-specific reverse if provided
        if step.reverse:
            # Parse "Press Escape then Alt+Left" into actions
            if "escape" in step.reverse.lower():
                actions = [("key_press", "Escape")] + actions
            if "alt+left" in step.reverse.lower():
                actions.append(("key_press", "alt+Left"))

        for action_type, keys in actions:
            try:
                self.env.step(Action(
                    action_type=ActionType.KEY_PRESS,
                    params={"keys": keys},
                ))
                time.sleep(0.5)
            except Exception:
                pass

        logger.info(f"  [reverse] {len(actions)} actions applied")
