"""WorkflowRunner — dynamic loop execution with pagination.

Instead of unrolling loops into N fixed sections, WorkflowRunner
iterates dynamically: process items until none remain, then paginate.

Each iteration is a bounded GymRunner.run() call with independent
retry support. The model signals "no more items" or "no more pages"
via its terminate() message to control loop flow.

Usage:
    from mantis_agent.gym.workflow_runner import WorkflowRunner, LoopConfig

    config = LoopConfig(
        iteration_intent="Process the {ORDINAL} listing: ...",
        pagination_intent="Click the Next page button",
    )
    runner = WorkflowRunner(brain=brain, env=env, loop_config=config)
    results = runner.run_loop()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .runner import GymRunner

logger = logging.getLogger(__name__)

ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
}

NO_MORE_SIGNALS = [
    "no more", "no listing", "no items", "no results",
    "empty page", "end of", "last page", "no next",
    "all processed", "none found", "none left",
]


@dataclass
class LoopConfig:
    """Configuration for a dynamic loop."""
    iteration_intent: str       # Template with {ORDINAL}, {N} placeholders
    pagination_intent: str = "Scroll to the bottom and click the Next page button or '>' arrow. If there is no next page, call terminate('failure') with 'no more pages'."
    max_iterations: int = 50    # Safety cap
    max_pages: int = 10
    max_steps_per_iteration: int = 60
    max_retries_per_iteration: int = 2
    max_steps_pagination: int = 20


@dataclass
class IterationResult:
    """Result of a single loop iteration."""
    iteration: int
    page: int
    success: bool
    data: str               # Model's terminate summary (extracted data)
    no_more_items: bool     # Model signaled end of items on page
    steps: int
    duration: float


class WorkflowRunner:
    """Drives dynamic loops and pagination over GymRunner.

    Args:
        brain: Brain backend (any with think() method).
        env: GymEnvironment (PlaywrightGymEnv etc).
        loop_config: Loop parameters.
        session_name: For session persistence.
    """

    def __init__(self, brain: Any, env: Any, loop_config: LoopConfig,
                 session_name: str = "workflow"):
        self.brain = brain
        self.env = env
        self.config = loop_config
        self.session_name = session_name

    def run_loop(self) -> list[IterationResult]:
        """Execute the full loop: iterate items on each page, paginate."""
        results: list[IterationResult] = []
        page = 1
        global_iteration = 0
        page_iteration = 0

        logger.info(f"Starting dynamic loop (max {self.config.max_iterations} items, {self.config.max_pages} pages)")

        while page <= self.config.max_pages and global_iteration < self.config.max_iterations:
            page_iteration += 1

            # Build iteration intent with ordinal
            ordinal = ORDINALS.get(page_iteration, f"#{page_iteration}")
            global_iteration += 1

            intent = self.config.iteration_intent
            intent = intent.replace("{ORDINAL}", ordinal)
            intent = intent.replace("{N}", str(global_iteration))
            intent = intent.replace("{PAGE_N}", str(page_iteration))
            intent = intent.replace("{PAGE}", str(page))

            # Add context about position
            if page_iteration > 1:
                intent += f"\n\nYou are on page {page}. You have already processed {page_iteration - 1} listings on this page. Scroll past them to find the {ordinal} one."

            logger.info(f"Iteration {global_iteration} (page {page}, item {page_iteration})")
            t0 = time.time()

            # Run bounded iteration
            result = self._run_iteration(intent, f"iter_{global_iteration}")

            iter_result = IterationResult(
                iteration=global_iteration,
                page=page,
                success=result.success,
                data=self._extract_data(result),
                no_more_items=self._check_no_more(result),
                steps=result.total_steps,
                duration=time.time() - t0,
            )
            results.append(iter_result)

            status = "VIABLE" if result.success else ("END_OF_PAGE" if iter_result.no_more_items else "SKIP")
            logger.info(f"  → {status} ({result.total_steps} steps, {iter_result.duration:.0f}s)")
            if iter_result.data:
                logger.info(f"  Data: {iter_result.data[:100]}")

            # Check for end of page
            if iter_result.no_more_items:
                logger.info(f"No more items on page {page}. Attempting pagination...")
                paginated = self._run_pagination()
                if paginated:
                    page += 1
                    page_iteration = 0
                    logger.info(f"Advanced to page {page}")
                else:
                    logger.info("No more pages. Loop complete.")
                    break

            # Check for hard failure (loop/max_steps without useful output)
            if result.termination_reason == "loop" and not result.success and not iter_result.data:
                logger.warning(f"Iteration stuck (loop). Moving to pagination.")
                paginated = self._run_pagination()
                if paginated:
                    page += 1
                    page_iteration = 0
                else:
                    break

        logger.info(f"Loop complete: {len(results)} iterations, {page} pages")
        return results

    def _run_iteration(self, intent: str, task_id: str):
        """Run a single iteration as a bounded GymRunner call."""
        runner = GymRunner(
            brain=self.brain,
            env=self.env,
            max_steps=self.config.max_steps_per_iteration,
            frames_per_inference=3,
        )

        # Retry loop
        best_result = None
        for attempt in range(1, self.config.max_retries_per_iteration + 1):
            attempt_intent = intent
            if attempt > 1 and best_result:
                attempt_intent += (
                    f"\n\nPrevious attempt failed ({best_result.termination_reason}). "
                    f"Try a different approach."
                )

            result = runner.run(task=attempt_intent, task_id=task_id)
            best_result = result

            if result.success or self._check_no_more(result):
                break

        return best_result

    def _run_pagination(self) -> bool:
        """Run a pagination step. Returns True if next page loaded."""
        runner = GymRunner(
            brain=self.brain,
            env=self.env,
            max_steps=self.config.max_steps_pagination,
            frames_per_inference=2,
        )

        result = runner.run(
            task=self.config.pagination_intent,
            task_id="pagination",
        )

        # If model signaled success, pagination worked
        if result.success:
            return True

        # Check if model signaled "no more pages"
        if self._check_no_more(result, check_pages=True):
            return False

        # Ambiguous — assume no more pages
        return False

    def _check_no_more(self, result, check_pages: bool = False) -> bool:
        """Check if the model signaled no more items/pages."""
        # Check terminate message in trajectory
        for step in result.trajectory:
            if step.action.action_type.value == "done":
                summary = step.action.params.get("summary", "").lower()
                thinking = (step.thinking or "").lower()
                text = summary + " " + thinking

                for signal in NO_MORE_SIGNALS:
                    if signal in text:
                        return True

                # Check for explicit failure with "no more" context
                if not step.action.params.get("success", True):
                    if any(s in text for s in ["no more", "empty", "end", "last"]):
                        return True

        return False

    @staticmethod
    def _extract_data(result) -> str:
        """Extract data from the model's terminate message."""
        for step in reversed(result.trajectory):
            if step.action.action_type.value == "done":
                summary = step.action.params.get("summary", "")
                if summary:
                    return summary
            if step.thinking and len(step.thinking) > 50:
                return step.thinking[:500]
        return ""
