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
import random
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
                 session_name: str = "workflow",
                 on_iteration: Any = None):
        self.brain = brain
        self.env = env
        self.config = loop_config
        self.session_name = session_name
        self.on_iteration = on_iteration  # Callback: fn(iteration_num, result, all_results)

    def run_loop(self) -> list[IterationResult]:
        """Execute the full loop: iterate items on each page, paginate."""
        results: list[IterationResult] = []
        page = 1
        global_iteration = 0
        page_iteration = 0
        learnings: list[str] = []  # Agentic: accumulate learnings from prior iterations

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

            # Agentic learning: inject what worked/failed in prior iterations
            if learnings:
                intent += "\n\nLEARNINGS FROM PRIOR LISTINGS (apply these):"
                for learning in learnings[-5:]:  # Last 5 learnings
                    intent += f"\n- {learning}"

            logger.info(f"Iteration {global_iteration} (page {page}, item {page_iteration})")
            t0 = time.time()

            # Human-like delay between iterations to avoid bot detection
            if global_iteration > 1:
                delay = random.uniform(2.0, 5.0)
                logger.info(f"  Inter-iteration delay: {delay:.1f}s")
                time.sleep(delay)

            # Run bounded iteration
            result = self._run_iteration(intent, f"iter_{global_iteration}")

            extracted = self._extract_data(result)
            viable = self._validate_viable(extracted)

            iter_result = IterationResult(
                iteration=global_iteration,
                page=page,
                success=viable,
                data=extracted,
                no_more_items=self._check_no_more(result),
                steps=result.total_steps,
                duration=time.time() - t0,
            )
            results.append(iter_result)

            if result.success and not viable:
                logger.warning(f"  Model claimed success but data failed validation (Cloudflare/empty)")

            # Agentic learning: distill what happened for future iterations
            learning = self._distill_learning(result, iter_result, viable)
            if learning:
                learnings.append(learning)
                logger.info(f"  Learning: {learning}")

            status = "VIABLE" if viable else ("END_OF_PAGE" if iter_result.no_more_items else "SKIP")
            logger.info(f"  → {status} ({result.total_steps} steps, {iter_result.duration:.0f}s)")
            if iter_result.data:
                logger.info(f"  Data: {iter_result.data[:100]}")

            # Progress callback — allows caller to write intermediate results
            if self.on_iteration:
                try:
                    self.on_iteration(global_iteration, iter_result, results)
                except Exception as e:
                    logger.warning(f"on_iteration callback error: {e}")

            # Check for end of page
            if iter_result.no_more_items:
                logger.info(f"No more items on page {page}. Attempting pagination...")
                time.sleep(random.uniform(3.0, 6.0))  # Pause before pagination
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
    def _distill_learning(result, iter_result, viable: bool) -> str:
        """Distill a single actionable learning from this iteration.

        These learnings are injected into future iteration prompts so
        the model improves over time without hardcoded hints.
        """
        data = iter_result.data.lower() if iter_result.data else ""
        steps = result.total_steps

        # Cloudflare block
        if "cloudflare" in data or "verify you are human" in data:
            return "Cloudflare blocked a listing. Press Alt+Left and try the next one instead of retrying."

        # Model didn't click into listing (stayed on search results)
        actions = [str(s.action)[:60] for s in result.trajectory] if result.trajectory else []
        clicks = sum(1 for a in actions if "click" in a.lower())
        scrolls = sum(1 for a in actions if "scroll" in a.lower())
        if scrolls > clicks * 2 and not viable:
            return "Too much scrolling without clicking. CLICK the listing card/image/title first, THEN scroll on the detail page."

        # Model used all steps without extracting
        if steps >= result.total_steps and not viable and "cookie" in data:
            return "Cookie popup appeared. Dismiss it immediately by clicking Accept/X, then proceed."

        # Model couldn't go back
        if "timeout" in data or "go_back" in data:
            return "Page.go_back timed out. Use key_press('Alt+ArrowLeft') instead of browser back button."

        # Successful extraction — note what worked
        if viable and iter_result.data:
            # Extract the approach that worked
            return f"Successfully extracted data. Keep using the same approach: click→scroll down→read description→back."

        # Model ran out of steps
        if result.termination_reason == "max_steps" and not viable:
            return "Ran out of steps. Be more efficient: click listing, scroll down 5x fast, read text, go back. Don't re-scroll the same area."

        # Generic skip
        if not viable and iter_result.data:
            return "Listing had no phone number. That's OK — move to next listing quickly."

        return ""

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

    @staticmethod
    def _validate_viable(data: str) -> bool:
        """Check if extracted data contains actual lead content.

        A viable extraction must contain at least one of:
        - A phone number pattern (3+ digit groups)
        - A price/dollar amount
        - The word VIABLE (model's explicit signal)
        - Boat-related keywords + data (year, make, model together)

        Rejects iterations that only mention Cloudflare, verification,
        blank pages, or generic failure messages.
        """
        if not data:
            return False

        text = data.lower()

        # Reject obvious non-extractions
        reject_signals = [
            "cloudflare", "verify you are human", "verifying",
            "about:blank", "blank page", "captcha",
            "page isn't loading", "not on the search",
        ]
        if any(sig in text for sig in reject_signals):
            # Even if model said success, CF page = not viable
            return False

        # Check for explicit VIABLE signal from model
        if "viable" in text:
            return True

        # Check for phone number pattern (3+ digits separated by dashes/dots/parens)
        import re
        phone_pattern = re.compile(r'\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}')
        if phone_pattern.search(data):
            return True

        # Check for price/dollar amount
        if re.search(r'\$\s*[\d,]+', data) or re.search(r'(?:price|asking)[:\s]*\d', text):
            return True

        # Check for boat data keywords (need at least 2 of: year, make, model)
        boat_signals = 0
        if re.search(r'20[12]\d', data):  # Year like 2019-2026
            boat_signals += 1
        for kw in ["make", "model", "hull", "engine", "footer", "console", "cabin"]:
            if kw in text:
                boat_signals += 1
        if boat_signals >= 2:
            return True

        return False
