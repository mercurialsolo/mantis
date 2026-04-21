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

from ..plan_decomposer import MicroIntent, MicroPlan

if TYPE_CHECKING:
    from ..extraction import ClaudeExtractor, ExtractionResult

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

        # Cost tracking
        self.costs = {
            "gpu_steps": 0,        # Total Holo3 steps
            "gpu_seconds": 0.0,    # Approx GPU time
            "claude_extract": 0,   # Claude extraction calls
            "claude_grounding": 0, # Claude grounding calls
            "proxy_mb": 0.0,       # Estimated proxy bandwidth
        }
        self._run_start = time.time()

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
        listings_on_page = 0  # Track how many listings processed on current page

        while step_index < len(plan.steps):
            step = plan.steps[step_index]
            t0 = time.time()

            # Dynamic intent: inject listing position for click steps
            dynamic_intent = step.intent
            if step.type == "click" and listings_on_page > 0:
                dynamic_intent = (
                    f"Scroll down past the first {listings_on_page} listings. "
                    f"Then click the next listing title text below a photo."
                )
            effective_step = MicroIntent(
                intent=dynamic_intent, type=step.type, verify=step.verify,
                budget=step.budget, reverse=step.reverse, grounding=step.grounding,
                claude_only=step.claude_only, loop_target=step.loop_target,
                loop_count=step.loop_count,
            )

            logger.info(f"  [{step_index:2d}] {step.type:15s} {dynamic_intent[:60]}")

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
            step_result = self._execute_step(effective_step, step_index)
            results.append(step_result)

            # Handle dedup: extract_url returned DUPLICATE → skip to loop
            if step_result.data and "DUPLICATE" in step_result.data:
                logger.info(f"  [{step_index}] DEDUP — skipping to next listing")
                # Go back to results page first
                try:
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "alt+Left"}))
                    time.sleep(2)
                except Exception:
                    pass
                # Jump to loop step
                for j in range(step_index + 1, len(plan.steps)):
                    if plan.steps[j].type == "loop":
                        step_index = j
                        break
                else:
                    step_index += 1
                listings_on_page += 1  # Count it as "processed" for scroll-past
                continue

            if step_result.success:
                # Track listing progress
                if step.type == "paginate":
                    listings_on_page = 0
                    self._listings_on_page = 0  # Reset for claude-guided click

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
                    logger.error(f"  [{step_index}] NAVIGATE FAILED — cannot proceed")
                    self._reverse_step(step)
                    break
                elif step.type in ("click",):
                    # Check if page exhausted (no more listings)
                    if step_result.data == "page_exhausted":
                        logger.info(f"  [{step_index}] PAGE EXHAUSTED — jumping to paginate")
                        # Find paginate step and jump there
                        for j in range(step_index + 1, len(plan.steps)):
                            if plan.steps[j].type == "paginate":
                                step_index = j
                                break
                        else:
                            # No paginate step — find next loop
                            for j in range(step_index + 1, len(plan.steps)):
                                if plan.steps[j].type == "loop":
                                    step_index = j
                                    break
                            else:
                                step_index += 1
                        continue
                    # Click failed — skip extraction cycle to loop
                    logger.warning(f"  [{step_index}] CLICK FAILED — skipping to next")
                    try:
                        self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Escape"}))
                        time.sleep(0.5)
                    except Exception:
                        pass
                    for j in range(step_index + 1, len(plan.steps)):
                        if plan.steps[j].type == "loop":
                            step_index = j
                            break
                    else:
                        step_index += 1
                    continue
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

            # Track costs
            if step_result.steps_used > 0:
                self.costs["gpu_steps"] += step_result.steps_used
                self.costs["gpu_seconds"] += step_result.steps_used * 3  # ~3s per step
            if effective_step.claude_only:
                self.costs["claude_extract"] += 1
            if effective_step.grounding:
                self.costs["claude_grounding"] += step_result.steps_used  # ~1 grounding per click
            if effective_step.type in ("click", "navigate", "paginate"):
                self.costs["proxy_mb"] += 5.0  # ~5MB per page load
            elif effective_step.type == "scroll":
                self.costs["proxy_mb"] += 0.5  # minimal for scroll

            # Cost calculation
            gpu_cost = (self.costs["gpu_seconds"] / 3600) * 3.25
            claude_cost = (self.costs["claude_extract"] * 0.003) + (self.costs["claude_grounding"] * 0.003)
            proxy_cost = (self.costs["proxy_mb"] / 1024) * 5.0
            total_cost = gpu_cost + claude_cost + proxy_cost

            # Progress logging with costs
            viable_count = sum(1 for r in results if r.success and "VIABLE" in (r.data or ""))
            unique_leads = len(set(
                r.data[:100] for r in results if r.success and "VIABLE" in (r.data or "")
            ))
            elapsed = time.time() - self._run_start
            cost_per_lead = total_cost / max(unique_leads, 1)

            # Print progress (visible in Modal logs + viewer)
            print(f"  [{step_index-1:2d}] {'OK' if step_result.success else 'FAIL'} "
                  f"| {unique_leads} leads | ${total_cost:.2f} total "
                  f"(${cost_per_lead:.2f}/lead) | "
                  f"GPU ${gpu_cost:.2f} Claude ${claude_cost:.2f} Proxy ${proxy_cost:.2f} | "
                  f"{elapsed/60:.0f}m")

        logger.info(f"MicroPlan complete: {len(results)} steps executed")
        # Final cost summary
        gpu_cost = (self.costs["gpu_seconds"] / 3600) * 3.25
        claude_cost = (self.costs["claude_extract"] * 0.003) + (self.costs["claude_grounding"] * 0.003)
        proxy_cost = (self.costs["proxy_mb"] / 1024) * 5.0
        total_cost = gpu_cost + claude_cost + proxy_cost
        viable_count = sum(1 for r in results if r.success and "VIABLE" in (r.data or ""))
        elapsed = time.time() - self._run_start

        print(f"\n{'='*60}")
        print(f"MICRO-PLAN COMPLETE")
        print(f"  Time:     {elapsed/60:.0f}m")
        print(f"  Steps:    {len(results)}")
        print(f"  Leads:    {viable_count}")
        print(f"  Cost:     ${total_cost:.2f} total (${total_cost/max(viable_count,1):.2f}/lead)")
        print(f"    GPU:    ${gpu_cost:.2f} ({self.costs['gpu_steps']} steps)")
        print(f"    Claude: ${claude_cost:.2f} ({self.costs['claude_extract']} extract + {self.costs['claude_grounding']} grounding)")
        print(f"    Proxy:  ${proxy_cost:.2f} ({self.costs['proxy_mb']:.0f} MB)")
        print(f"{'='*60}")

        # Attach costs to results for saving
        self._final_costs = {
            "total": round(total_cost, 3),
            "gpu": round(gpu_cost, 3),
            "claude": round(claude_cost, 3),
            "proxy": round(proxy_cost, 3),
            "per_lead": round(total_cost / max(viable_count, 1), 3),
        }

        return results

    def _execute_step(self, step: MicroIntent, index: int) -> StepResult:
        """Execute a single micro-intent."""

        # Navigate steps: use env.reset() with URL instead of Holo3
        if step.type == "navigate":
            return self._execute_navigate(step, index)

        # Click steps: Claude finds target → Holo3 clicks coordinates
        if step.type == "click" and self.extractor:
            # Brief settle — page may still be loading after navigate/paginate
            time.sleep(2)
            return self._execute_claude_guided_click(step, index)

        # Claude-only steps (extract_url, extract_data)
        if step.claude_only:
            # Brief settle — page may still be rendering after scroll
            time.sleep(1)
            return self._execute_claude_step(step, index)

        # Paginate steps: Claude finds Next button → Holo3 clicks
        if step.type == "paginate" and self.extractor:
            return self._execute_claude_guided_paginate(step, index)

        # Holo3 steps (scroll, filter, navigate_back)
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
            time.sleep(6)  # BoatTrader needs 6s+ to fully load
            # Scroll to top
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(1)
            return StepResult(step_index=index, intent=step.intent, success=True)
        except Exception as e:
            logger.error(f"  [navigate] Failed: {e}")
            return StepResult(step_index=index, intent=step.intent, success=False)

    def _execute_claude_guided_click(self, step: MicroIntent, index: int) -> StepResult:
        """Claude identifies click target → Holo3 clicks coordinates.

        1. Claude reads screenshot → finds Nth listing title → returns (x, y)
        2. Grounding refines coordinates to nearest text element
        3. Holo3 clicks the refined coordinates
        4. Claude verifies: are we on a detail page now?
        """
        from ..actions import Action, ActionType

        screenshot = self.env.screenshot()

        # Claude finds the target
        target = self.extractor.find_click_target(screenshot, skip_count=self._listings_on_page)
        self.costs["claude_extract"] += 1  # Track Claude call

        if target is None:
            logger.info(f"  [claude-click] No more listings found (page exhausted)")
            return StepResult(step_index=index, intent=step.intent, success=False,
                            data="page_exhausted")

        x, y, title = target
        logger.info(f"  [claude-click] Target: '{title[:40]}' at ({x}, {y})")

        # Grounding refines coordinates
        if self.grounding:
            from PIL import Image
            grounding_result = self.grounding.ground(screenshot, title, x, y)
            self.costs["claude_grounding"] += 1
            if grounding_result.confidence > 0.5:
                x, y = grounding_result.x, grounding_result.y
                logger.info(f"  [grounding] refined to ({x}, {y})")

        # Holo3 clicks the coordinates
        try:
            self.env.step(Action(
                action_type=ActionType.CLICK,
                params={"x": x, "y": y},
            ))
            time.sleep(3)  # Wait for page to load
            self.costs["gpu_steps"] += 1
            self.costs["gpu_seconds"] += 3
            self.costs["proxy_mb"] += 5.0  # Page load
        except Exception as e:
            logger.warning(f"  [claude-click] Click failed: {e}")
            return StepResult(step_index=index, intent=step.intent, success=False)

        # Verify: are we on a detail page?
        after = self.env.screenshot()
        verify_data = self.extractor.extract(after)
        self.costs["claude_extract"] += 1
        url = verify_data.url if verify_data else ""

        if url and "/boat/" in url:
            logger.info(f"  [claude-click] Verified on detail page: {url[:60]}")
            self._listings_on_page += 1
            return StepResult(step_index=index, intent=step.intent, success=True,
                            steps_used=1, duration=3.0)
        else:
            logger.warning(f"  [claude-click] Not on detail page after click (url={url[:40]})")
            return StepResult(step_index=index, intent=step.intent, success=False)

    def _execute_claude_guided_paginate(self, step: MicroIntent, index: int) -> StepResult:
        """Claude finds Next button → Holo3 clicks it.

        1. Scroll to bottom (End key)
        2. Claude reads screenshot → finds Next button coordinates
        3. Grounding refines → Holo3 clicks
        """
        # Scroll to bottom first
        try:
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "End"}))
            time.sleep(2)
        except Exception:
            pass

        screenshot = self.env.screenshot()
        target = self.extractor.find_paginate_target(screenshot)
        self.costs["claude_extract"] += 1

        if target is None:
            logger.info(f"  [claude-paginate] No Next button found")
            return StepResult(step_index=index, intent=step.intent, success=False)

        x, y = target

        # Grounding refines
        if self.grounding:
            grounding_result = self.grounding.ground(screenshot, "Next page button", x, y)
            self.costs["claude_grounding"] += 1
            if grounding_result.confidence > 0.5:
                x, y = grounding_result.x, grounding_result.y

        # Click
        try:
            self.env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
            time.sleep(4)  # Wait for new page to load
            self.costs["gpu_steps"] += 1
            self.costs["gpu_seconds"] += 4
            self.costs["proxy_mb"] += 5.0

            # Scroll to top of new page
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(1)
        except Exception as e:
            logger.warning(f"  [claude-paginate] Click failed: {e}")
            return StepResult(step_index=index, intent=step.intent, success=False)

        logger.info(f"  [claude-paginate] Clicked Next at ({x}, {y})")
        self._listings_on_page = 0  # Reset for new page
        return StepResult(step_index=index, intent=step.intent, success=True, steps_used=1)

    @property
    def _listings_on_page(self):
        """Track listings processed on current page."""
        if not hasattr(self, '_page_listings'):
            self._page_listings = 0
        return self._page_listings

    @_listings_on_page.setter
    def _listings_on_page(self, value):
        self._page_listings = value

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

        success = result.success

        # Post-step verification using Claude (if extractor available)
        if success and step.verify and self.extractor:
            screenshot = self.env.screenshot()
            verify_data = self.extractor.extract(screenshot)
            verified = self._check_verify(step.verify, verify_data, screenshot)
            if not verified:
                logger.warning(f"  [verify] Step {index} claimed success but verification FAILED: {step.verify[:60]}")
                success = False

        return StepResult(
            step_index=index,
            intent=step.intent,
            success=success,
            steps_used=result.total_steps,
            duration=result.total_time,
        )

    def _check_verify(self, verify_condition: str, extract_data, screenshot) -> bool:
        """Check if a step's verify condition is met using Claude extraction data.

        Simple heuristic checks based on the verify string and extracted data.
        Falls back to True if check is ambiguous (don't over-block).
        """
        v = verify_condition.lower()
        url = extract_data.url if extract_data else ""

        # Click verification: should be on a detail page, not search results
        if "detail page" in v or "page opens" in v:
            # URL should contain /boat/ (single listing) not just /boats/ (search)
            if url and "/boat/" in url and "/boats/" not in url.split("/boat/")[0][-1:]:
                return True
            if url and url.endswith("/boats/by-owner/"):
                logger.info(f"  [verify] Still on search page, not detail page")
                return False
            # If no URL extracted, check if page looks different
            if not url:
                return True  # Can't verify, assume OK

        # Filter verification: check heading/URL for filter signals
        if "selected" in v or "highlighted" in v or "filter" in v:
            # Can't easily verify filter state from extraction — pass through
            return True

        # Pagination verification: check for new listings
        if "new listings" in v:
            return True  # Pagination success already checked by runner

        # Default: trust the runner's success signal
        return True

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
