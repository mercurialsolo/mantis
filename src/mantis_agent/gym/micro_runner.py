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
        max_cost: float = 2.0,      # Stop if total cost exceeds this
        max_time_minutes: int = 30,  # Stop if runtime exceeds this
    ):
        self.brain = brain
        self.env = env
        self.grounding = grounding
        self.extractor = extractor
        self.on_step = on_step
        self.max_retries = max_retries
        self.checkpoint_path = checkpoint_path
        self._seen_urls: set[str] = set()
        self._extracted_titles: list[str] = []  # Exact titles Claude returned, for skip list
        self._page_listings: list[tuple[int, int, str]] = []  # Cached card coords for current viewport
        self._page_listing_index: int = 0  # Next card to click from cache
        self._viewport_stage: int = 0  # 0=Home, 1=Page_Down, 2=Page_Down×2
        self._max_viewport_stages: int = 3
        self.max_cost = max_cost
        self.max_time = max_time_minutes * 60

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
        step_retry_counts: dict[int, int] = {}
        loop_counters: dict[int, int] = {}  # Per-step loop iteration counts
        max_loop_iterations = 200  # Safety cap
        listings_on_page = 0  # Track how many listings processed on current page

        while step_index < len(plan.steps):
            # Budget + time checks
            elapsed = time.time() - self._run_start
            gpu_cost = (self.costs["gpu_seconds"] / 3600) * 3.25
            claude_cost = (self.costs["claude_extract"] * 0.003) + (self.costs["claude_grounding"] * 0.003)
            proxy_cost = (self.costs["proxy_mb"] / 1024) * 5.0
            total_cost = gpu_cost + claude_cost + proxy_cost

            if total_cost >= self.max_cost:
                print(f"  BUDGET CAP: ${total_cost:.2f} >= ${self.max_cost:.2f} — stopping")
                break
            if elapsed >= self.max_time:
                print(f"  TIME CAP: {elapsed/60:.0f}m >= {self.max_time/60:.0f}m — stopping")
                break

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

            # Handle loop steps — each loop step has its own counter
            if step.type == "loop":
                loop_counters[step_index] = loop_counters.get(step_index, 0) + 1
                count = loop_counters[step_index]
                max_count = step.loop_count or max_loop_iterations
                if count < max_count:
                    target = step.loop_target if step.loop_target >= 0 else step_index
                    step_index = target
                    logger.info(f"  [loop@{step_index}] iteration {count}/{max_count} → step {step_index}")
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
                step_retry_counts.pop(step_index, None)
                # Track listing progress
                if step.type == "paginate":
                    listings_on_page = 0
                    self._listings_on_page = 0
                    self._extracted_titles = []  # New page = new listings
                    self._page_listings = []   # Reset card cache
                    self._page_listing_index = 0
                    self._viewport_stage = 0  # Start from Home on new page
                    # Reset inner loop counters — new page means fresh listing loop
                    for k in list(loop_counters.keys()):
                        if k != step_index:  # Don't reset the outer loop's own counter
                            loop_counters[k] = 0
                    # Scroll to top of new page + wait for load
                    try:
                        self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
                        time.sleep(8)
                    except Exception:
                        pass
                    logger.info(f"  [paginate] Success — reset to top of new page")

                # Verify navigate_back: check if we left the detail page
                if step.type == "navigate_back" and self.extractor:
                    time.sleep(2)
                    screenshot = self.env.screenshot()
                    check = self.extractor.extract(screenshot)
                    self.costs["claude_extract"] += 1
                    url = check.url if check else ""
                    if url and "/boat/" in url and "/boats/" not in url.split("/boat/")[0]:
                        # Still on detail page — give the CUA a recovery task
                        # Use the plan's reverse intent, not hardcoded site knowledge
                        recovery_intent = step.reverse or "Go back to the previous page."
                        logger.warning(f"  [back-verify] Still on detail page — CUA recovery: {recovery_intent[:50]}")
                        recovery = self._execute_holo3_step(
                            MicroIntent(
                                intent=recovery_intent,
                                type="navigate_back",
                                budget=8,
                                grounding=True,
                            ),
                            step_index,
                        )
                        self.costs["gpu_steps"] += recovery.steps_used

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
                    if step_result.data in ("scan_error", "page_blocked"):
                        attempt = step_retry_counts.get(step_index, 0) + 1
                        if attempt <= self.max_retries:
                            step_retry_counts[step_index] = attempt
                            wait_s = 12 if step_result.data == "page_blocked" else 4
                            logger.warning(
                                f"  [{step_index}] {step_result.data.upper()} — "
                                f"waiting {wait_s}s and retrying ({attempt}/{self.max_retries})"
                            )
                            time.sleep(wait_s)
                            continue
                        logger.warning(f"  [{step_index}] {step_result.data.upper()} — retry budget exhausted")
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
                    # Back failure — try multiple times and verify
                    logger.warning(f"  [{step_index}] BACK FAILED — retrying Alt+Left")
                    for back_attempt in range(3):
                        try:
                            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "alt+Left"}))
                            time.sleep(3)
                        except Exception:
                            pass
                        # Verify: check if URL is back on search results
                        if self.extractor:
                            screenshot = self.env.screenshot()
                            check = self.extractor.extract(screenshot)
                            url = check.url if check else ""
                            if url and "/boats/" in url and "/boat/" not in url:
                                logger.info(f"  [back] Verified on results page after {back_attempt+1} attempts")
                                break
                    step_index += 1
                elif step.type in ("paginate",):
                    # Paginate failed — no new page loaded, stop the pipeline
                    logger.warning(f"  [{step_index}] PAGINATE FAILED — no more pages, ending")
                    # Exhaust the outer loop so it doesn't restart on the same page
                    for k in list(loop_counters.keys()):
                        loop_counters[k] = 999999
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

        # Paginate: try Holo3 first. If it fails without taking any actions,
        # fall back to Claude-guided pagination on the unchanged page state.
        if step.type == "paginate" and self.extractor:
            holo_result = self._execute_holo3_step(step, index)
            if holo_result.success:
                return holo_result
            if holo_result.steps_used == 0:
                logger.info("  [paginate] Holo3 failed with 0 steps — falling back to Claude-guided paginate")
                return self._execute_claude_guided_paginate(step, index)
            return holo_result

        # Holo3 steps (scroll, filter, navigate_back, paginate)
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
            time.sleep(12)  # BoatTrader needs 8-12s to fully render
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(1)
            return StepResult(step_index=index, intent=step.intent, success=True)
        except Exception as e:
            logger.error(f"  [navigate] Failed: {e}")
            return StepResult(step_index=index, intent=step.intent, success=False)

    def _execute_claude_guided_click(self, step: MicroIntent, index: int) -> StepResult:
        """Claude finds ALL listings once per page, clicks them one by one.

        First call on a page: find_all_listings() → cache coordinates (1 Claude call)
        Subsequent calls: pop next from cache (0 Claude calls)
        Page exhausted when cache is empty.
        """
        # If no cached listings, scan the page (one Claude call for ALL cards)
        if self._page_listing_index >= len(self._page_listings):
            # Staged per-viewport scan: scan ONE viewport, cache its cards.
            # When cache empties, advance to next viewport (Page_Down).
            # Only page_exhausted after all viewport stages return empty.
            while self._viewport_stage < self._max_viewport_stages:
                # Reconstruct the current viewport deterministically:
                # always start from Home, then Page_Down N times.
                try:
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
                    time.sleep(0.5)
                    for _ in range(self._viewport_stage):
                        self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                        time.sleep(0.5)
                except Exception:
                    pass

                screenshot = self.env.screenshot()
                scan_result = self.extractor.find_all_listings(screenshot)
                self.costs["claude_extract"] += 1

                if isinstance(scan_result, tuple):
                    status = scan_result[0]
                    if status == "blocked":
                        logger.warning(f"  [claude-click] Viewport {self._viewport_stage}: blocked/error page")
                        return StepResult(
                            step_index=index,
                            intent=step.intent,
                            success=False,
                            data="page_blocked",
                        )
                    if status == "error":
                        logger.warning(f"  [claude-click] Viewport {self._viewport_stage}: parse/API failure")
                        return StepResult(
                            step_index=index,
                            intent=step.intent,
                            success=False,
                            data="scan_error",
                        )
                    cards = []
                else:
                    cards = scan_result

                # Filter out already-extracted titles
                skip = set(t.lower() for t in self._extracted_titles)
                filtered = [(x, y, t) for x, y, t in cards
                           if t.lower() not in skip and t != "unknown"]
                unknown_cards = [(x, y, t) for x, y, t in cards if t == "unknown"]
                filtered.extend(unknown_cards)
                filtered.sort(key=lambda c: c[1])

                logger.info(f"  [claude-click] Viewport {self._viewport_stage}: {len(cards)} cards, {len(filtered)} new")

                if filtered:
                    self._page_listings = filtered
                    self._page_listing_index = 0
                    break  # Found cards in this viewport — click them
                else:
                    self._viewport_stage += 1  # Try next viewport

            if not self._page_listings or self._page_listing_index >= len(self._page_listings):
                logger.info(f"  [claude-click] All {self._max_viewport_stages} viewports exhausted")
                return StepResult(step_index=index, intent=step.intent, success=False,
                                data="page_exhausted")

        # Pop next card from cache — scroll to the viewport where it was found
        x, y, title = self._page_listings[self._page_listing_index]
        self._page_listing_index += 1
        self._last_click_title = title

        # Scroll to the correct viewport (Home + N Page_Downs)
        try:
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(0.5)
            for _ in range(self._viewport_stage):
                self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                time.sleep(0.5)
        except Exception:
            pass

        logger.info(f"  [claude-click] Card {self._page_listing_index}/{len(self._page_listings)}: '{title[:40]}' at ({x}, {y}) viewport={self._viewport_stage}")

        # Delay before the final screenshot so grounding sees the frame we will actually click.
        import random
        time.sleep(random.uniform(1.5, 3.5))

        # Grounding refines — but only accept if the delta is small
        if self.grounding:
            screenshot = self.env.screenshot()
            grounding_result = self.grounding.ground(screenshot, title, x, y)
            self.costs["claude_grounding"] += 1
            dx = abs(grounding_result.x - x)
            dy = abs(grounding_result.y - y)
            if grounding_result.confidence > 0.5 and dx < 200 and dy < 200:
                x, y = grounding_result.x, grounding_result.y
                logger.info(f"  [grounding] refined to ({x}, {y}) delta=({dx},{dy})")
            else:
                logger.info(f"  [grounding] rejected: delta=({dx},{dy}) conf={grounding_result.confidence}")

        # Click
        try:
            self.env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
            self.costs["gpu_steps"] += 1
            self.costs["gpu_seconds"] += 3
            self.costs["proxy_mb"] += 5.0
        except Exception as e:
            logger.warning(f"  [claude-click] Click failed: {e}")
            return StepResult(step_index=index, intent=step.intent, success=False)

        # Verify: are we on a detail page? Retry once (page may still load)
        for verify_attempt in range(2):
            time.sleep(3 + verify_attempt * 3)  # 3s first, 6s retry
            after = self.env.screenshot()
            verify_data = self.extractor.extract(after)
            self.costs["claude_extract"] += 1
            url = verify_data.url if verify_data else ""

            if url and "/boat/" in url and "/boats/" not in url.split("/boat/")[0]:
                logger.info(f"  [claude-click] Verified on detail page: {url[:60]}")
                self._listings_on_page += 1
                # Store the exact title Claude found for skip list
                if hasattr(self, '_last_click_title') and self._last_click_title:
                    self._extracted_titles.append(self._last_click_title)
                return StepResult(step_index=index, intent=step.intent, success=True,
                                steps_used=1, duration=3.0 + verify_attempt * 3)

            if verify_attempt == 0:
                logger.info(f"  [claude-click] Not on detail page yet (url={url[:40]}) — retrying verify")

        logger.warning(f"  [claude-click] Failed verification after retries (url={url[:40]})")
        return StepResult(step_index=index, intent=step.intent, success=False)

    def _execute_claude_guided_paginate(self, step: MicroIntent, index: int) -> StepResult:
        """Claude finds Next button → Holo3 clicks it.

        Scrolls near the bottom, Claude finds pagination, retry on error, bounded grounding.
        """
        # Go to bottom first so the pagination bar is likely on screen.
        try:
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "End"}))
            time.sleep(3)
        except Exception:
            pass

        # Find pagination target with retry. On retry, move slightly up so the
        # pagination bar is not flush with the screen edge or hidden by footer UI.
        target = None
        screenshot = None
        for attempt in range(3):
            if attempt == 1:
                try:
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Up"}))
                    time.sleep(1.5)
                except Exception:
                    pass
            elif attempt == 2:
                try:
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "End"}))
                    time.sleep(2)
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Up"}))
                    time.sleep(1.5)
                except Exception:
                    pass

            screenshot = self.env.screenshot()
            target = self.extractor.find_paginate_target(screenshot)
            self.costs["claude_extract"] += 1

            if isinstance(target, tuple) and len(target) == 3:
                break
            if isinstance(target, tuple) and target[0] == "not_found":
                logger.warning(f"  [claude-paginate] no control visible on attempt {attempt+1}/3")
                continue
            if isinstance(target, tuple) and target[0] == "error":
                logger.warning(f"  [claude-paginate] parse/error on attempt {attempt+1}/3")
                continue

            logger.warning(f"  [claude-paginate] empty response on attempt {attempt+1}/3")

        if not isinstance(target, tuple) or len(target) != 3:
            logger.info(f"  [claude-paginate] No Next control found after retries")
            return StepResult(step_index=index, intent=step.intent, success=False)

        x, y, label = target

        # Grounding with delta bound
        if self.grounding:
            grounding_result = self.grounding.ground(screenshot, f"pagination control {label or 'Next'}", x, y)
            self.costs["claude_grounding"] += 1
            dx = abs(grounding_result.x - x)
            dy = abs(grounding_result.y - y)
            if grounding_result.confidence > 0.5 and dx < 150 and dy < 150:
                x, y = grounding_result.x, grounding_result.y
                logger.info(f"  [grounding] paginate refined to ({x}, {y}) delta=({dx},{dy})")
            else:
                logger.info(f"  [grounding] paginate rejected: delta=({dx},{dy}) conf={grounding_result.confidence}")

        # Click
        try:
            self.env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
            self.costs["gpu_steps"] += 1
            self.costs["gpu_seconds"] += 4
            self.costs["proxy_mb"] += 5.0
        except Exception as e:
            logger.warning(f"  [claude-paginate] Click failed: {e}")
            return StepResult(step_index=index, intent=step.intent, success=False)

        # Wait for page load, then scroll to top
        time.sleep(8)
        try:
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(2)
        except Exception:
            pass

        logger.info(f"  [claude-paginate] Clicked '{label[:20]}' at ({x}, {y})")
        self._listings_on_page = 0  # Reset for new page
        return StepResult(step_index=index, intent=step.intent, success=True, steps_used=1)

    @property
    def _listings_on_page(self):
        """Track listings processed on current page."""
        if not hasattr(self, '_page_listing_count'):
            self._page_listing_count = 0
        return self._page_listing_count

    @_listings_on_page.setter
    def _listings_on_page(self, value):
        self._page_listing_count = value

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
