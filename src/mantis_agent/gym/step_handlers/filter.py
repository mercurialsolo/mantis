"""ClaudeGuidedFilterHandler — sidebar filter dispatch.

Phase 2 of EPIC #161, sixth handler extraction. Lifts
``MicroPlanRunner._execute_claude_guided_filter`` (147 LOC) verbatim
into a standalone class. The runner method becomes a delegating shim.

Holo3 is 0% reliable on sidebar filters (clicks wrong elements).
Claude reads the screenshot, identifies exact coordinates and action
type (``click``/``type``/``select``), then we execute directly — no
Holo3 involved. If not found in the current viewport, scrolls the
sidebar down and retries up to 8 times.

Three action types share this body:

- ``click``  — checkbox / radio / toggle. Single click + 2s settle.
- ``type``   — text-input filter. Click + triple-click + ctrl+a + Delete + TYPE + Return.
- ``select`` — dropdown filter. Click to open + Claude finds option in
  open menu + click option. Escape on miss.

What the handler reads from :class:`StepContext`:

- ``env``         — screenshot / step (SCROLL / CLICK / KEY_PRESS / TYPE)
- ``extractor``   — find_filter_target (Claude vision)
- ``grounding``   — coordinate refinement (150px delta bound)

What it reads from the runner via parent back-reference:

- ``costs``                       — cost meter aliased dict
- ``_last_known_url``             — anchor URL after filter applies
- ``_results_base_url``           — fallback when current URL not set
- ``_set_scroll_state``           — delegates to BrowserState
- ``_current_results_page_url``   — delegates to BrowserState
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING

from ...actions import Action, ActionType
from ..checkpoint import StepResult
from ..step_context import StepContext

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner
    from ...plan_decomposer import MicroIntent

logger = logging.getLogger(__name__)


class ClaudeGuidedFilterHandler:
    """Implements :class:`~..step_context.StepHandler` for ``filter`` steps."""

    step_type = "filter"

    def __init__(self, runner: "MicroPlanRunner") -> None:
        self.parent = runner

    def execute(self, step: "MicroIntent", ctx: StepContext) -> StepResult:
        runner = self.parent
        env = ctx.env
        extractor = ctx.extractor
        grounding = ctx.grounding
        index = int(ctx.state.get("index", 0))

        # Reset sidebar to top before each filter step (scroll persists between steps).
        # Filters are spread across the sidebar: Location near top, Seller Type near bottom.
        try:
            for _ in range(10):
                env.step(Action(action_type=ActionType.SCROLL,
                                params={"direction": "up", "amount": 5,
                                        "x": 150, "y": 400}))
                time.sleep(0.3)
        except Exception:
            pass
        time.sleep(1)

        # Scan sidebar top-to-bottom with small scroll increments.
        # Check each viewport position for the target filter element.
        target = None
        screenshot = None
        for scroll_attempt in range(8):
            if scroll_attempt > 0:
                # Scroll sidebar down in small increments (3 clicks ≈ ~100px)
                try:
                    env.step(Action(action_type=ActionType.SCROLL,
                                    params={"direction": "down", "amount": 3,
                                            "x": 150, "y": 400}))
                    time.sleep(1)
                except Exception:
                    pass

            screenshot = env.screenshot()
            target = extractor.find_filter_target(screenshot, step.intent)
            runner.costs["claude_extract"] += 1

            if target:
                break
            print(f"  [claude-filter] Not found in viewport {scroll_attempt}, scrolling sidebar")

        if not target:
            logger.warning("  [claude-filter] Could not find filter element")
            return StepResult(step_index=index, intent=step.intent, success=False)

        x, y = target["x"], target["y"]
        action = target["action"]
        value = target["value"]
        label = target["label"]

        # Grounding refines coordinates (bounded delta)
        if grounding:
            grounding_result = grounding.ground(screenshot, label or step.intent, x, y)
            runner.costs["claude_grounding"] += 1
            dx = abs(grounding_result.x - x)
            dy = abs(grounding_result.y - y)
            if grounding_result.confidence > 0.5 and dx < 150 and dy < 150:
                x, y = grounding_result.x, grounding_result.y
                logger.info(f"  [grounding] filter refined to ({x},{y}) delta=({dx},{dy})")
            else:
                logger.info(f"  [grounding] filter rejected: delta=({dx},{dy}) conf={grounding_result.confidence}")

        # Human-like delay before interaction
        time.sleep(random.uniform(0.5, 1.5))

        try:
            if action == "click":
                # Simple click — checkbox, radio, toggle
                env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                runner.costs["gpu_steps"] += 1
                time.sleep(2)  # Wait for filter to apply

            elif action == "type":
                # Click input → clear → type value → Enter
                env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                time.sleep(0.5)
                # Triple-click to select all existing text
                env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                time.sleep(0.1)
                env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                time.sleep(0.3)
                # Select all and delete
                env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "ctrl+a"}))
                time.sleep(0.2)
                env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Delete"}))
                time.sleep(0.3)
                # Type the value
                if value:
                    env.step(Action(action_type=ActionType.TYPE, params={"text": value}))
                    time.sleep(0.5)
                    env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Return"}))
                    time.sleep(3)  # Wait for results to update
                runner.costs["gpu_steps"] += 1

            elif action == "select":
                # Click dropdown to open → wait → screenshot → find option → click
                env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                runner.costs["gpu_steps"] += 1
                time.sleep(1.5)

                # Take new screenshot with dropdown open
                dropdown_shot = env.screenshot()
                # Ask Claude to find the specific option in the dropdown
                option_target = extractor.find_filter_target(
                    dropdown_shot,
                    f"Find and click the option '{value}' in the open dropdown menu"
                )
                runner.costs["claude_extract"] += 1

                if option_target:
                    ox, oy = option_target["x"], option_target["y"]
                    time.sleep(random.uniform(0.3, 0.8))
                    env.step(Action(action_type=ActionType.CLICK, params={"x": ox, "y": oy}))
                    time.sleep(2)
                else:
                    # Dropdown option not found — close dropdown
                    env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Escape"}))
                    time.sleep(0.5)
                    logger.warning(f"  [claude-filter] Dropdown option '{value}' not found")
                    return StepResult(step_index=index, intent=step.intent, success=False)

            else:
                # Unknown action — fall back to click
                env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                runner.costs["gpu_steps"] += 1
                time.sleep(2)

        except Exception as e:
            logger.warning(f"  [claude-filter] Action failed: {e}")
            return StepResult(step_index=index, intent=step.intent, success=False)

        logger.info(f"  [claude-filter] {action}@({x},{y}) '{label[:30]}' value='{value[:20]}'")
        runner._last_known_url = runner._current_results_page_url() or runner._results_base_url
        runner._set_scroll_state(
            context="results_after_filter",
            url=runner._last_known_url,
            page_downs=0,
            wheel_downs=0,
        )
        return StepResult(
            step_index=index, intent=step.intent, success=True,
            steps_used=1, duration=3.0,
        )
