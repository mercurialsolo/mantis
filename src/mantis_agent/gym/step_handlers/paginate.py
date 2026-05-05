"""PaginateHandler — layered Next-page strategy.

Phase 2 of EPIC #161, fifth handler extraction. Combines the two
pagination methods on the runner that work as a unit:

- ``_execute_paginate_layered`` — three-layer fallback orchestrator
  (URL-based → Claude-guided → Holo3 fallback) with success bookkeeping
  per layer
- ``_execute_claude_guided_paginate`` — Layer 2 implementation (Escape
  → End → Claude finds Next → grounding refine → click → 8s settle)

Lifted into one cohesive handler so the layered strategy and its
implementation live in the same module. The Holo3 fallback layer
calls back into the runner for now (``runner._execute_holo3_step``) —
that hand-off goes away when Holo3StepHandler is extracted in a
follow-up PR.

What the handler reads from :class:`StepContext`:

- ``env``               — screenshot / step / reset
- ``extractor``         — find_paginate_target (Claude vision)
- ``grounding``         — coordinate refinement
- ``dynamic_verifier``  — record_pagination / record_page_start
- ``site_config``       — pagination_format / paginated_url

What it reads from the runner via parent back-reference:

- ``costs``                     — cost meter aliased dict
- ``_current_page``             — pagination counter
- ``_results_base_url``         — anchor URL set by NavigateHandler
- ``_last_known_url``
- ``_listings_on_page``         — counter, reset on success
- ``_set_scroll_state``         — delegates to BrowserState
- ``_current_results_page_url`` — delegates to BrowserState
- ``_execute_holo3_step``       — Layer 3 fallback (will move when
                                  Holo3StepHandler lands)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ...actions import Action, ActionType
from ..checkpoint import StepResult
from ..step_context import StepContext

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner
    from ...plan_decomposer import MicroIntent

logger = logging.getLogger(__name__)


class PaginateHandler:
    """Implements :class:`~..step_context.StepHandler` for ``paginate`` steps.

    Three layers, applied in order:

    1. **URL-based** — fastest, most reliable. Detects current page URL
       pattern, constructs page N+1 URL, navigates directly. Skips if
       ``site_config.pagination_format`` is unset.
    2. **Claude-guided** — Escape clears focus traps, End scrolls to
       footer, Claude finds Next-button coordinates, grounding refines,
       click + 8s settle.
    3. **Holo3 fallback** — calculated scroll then a 1-sentence
       Holo3 task. Last resort when Claude can't see pagination.
    """

    step_type = "paginate"

    def __init__(self, runner: "MicroPlanRunner") -> None:
        self.parent = runner

    def execute(self, step: "MicroIntent", ctx: StepContext) -> StepResult:
        runner = self.parent
        env = ctx.env
        dynamic_verifier = ctx.dynamic_verifier
        site_config = ctx.site_config
        index = int(ctx.state.get("index", 0))

        # Track current page number
        if not hasattr(runner, "_current_page"):
            runner._current_page = 1
        current_page = runner._current_page

        # ── Layer 1: URL-based pagination ──
        # Use the stored results base URL (from initial navigate), NOT the current page URL
        # (which might be a detail page after extraction)
        base_url = getattr(runner, "_results_base_url", "")
        if base_url and site_config.pagination_format:
            next_page = runner._current_page + 1
            next_url = site_config.paginated_url(base_url, next_page)

            # Ensure full URL
            if not next_url.startswith("http"):
                next_url = f"https://www.{next_url}"

            logger.info(f"  [paginate] Layer 1: URL-based → {next_url[:80]}")
            try:
                env.reset(task="paginate_url", start_url=next_url)
                time.sleep(10)
                env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
                time.sleep(2)
                runner._current_page = next_page
                runner._last_known_url = next_url
                runner._set_scroll_state(context="results_top", url=next_url, page_downs=0, wheel_downs=0)
                dynamic_verifier.record_pagination(
                    page=current_page,
                    success=True,
                    method="url",
                    next_url=next_url,
                )
                dynamic_verifier.record_page_start(page=next_page, url=next_url)
                return StepResult(step_index=index, intent=step.intent, success=True,
                                steps_used=0, data=f"url_paginate_page{next_page}")
            except Exception as e:
                logger.warning(f"  [paginate] Layer 1 failed: {e}")
                dynamic_verifier.record_pagination(
                    page=current_page,
                    success=False,
                    method="url",
                    next_url=next_url,
                    reason=f"url_navigation_failed:{e}",
                )

        # ── Layer 2: Claude-guided ──
        logger.info("  [paginate] Layer 2: Claude-guided (End → Page_Up)")
        claude_result = self._claude_guided_paginate(step, ctx, index)
        if claude_result.success:
            runner._current_page += 1
            runner._last_known_url = runner._current_results_page_url()
            runner._set_scroll_state(context="results_top", url=runner._last_known_url, page_downs=0, wheel_downs=0)
            dynamic_verifier.record_pagination(
                page=current_page,
                success=True,
                method="claude_guided",
                next_url=runner._last_known_url,
            )
            dynamic_verifier.record_page_start(page=runner._current_page, url=runner._last_known_url)
            return claude_result

        # ── Layer 3: Holo3 fallback ──
        logger.info("  [paginate] Layer 3: Holo3 fallback")
        # Scroll to a calculated position: End then 2x Page_Up to avoid sidebar
        try:
            env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(0.5)
            # Scroll down ~80% of the page (past listings, before footer/sidebar bottom)
            for _ in range(6):
                env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                time.sleep(0.5)
        except Exception:
            pass

        # Late import to avoid a cycle: holo3 fallback still lives on the
        # runner. When Holo3StepHandler lands, this dispatches via
        # ctx.handler_registry instead of the parent back-reference.
        from ...plan_decomposer import MicroIntent as _MicroIntent
        holo_result = runner._execute_holo3_step(
            _MicroIntent(
                intent="Click the Next page button or the next page number.",
                type="paginate",
                budget=8,
                grounding=True,
            ),
            index,
        )
        if holo_result.success:
            runner._current_page += 1
            runner._last_known_url = runner._current_results_page_url()
            runner._set_scroll_state(context="results_top", url=runner._last_known_url, page_downs=0, wheel_downs=0)
            dynamic_verifier.record_pagination(
                page=current_page,
                success=True,
                method="holo3",
                next_url=runner._last_known_url,
            )
            dynamic_verifier.record_page_start(page=runner._current_page, url=runner._last_known_url)
        else:
            dynamic_verifier.record_pagination(
                page=current_page,
                success=False,
                method="all_layers",
                reason="next_control_not_found",
            )
        return holo_result

    def _claude_guided_paginate(
        self, step: "MicroIntent", ctx: StepContext, index: int,
    ) -> StepResult:
        """Layer 2 — Claude finds Next button → click + grounding refine.

        Scrolls near the bottom, Claude finds pagination, retry on
        error, bounded grounding delta.
        """
        runner = self.parent
        env = ctx.env
        extractor = ctx.extractor
        grounding = ctx.grounding

        # Clear focus traps such as open menus or overlays before repositioning.
        try:
            env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Escape"}))
            time.sleep(0.5)
        except Exception:
            pass

        # Go to bottom first so the pagination bar is likely on screen.
        try:
            env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "End"}))
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
                    env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Up"}))
                    time.sleep(1.5)
                except Exception:
                    pass
            elif attempt == 2:
                try:
                    env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "End"}))
                    time.sleep(2)
                    env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Up"}))
                    time.sleep(1.5)
                except Exception:
                    pass

            screenshot = env.screenshot()
            target = extractor.find_paginate_target(screenshot)
            runner.costs["claude_extract"] += 1

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
            logger.info("  [claude-paginate] No Next control found after retries")
            return StepResult(step_index=index, intent=step.intent, success=False)

        x, y, label = target

        # Grounding with delta bound
        if grounding:
            grounding_result = grounding.ground(screenshot, f"pagination control {label or 'Next'}", x, y)
            runner.costs["claude_grounding"] += 1
            dx = abs(grounding_result.x - x)
            dy = abs(grounding_result.y - y)
            if grounding_result.confidence > 0.5 and dx < 150 and dy < 150:
                x, y = grounding_result.x, grounding_result.y
                logger.info(f"  [grounding] paginate refined to ({x}, {y}) delta=({dx},{dy})")
            else:
                logger.info(f"  [grounding] paginate rejected: delta=({dx},{dy}) conf={grounding_result.confidence}")

        # Click
        try:
            env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
            runner.costs["gpu_steps"] += 1
            runner.costs["gpu_seconds"] += 4
            runner.costs["proxy_mb"] += 5.0
        except Exception as e:
            logger.warning(f"  [claude-paginate] Click failed: {e}")
            return StepResult(step_index=index, intent=step.intent, success=False)

        # Wait for page load, then scroll to top
        time.sleep(8)
        try:
            env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(2)
        except Exception:
            pass

        logger.info(f"  [claude-paginate] Clicked '{label[:20]}' at ({x}, {y})")
        runner._listings_on_page = 0  # Reset for new page
        runner._set_scroll_state(context="pagination_clicked", page_downs=0, wheel_downs=0)
        return StepResult(step_index=index, intent=step.intent, success=True, steps_used=1)
