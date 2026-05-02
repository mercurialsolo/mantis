"""Browser-state helpers for MicroPlanRunner — extracted from
micro_runner.py (#115, step 4).

Owns the scroll-position / viewport-stage / pagination-URL machinery used
to re-enter a browser session after a checkpoint resume or a navigation
detour. The runner keeps the underlying state attributes
(``_scroll_state``, ``_viewport_stage``, ``_current_page``,
``_last_known_url``, ``_results_base_url``, ``_required_filter_tokens``,
``_opened_detail_in_new_tab``) for now — there are 170+ scattered
read/write sites inside the runner and migrating them all in one PR
would be unreviewable. This split moves the **method bodies** out and
leaves the state references where they are.

Behavior is unchanged from the previous in-place implementation. The
runner composes a single :class:`BrowserState` and the previous
``_current_results_page_url`` / ``_reentry_url_for_step`` /
``_set_scroll_state`` / ``_update_scroll_state_from_trajectory`` /
``_restore_scroll_position`` / ``_resume_browser_state`` methods become
one-line shims that delegate here.

Future steps in #115 can migrate state ownership into this class with a
property-descriptor shim or by refactoring the call sites in batches.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any

from ..actions import Action, ActionType

if TYPE_CHECKING:
    from ..plan_decomposer import MicroPlan
    from .micro_runner import MicroPlanRunner

logger = logging.getLogger(__name__)


class BrowserState:
    """Stateless helper bound to one :class:`MicroPlanRunner` instance.

    Methods read/write the runner's underlying attributes through the
    ``parent`` back-reference. This keeps the 170+ scattered state-access
    sites in micro_runner.py untouched — only the method bodies move here.
    """

    def __init__(self, parent: "MicroPlanRunner") -> None:
        self.parent = parent

    # ── URL composition ─────────────────────────────────────────────────

    def current_results_page_url(self) -> str:
        p = self.parent
        if not p._results_base_url:
            return ""
        if p._current_page <= 1:
            base_clean = re.sub(
                p.site_config.pagination_strip_pattern or r"/page-\d+/?$",
                "",
                p._results_base_url.rstrip("/"),
            )
            return f"{base_clean}/"
        return p.site_config.paginated_url(p._results_base_url, p._current_page)

    def reentry_url_for_step(
        self, plan: "MicroPlan", next_step_index: int
    ) -> str:
        p = self.parent
        next_step = (
            plan.steps[next_step_index]
            if 0 <= next_step_index < len(plan.steps)
            else None
        )
        results_url = self.current_results_page_url() or p._results_base_url
        if next_step and next_step.type in {"click", "paginate", "loop", "filter"}:
            return results_url or p._last_known_url
        if next_step and next_step.type in {
            "extract_url", "scroll", "extract_data", "navigate_back",
        }:
            return p._last_known_url or results_url
        return p._last_known_url or results_url

    # ── Scroll state ────────────────────────────────────────────────────

    def set_scroll_state(
        self,
        *,
        context: str,
        url: str = "",
        page_downs: int | None = None,
        wheel_downs: int | None = None,
        viewport_stage: int | None = None,
        label: str = "",
        flush: bool = False,
    ) -> None:
        p = self.parent
        state = dict(p._scroll_state)
        state["context"] = context
        state["url"] = url or p._last_known_url or self.current_results_page_url()
        state["updated_at"] = time.time()
        if page_downs is not None:
            state["page_downs"] = max(0, page_downs)
        if wheel_downs is not None:
            state["wheel_downs"] = max(0, wheel_downs)
        if viewport_stage is not None:
            state["viewport_stage"] = max(0, viewport_stage)
        if label:
            state["label"] = label
        p._scroll_state = state
        if flush:
            p._checkpoint_active_progress(f"scroll_state:{context}")

    def update_scroll_state_from_trajectory(
        self, result: Any, context: str
    ) -> None:
        p = self.parent
        page_downs = int(p._scroll_state.get("page_downs", 0) or 0)
        wheel_downs = int(p._scroll_state.get("wheel_downs", 0) or 0)
        for item in getattr(result, "trajectory", []) or []:
            action = getattr(item, "action", None)
            if not action:
                continue
            if action.action_type == ActionType.KEY_PRESS:
                keys = str(
                    action.params.get("keys") or action.params.get("key") or ""
                ).lower()
                if "home" in keys:
                    page_downs = 0
                    wheel_downs = 0
                elif "page_down" in keys or "pagedown" in keys:
                    page_downs += 1
                elif "page_up" in keys or "pageup" in keys:
                    page_downs = max(0, page_downs - 1)
                elif keys == "end":
                    p._scroll_state["end_reached"] = True
            elif action.action_type == ActionType.SCROLL:
                direction = str(action.params.get("direction", "down")).lower()
                amount = int(action.params.get("amount", 3) or 0)
                if direction == "down":
                    wheel_downs += amount
                elif direction == "up":
                    wheel_downs = max(0, wheel_downs - amount)
        self.set_scroll_state(
            context=context,
            page_downs=page_downs,
            wheel_downs=wheel_downs,
            viewport_stage=p._viewport_stage,
        )

    def restore_scroll_position(self) -> None:
        """Replay logical scroll depth after URL re-entry in screen-only envs."""
        p = self.parent
        if not p._scroll_state:
            return
        state_url = str(p._scroll_state.get("url") or "")
        current_url = p._last_known_url or self.current_results_page_url()
        if state_url and current_url:
            def normalize(url: str) -> str:
                return re.sub(r"^https?://(www\.)?", "", url).rstrip("/")

            if normalize(state_url) != normalize(current_url):
                logger.info(
                    "  [resume] Skipping scroll restore for different URL "
                    "(state=%s current=%s)",
                    state_url[:80],
                    current_url[:80],
                )
                return
        page_downs = int(p._scroll_state.get("page_downs", 0) or 0)
        wheel_downs = int(p._scroll_state.get("wheel_downs", 0) or 0)
        if page_downs <= 0 and wheel_downs <= 0:
            return
        try:
            p.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(0.5)
            for _ in range(min(page_downs, 12)):
                p.env.step(
                    Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"})
                )
                time.sleep(0.3)
            if wheel_downs:
                p.env.step(
                    Action(
                        action_type=ActionType.SCROLL,
                        params={"direction": "down", "amount": min(wheel_downs, 40)},
                    )
                )
                time.sleep(0.5)
            logger.info(
                "  [resume] Restored scroll depth page_downs=%s wheel_downs=%s context=%s",
                page_downs,
                wheel_downs,
                p._scroll_state.get("context", ""),
            )
        except Exception as e:
            logger.warning("  [resume] Failed to restore scroll position: %s", e)

    # ── Browser-session resume ──────────────────────────────────────────

    def resume_browser_state(self, url: str) -> bool:
        """Re-enter the browser at ``url`` and replay scroll depth."""
        p = self.parent
        if not url:
            return False
        logger.info("  [resume] Re-entering browser state at %s", url[:140])
        try:
            p.env.reset(task="resume", start_url=url)
            time.sleep(12)
            try:
                p.env.step(
                    Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"})
                )
                time.sleep(1)
            except Exception:
                pass
            p._last_known_url = url
            self.restore_scroll_position()
            return True
        except Exception as e:
            logger.warning("  [resume] Failed to restore browser at %s: %s", url[:120], e)
            return False


__all__ = ["BrowserState"]
