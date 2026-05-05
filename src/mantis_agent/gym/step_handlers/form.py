"""ClaudeGuidedFormHandler — fill_field / submit / select_option dispatch.

Phase 2 of EPIC #161, third handler extraction. Lifts
``MicroPlanRunner._execute_claude_guided_form`` (252 LOC) verbatim
into a standalone class. The runner method becomes a delegating shim.

Form-shaped steps use ``ClaudeExtractor.find_form_target`` (single
labelled element) instead of ``find_all_listings`` (the listings
extractor that returns zero cards on a login form). Operates on
whatever page is currently loaded — does not assume listings/results
semantics, does not call ``_ensure_results_filters``.

Step-type contract (carried on ``MicroIntent.params``):

==================  ============================================================
Step type           ``params`` keys
==================  ============================================================
``fill_field``      ``label``, ``value``, optional ``aliases``
``submit``          ``label``, optional ``aliases``
``select_option``   ``dropdown_label`` (or ``label``), ``option_label`` (or ``value``)
==================  ============================================================

The runner trusts ``params`` over the prose ``intent`` when both are
present — same behaviour as the in-runner version.

What the handler reads from :class:`StepContext`:

- ``env``         — screenshot / step
- ``extractor``   — find_form_target (Claude vision)

What it reads from the runner via parent back-reference:

- ``costs``                       — cost meter aliased dict
- ``_best_effort_current_url``    — pre/post URL snapshot for adaptive settle
- ``_adaptive_submit_settle``     — bounded URL-poll until navigation
- ``_dump_debug_screenshot``      — MANTIS_DEBUG_DUMP_DIR opt-in
- ``_safe_screenshot``            — exception-tolerant capture for debug dumps

Phase 4 will lift ``_adaptive_submit_settle`` and the debug-dump helpers
onto a smaller "BrowserActions" service so the parent back-reference
shrinks; that's a separate ergonomic cleanup.
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


class ClaudeGuidedFormHandler:
    """Implements :class:`~..step_context.StepHandler` for form-shaped steps.

    Bound to ``fill_field`` / ``submit`` / ``select_option``. Also
    invoked by :meth:`MicroPlanRunner._execute_step`'s "click" branch
    when the layout hint indicates a single-element click (the runner
    synthesises a ``submit``-typed :class:`MicroIntent` and calls into
    the form handler).
    """

    # The handler claims one canonical step type for HandlerRegistry.register;
    # MicroPlanRunner.__init__ wires it to all three form types via
    # register_for_types.
    step_type = "submit"

    def __init__(self, runner: "MicroPlanRunner") -> None:
        self.parent = runner

    def execute(self, step: "MicroIntent", ctx: StepContext) -> StepResult:
        runner = self.parent
        env = ctx.env
        extractor = ctx.extractor
        index = int(ctx.state.get("index", 0))

        params = dict(getattr(step, "params", {}) or {})
        # Brief settle — form pages frequently finish hydrating after the
        # navigate that brought us here.
        time.sleep(2)
        screenshot = env.screenshot()

        if step.type == "fill_field":
            label = str(params.get("label") or "").strip()
            value = str(params.get("value") or "").strip()
            aliases = params.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            aliases = [str(a).strip() for a in aliases if str(a).strip()]
            search_intent = (
                f"Click the input field labelled '{label}' so we can type into it"
                if label
                else step.intent
            )
            target = extractor.find_form_target(
                screenshot,
                search_intent,
                target_label=label,
                target_value=value,
                target_aliases=aliases,
            )
            runner.costs["claude_extract"] += 1
            if not target:
                logger.warning(f"  [claude-form] fill_field: target '{label}' not found")
                return StepResult(step_index=index, intent=step.intent, success=False, data="form_target_not_found")
            x, y = target["x"], target["y"]
            type_value = value or target.get("value") or ""
            try:
                # Click the field, clear any pre-filled value, then type.
                time.sleep(random.uniform(0.3, 0.8))
                env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                runner.costs["gpu_steps"] += 1
                time.sleep(0.4)
                # Triple-click to select existing text (more reliable than ctrl+a in some inputs)
                env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                time.sleep(0.05)
                env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                time.sleep(0.2)
                env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "ctrl+a"}))
                time.sleep(0.15)
                env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Delete"}))
                time.sleep(0.2)
                if type_value:
                    env.step(Action(action_type=ActionType.TYPE, params={"text": type_value}))
                    time.sleep(0.4)
                logger.info(f"  [claude-form] fill_field '{label[:40]}' = '{type_value[:30]}'")
                return StepResult(
                    step_index=index, intent=step.intent, success=True,
                    steps_used=2, duration=2.0,
                    data=f"fill:{label[:40]}",
                )
            except Exception as e:
                logger.warning(f"  [claude-form] fill_field failed: {e}")
                return StepResult(step_index=index, intent=step.intent, success=False, data=f"fill_error:{e}")

        if step.type == "submit":
            label = str(params.get("label") or "").strip()
            aliases = params.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            aliases = [str(a).strip() for a in aliases if str(a).strip()]
            search_intent = (
                f"Click the '{label}' button to submit the form" if label else step.intent
            )
            # Scroll-and-rescan loop — issue #89 §2. Long forms (CRMs,
            # settings panels) often render the primary submit button below
            # the fold; the previous single-screenshot path declared the
            # button missing without ever checking lower viewports.
            target = extractor.find_form_target(
                screenshot, search_intent,
                target_label=label, target_aliases=aliases,
            )
            runner.costs["claude_extract"] += 1
            scroll_steps = 0
            max_scrolls = 4
            while target is None and scroll_steps < max_scrolls:
                logger.info(
                    f"  [claude-form] submit '{label}' not in viewport — "
                    f"scrolling Page_Down ({scroll_steps + 1}/{max_scrolls})"
                )
                try:
                    env.step(Action(
                        action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"},
                    ))
                except Exception:
                    break
                time.sleep(0.6)
                screenshot = env.screenshot()
                target = extractor.find_form_target(
                    screenshot, search_intent,
                    target_label=label, target_aliases=aliases,
                )
                runner.costs["claude_extract"] += 1
                scroll_steps += 1
            if not target:
                logger.warning(
                    f"  [claude-form] submit: button '{label}' not found "
                    f"after {scroll_steps} scroll(s)"
                )
                # Reset scroll position so the next step starts at top.
                try:
                    env.step(Action(
                        action_type=ActionType.KEY_PRESS, params={"keys": "Home"},
                    ))
                except Exception:
                    pass
                return StepResult(step_index=index, intent=step.intent, success=False, data="form_target_not_found")
            x, y = target["x"], target["y"]
            try:
                time.sleep(random.uniform(0.4, 0.9))
                # Snapshot URL before click so the adaptive settle below
                # can detect the moment the page actually navigates.
                url_before = runner._best_effort_current_url()
                runner._dump_debug_screenshot(
                    f"submit_step{index}_pre_click", screenshot,
                )
                env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                runner.costs["gpu_seconds"] += runner._adaptive_submit_settle(
                    url_before=url_before,
                )
                runner.costs["gpu_steps"] += 1

                # Enter-key fallback: HTML forms whose JS swallows the click
                # event still submit on Return in a focused input (the
                # browser's native form behaviour). When the click + adaptive
                # settle didn't produce navigation, fire Enter and give it
                # a short additional window. Common reason this is needed:
                # the click landed on the right pixel but the button's
                # onclick handler is conditioned on something we can't see
                # from the screenshot (CSRF token, validation state).
                url_after_click = runner._best_effort_current_url()
                if url_before and url_after_click == url_before:
                    logger.info(
                        "  [claude-form] click did not navigate — trying "
                        "Enter-key fallback on focused field"
                    )
                    try:
                        env.step(Action(
                            action_type=ActionType.KEY_PRESS,
                            params={"keys": "Return"},
                        ))
                    except Exception as enter_exc:  # noqa: BLE001
                        logger.debug("Enter fallback failed: %s", enter_exc)
                    else:
                        runner.costs["gpu_seconds"] += runner._adaptive_submit_settle(
                            url_before=url_before,
                        )
                        runner.costs["gpu_steps"] += 1
                    runner._dump_debug_screenshot(
                        f"submit_step{index}_post_enter",
                        runner._safe_screenshot(),
                    )
                else:
                    runner._dump_debug_screenshot(
                        f"submit_step{index}_post_click",
                        runner._safe_screenshot(),
                    )

                logger.info(f"  [claude-form] submit '{label[:40]}'")
                return StepResult(
                    step_index=index, intent=step.intent, success=True,
                    steps_used=1, duration=3.0,
                    data=f"submit:{label[:40]}",
                )
            except Exception as e:
                logger.warning(f"  [claude-form] submit failed: {e}")
                return StepResult(step_index=index, intent=step.intent, success=False, data=f"submit_error:{e}")

        if step.type == "select_option":
            dropdown = str(params.get("dropdown_label") or params.get("label") or "").strip()
            option = str(params.get("option_label") or params.get("value") or "").strip()
            # Phase 1: open the dropdown.
            open_intent = (
                f"Click the '{dropdown}' dropdown to open its option list"
                if dropdown else step.intent
            )
            target = extractor.find_form_target(
                screenshot, open_intent, target_label=dropdown,
            )
            runner.costs["claude_extract"] += 1
            if not target:
                logger.warning(f"  [claude-form] select_option: dropdown '{dropdown}' not found")
                return StepResult(step_index=index, intent=step.intent, success=False, data="form_target_not_found")
            try:
                time.sleep(random.uniform(0.3, 0.8))
                env.step(Action(action_type=ActionType.CLICK, params={"x": target["x"], "y": target["y"]}))
                runner.costs["gpu_steps"] += 1
                time.sleep(1.5)  # Allow option list to render
            except Exception as e:
                logger.warning(f"  [claude-form] select_option open failed: {e}")
                return StepResult(step_index=index, intent=step.intent, success=False, data=f"select_open_error:{e}")

            # Phase 2: pick the option. Re-screenshot so Claude sees the open menu.
            opened_shot = env.screenshot()
            pick_intent = (
                f"Click the '{option}' option in the open dropdown menu"
                if option else step.intent
            )
            option_target = extractor.find_form_target(
                opened_shot, pick_intent, target_label=option,
            )
            runner.costs["claude_extract"] += 1
            if not option_target:
                # Close the dropdown to keep the page in a clean state.
                try:
                    env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Escape"}))
                except Exception:
                    pass
                logger.warning(f"  [claude-form] select_option: option '{option}' not found in open menu")
                return StepResult(step_index=index, intent=step.intent, success=False, data="option_not_found")
            try:
                time.sleep(random.uniform(0.2, 0.6))
                env.step(Action(action_type=ActionType.CLICK, params={"x": option_target["x"], "y": option_target["y"]}))
                runner.costs["gpu_steps"] += 1
                time.sleep(1.5)
                logger.info(f"  [claude-form] select_option '{dropdown[:30]}' = '{option[:30]}'")
                return StepResult(
                    step_index=index, intent=step.intent, success=True,
                    steps_used=2, duration=4.0,
                    data=f"select:{dropdown[:30]}={option[:30]}",
                )
            except Exception as e:
                logger.warning(f"  [claude-form] select_option pick failed: {e}")
                return StepResult(step_index=index, intent=step.intent, success=False, data=f"select_pick_error:{e}")

        # Unknown form type — shouldn't reach here.
        logger.warning(f"  [claude-form] unsupported form step type: {step.type}")
        return StepResult(step_index=index, intent=step.intent, success=False)
