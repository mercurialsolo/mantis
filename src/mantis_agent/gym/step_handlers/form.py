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
from typing import TYPE_CHECKING, Any

from PIL import Image

from ...actions import Action, ActionType
from ..checkpoint import StepResult
from ..step_context import StepContext

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner
    from ...plan_decomposer import MicroIntent

logger = logging.getLogger(__name__)


# ── Render-wait primitives (#209 follow-up: blank-screenshot race) ──────
#
# Form steps frequently fire ``find_form_target`` against a screenshot
# the page hasn't finished painting yet — typically the first action
# after a navigate to a slow / proxied / SPA-cold-start page. The
# extractor then correctly returns ``not_found`` ("blank white page
# with no visible form elements") and the runner halts a required
# step that would have succeeded a few seconds later. Confirmed via
# /tmp/mantis_debug screenshots from a staff-crm smoke run: pure-white
# 1280x720 PNGs at attempt 1+2, fully-rendered login form at attempt 3.
#
# The helpers below detect a blank screenshot via a cheap pixel-count
# heuristic and re-screenshot until the page has visible content or
# the deadline expires. Generic — no domain knowledge; works for any
# site whose first paint is slower than the form handler's static
# settle.


def _is_blank_screenshot(img: Image.Image, threshold: float = 0.99) -> bool:
    """True iff ``img`` is at least ``threshold`` fraction near-white.

    Cheap heuristic for "page hasn't rendered yet". Even a sparse
    real page (login form, settings panel) carries 5-15% non-white
    pixels — the threshold is intentionally conservative to avoid
    false positives on legitimately white-themed UIs. Downsamples to
    a 64×64 grid before counting so the cost is fixed regardless of
    viewport size.
    """
    try:
        # NEAREST avoids anti-aliasing at boundaries — the helper counts
        # near-white pixels, and bicubic smoothing would shift the
        # ratio at the threshold edge unpredictably across viewport
        # sizes. NEAREST preserves the original pixel population.
        small = img.convert("L").resize((64, 64), Image.NEAREST)
    except Exception:
        return False
    pixels = list(small.getdata())
    if not pixels:
        return False
    near_white = sum(1 for p in pixels if p >= 250)
    return near_white / len(pixels) >= threshold


def _wait_for_rendered_screenshot(
    env: Any,
    *,
    max_retries: int = 5,
    poll_seconds: float = 2.0,
) -> Image.Image:
    """Re-screenshot until the page is non-blank or ``max_retries`` is hit.

    Returns the most recent screenshot regardless — callers should not
    assume a guaranteed-rendered frame after the cap (the page might
    genuinely be white). The retry pattern keeps the page-not-rendered
    race from cascading into a ``not_found`` response that the form
    handler treats as a real verify failure.

    The retry cap is a count rather than a wall-clock deadline so tests
    can drive the helper deterministically by mocking only
    ``time.sleep`` (a no-op sleep moves the loop forward in zero real
    time without needing to also mock ``time.time``).
    """
    img = env.screenshot()
    for _ in range(max(0, max_retries)):
        if not _is_blank_screenshot(img):
            return img
        logger.info(
            "  [claude-form] screenshot is blank (page still painting); "
            "waiting %.1fs and retrying",
            poll_seconds,
        )
        time.sleep(poll_seconds)
        img = env.screenshot()
    return img


# Visual-affordance kinds for ``submit`` steps. The decomposer classifies
# each submit step into one of these so the search prompt sent to
# ``find_form_target`` frames the target correctly. Default is "button"
# — the previous unconditional "click the X button to submit the form"
# wording — so submit steps without a ``kind`` hint behave exactly as
# before. Adding a new kind requires a template entry here AND a worked
# example in DECOMPOSE_PROMPT — that friction is intentional. (#209
# Symptom 4)
_SUBMIT_KIND_INTENT_TEMPLATES: dict[str, str] = {
    "nav_link": (
        "Click the '{label}' navigation link in the sidebar or top-level "
        "navigation. This is a primary nav item, not a button or form control."
    ),
    "tab": "Click the '{label}' tab in the page's tab bar.",
    "menu_item": "Click the '{label}' menu item — an entry in an open menu, dropdown, or kebab.",
    "row_link": (
        "Click the '{label}' link in a table row's body cell — a record-name "
        "or detail-link inside a data table that opens that row's detail page. "
        "This is the row's primary clickable cell text (often blue / underlined), "
        "NOT a column header, NOT a status badge, NOT a sort/filter control, "
        "NOT a row checkbox or action icon."
    ),
    "cell_link": (
        "Click the '{label}' link inside a table cell — a hyperlinked value "
        "embedded in a list/table cell. Click the underlined cell text itself, "
        "not the surrounding row, header, or any inline action icon."
    ),
    "button": "Click the '{label}' button to submit the form.",
}
_SUBMIT_KIND_DEFAULT = "button"


def _build_submit_search_intent(label: str, kind: str, fallback: str) -> str:
    """Construct the find_form_target prompt for a submit step.

    Generic primitive — the kind comes from the analysis/decomposition
    stage, not from any domain knowledge in this handler. Unknown kinds
    fall back to the default ``button`` framing to preserve existing
    behaviour for cached plans that predate the ``kind`` hint.
    """
    if not label:
        return fallback
    template = _SUBMIT_KIND_INTENT_TEMPLATES.get(
        kind, _SUBMIT_KIND_INTENT_TEMPLATES[_SUBMIT_KIND_DEFAULT],
    )
    return template.format(label=label)


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
        # Combined settle — form pages frequently finish hydrating after the
        # navigate that brought us here. EPIC #161 cleanup merges the
        # pre-handler 2s sleep that used to live in MicroPlanRunner
        # ._execute_step's form-types branch with the existing 2s settle
        # at the top of this method, preserving the legacy 4s total. The
        # synthesised click→submit path that calls form via the runner
        # shim no longer adds its own pre-settle either.
        time.sleep(4)
        screenshot = _wait_for_rendered_screenshot(env)

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
            kind = str(params.get("kind") or _SUBMIT_KIND_DEFAULT).strip().lower() or _SUBMIT_KIND_DEFAULT
            search_intent = _build_submit_search_intent(label, kind, step.intent)
            # Agentic scroll search — issue #89 §2 + staff-crm post-PR-#228.
            # Long forms (CRMs, settings panels, account-edit pages) almost
            # always pin the primary submit button at the absolute bottom
            # OR the top of the form. Probe the ends first (cheap) before
            # falling back to progressive Page_Down sweeps.
            #
            # Order:
            #   0. current viewport (no scroll — covers buttons in initial
            #      view, the common case)
            #   1. End (jump to bottom — covers Save / Update Lead / Submit
            #      buttons pinned at the form footer)
            #   2. Home (jump to top — covers nav-bar action buttons like
            #      "Save" pinned at a sticky header)
            #   3. progressive Page_Down sweep from top — covers buttons
            #      mid-form (rare on action submits but possible on
            #      multi-section settings pages)
            #
            # Cost: each probe = 1 ``find_form_target`` call. Steps 0+1+2
            # cost ~3 calls in the worst case before the Page_Down loop
            # starts; with the loop the budget is ~3 + 6 = 9 calls.
            target = extractor.find_form_target(
                screenshot, search_intent,
                target_label=label, target_aliases=aliases,
            )
            runner.costs["claude_extract"] += 1
            probe_attempts = ["initial"]

            def _probe(keys: str, label_for_log: str) -> dict | None:
                """One scroll-then-search step. Returns the target or None."""
                try:
                    env.step(Action(
                        action_type=ActionType.KEY_PRESS, params={"keys": keys},
                    ))
                except Exception:
                    return None
                time.sleep(0.6)
                shot = _wait_for_rendered_screenshot(env)
                # Update the closure's ``screenshot`` so the
                # downstream click uses the freshest frame for
                # debug-screenshot capture.
                nonlocal screenshot
                screenshot = shot
                t = extractor.find_form_target(
                    shot, search_intent,
                    target_label=label, target_aliases=aliases,
                )
                runner.costs["claude_extract"] += 1
                logger.info(
                    f"  [claude-form] submit '{label}' probe {label_for_log}: "
                    f"{'found' if t else 'not found'}"
                )
                probe_attempts.append(label_for_log)
                return t

            if target is None:
                target = _probe("End", "End→bottom")
            if target is None:
                target = _probe("Home", "Home→top")

            scroll_steps = 0
            max_scrolls = 6
            while target is None and scroll_steps < max_scrolls:
                target = _probe(
                    "Page_Down",
                    f"Page_Down{scroll_steps + 1}/{max_scrolls}",
                )
                scroll_steps += 1

            if not target:
                logger.warning(
                    f"  [claude-form] submit: button '{label}' not found "
                    f"after probes: {probe_attempts}"
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
                # Stash the pre-click screenshot so run_executor's
                # ``_maybe_demote_form_no_change`` can run a SPA-aware
                # visual diff before demoting a same-URL submit. Same
                # pattern PR #222 added for clicks: a successful login
                # / form submit on a CRM SPA (staff-crm is the canonical
                # case) often replaces the form with a dashboard at the
                # SAME URL; the runner-state snapshot can't see that
                # delta, so without the visual diff every such submit
                # was being demoted to failure.
                runner._last_submit_pre_screenshot = screenshot
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
                        url_after_click = runner._best_effort_current_url()
                    runner._dump_debug_screenshot(
                        f"submit_step{index}_post_enter",
                        runner._safe_screenshot(),
                    )
                else:
                    runner._dump_debug_screenshot(
                        f"submit_step{index}_post_click",
                        runner._safe_screenshot(),
                    )

                # Propagate any post-submit URL change to runner._last_known_url
                # so step_snapshot.diff() picks it up. Without this, a successful
                # navigation (Leads link, Sign In button, etc.) is invisible to
                # the snapshot diff — _last_known_url only updates in the click
                # handler's detail-page branch — and run_executor's
                # _maybe_demote_form_no_change demotes the success to failure.
                if url_after_click and url_after_click != url_before:
                    runner._last_known_url = url_after_click

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
