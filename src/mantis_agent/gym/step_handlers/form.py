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
from .. import adaptive_settle
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
        # #294 adaptive-settle: cap at 4s but exit early when the form
        # is rendered.
        adaptive_settle.settle_after_action(env, max_seconds=4.0)
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
                # Visual-affordance fallback before failing — covers
                # non-English / icon-only inputs whose labels don't
                # match the configured aliases. Defocus first so any
                # currently-focused control doesn't absorb the
                # subsequent search's interactions.
                logger.warning(
                    "  [claude-form] fill_field: label-match exhausted "
                    "for '%s' — trying visual-affordance fallback", label,
                )
                try:
                    env.step(Action(
                        action_type=ActionType.KEY_PRESS, params={"keys": "Tab"},
                    ))
                    time.sleep(0.3)
                except Exception:
                    pass
                shot = _wait_for_rendered_screenshot(env)
                target = extractor.find_target_by_affordance(shot, search_intent)
                runner.costs["claude_extract"] += 1
            if not target:
                logger.warning(f"  [claude-form] fill_field: target '{label}' not found")
                return StepResult(step_index=index, intent=step.intent, success=False, data="form_target_not_found")
            x, y = target["x"], target["y"]
            type_value = value or target.get("value") or ""
            try:
                # Click the field, clear any pre-filled value, then type.
                time.sleep(random.uniform(0.3, 0.8))
                # #300: SoM-anchored focus click. Only the FIRST click
                # gets the SoM dispatch — the triple-click that follows
                # is xdotool-only because it needs to land at the same
                # X-level coordinate to trigger text selection. The CDP
                # ``el.click()`` doesn't reliably emit the
                # ``selectionchange`` event a triple-click depends on.
                from ..som_dispatch import try_som_click
                if try_som_click(env, x, y, ctx.routing_policy):
                    ctx.state["_executor_backend"] = "som"
                else:
                    env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                    ctx.state["_executor_backend"] = "vision"
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
            # Agentic-recovery hints — issue #224 follow-up; epic #377
            # Phase A.3 lifts the consumption side into a shared helper
            # so any handler that builds a Claude prompt from
            # ``step.intent`` can splice the accumulated hints in with
            # one call. Hints come from Claude analysing the failure
            # screenshot — much more specific than the generic "avoid
            # these coords" feedback the snapshot-diff path produces.
            from .. import recovery_hints as _hints
            hint_block = _hints.get_hint_block(runner, index)
            if hint_block:
                search_intent = search_intent + hint_block
                logger.warning(
                    "  [claude-form] submit '%s' retry — applying %d "
                    "recovery hint(s) from agentic-recovery loop",
                    label, _hints.count(runner, index),
                )
            # Agentic retry feedback — issue #224 follow-up. When a
            # previous attempt at this step failed with no_state_change
            # / wrong_target, the runner records the click target + label
            # on ``_step_failure_history``. We surface that to
            # ``find_form_target`` so the LLM avoids picking the same
            # broken target again. Without this, retries blindly re-pick
            # the same coordinates and fail identically (the canonical
            # case is the staff-crm "Click Qualified" step where the
            # label-text matched a status pill rather than the row link).
            failure_history = (
                runner._step_failure_history.get(index, [])
                if hasattr(runner, "_step_failure_history") else []
            )
            if failure_history:
                avoid_lines = []
                for record in failure_history[-3:]:  # last 3 failures
                    avoid_lines.append(
                        f"  - target ({record.get('x', '?')}, {record.get('y', '?')}) "
                        f"matched label='{record.get('label', '?')}' "
                        f"but {record.get('kind', 'no_state_change')} "
                        f"(reason: {record.get('reason', '')[:80]})"
                    )
                search_intent = (
                    search_intent
                    + "\n\nNOTE: previous attempts at this step did not change "
                    "the page UI:\n" + "\n".join(avoid_lines)
                    + "\n\nThis usually means the previous click landed on a "
                    "non-action element (a status badge, a filter chip, a "
                    "label rather than a link / button). Pick a DIFFERENT "
                    "target — different coordinates AND ideally a different "
                    "kind of element (e.g. the row container instead of the "
                    "status pill, the link rather than the surrounding text)."
                )
                # Log at WARNING so the trace is visible in Modal's
                # default INFO-suppressed log capture; this firing is
                # the canonical signal that the agentic-retry loop
                # actually engaged on a retry attempt.
                logger.warning(
                    f"  [claude-form] submit '{label}' retry — feeding "
                    f"{len(failure_history)} prior failure(s) into search prompt"
                )
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

            # Vision-affordance fallback — the label-driven scroll-probe
            # above can't find elements whose actual text differs from
            # any configured alias (canonical case: a French CRM whose
            # ``Update Lead`` button reads ``Enregistrer`` and the plan
            # didn't enumerate that, OR an icon-only checkmark button).
            # Before halting, do ONE final pass that asks Claude to
            # identify the right element FOR THE INTENT by VISUAL
            # AFFORDANCE — shape, position, styling — independent of
            # label. The intent prose drives element-type selection
            # (button / input / dropdown), so this fallback handles
            # any step type, not just submits.
            #
            # Defocus any active input first (Tab key) so the prior
            # keyboard-scroll attempts that may have been eaten by an
            # open dropdown / focused field can finally move the page —
            # without this, End/Page_Down on a focused <select> are
            # no-ops and the search saw the same viewport repeatedly.
            if target is None:
                logger.warning(
                    "  [claude-form] submit: label-match exhausted after "
                    "probes %s — trying visual-affordance fallback",
                    probe_attempts,
                )
                try:
                    env.step(Action(
                        action_type=ActionType.KEY_PRESS, params={"keys": "Tab"},
                    ))
                    time.sleep(0.3)
                    env.step(Action(
                        action_type=ActionType.KEY_PRESS, params={"keys": "End"},
                    ))
                except Exception:
                    pass
                time.sleep(0.6)
                shot = _wait_for_rendered_screenshot(env)
                screenshot = shot
                target = extractor.find_target_by_affordance(shot, search_intent)
                runner.costs["claude_extract"] += 1
                if target:
                    probe_attempts.append("vision-affordance")

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
            # Stash the click target so the demote check can record
            # the target into ``_step_failure_history`` if the click
            # turns out to produce no state change. Cleared by the
            # demote check after it consumes the value.
            runner._last_submit_target = {
                "x": x, "y": y,
                "label": label,
                "matched_label": str(target.get("label") or ""),
                "step_index": index,
            }
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
                # #300: SoM-anchored submit click — the canonical #88
                # failure mode (SPA Login / Update button whose
                # onPointerDown stops propagation so xdotool's
                # mousedown doesn't reach the onClick handler). CDP
                # ``el.click()`` dispatches the synthetic event chain
                # React / Vue / Svelte actually listen for. Falls
                # through to xdotool on policy-off / no CDP / no
                # element at point.
                from ..som_dispatch import try_som_click
                if try_som_click(env, x, y, ctx.routing_policy):
                    ctx.state["_executor_backend"] = "som"
                else:
                    env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                    ctx.state["_executor_backend"] = "vision"
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

                logger.info(f"  [claude-form] submit '{label[:40]}' at ({x}, {y})")
                # Include the click coordinates in result.data so a
                # post-mortem (or the agentic-retry feedback) can see
                # whether successive retries actually picked different
                # targets. Without coordinates, two failed retries with
                # ``submit:Login`` data look identical even when they
                # clicked different pixels.
                return StepResult(
                    step_index=index, intent=step.intent, success=True,
                    steps_used=1, duration=3.0,
                    data=f"submit:{label[:40]}@({x},{y})",
                )
            except Exception as e:
                logger.warning(f"  [claude-form] submit failed: {e}")
                return StepResult(step_index=index, intent=step.intent, success=False, data=f"submit_error:{e}")

        if step.type == "right_click":
            # #373: open the browser's native context menu on a single
            # labelled element. Uses find_form_target (Claude vision)
            # to locate the target by label, then dispatches a
            # button=right click. No URL-change verification — a
            # right-click that succeeds opens a menu, it does not
            # navigate. SoM dispatch is left-click-only (``el.click()``
            # synthesises a left-button event chain), so this path
            # goes straight through xdotool / Playwright button=3.
            label = str(params.get("label") or "").strip()
            aliases = params.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            aliases = [str(a).strip() for a in aliases if str(a).strip()]
            search_intent = (
                f"Find the '{label}' element so we can right-click on it "
                f"to open its native context menu"
                if label else step.intent
            )
            target = extractor.find_form_target(
                screenshot, search_intent,
                target_label=label, target_aliases=aliases,
            )
            runner.costs["claude_extract"] += 1
            if not target:
                logger.warning(
                    f"  [claude-form] right_click: target '{label}' not found"
                )
                return StepResult(
                    step_index=index, intent=step.intent, success=False,
                    data="form_target_not_found",
                    failure_class="selector_miss",
                )
            x, y = int(target["x"]), int(target["y"])
            try:
                time.sleep(random.uniform(0.2, 0.5))
                env.step(Action(
                    action_type=ActionType.CLICK,
                    params={"x": x, "y": y, "button": "right"},
                ))
                ctx.state["_executor_backend"] = "vision"
                runner.costs["gpu_steps"] += 1
                # Brief settle to let the context menu render — caller
                # typically follows with a ``submit`` step keyed on a
                # menu item ("Open Link in New Tab", "Copy Link", …).
                time.sleep(0.6)
                logger.info(
                    f"  [claude-form] right_click '{label[:40]}' at ({x}, {y})"
                )
                return StepResult(
                    step_index=index, intent=step.intent, success=True,
                    steps_used=1, duration=1.0,
                    data=f"right_click:{label[:40]}@({x},{y})",
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  [claude-form] right_click failed: {e}")
                return StepResult(
                    step_index=index, intent=step.intent, success=False,
                    data=f"right_click_error:{e}",
                )

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
                # #300: SoM-anchored dropdown-open click. Same React
                # onClick / onPointerDown story as the submit click —
                # custom-styled dropdowns (Headless UI, MUI Select)
                # often miss xdotool mousedown.
                from ..som_dispatch import try_som_click
                if try_som_click(env, target["x"], target["y"], ctx.routing_policy):
                    ctx.state["_executor_backend"] = "som"
                else:
                    env.step(Action(action_type=ActionType.CLICK, params={"x": target["x"], "y": target["y"]}))
                    ctx.state["_executor_backend"] = "vision"
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
                # #300: SoM-anchored option-pick click (Phase 2 of
                # select_option). The picked option is the
                # ``executor_backend`` we want to record on the
                # StepResult — overwrite the open-click tag set above.
                from ..som_dispatch import try_som_click
                if try_som_click(env, option_target["x"], option_target["y"], ctx.routing_policy):
                    ctx.state["_executor_backend"] = "som"
                else:
                    env.step(Action(action_type=ActionType.CLICK, params={"x": option_target["x"], "y": option_target["y"]}))
                    ctx.state["_executor_backend"] = "vision"
                runner.costs["gpu_steps"] += 1
                time.sleep(0.8)
                # Blur the dropdown so any pending onChange / onBlur
                # handler commits the new value to the underlying form
                # state. Many React-style controlled selects only fire
                # onChange on blur — without this Tab, the dropdown
                # *visually* shows the new option (the verifier reads
                # it correctly) but the form serializes the previous
                # default on submit. Diagnosed against staff-crm:
                # Priority=High clicked + visually displayed, but
                # Update Lead persisted Priority=Critical.
                try:
                    env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Tab"}))
                except Exception:
                    pass
                time.sleep(0.7)
            except Exception as e:
                logger.warning(f"  [claude-form] select_option pick failed: {e}")
                return StepResult(step_index=index, intent=step.intent, success=False, data=f"select_pick_error:{e}")

            # Post-click verification — the two-phase open+pick flow can
            # land on an adjacent menu item (y-coordinate disambiguation
            # between visually similar options). Without this read-back,
            # the runner reports ``select:Priority=High`` when the
            # dropdown actually committed ``Critical``, the verify gate
            # later sees the wrong value, and recovery wastes a budget.
            if dropdown and option:
                verify: dict | None = None
                try:
                    verify_shot = env.screenshot()
                    verify = extractor.verify_dropdown_value(
                        verify_shot, dropdown_label=dropdown, expected_value=option,
                    )
                    runner.costs["claude_extract"] += 1
                except Exception as ve:  # noqa: BLE001
                    logger.warning("  [claude-form] verify_dropdown_value raised: %s", ve)
                # Always log the verifier's verdict so post-mortem on a
                # mis-click vs lying-LLM is possible from the modal logs
                # alone (without re-running with a debugger).
                if verify is None:
                    logger.warning(
                        "  [claude-form] verify_dropdown_value returned None "
                        "(API failure or empty schema response) — proceeding "
                        "without verification; downstream verify gate is the "
                        "safety net"
                    )
                else:
                    logger.info(
                        "  [claude-form] verify_dropdown: dropdown='%s' "
                        "expected='%s' observed='%s' matches=%s",
                        dropdown[:30], option[:30],
                        (verify.get("observed") or "<empty>")[:40],
                        verify.get("matches"),
                    )
                if verify is not None and not verify.get("matches", False):
                    observed = (verify.get("observed") or "").strip() or "<unknown>"
                    logger.warning(
                        "  [claude-form] select_option mismatch: dropdown '%s' "
                        "shows '%s' but expected '%s'",
                        dropdown[:30], observed[:40], option[:30],
                    )
                    runner._dump_debug_screenshot(
                        f"select_mismatch_step{index}",
                        verify_shot,
                    )
                    # Close any stray menu so the page is clean for retry.
                    try:
                        env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Escape"}))
                    except Exception:
                        pass
                    return StepResult(
                        step_index=index, intent=step.intent, success=False,
                        data=f"select_mismatch:got={observed[:40]}_wanted={option[:30]}",
                    )

            logger.info(f"  [claude-form] select_option '{dropdown[:30]}' = '{option[:30]}'")
            return StepResult(
                step_index=index, intent=step.intent, success=True,
                steps_used=2, duration=4.0,
                data=f"select:{dropdown[:30]}={option[:30]}",
            )

        # Unknown form type — shouldn't reach here.
        logger.warning(f"  [claude-form] unsupported form step type: {step.type}")
        return StepResult(step_index=index, intent=step.intent, success=False)
