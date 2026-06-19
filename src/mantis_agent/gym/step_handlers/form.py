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

import json
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


# #435 follow-up: auto-region inference, context-aware. Originally a
# flat (type, kind) → region table — but the staff-crm-long run
# revealed that's too coarse: a ``(submit, button)`` step can be
# either a form-finalize action (Update / Save / Submit, lives in
# the footer) OR a filter-toolbar action (Apply / Search / Sort,
# lives at the top). The flat default mapped both to ``"form-footer"``
# and cropped the toolbar buttons out of frame.
#
# The fix is to peek at preceding plan steps. A submit immediately
# preceded by ``fill_field`` actions is almost certainly a
# form-finalize submit; one preceded by ``select_option`` /
# ``filter`` (with no ``fill_field`` in the recent window) is
# ambiguous and gets NO default — the plan author / decomposer
# sets ``hints.region`` explicitly when scoping matters there.
#
# The "explicit hint always wins" precedence is enforced at the
# call site in :meth:`ClaudeGuidedFormHandler.execute`.
_AUTO_REGION_WINDOW_SIZE: int = 4


def _auto_region_for_step(
    step: Any,
    params: dict[str, Any],
    *,
    plan_steps: list[Any] | None = None,
    step_index: int | None = None,
) -> str:
    """Default region hint for a step that didn't set ``hints.region``.

    Returns empty string when no auto-default fits — caller treats
    that as "no crop, same as pre-#435 behaviour". Operators who
    want to force the unscoped path can set
    ``hints.region: "full"`` (a no-op named region defined in
    :mod:`..form_targeting.region`); both routes leave the
    screenshot untouched but ``"full"`` is more explicit.

    Context-aware: when ``plan_steps`` and ``step_index`` are both
    provided, a ``(submit, button)`` step only auto-crops to
    ``"form-footer"`` if the preceding ``_AUTO_REGION_WINDOW_SIZE``
    steps include at least one ``fill_field``. That signal
    distinguishes form-finalize submits (which want the footer
    crop) from filter-toolbar / list-page submits (which don't).

    When the context is missing (legacy callers, unit tests that
    build a StepContext directly), the helper returns empty rather
    than fall back to the old flat default — better to under-crop
    than mis-crop.
    """
    step_type = str(getattr(step, "type", "") or "")
    kind = str(params.get("kind") or "").lower()
    if (step_type, kind) != ("submit", "button"):
        return ""
    if plan_steps is None or step_index is None:
        return ""
    window_start = max(0, int(step_index) - _AUTO_REGION_WINDOW_SIZE)
    recent = plan_steps[window_start:int(step_index)]
    if any(
        str(getattr(s, "type", "") or "") == "fill_field"
        for s in recent
    ):
        return "form-footer"
    return ""


# PR-D follow-up — Tier-2 fallback for nav_link sidebar clicks.
# When vision + region cropping + retry context all fail to land on the
# target sidebar anchor (canonical case: cross-session layout drift
# where ``y=390`` hits Qualified in one tenant and Contacted in
# another), Tab through the DOM's focusable elements until the focus
# ring lands on an anchor whose accessible name matches the plan's
# label. Then fire Enter. Layout-drift-immune because Tab order is
# DOM-anchored, not pixel-anchored.
#
# Why this is CUA-compliant: per ``feedback_cua_no_dom_access`` rule,
# the runner must derive click TARGETS from screenshots only. Tab is
# a procedural keyboard action (no derivation); reading
# ``document.activeElement.textContent`` is a state observation, not a
# target choice — same pattern the SoM diagnostic already uses to
# inspect what element is at a clicked point.
#
# Defaults: 100 Tab presses (covers staff-crm's full focus order
# including table rows after the sidebar/top-nav prelude).
#
# History:
# - 30 (PR #448 v1) — too short, hit the sidebar's Contacted at Tab+34
# - 60 (PR #448 v2) — covered sidebar but ran out before row-link
#   table entries. Run 13288dc0 visited 60 elements: top nav + entire
#   sidebar (LEAD VIEWS, BY PRIORITY, ACTIONS, SYSTEM) + footer links,
#   ending at "A(Home), A(Leads)" with the table-row anchors
#   ("Tempest Cleaner" etc.) still ~15-25 Tabs further down.
# - 100 (current) — empirically clears all preludes + several rows
#   of the data table. Cost is still ~$0.02 / walk because it's CDP
#   introspection, not vision (~$0.0002/Tab).
_TAB_WALK_MAX_TABS = 100
_TAB_WALK_KEY_DELAY = 0.12  # seconds between Tab keypresses


def _label_matches(needle: str, candidate: str) -> bool:
    """Tab-walk anchor/input label matcher.

    Returns True when ``needle`` should be considered a match for
    ``candidate``. Lifts a strict exact/substring check (which rejects
    "Estimated Value" against "Estimated Deal Value:" because the
    word "Deal" interrupts the substring) into token-subset matching:
    all whitespace-split tokens of ``needle`` must appear among the
    tokens of ``candidate``, order-independent.

    Both arguments are expected pre-lowercased and stripped. Empty
    needle returns False; empty candidate returns False unless needle
    is also empty (caller-side guard).
    """
    if not needle or not candidate:
        return False
    if needle == candidate or needle in candidate:
        return True
    needle_tokens = {t for t in needle.split() if t}
    if not needle_tokens:
        return False
    cand_tokens = {t.strip(":,.()[]") for t in candidate.split() if t}
    return needle_tokens.issubset(cand_tokens)


def _tab_walk_to_nav_link(
    env: Any,
    target_label: str,
    *,
    max_tabs: int = _TAB_WALK_MAX_TABS,
) -> dict[str, Any] | None:
    """Tab through focusable elements until the focus ring lands on an
    anchor whose accessible name matches ``target_label``.

    Returns the match dict ``{"tabs": int, "tag": str, "name": str}``
    when found; returns ``None`` otherwise. On match, callers should
    fire ``Enter`` to activate the focused anchor.

    Requires the env to expose ``cdp_evaluate`` (CDP Runtime.evaluate).
    Without it, returns ``None`` immediately — no fallback to
    blind-tab-and-screenshot because the cost would balloon past the
    value of a single screenshot-driven retry.
    """
    if not target_label:
        return None
    needle = target_label.strip().lower()
    if not needle:
        return None
    if not hasattr(env, "cdp_evaluate"):
        logger.debug(
            "  [claude-form] tab-walk: env lacks cdp_evaluate — skipping",
        )
        return None

    # Reset focus to document.body so the first Tab lands on the FIRST
    # tabbable element in DOM order — not on whatever was focused by
    # the prior click + Enter. Without this, the walk skips items
    # earlier than the previous click position. Run `012bbc94` showed
    # the trail starting mid-sidebar (Proposal, Negotiation, ...)
    # because focus was on Qualified after the prior click; we were
    # tabbing FORWARD from there and missing New Leads / Contacted /
    # Qualified.
    #
    # We use CDP rather than keypresses because no key (Escape, Home,
    # End, Tab+Shift, …) reliably moves focus to body across browsers.
    # ``document.body.focus()`` requires a brief tabindex assignment
    # because body is not focusable by default.
    try:
        env.cdp_evaluate(
            "(() => {"
            "try { if (document.activeElement) document.activeElement.blur(); } catch (e) {}"
            "try {"
            "  document.body.tabIndex = -1;"
            "  document.body.focus();"
            "  document.body.removeAttribute('tabindex');"
            "} catch (e) {}"
            "return true;"
            "})()"
        )
        # Also Home for scroll reset (so the FIRST tabbable element is
        # actually visible if vision wants to corroborate later).
        env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Escape"}))
        env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
        time.sleep(0.3)
    except Exception as exc:  # noqa: BLE001
        logger.debug("tab-walk reset failed: %s", exc)

    js_inspect = (
        "(() => {"
        "const el = document.activeElement;"
        "if (!el) return {tag: '', name: ''};"
        "const aria = el.getAttribute && el.getAttribute('aria-label');"
        "const txt = (el.textContent || el.innerText || '').trim();"
        "const val = (el.value || '').trim();"
        "const name = (aria || txt || val || '').trim().slice(0, 120);"
        "return {tag: el.tagName || '', name: name};"
        "})()"
    )

    # Trail of (tag, short_name) tuples we visited — surfaced on
    # no-match for postmortem. Bounded by max_tabs so it never grows
    # beyond ~60 entries.
    visited: list[tuple[str, str]] = []
    for i in range(1, max_tabs + 1):
        try:
            env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Tab"}))
        except Exception as exc:  # noqa: BLE001
            logger.debug("tab-walk: Tab dispatch raised: %s", exc)
            return None
        time.sleep(_TAB_WALK_KEY_DELAY)
        try:
            result = env.cdp_evaluate(js_inspect)
        except Exception as exc:  # noqa: BLE001
            logger.debug("tab-walk: cdp_evaluate raised: %s", exc)
            continue
        if not isinstance(result, dict):
            continue
        name = str(result.get("name") or "").strip().lower()
        tag = str(result.get("tag") or "").strip().upper()
        visited.append((tag, str(result.get("name") or "")[:40]))
        # Accept `<a>` (nav/row/cell links) AND `<button>` — both are
        # primary interactive elements that activate via Enter from
        # focused state. Extended from anchor-only after staff-crm-long
        # step 10 surfaced a `kind=button` step (Edit Lead) hitting the
        # parent-container click-trap pattern: vision picks coords that
        # resolve to the toolbar `<div>` wrapping the button, not the
        # `<button>` itself. Tab-walk reaches the button via DOM order.
        if tag in ("A", "BUTTON") and name and _label_matches(needle, name):
            logger.warning(
                "  [claude-form] tab-walk: matched %s at Tab+%d "
                "(tag=%s name=%r) for target=%r",
                "button" if tag == "BUTTON" else "anchor",
                i, tag, result.get("name"), target_label,
            )
            return {"tabs": i, "tag": tag, "name": str(result.get("name") or "")}

    # Compact trail for postmortem: tag+name of each focused element.
    # Truncate per-element name to 20 chars so the log line stays
    # readable even at max_tabs=60.
    trail = ", ".join(f"{t}({n[:20]})" for t, n in visited)
    logger.warning(
        "  [claude-form] tab-walk: no anchor matched %r after %d Tabs — "
        "visited: %s",
        target_label, max_tabs, trail[:1200],
    )
    return None


def _tab_walk_to_input(
    env: Any,
    target_label: str,
    *,
    max_tabs: int = _TAB_WALK_MAX_TABS,
) -> dict[str, Any] | None:
    """Tab through focusables until focus lands on an `<input>` /
    `<textarea>` / `[contenteditable]` whose label/placeholder/aria
    matches ``target_label``.

    Counterpart to ``_tab_walk_to_nav_link`` for fill_field steps. The
    canonical failure mode (staff-crm-long step 13): vision picks
    coordinates that resolve to a `<td>` wrapping the actual `<input>`;
    the form handler's tag-guard correctly refuses to click the TD;
    fill_field halts. Tab order is DOM-anchored and reaches the input
    regardless of what the TD's pixel area covers.

    Returns ``{"tabs": int, "tag": str, "name": str, "value": str}``
    on match — including the field's CURRENT value so the caller can
    apply ``skip_if_field_has_value`` semantics without a second
    CDP round-trip.

    CUA-compliant: Tab is procedural; reading
    ``document.activeElement.{name,placeholder,value}`` is state
    observation, same as the link-Tab-walk's textContent read.

    Match criteria, in priority order (first match wins):
    1. ``<label for=ID>`` text matches ``target_label``
    2. ``aria-label`` / ``aria-labelledby`` resolves to matching text
    3. ``placeholder`` contains target_label
    4. ``name`` attribute matches target_label (case-insensitive)
    5. Previous-sibling text contains target_label (staff-crm pattern)
    """
    if not target_label:
        return None
    needle = target_label.strip().lower()
    if not needle:
        return None
    if not hasattr(env, "cdp_evaluate"):
        return None

    try:
        env.cdp_evaluate(
            "(() => {"
            "try { if (document.activeElement) document.activeElement.blur(); } catch (e) {}"
            "try {"
            "  document.body.tabIndex = -1;"
            "  document.body.focus();"
            "  document.body.removeAttribute('tabindex');"
            "} catch (e) {}"
            "return true;"
            "})()"
        )
        env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Escape"}))
        env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
        time.sleep(0.3)
    except Exception as exc:  # noqa: BLE001
        logger.debug("tab-walk-input reset failed: %s", exc)

    # Probe each focused element for input-shape + label/placeholder/aria
    # matching. Same JS pattern as ``_tab_walk_to_nav_link`` but with
    # extra fields collected and a different ``matches`` predicate.
    js_inspect = (
        "(() => {"
        "const el = document.activeElement;"
        "if (!el) return null;"
        "const tag = (el.tagName || '').toUpperCase();"
        "const editable = el.isContentEditable === true;"
        "const isInputLike = tag === 'INPUT' || tag === 'TEXTAREA' || editable;"
        "if (!isInputLike) return {tag: tag, name: '', placeholder: '', "
        "  ariaLabel: '', siblingText: '', labelText: '', value: '', isInputLike: false};"
        "const id = el.id || '';"
        "let labelText = '';"
        "if (id) {"
        "  const lab = document.querySelector('label[for=\"' + id.replace(/\"/g, '\\\\\"') + '\"]');"
        "  if (lab) labelText = (lab.innerText || lab.textContent || '').trim();"
        "}"
        "let prevText = '';"
        "let cur = el.previousElementSibling;"
        "for (let i = 0; i < 2 && cur; i++, cur = cur.previousElementSibling) {"
        "  const t = (cur.innerText || cur.textContent || '').trim();"
        "  if (t) { prevText = t; break; }"
        "}"
        "if (!prevText && el.parentElement) {"
        "  const pcur = el.parentElement.previousElementSibling;"
        "  if (pcur) prevText = (pcur.innerText || pcur.textContent || '').trim();"
        "}"
        "return {"
        "  tag: tag,"
        "  name: (el.getAttribute('name') || '').trim(),"
        "  placeholder: (el.getAttribute('placeholder') || '').trim(),"
        "  ariaLabel: (el.getAttribute('aria-label') || '').trim(),"
        "  siblingText: prevText.slice(0, 120),"
        "  labelText: labelText.slice(0, 120),"
        "  value: (el.value === undefined ? '' : String(el.value)).slice(0, 200),"
        "  isInputLike: true,"
        "};"
        "})()"
    )

    visited: list[tuple[str, str]] = []
    for i in range(1, max_tabs + 1):
        try:
            env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Tab"}))
        except Exception as exc:  # noqa: BLE001
            logger.debug("tab-walk-input: Tab dispatch raised: %s", exc)
            return None
        time.sleep(_TAB_WALK_KEY_DELAY)
        try:
            result = env.cdp_evaluate(js_inspect)
        except Exception as exc:  # noqa: BLE001
            logger.debug("tab-walk-input: cdp_evaluate raised: %s", exc)
            continue
        if not isinstance(result, dict):
            continue
        tag = str(result.get("tag") or "").upper()
        # Always record for trail, even non-input focuses
        summary = (
            result.get("labelText") or result.get("ariaLabel")
            or result.get("placeholder") or result.get("name")
            or result.get("siblingText") or ""
        )
        visited.append((tag, str(summary)[:40]))
        if not result.get("isInputLike"):
            continue
        # Match against label / aria / placeholder / name / sibling-text
        candidates = [
            str(result.get(k) or "").strip().lower()
            for k in ("labelText", "ariaLabel", "placeholder", "name", "siblingText")
        ]
        if any(_label_matches(needle, c) for c in candidates):
            logger.warning(
                "  [claude-form] tab-walk-input: matched %s at Tab+%d "
                "(label=%r aria=%r placeholder=%r name=%r) for target=%r",
                tag, i,
                result.get("labelText"), result.get("ariaLabel"),
                result.get("placeholder"), result.get("name"),
                target_label,
            )
            return {
                "tabs": i,
                "tag": tag,
                "name": str(result.get("name") or ""),
                "value": str(result.get("value") or ""),
                "label": str(result.get("labelText") or result.get("ariaLabel") or ""),
            }

    trail = ", ".join(f"{t}({n[:20]})" for t, n in visited)
    logger.warning(
        "  [claude-form] tab-walk-input: no input matched %r after %d Tabs — "
        "visited: %s",
        target_label, max_tabs, trail[:1200],
    )
    return None


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
        # #523 PR B-3 — wrap the dispatch in a ``grounding`` modelio
        # context so every grounding call this handler makes (the
        # ~10 ``target_provider.find_*`` invocations across the form
        # branches) is captured at the AnthropicToolUseClient layer.
        # ``publish_modelio_context`` is a no-op when augur is None or
        # inactive (telemetry never breaks runs).
        from ...observability.modelio import publish_modelio_context
        runner = self.parent
        index = int(ctx.state.get("index", 0))
        with publish_modelio_context(
            getattr(runner, "_augur", None),
            layer="grounding", step_index=index,
        ):
            return self._dispatch(step, ctx)

    def _dispatch(self, step: "MicroIntent", ctx: StepContext) -> StepResult:
        runner = self.parent
        env = ctx.env
        extractor = ctx.extractor
        # #406: prefer the explicit FormTargetProvider on the context.
        # Fall back to the extractor (which exposes find_form_target /
        # find_target_by_affordance / verify_dropdown_value as back-
        # compat shims) so legacy callers that build a StepContext
        # without a form_target_provider keep working.
        target_provider = ctx.form_target_provider or extractor
        index = int(ctx.state.get("index", 0))

        params = dict(getattr(step, "params", {}) or {})
        # #435 item 1: extract the optional region hint once for the
        # whole step. Every ``find_form_target`` call this handler
        # makes (initial, scroll-probe sweep, retry-with-hint) passes
        # ``region`` through, so the executor sees a cropped frame on
        # each call — coordinates re-projected by the provider before
        # they reach the runner.
        #
        # #435 follow-up: auto-region inference, context-aware. For the
        # canonical submit-on-button pattern that comes AFTER form
        # fills (Update Lead, Save, Submit), the action-button row
        # almost always lives in the form footer. But not all
        # ``(submit, button)`` steps are footer submits — filter-bar
        # Apply / Search buttons share the same shape and live at the
        # top of the page. The helper peeks at preceding plan steps
        # to disambiguate: a fill_field within the last few steps =
        # form-finalize submit = footer; otherwise no default.
        #
        # Explicit ``hints.region`` always wins (set ``"full"`` for
        # an explicit no-crop opt-out).
        step_region = (getattr(step, "hints", {}) or {}).get("region")
        if not step_region:
            ctx_blob = getattr(runner, "_active_checkpoint_context", None) or {}
            plan_for_ctx = ctx_blob.get("plan")
            plan_steps_for_ctx = getattr(plan_for_ctx, "steps", None) if plan_for_ctx else None
            step_index_for_ctx = ctx_blob.get("step_index")
            step_region = _auto_region_for_step(
                step, params,
                plan_steps=plan_steps_for_ctx,
                step_index=step_index_for_ctx,
            )
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
            base_intent = (
                f"Click the input field labelled '{label}' so we can type into it"
                if label
                else step.intent
            )
            # #411: splice any accumulated recovery hints into the
            # search prompt — same protocol the submit handler uses
            # (form.py:428). The producer is the tag-guard below: when
            # a previous attempt's coord-pick landed on a BUTTON/A/DIV
            # we record where, and the LLM is told to avoid that
            # rectangle on the next try. Without this, Claude vision
            # on lu.ma's "Your Info" modal keeps returning the same
            # (618, 587) coord that overlaps the Register button.
            from .. import recovery_hints as _hints
            hint_block = _hints.get_hint_block(runner, index)
            search_intent = base_intent + hint_block if hint_block else base_intent
            if hint_block:
                logger.warning(
                    "  [claude-form] fill_field '%s' retry — applying %d "
                    "recovery hint(s) from prior attempts",
                    label[:30], _hints.count(runner, index),
                )
            target = target_provider.find_form_target(
                screenshot,
                search_intent,
                target_label=label,
                target_value=value,
                target_aliases=aliases,
                region=step_region,
            )
            runner.costs["claude_extract"] += 1
            probe_attempts = ["initial"]

            # Scroll-probe before falling back to affordance search —
            # mirror of the ``submit`` handler's End/Home/Page_Down
            # sweep (form.py:430). The lu.ma host-question modal is
            # internally scrollable: after fill_field 1+2 succeed,
            # the third question is below the visible fold and
            # find_form_target returns None. Without this probe the
            # visual-affordance fallback kicks in immediately and is
            # liable to click the modal's submit button (it's the
            # most prominent element left on screen), prematurely
            # submitting a half-filled form. Probing via mouse-wheel
            # scroll instead of KEY_PRESS Page_Down is intentional:
            # after the previous fill_field, focus is on its input
            # and Page_Down would move the cursor inside the input
            # rather than scroll the page/modal. Mouse wheel events
            # ignore input focus.
            def _scroll_probe(direction: str, amount: int, log_tag: str) -> dict | None:
                try:
                    env.step(Action(
                        action_type=ActionType.SCROLL,
                        params={"direction": direction, "amount": amount},
                    ))
                except Exception:
                    return None
                time.sleep(0.5)
                shot = _wait_for_rendered_screenshot(env)
                t = target_provider.find_form_target(
                    shot,
                    search_intent,
                    target_label=label,
                    target_value=value,
                    target_aliases=aliases,
                    region=step_region,
                )
                runner.costs["claude_extract"] += 1
                probe_attempts.append(log_tag)
                logger.info(
                    "  [claude-form] fill_field '%s' probe %s: %s",
                    label[:30], log_tag, "found" if t else "not found",
                )
                return t

            if target is None:
                # First jump to top of the form so progressive scrolls
                # below cover the whole scroll range deterministically.
                target = _scroll_probe("up", 10, "scroll-top")
            scroll_steps = 0
            max_scrolls = 5
            while target is None and scroll_steps < max_scrolls:
                target = _scroll_probe(
                    "down", 3, f"scroll-down-{scroll_steps + 1}/{max_scrolls}",
                )
                scroll_steps += 1

            if not target:
                # Visual-affordance fallback before failing — covers
                # non-English / icon-only inputs whose labels don't
                # match the configured aliases. Defocus first so any
                # currently-focused control doesn't absorb the
                # subsequent search's interactions.
                logger.warning(
                    "  [claude-form] fill_field: label-match exhausted "
                    "for '%s' after probes %s — trying visual-affordance "
                    "fallback", label, probe_attempts,
                )
                try:
                    env.step(Action(
                        action_type=ActionType.KEY_PRESS, params={"keys": "Tab"},
                    ))
                    time.sleep(0.3)
                except Exception:
                    pass
                shot = _wait_for_rendered_screenshot(env)
                affordance = target_provider.find_target_by_affordance(shot, search_intent)
                runner.costs["claude_extract"] += 1
                # Refuse affordance results whose recommended action is
                # not text-input shaped. A ``fill_field`` step that
                # falls back to an affordance pass returning
                # ``action=click`` / ``select`` / ``right_click`` means
                # Claude could not find any visible input matching the
                # intent and picked the next-most-prominent element
                # instead — almost always the modal's submit button.
                # Clicking it would submit a half-filled form rather
                # than fill the missing field.
                if affordance is not None:
                    aff_action = str(affordance.get("action") or "").lower()
                    if aff_action and aff_action != "type":
                        logger.warning(
                            "  [claude-form] fill_field: affordance "
                            "returned action=%s for fill intent — "
                            "refusing to click (would dismiss form / "
                            "submit partial data)",
                            aff_action,
                        )
                        affordance = None
                target = affordance
            if not target:
                logger.warning(f"  [claude-form] fill_field: target '{label}' not found")
                return StepResult(step_index=index, intent=step.intent, success=False, data="form_target_not_found")
            x, y = target["x"], target["y"]
            # Tag-guard: refuse a click whose elementFromPoint is not
            # input-shaped. Without this, a stale or mis-grounded
            # target whose coordinates fall on the modal backdrop /
            # close-X / submit button dismisses the form on the
            # subsequent SoM ``el.click()``. Returns None on envs
            # without CDP (tests / Playwright path) so legacy
            # behaviour is preserved where the guard can't run.
            from ..som_dispatch import is_input_like, probe_element_tag_at
            tag_info = probe_element_tag_at(env, x, y)
            if not is_input_like(tag_info):
                tag_name = (tag_info or {}).get("tag", "?")
                logger.warning(
                    "  [claude-form] fill_field: tag-guard refused click at "
                    "(%d, %d) — elementFromPoint=%s is not input-shaped — "
                    "trying Tab-walk DOM-input fallback",
                    x, y, tag_name,
                )
                # PR-L: Tab-walk fallback for fill_field. The brain
                # picked a non-input target (canonical staff-crm step
                # 13: clicked the `<td>` wrapping the Estimated Value
                # `<input>`). Tab through focusables looking for an
                # input whose label / placeholder / aria matches —
                # same primitive as PR-J's button-kind Tab-walk but
                # matching <input>/<textarea> instead of <a>/<button>.
                tab_match = _tab_walk_to_input(env, label)
                if tab_match is not None:
                    # Tab-walk landed on the input — it's now focused.
                    # Apply the same skip-if-field-has-value semantics
                    # as the vision-grounded path below; otherwise
                    # clear via ctrl+a+Delete and type the new value.
                    skip_if_value = bool((step.params or {}).get(
                        "skip_if_field_has_value", False,
                    ))
                    current_value = tab_match.get("value", "").strip()
                    if skip_if_value and current_value:
                        logger.warning(
                            "  [claude-form] fill_field (tab-walk): field "
                            "'%s' already has value %r — skipping (plan "
                            "opted into skip_if_field_has_value)",
                            label, current_value[:40],
                        )
                        ctx.state["_executor_backend"] = "som"
                        return StepResult(
                            step_index=index, intent=step.intent,
                            success=True, steps_used=1, duration=0.5,
                            data=f"fill:{label}:skipped_existing_value",
                        )
                    type_value = value or ""
                    try:
                        env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "ctrl+a"}))
                        time.sleep(0.15)
                        env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Delete"}))
                        time.sleep(0.15)
                        if type_value:
                            env.step(Action(action_type=ActionType.TYPE, params={"text": type_value}))
                            time.sleep(0.3)
                        ctx.state["_executor_backend"] = "som"
                        runner.costs["gpu_steps"] += 1
                        logger.info(
                            "  [claude-form] fill_field (tab-walk): "
                            "filled %s='%s' via DOM-anchored focus",
                            label[:30], (type_value or "")[:30],
                        )
                        return StepResult(
                            step_index=index, intent=step.intent,
                            success=True, steps_used=1, duration=1.0,
                            data=f"fill:{label}",
                        )
                    except Exception as fill_exc:  # noqa: BLE001
                        logger.warning(
                            "  [claude-form] fill_field tab-walk type "
                            "raised: %s — falling through to recovery hint",
                            fill_exc,
                        )

                # Fall through to legacy recovery-hint path when
                # Tab-walk also misses (input not tab-reachable, label
                # match failed, etc.). #411: feed the exact failure
                # back as a recovery hint so the next attempt's
                # find_form_target doesn't re-pick the same wrong
                # coordinate.
                avoid_label = label or step.intent[:40]
                _hints.add_hint(
                    runner, index,
                    f"Your previous coordinate pick for "
                    f"'{avoid_label}' was ({x}, {y}), which "
                    f"document.elementFromPoint resolves to a "
                    f"<{tag_name}> element (a button or container, "
                    f"NOT an input). DO NOT return coordinates inside "
                    f"the box ({max(0, x - 40)}, {max(0, y - 20)}) to "
                    f"({x + 40}, {y + 20}). The input you want is "
                    f"almost certainly ABOVE that point — find the "
                    f"text-input rectangle adjacent to the label and "
                    f"return its center."
                )
                return StepResult(
                    step_index=index, intent=step.intent, success=False,
                    data=f"form_target_not_input:{tag_name}",
                )
            type_value = value or target.get("value") or ""
            # PR-L Fix 2: honor ``skip_if_field_has_value`` opt-in.
            # When the plan author marks a fill_field as conditional
            # (canonical staff-crm-long step 13: "Enter 250000 ... if
            # empty"), peek at the input's current value via CDP and
            # short-circuit when non-empty. Reading the value is a
            # state observation (CUA-compliant) — same as the SoM
            # diagnostic that already reads elementFromPoint text.
            if (step.params or {}).get("skip_if_field_has_value", False):
                try:
                    current = env.cdp_evaluate(
                        "(() => {"
                        f"const oh = window.outerHeight, ih = window.innerHeight;"
                        f"const chromeH = Math.max(0, oh - ih);"
                        f"const el = document.elementFromPoint({int(x)}, {int(y)} - chromeH);"
                        "if (!el || (el.value === undefined && !el.isContentEditable)) return '';"
                        "return String(el.value || el.textContent || '').trim();"
                        "})()"
                    )
                except Exception:  # noqa: BLE001
                    current = ""
                if isinstance(current, str) and current.strip():
                    logger.warning(
                        "  [claude-form] fill_field: field '%s' already "
                        "has value %r — skipping (plan opted into "
                        "skip_if_field_has_value)",
                        label, current[:40],
                    )
                    return StepResult(
                        step_index=index, intent=step.intent,
                        success=True, steps_used=1, duration=0.5,
                        data=f"fill:{label}:skipped_existing_value",
                    )
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
                    type_res = env.step(Action(action_type=ActionType.TYPE, params={"text": type_value}))
                    time.sleep(0.4)
                    # #931 P0: honor the post-type read-back verdict
                    # (``type_verified``). When it positively shows the
                    # text didn't land, re-type once, then demote so the
                    # verdict/log matches reality instead of optimistic
                    # success. ``tv is None`` (CDP off / no focused field)
                    # preserves the legacy behavior — no new failures.
                    tv = (getattr(type_res, "info", None) or {}).get("type_verified")
                    if tv is not None and not tv.get("success"):
                        logger.warning(
                            "  [claude-form] fill_field '%s': typed text not "
                            "detected (expected %r, field shows %r) — retrying once",
                            label[:40], type_value[:40], str(tv.get("actual"))[:40],
                        )
                        env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "ctrl+a"}))
                        time.sleep(0.1)
                        env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Delete"}))
                        time.sleep(0.1)
                        retry_res = env.step(Action(action_type=ActionType.TYPE, params={"text": type_value}))
                        time.sleep(0.4)
                        tv2 = (getattr(retry_res, "info", None) or {}).get("type_verified")
                        if tv2 is not None and not tv2.get("success"):
                            logger.warning(
                                "  [claude-form] fill_field '%s': retype still not "
                                "detected (field shows %r) — failing step",
                                label[:40], str(tv2.get("actual"))[:40],
                            )
                            failed = StepResult(
                                step_index=index, intent=step.intent, success=False,
                                steps_used=2, duration=2.0,
                                data=f"type_not_landed:{label[:30]}",
                            )
                            failed.failure_class = "type_not_landed"
                            return failed
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
            # NOTE: ``failure_history`` was originally inlined here for
            # a DOM-mode tier-0 path that was reverted (CUA contract:
            # the runner must derive click targets from screenshots
            # only, never from a direct DOM query). Lookup retained
            # at this position so the original feedback-into-prompt
            # block below can reuse it without re-querying.
            failure_history = (
                runner._step_failure_history.get(index, [])
                if hasattr(runner, "_step_failure_history") else []
            )
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
            #
            # ``failure_history`` was looked up above (DOM-mode tier-0
            # block) — reuse it here rather than re-querying the runner.
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
            target = target_provider.find_form_target(
                screenshot, search_intent,
                target_label=label, target_aliases=aliases,
                region=step_region,
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
                t = target_provider.find_form_target(
                    shot, search_intent,
                    target_label=label, target_aliases=aliases,
                    region=step_region,
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
                target = target_provider.find_target_by_affordance(shot, search_intent)
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

                # #audit batch follow-up: SoM ``el.click()`` returns
                # ok=True but the page didn't navigate. Trust-gate-
                # bypass logic extracted to ``pointer_retry`` (PR-G
                # follow-up) so the click handler can call the same
                # primitive. See that module's docstring for the full
                # rationale, gate conditions, and bounce-detection.
                from ..pointer_retry import pointer_retry_if_unchanged
                url_after_click = pointer_retry_if_unchanged(
                    env, runner, x, y,
                    url_before=url_before,
                    executor_backend=str(ctx.state.get("_executor_backend") or ""),
                    log_prefix="[claude-form]",
                )

                # Enter-key fallback: HTML forms whose JS swallows the click
                # event still submit on Return in a focused input (the
                # browser's native form behaviour). When the click + adaptive
                # settle didn't produce navigation, fire Enter and give it
                # a short additional window. Common reason this is needed:
                # the click landed on the right pixel but the button's
                # onclick handler is conditioned on something we can't see
                # from the screenshot (CSRF token, validation state).
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

                # PR-E: Tab-walk fallback for ``kind=nav_link``. The
                # canonical failure mode (staff-crm-long step 6): vision
                # + region cropping + retry context all pick coordinates
                # that resolve to the wrong sibling anchor (cross-
                # session sidebar layout drift). Tab order is DOM-
                # anchored and immune to that drift — walk Tab until a
                # focused anchor matches the plan's label, then fire
                # Enter. Bounded to 30 Tab presses (~$0.05 of grounding
                # if every step took a screenshot, but we use CDP
                # introspection here so it's near-free).
                if (
                    url_before
                    and url_after_click == url_before
                    and (params.get("kind") or "") in {"nav_link", "row_link", "cell_link", "button"}
                ):
                    # For row_link / cell_link the plan often doesn't
                    # carry a label (the anchor text is dynamic per row).
                    # Recover the label from the most-recent SoM diag
                    # in this step's failure history — the failed click
                    # captured the visible text via elementFromPoint.
                    match_label = label
                    if not match_label:
                        history_records = (
                            (getattr(runner, "_step_failure_history", {}) or {})
                            .get(index, [])
                        )
                        if history_records:
                            last = history_records[-1]
                            match_label = str(last.get("som_elv_text") or "").strip()
                            if match_label:
                                logger.warning(
                                    "  [claude-form] Tab-walk: no plan label "
                                    "for %s; using SoM-captured elv_text %r",
                                    params.get("kind") or "?", match_label,
                                )
                    logger.warning(
                        "  [claude-form] click + Enter both no-op'd on "
                        "%s '%s' — trying Tab-walk DOM-anchor fallback",
                        params.get("kind") or "?",
                        match_label or "(no label)",
                    )
                    match = _tab_walk_to_nav_link(env, match_label)
                    if match is not None:
                        # Two-stage activation. Enter is the cheap, most-
                        # browsers-do-the-right-thing path; it works for
                        # `<a href="...">`. But many SPAs use anchors
                        # without href + JS routing — for those Enter
                        # doesn't trigger nav, and we need a real-pointer
                        # click at the focused element's center to fire
                        # React's onClick handler (real keyboard event
                        # turns out to be insufficient when the framework
                        # listens for click only). Run `9e51591d` confirmed
                        # this: Tab-walk found Contacted (8) at Tab+34
                        # twice, Enter didn't navigate either time.
                        try:
                            env.step(Action(
                                action_type=ActionType.KEY_PRESS,
                                params={"keys": "Return"},
                            ))
                        except Exception as tab_enter_exc:  # noqa: BLE001
                            logger.debug(
                                "tab-walk Enter dispatch raised: %s",
                                tab_enter_exc,
                            )
                        else:
                            runner.costs["gpu_seconds"] += runner._adaptive_submit_settle(
                                url_before=url_before,
                            )
                            runner.costs["gpu_steps"] += 1
                            time.sleep(0.6)
                            url_after_click = runner._best_effort_current_url()
                        # Stage 2: if Enter didn't navigate, real-pointer
                        # click at the focused anchor's center. Requires
                        # CDP introspection of the active element's
                        # bounding rect + chrome offset, then dispatches
                        # via Input.dispatchMouseEvent (isTrusted=true).
                        if (
                            url_before
                            and url_after_click == url_before
                            and hasattr(env, "cdp_evaluate")
                            and hasattr(env, "cdp_click_via_pointer")
                            and hasattr(env, "_chrome_offset_px")
                        ):
                            logger.warning(
                                "  [claude-form] tab-walk: Enter didn't navigate "
                                "— dispatching real-pointer click at focused "
                                "anchor's center"
                            )
                            try:
                                rect = env.cdp_evaluate(
                                    "(() => {"
                                    "const el = document.activeElement;"
                                    "if (!el || !el.getBoundingClientRect) return null;"
                                    "const r = el.getBoundingClientRect();"
                                    "return {"
                                    "  x: Math.round(r.left + r.width / 2),"
                                    "  y: Math.round(r.top + r.height / 2)"
                                    "};"
                                    "})()"
                                )
                            except Exception as rect_exc:  # noqa: BLE001
                                logger.debug("tab-walk rect read raised: %s", rect_exc)
                                rect = None
                            if isinstance(rect, dict) and "x" in rect and "y" in rect:
                                try:
                                    chrome_h = int(env._chrome_offset_px() or 0)
                                except Exception:  # noqa: BLE001
                                    chrome_h = 0
                                screen_x = int(rect["x"])
                                screen_y = int(rect["y"]) + chrome_h
                                logger.warning(
                                    "  [claude-form] tab-walk: focused-anchor "
                                    "rect center=(%d,%d) screen=(%d,%d)",
                                    int(rect["x"]), int(rect["y"]),
                                    screen_x, screen_y,
                                )
                                try:
                                    env.cdp_click_via_pointer(screen_x, screen_y)
                                except Exception as click_exc:  # noqa: BLE001
                                    logger.debug(
                                        "tab-walk real-pointer click raised: %s",
                                        click_exc,
                                    )
                                else:
                                    runner.costs["gpu_seconds"] += runner._adaptive_submit_settle(
                                        url_before=url_before,
                                    )
                                    runner.costs["gpu_steps"] += 1
                                    time.sleep(0.6)
                                    url_after_click = runner._best_effort_current_url()
                        runner._dump_debug_screenshot(
                            f"submit_step{index}_post_tab_walk",
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
            target = target_provider.find_form_target(
                screenshot, search_intent,
                target_label=label, target_aliases=aliases,
                region=step_region,
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
            target = target_provider.find_form_target(
                screenshot, open_intent, target_label=dropdown,
                region=step_region,
            )
            runner.costs["claude_extract"] += 1
            if not target:
                logger.warning(f"  [claude-form] select_option: dropdown '{dropdown}' not found")
                return StepResult(step_index=index, intent=step.intent, success=False, data="form_target_not_found")
            # Native `<select>` fast path — if SoM resolves the
            # dropdown target to a `<select>` element, skip the
            # click-to-open + click-an-option dance entirely. Chrome
            # renders native `<select>` option lists as OS-level UI
            # widgets (not page DOM), so vision can't find option text
            # in a screenshot after opening. The standard programmatic
            # interaction is `el.value = X; el.dispatchEvent('change')`
            # — what MCP / Playwright / Selenium all use.
            #
            # CUA-compliant: the brain's option choice (option_label
            # from the plan) is the input; we use CDP to dispatch the
            # change at the protocol level. Same pattern PR-G uses for
            # real-pointer click dispatch — vision-derived choice,
            # procedural CDP dispatch.
            try:
                # Probe the target via SoM. cdp_click_at_point stashes
                # the elv tag/text on env._last_som_diag, which lets us
                # detect a native `<select>` without an extra round-trip.
                if hasattr(env, "cdp_click_at_point") and hasattr(env, "cdp_evaluate"):
                    # Peek at what's under the dropdown coordinates
                    # WITHOUT clicking — use elementFromPoint directly.
                    elv_probe = env.cdp_evaluate(
                        "(() => {"
                        f"const oh = window.outerHeight, ih = window.innerHeight;"
                        f"const chromeH = Math.max(0, oh - ih);"
                        f"const el = document.elementFromPoint({int(target['x'])}, {int(target['y'])} - chromeH);"
                        "if (!el) return null;"
                        "return {tag: el.tagName, name: el.name || '', id: el.id || ''};"
                        "})()"
                    )
                    if isinstance(elv_probe, dict) and elv_probe.get("tag") == "SELECT":
                        logger.warning(
                            "  [claude-form] select_option: native <select> "
                            "detected at (%s,%s) — using programmatic value-set",
                            target["x"], target["y"],
                        )
                        # Find the option by visible text and dispatch
                        # change. Case-insensitive substring match so
                        # plan option_label "High" matches option "High"
                        # even with whitespace or icon prefixes.
                        # JSON-escape option to survive `'`/`"` in the
                        # plan-supplied label.
                        js_set = (
                            "(() => {"
                            f"const oh = window.outerHeight, ih = window.innerHeight;"
                            f"const chromeH = Math.max(0, oh - ih);"
                            f"const el = document.elementFromPoint({int(target['x'])}, {int(target['y'])} - chromeH);"
                            "if (!el || el.tagName !== 'SELECT') return {ok: false, reason: 'not_a_select'};"
                            f"const target_text = {json.dumps(option)}.toLowerCase().trim();"
                            "let picked = null;"
                            "for (const opt of el.options) {"
                            "  const t = (opt.text || '').toLowerCase().trim();"
                            "  if (t === target_text || t.includes(target_text)) { picked = opt; break; }"
                            "}"
                            "if (!picked) return {ok: false, reason: 'option_not_in_select', "
                            "  available: Array.from(el.options).map(o => o.text)};"
                            "const setter = Object.getOwnPropertyDescriptor("
                            "  window.HTMLSelectElement.prototype, 'value').set;"
                            "setter.call(el, picked.value);"
                            "el.dispatchEvent(new Event('input', {bubbles: true}));"
                            "el.dispatchEvent(new Event('change', {bubbles: true}));"
                            "return {ok: true, value: picked.value, text: picked.text};"
                            "})()"
                        )
                        set_result = env.cdp_evaluate(js_set)
                        if isinstance(set_result, dict) and set_result.get("ok"):
                            logger.warning(
                                "  [claude-form] select_option (native) "
                                "'%s' = '%s'",
                                dropdown[:30], option[:30],
                            )
                            ctx.state["_executor_backend"] = "som"
                            runner.costs["gpu_steps"] += 1
                            # Mirror the click-based select path below: many
                            # controlled forms do not commit select state until
                            # focus leaves the field. Returning success before
                            # blur/readback is how a long plan silently carries
                            # stale form state into the final submit.
                            try:
                                env.step(Action(
                                    action_type=ActionType.KEY_PRESS,
                                    params={"keys": "Tab"},
                                ))
                            except Exception:
                                pass
                            time.sleep(0.7)
                            if dropdown and option:
                                verify: dict | None = None
                                verify_shot = None
                                try:
                                    verify_shot = env.screenshot()
                                    verify = target_provider.verify_dropdown_value(
                                        verify_shot,
                                        dropdown_label=dropdown,
                                        expected_value=option,
                                    )
                                    runner.costs["claude_extract"] += 1
                                except Exception as ve:  # noqa: BLE001
                                    logger.warning(
                                        "  [claude-form] native select "
                                        "verify_dropdown_value raised: %s",
                                        ve,
                                    )
                                if verify is None:
                                    logger.warning(
                                        "  [claude-form] native select verifier "
                                        "returned None — proceeding without "
                                        "readback; downstream verify gate is "
                                        "the safety net"
                                    )
                                else:
                                    logger.warning(
                                        "  [claude-form] native select verify: "
                                        "dropdown='%s' expected='%s' observed='%s' "
                                        "matches=%s",
                                        dropdown[:30], option[:30],
                                        (verify.get("observed") or "<empty>")[:40],
                                        verify.get("matches"),
                                    )
                                if verify is not None and not verify.get("matches", False):
                                    observed = (verify.get("observed") or "").strip() or "<unknown>"
                                    logger.warning(
                                        "  [claude-form] native select mismatch: "
                                        "dropdown '%s' shows '%s' but expected '%s'",
                                        dropdown[:30], observed[:40], option[:30],
                                    )
                                    if verify_shot is not None:
                                        runner._dump_debug_screenshot(
                                            f"select_mismatch_step{index}",
                                            verify_shot,
                                        )
                                    return StepResult(
                                        step_index=index, intent=step.intent,
                                        success=False,
                                        data=(
                                            f"select_mismatch:got={observed[:40]}"
                                            f"_wanted={option[:30]}"
                                        ),
                                    )
                            return StepResult(
                                step_index=index, intent=step.intent,
                                success=True, steps_used=1, duration=1.0,
                                data=f"select:{dropdown[:30]}={option[:30]}",
                            )
                        else:
                            logger.warning(
                                "  [claude-form] select_option native fast-"
                                "path failed: %s — falling through to "
                                "click-based open+pick",
                                set_result,
                            )
            except Exception as native_exc:  # noqa: BLE001
                logger.debug(
                    "select_option native fast-path raised: %s — "
                    "falling through to click-based path",
                    native_exc,
                )

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
            option_target = target_provider.find_form_target(
                opened_shot, pick_intent, target_label=option,
            )  # Note: no region= here — the opened dropdown menu lives at
            # an unpredictable position relative to the trigger, so a
            # step-level region hint (typed for the closed-form layout)
            # doesn't apply.
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
                    verify = target_provider.verify_dropdown_value(
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
