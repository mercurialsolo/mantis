"""Shared SoM-anchored click dispatch (#300 follow-up).

Both :class:`~.runner.GymRunner` (the ``/v1/cua`` engine) and
:class:`~.micro_runner.MicroPlanRunner` (the ``/v1/predict`` engine,
serving recipe-based / host-integration plans) need the same
Set-of-Mark click semantics: when the routing policy promotes SoM
and the env exposes a Chrome DevTools Protocol click shim
(:meth:`~.xdotool_env.XdotoolGymEnv.cdp_click_at_point`), dispatch
the click via ``document.elementFromPoint(x, y).click()`` instead of
xdotool's X-level mouse pipeline. That sidesteps the well-known #88
row-click failure on SPA layouts whose ``mousedown`` handler stops
propagation or whose React onClick never runs from a synthetic xdotool
mousedown.

This module exists so the policy/capability/fallback logic lives in
exactly one place — :func:`try_som_click` — and both runners call into
it without duplicating the check.

Design contract:

* The helper does NOT call :meth:`env.step`. The caller decides what
  the fallback is (typically ``env.step(Action(CLICK, ...))`` for
  xdotool, but a Playwright-backed runner may want
  ``page.mouse.click``). The helper just tells the caller "I did
  SoM-anchor this; you don't need to fall back."
* The helper does NOT raise. Every failure path returns ``False`` so
  the caller's branchless fallback works without a try/except wrapper.
* The helper does NOT log at WARNING for the routine "no policy / no
  capability / no element at point" cases — those are expected during
  normal traffic. CDP-call failures already log inside
  :meth:`XdotoolGymEnv.cdp_click_at_point`.

Reads:

* :class:`~.runner.RoutingPolicy.som_for_unstructured_clicks` —
  the policy gate. Toggle on via
  ``MANTIS_ROUTE_SOM_CLICKS=enabled`` or by passing a custom
  policy.
* ``env.cdp_click_at_point(x, y) -> bool`` — the capability gate.
  Returns ``True`` iff an element exists at the point and the JS
  click dispatched.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def probe_element_tag_at(env: Any, x: int, y: int) -> dict | None:
    """Best-effort: report the tag + contentEditable of the element at
    SCREEN (x, y) — the same coordinates xdotool would click.

    Returns ``None`` when:

    - The env does not expose ``cdp_evaluate`` (Playwright path, tests).
    - The CDP call fails or throws.
    - ``document.elementFromPoint`` returns ``null`` (point off-screen).

    On success returns ``{"tag": str, "contentEditable": bool}`` where
    ``tag`` is the upper-case ``tagName`` of the element and
    ``contentEditable`` reflects ``el.isContentEditable`` (covers rich
    text editors that are technically ``<div>`` but accept typed input).

    Used by :class:`~.step_handlers.form.ClaudeGuidedFormHandler` to
    refuse a ``fill_field`` click when the element at the chosen point
    is a button / backdrop / modal close-X — clicking those dismisses
    the form instead of focusing an input.

    Coordinate system note (#413): caller passes screen-space (x, y) —
    the same numbers xdotool will receive. But ``document.element
    FromPoint`` takes CSS-viewport coords (origin = top-left of the
    page area, BELOW the browser's tabs + URL bar). Without the
    chrome-offset subtraction below, a tag-guard call for a Title
    input visually at screen-y=588 would resolve to whatever element
    sits at viewport-y=588 instead — which on Lu.ma's modal is the
    Register button ~85 px below the input. The runner then refused
    correct clicks as ``form_target_not_input:BUTTON``. Fix: subtract
    ``window.outerHeight - window.innerHeight`` so screen-y is
    translated into viewport-y before the elementFromPoint call.

    Tests that don't wire CDP get ``None`` and the caller falls through
    to legacy behaviour — no breakage on the existing form-handler
    suite. Real Modal/Baseten envs that DO wire CDP get the guard.
    """
    eval_fn = getattr(env, "cdp_evaluate", None)
    if not callable(eval_fn):
        return None
    js = (
        "(() => {"
        # screen-y → viewport-y by subtracting the chrome offset.
        # ``outerHeight - innerHeight`` is the canonical browser
        # measurement that covers tabs + URL bar (+ OS window
        # decorations when present). On Xvfb without a WM this is
        # just Chrome's tabs+URL bar (~85 px); on a desktop it
        # additionally includes the OS title bar. ``Math.max(0, …)``
        # guards against headless modes that report 0 chrome.
        "const chromeH = Math.max(0, window.outerHeight - window.innerHeight);"
        f"const vx = {int(x)};"
        f"const vy = {int(y)} - chromeH;"
        "const el = document.elementFromPoint(vx, vy);"
        "if (!el) return null;"
        "return {"
        "tag: (el.tagName || '').toUpperCase(),"
        "contentEditable: !!el.isContentEditable"
        "};"
        "})()"
    )
    try:
        result = eval_fn(js)
    except Exception as exc:  # noqa: BLE001
        logger.debug("probe_element_tag_at CDP eval raised: %s", exc)
        return None
    if not isinstance(result, dict):
        return None
    # Normalise shape so callers don't have to defend against missing keys.
    return {
        "tag": str(result.get("tag") or "").upper(),
        "contentEditable": bool(result.get("contentEditable")),
    }


# Tags whose clicked element legitimately receives keyboard input. The
# allow-list is intentionally narrow: anything outside it (BUTTON, A,
# DIV-as-backdrop, SVG icons) means the chosen coords are NOT on an
# input, so a click there dismisses the form rather than focusing it.
# ``LABEL`` stays in the list because clicking a ``<label for="x">``
# focuses the bound input — a legitimate fill_field flow.
INPUT_LIKE_TAGS: frozenset[str] = frozenset({"INPUT", "TEXTAREA", "LABEL"})


def is_input_like(tag_info: dict | None) -> bool:
    """True iff the probed element is something `fill_field` may type into.

    Returns ``True`` when ``tag_info`` is ``None`` (CDP unavailable — we
    can't refute, so don't block). Returns ``True`` for INPUT / TEXTAREA
    / LABEL tags, and for any element with ``isContentEditable`` (rich
    text editors). Everything else returns ``False`` and is the caller's
    cue to refuse the click and report ``form_target_not_found`` / a
    typed variant rather than dispatching a misdirected click that
    would dismiss the form modal.
    """
    if tag_info is None:
        return True
    if tag_info.get("contentEditable"):
        return True
    return tag_info.get("tag", "") in INPUT_LIKE_TAGS


def try_som_click(env: Any, x: int, y: int, policy: Any) -> bool:
    """Attempt a SoM-anchored click via CDP.

    Returns ``True`` iff:

    1. ``policy`` is non-None and has
       ``som_for_unstructured_clicks=True``.
    2. ``env`` exposes a callable ``cdp_click_at_point``.
    3. The CDP call returned ``True`` (an element existed at (x, y)
       and ``el.click()`` dispatched without raising).

    On any False return, the caller is expected to fall back to its
    legacy click pipeline (xdotool, Playwright mouse, …). The caller
    is responsible for tagging the resulting backend (``"som"`` vs
    ``"vision"``) — this helper only reports SoM success/failure.

    Why this isn't a context-managed wrapper: callers want to keep
    their existing ``env.step`` call sites unchanged. The cleanest
    integration is a single ``if try_som_click(...): backend = "som"``
    branch before the legacy click.
    """
    if policy is None:
        return False
    if not getattr(policy, "som_for_unstructured_clicks", False):
        return False
    handler = getattr(env, "cdp_click_at_point", None)
    if not callable(handler):
        return False
    try:
        return bool(handler(int(x), int(y)))
    except Exception as exc:
        # CDP call failures already log inside ``cdp_click_at_point``;
        # debug-level here so the caller's fallback path stays clean.
        logger.debug("SoM CDP click raised: %s", exc)
        return False


def try_dom_labeled_click(
    env: Any,
    *,
    label: str,
    aliases: list[str] | None = None,
    kind: str = "button",
) -> dict | None:
    """DOM-mode tier-0 click on a button whose visible text matches a
    plan-supplied label, dispatched via CDP ``el.click()`` before any
    vision-based grounding runs.

    Returns ``{"tag": str, "label": str, "rect": {x,y,w,h}}`` on success
    and ``None`` on miss / no CDP / wrong kind. Designed as a cheap
    deterministic tier-0 — the caller pre-checks the result and falls
    through to vision-based grounding when ``None``.

    Why this exists (#staff-crm-long step 8): vision grounding
    reliably misses small toolbar buttons. Apply / Update Lead /
    Login / Send / Save all have exact-text labels and matching
    DOM elements. When the plan author names the label, an exact
    DOM query is strictly more reliable than running Holo3 + Claude
    over a screenshot. Each saved miss is ~$0.05 + 30s and avoids
    the cascade where bad clicks navigate the agent to the wrong
    page (the "succeeded onto the wrong URL" failure mode).

    Scoped narrowly:

    * Only fires on submit-button-shaped kinds (``button`` / ``submit``
      / empty). Skips ``nav_link`` / ``row_link`` where multiple
      elements legitimately share the same label text.
    * Targets ``<button>`` / ``<input type="submit|button">`` /
      ``[role="button"]`` only. Anchors (``<a>``) excluded — too many
      label collisions on nav menus.
    * Filters for visible (``offsetParent != null``) + enabled
      (``!el.disabled``) elements; uses the first DOM-order match.
    * Trims + case-folds before comparing; alias matches are first
      tried in order.

    The helper does NOT settle / verify navigation — the caller owns
    the post-click URL-change check, debug screenshot dump, and
    no_state_change demotion. The helper only reports whether the
    click reached an exact-labeled element.
    """
    if not label:
        return None
    eval_fn = getattr(env, "cdp_evaluate", None)
    if not callable(eval_fn):
        return None
    kind_l = (kind or "").strip().lower()
    if kind_l and kind_l not in ("button", "submit"):
        return None
    candidates: list[str] = [label]
    for alias in (aliases or []):
        alias_s = str(alias).strip()
        if alias_s and alias_s not in candidates:
            candidates.append(alias_s)
    js_labels = "[" + ",".join(json.dumps(c) for c in candidates) + "]"
    js = (
        "(() => {"
        f"const labels = {js_labels}.map(s => s.trim().toLowerCase());"
        "const sel = 'button, input[type=\"submit\"], input[type=\"button\"], [role=\"button\"]';"
        "const elems = Array.from(document.querySelectorAll(sel));"
        "const visible = el => {"
        "  if (el.disabled) return false;"
        "  if (el.offsetParent === null) return false;"
        "  return true;"
        "};"
        "const textOf = el => {"
        "  const tag = (el.tagName || '').toLowerCase();"
        "  const raw = tag === 'input' ? (el.value || '')"
        "    : (el.innerText || el.textContent || '');"
        "  return raw.trim().toLowerCase();"
        "};"
        "let hit = null;"
        "for (const want of labels) {"
        "  for (const el of elems) {"
        "    if (!visible(el)) continue;"
        "    if (textOf(el) !== want) continue;"
        "    hit = el; break;"
        "  }"
        "  if (hit) break;"
        "}"
        "if (!hit) return null;"
        "const r = hit.getBoundingClientRect();"
        "try {"
        "  hit.click();"
        "  return {"
        "    tag: (hit.tagName || '').toLowerCase(),"
        "    label: textOf(hit),"
        "    rect: {x: r.left, y: r.top, w: r.width, h: r.height}"
        "  };"
        "} catch (e) { return null; }"
        "})()"
    )
    try:
        result = eval_fn(js)
    except Exception as exc:  # noqa: BLE001 — CDP path; never crash the runner
        logger.debug("try_dom_labeled_click CDP eval raised: %s", exc)
        return None
    if not isinstance(result, dict):
        return None
    rect = result.get("rect") or {}
    return {
        "tag": str(result.get("tag") or "").lower(),
        "label": str(result.get("label") or ""),
        "rect": {
            "x": float(rect.get("x") or 0.0),
            "y": float(rect.get("y") or 0.0),
            "w": float(rect.get("w") or 0.0),
            "h": float(rect.get("h") or 0.0),
        },
    }


__all__ = [
    "try_som_click",
    "try_dom_labeled_click",
    "probe_element_tag_at",
    "is_input_like",
    "INPUT_LIKE_TAGS",
]
