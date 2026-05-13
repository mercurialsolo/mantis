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

import logging
from typing import Any

logger = logging.getLogger(__name__)


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


__all__ = ["try_som_click"]
