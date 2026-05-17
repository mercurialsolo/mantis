"""Click-dispatch trust-gate-bypass primitive.

Originally lived inside :class:`~.step_handlers.form.ClaudeGuidedFormHandler`
as PR #447 Fix B (`submit`-only). Extracted to a reusable helper so the
generic click handler and any future step handler can apply the same
post-dispatch URL gate.

The pattern: when a synthetic CDP ``el.click()`` returns ``ok=True``
but the page doesn't navigate, the click event likely had
``isTrusted=false`` and the SPA's framework rejected it silently. Retry
once with CDP ``Input.dispatchMouseEvent`` which produces a
protocol-level click that's indistinguishable from a real mouse click
(``isTrusted=true``).

Gate tightening (PR #447 follow-up): the URL check compares the FINAL
settled URL — re-polled after a stabilization window — not the
immediately-polled URL. The latter race-loses when the synthetic click
triggers a brief navigation that bounces back: settle returns early on
the intermediate URL, the immediate poll captures the (different)
intermediate, and the gate doesn't fire. The stabilization window
makes the gate read the true post-click steady-state URL.

Surfaced as a generic primitive in PR-G (run ``74add5d8``, 2026-05-17):
staff-crm-long step 8 (`type: click` row-link) halted with the same
trust-gated pattern as step 6 (`type: submit` sidebar anchor). The
form handler's inline Fix B logic was the right tool but lived in the
wrong handler.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .micro_runner import MicroPlanRunner

logger = logging.getLogger(__name__)


_STABILIZATION_WINDOW_SECONDS = 0.8


def pointer_retry_if_unchanged(
    env: Any,
    runner: "MicroPlanRunner",
    x: int,
    y: int,
    *,
    url_before: str,
    executor_backend: str,
    log_prefix: str = "[click]",
) -> str:
    """After a click + initial settle, re-poll URL and retry via
    ``Input.dispatchMouseEvent`` if the URL hasn't changed.

    Returns the final ``url_after_click`` — either the post-retry URL
    (if retry fired and changed things) or the post-stabilization URL
    (if retry was skipped or didn't change anything). Callers should
    update their step-result logic with this value rather than reading
    URL again themselves.

    The retry only fires when ALL of these hold:

    1. ``url_before`` is non-empty (otherwise we have no baseline)
    2. The FINAL settled URL equals ``url_before`` (click didn't navigate)
    3. ``executor_backend == "som"`` (the SoM path was used; trust-gated
       is the canonical SoM-path symptom; xdotool's already-real clicks
       don't need this retry)
    4. ``env`` exposes ``cdp_click_via_pointer`` (capability check)

    Any of these conditions failing → return the stabilization-window
    URL unchanged. Errors during retry dispatch (CDP unreachable,
    method missing, page closed) are caught and logged at DEBUG; the
    caller's existing fallback path (Enter key, scroll-and-rescan, etc.)
    handles the leftover failure.
    """
    url_after_click = _safe_url_read(runner)
    time.sleep(_STABILIZATION_WINDOW_SECONDS)
    final_url = _safe_url_read(runner)

    # A blank final_url means the runner couldn't read URL on the
    # second poll — treat as missing info rather than a navigation.
    # Otherwise a transient read-fail would mask a genuine no-navigation
    # case (gate fires on bogus inequality) or hide a genuine
    # navigation (gate misfires the retry).
    if final_url and final_url != url_after_click:
        logger.warning(
            "  %s post-click URL bounce detected: settle saw %r, "
            "final is %r — pointer-retry evaluates against final",
            log_prefix,
            (url_after_click or "")[:80],
            final_url[:80],
        )
        url_after_click = final_url

    if not (
        url_before
        and url_after_click == url_before
        and executor_backend == "som"
        and hasattr(env, "cdp_click_via_pointer")
    ):
        return url_after_click

    logger.warning(
        "  %s SoM click ok=True but URL stable — retrying via CDP "
        "Input.dispatchMouseEvent (real pointer events, isTrusted=true)",
        log_prefix,
    )
    try:
        ok = env.cdp_click_via_pointer(x, y)
    except Exception as exc:  # noqa: BLE001
        logger.debug("pointer-retry dispatch raised: %s", exc)
        ok = False
    if not ok:
        return url_after_click

    # Re-settle and re-read; preserve the original cost-accounting
    # pattern (one extra settle billed, one extra gpu_step billed).
    settle_seconds = _safe_adaptive_settle(runner, url_before=url_before)
    runner.costs["gpu_seconds"] += settle_seconds
    runner.costs["gpu_steps"] += 1
    time.sleep(_STABILIZATION_WINDOW_SECONDS)
    return _safe_url_read(runner)


def _safe_url_read(runner: "MicroPlanRunner") -> str:
    """Read current URL via ``_best_effort_current_url`` if available,
    falling back to ``""`` on any error.
    """
    reader = getattr(runner, "_best_effort_current_url", None)
    if not callable(reader):
        return ""
    try:
        return str(reader() or "")
    except Exception:  # noqa: BLE001
        return ""


def _safe_adaptive_settle(runner: "MicroPlanRunner", *, url_before: str) -> float:
    """Call ``_adaptive_submit_settle`` if available; return 0.0 otherwise.

    Both ``MicroPlanRunner`` and ``GymRunner`` (legacy) expose this
    method; tests with FakeRunners may stub it. Defensive default
    avoids attribute errors in unusual harness configurations.
    """
    settle = getattr(runner, "_adaptive_submit_settle", None)
    if not callable(settle):
        return 0.0
    try:
        return float(settle(url_before=url_before) or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


__all__ = ["pointer_retry_if_unchanged"]
