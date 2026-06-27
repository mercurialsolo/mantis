"""Frame-stability and network-idle gating to replace fixed ``time.sleep``
calls after actions (#294).

The runner and ``PlanExecutor`` previously paid the worst-case ``settle_time``
on every step even when the DOM settled in 50 ms; conversely, network-heavy
submits sometimes needed >2 s and silently capped. This module replaces
fixed sleeps with two adaptive primitives:

* :func:`wait_until_stable` — poll a screenshot supplier every
  ``poll_interval`` seconds, return when ``require_consecutive`` captures
  have the same :func:`phash_64`. Cap at ``max_seconds`` (the old fixed
  budget). Used by the xdotool path which has no DOM signal.

* :func:`wait_for_networkidle` — wraps Playwright's
  ``page.wait_for_load_state("networkidle", ...)`` with the same cap.
  Used by the CDP/Playwright path where the browser exposes network state
  directly.

Both functions return the seconds actually waited so callers can record
the savings — useful for the ``#261`` ablation discipline (we don't claim a
speedup we haven't measured).

The ablation toggle ``MANTIS_ADAPTIVE_SETTLE=disabled`` short-circuits both
functions back to a plain fixed sleep — flip it to compare wall time on the
same workload without a redeploy that switches the code path.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

from PIL import Image

from ..loop_detector import phash_64

logger = logging.getLogger(__name__)


_ENV_TOGGLE: str = "MANTIS_ADAPTIVE_SETTLE"


def is_enabled() -> bool:
    """``MANTIS_ADAPTIVE_SETTLE=disabled`` reverts to fixed sleeps."""
    return os.environ.get(_ENV_TOGGLE, "enabled").lower() != "disabled"


# #561: per-run ceiling applied to every ``settle_after_action`` call.
# Set via ``set_runtime_ceiling`` by ``MicroPlanRunner.__init__`` when
# the suite carries a ``settle_ceiling_seconds`` runtime field.
# ``None`` (default) preserves each call site's existing ``max_seconds``
# argument — no clamp. When set, every settle is clamped to
# ``min(call_site_max, ceiling)``.
#
# Module-level state is safe in this codebase: Modal executors are
# single-process per run; only one ``MicroPlanRunner`` instance is
# active at a time per process. Tests reset via ``set_runtime_ceiling(None)``.
_runtime_ceiling: float | None = None


def set_runtime_ceiling(seconds: float | None) -> None:
    """Set the per-run global ceiling for ``settle_after_action`` calls.

    Pass ``None`` to clear (every call uses its own ``max_seconds``).
    Pass a positive float to clamp every settle to at most that many
    seconds — typical pattern is the runner setting this at init from
    the suite's ``_settle_ceiling_seconds`` field.

    Non-positive values are treated as ``None`` (clear) — a ceiling
    of 0 would brick every settle, never intended.
    """
    global _runtime_ceiling
    if seconds is None or seconds <= 0:
        _runtime_ceiling = None
    else:
        _runtime_ceiling = float(seconds)


def get_runtime_ceiling() -> float | None:
    """Current ceiling, or ``None`` when unset."""
    return _runtime_ceiling


def _apply_ceiling(max_seconds: float) -> float:
    """Clamp ``max_seconds`` to ``_runtime_ceiling`` if set."""
    if _runtime_ceiling is None:
        return max_seconds
    return min(max_seconds, _runtime_ceiling)


def wait_until_stable(
    capture: Callable[[], "Image.Image | None"],
    *,
    max_seconds: float,
    poll_interval: float = 0.1,
    min_seconds: float = 0.0,
    require_consecutive: int = 2,
    sleep_fn: Callable[[float], None] = time.sleep,
    time_fn: Callable[[], float] = time.monotonic,
) -> float:
    """Poll ``capture()`` until the frame hash has stayed identical for
    ``require_consecutive`` consecutive captures, or ``max_seconds`` elapses.

    Returns the actual seconds waited.

    A capture returning ``None`` (or raising) is treated as "no signal yet"
    — the poll continues; stability is never declared on a missing frame.

    All clock and capture callables are injected so tests can drive the
    function deterministically. Production callers pass ``time.sleep`` and
    ``time.monotonic`` (the defaults).
    """
    if max_seconds <= 0:
        return 0.0

    start = time_fn()
    deadline = start + max_seconds

    if min_seconds > 0:
        sleep_fn(min(min_seconds, max_seconds))

    last_hash: str | None = None
    consecutive_matches: int = 0

    while time_fn() < deadline:
        try:
            frame = capture()
        except Exception as exc:
            logger.debug("adaptive-settle: capture raised %s", exc)
            frame = None

        if frame is None:
            sleep_fn(poll_interval)
            continue

        try:
            h = phash_64(frame)
        except Exception as exc:
            logger.debug("adaptive-settle: phash raised %s", exc)
            sleep_fn(poll_interval)
            continue

        if last_hash is not None and h == last_hash:
            consecutive_matches += 1
            if consecutive_matches + 1 >= require_consecutive:
                return time_fn() - start
        else:
            consecutive_matches = 0
        last_hash = h
        sleep_fn(poll_interval)

    return time_fn() - start


def _content_score(img: "Image.Image | None") -> int:
    """Theme-agnostic "how much is painted on screen" proxy.

    Counts strong edges (adjacent-pixel transitions) on a 48×48 grayscale
    downsample. A blank / skeleton frame has near-zero edges; a fully
    rendered form/page has hundreds. Edge count — not near-white-pixel
    count — so it works on dark-themed UIs too (a dark page still has
    edges where content is). Cost is fixed (~4.6k comparisons) regardless
    of viewport size.
    """
    if img is None:
        return 0
    try:
        small = img.convert("L").resize((48, 48), Image.NEAREST)
    except Exception:  # noqa: BLE001 — never fatal
        return 0
    px = list(small.getdata())
    if not px:
        return 0
    w = h = 48
    delta = 24  # grayscale step that counts as an "edge"
    edges = 0
    for y in range(h):
        row = y * w
        for x in range(w):
            p = px[row + x]
            if x + 1 < w and abs(p - px[row + x + 1]) > delta:
                edges += 1
            if y + 1 < h and abs(p - px[row + w + x]) > delta:
                edges += 1
    return edges


def wait_for_content_stable(
    capture: Callable[[], "Image.Image | None"],
    *,
    max_seconds: float,
    poll_interval: float = 0.25,
    growth_tolerance: float = 0.06,
    require_stable: int = 2,
    sleep_fn: Callable[[float], None] = time.sleep,
    time_fn: Callable[[], float] = time.monotonic,
) -> "tuple[Image.Image | None, float]":
    """Poll ``capture()`` until the page stops painting NEW content, or
    ``max_seconds`` elapses. Returns ``(last_frame, seconds_waited)``.

    Where :func:`wait_until_stable` waits for the frame to stop *changing*
    (it returns early on a momentarily-stable skeleton, the cold-SPA-mount
    trap), this waits for the visible-content metric to stop *growing*. A
    frame whose :func:`_content_score` doesn't rise beyond ``growth_tolerance``
    (relative to the running peak) for ``require_stable`` consecutive polls
    is "settled".

    This is the right signal because the only thing distinguishing a sparse
    page that's DONE (e.g. example.com) from a sparse page that's still
    LOADING (a LinkedIn skeleton) is temporal — content is still arriving in
    the second case. So:

    * fast / already-rendered pages return after ~1-2 polls (content isn't
      growing) — minimal added latency,
    * slow SPAs wait exactly until the form paints and the score plateaus —
      no per-site constant needed,
    * perpetual-motion pages (carousels, spinners) plateau once their peak
      stops climbing, and the ``max_seconds`` cap bounds the pathological
      case.

    Screenshot-only — no DOM/network signal — so it works on the production
    xdotool path and stays CUA-compliant (load-readiness observation, not
    target derivation). Clock + capture are injected for deterministic tests.
    """
    start = time_fn()
    frame = _safe_capture(capture)
    if max_seconds <= 0:
        return frame, 0.0
    deadline = start + max_seconds
    peak = _content_score(frame)
    stable = 0
    while time_fn() < deadline:
        sleep_fn(poll_interval)
        f = _safe_capture(capture)
        if f is None:
            continue
        frame = f
        score = _content_score(f)
        if score > peak * (1.0 + growth_tolerance):
            # Still painting new content — reset the plateau counter.
            stable = 0
        else:
            stable += 1
            if stable >= require_stable:
                elapsed = time_fn() - start
                _credit_settle(elapsed)
                return frame, elapsed
        if score > peak:
            peak = score
    elapsed = time_fn() - start
    _credit_settle(elapsed)
    return frame, elapsed


def _safe_capture(capture: Callable[[], "Image.Image | None"]) -> "Image.Image | None":
    """Call ``capture()``; return ``None`` on any error (never fatal)."""
    try:
        return capture()
    except Exception as exc:  # noqa: BLE001
        logger.debug("adaptive-settle: capture raised %s", exc)
        return None


def settle_after_action(
    env: Any,
    *,
    max_seconds: float,
    poll_interval: float = 0.1,
) -> float:
    """Step-handler shorthand: ``wait_until_stable`` on the env's screenshot.

    Replaces scattered ``time.sleep(N)`` calls in MicroPlanRunner step
    handlers (``filter.py``, ``form.py``, ``paginate.py``, ``click.py``,
    ``navigate.py``) that wait for the page to repaint after a browser
    action. Step handlers don't have a Playwright page (they go through
    :class:`XdotoolGymEnv` for action dispatch), so the screenshot-based
    gate is the right fit.

    Falls back to a plain fixed sleep when:

    * ``MANTIS_ADAPTIVE_SETTLE=disabled`` (ablation toggle), or
    * the env doesn't expose a callable ``_screenshot`` / ``screenshot``
      attribute — defensive against alternative env adapters that may
      lack the perceptual signal we need.

    Returns seconds actually waited.
    """
    # #561: apply the per-run ceiling BEFORE the early-exit and fixed-
    # sleep branches so all paths honor it consistently.
    effective_max = _apply_ceiling(max_seconds)
    if not is_enabled() or effective_max <= 0:
        if effective_max > 0:
            time.sleep(effective_max)
        elapsed = effective_max if effective_max > 0 else 0.0
        _credit_settle(elapsed)
        return elapsed
    capture = getattr(env, "_screenshot", None) or getattr(env, "screenshot", None)
    if not callable(capture):
        time.sleep(effective_max)
        _credit_settle(effective_max)
        return effective_max
    elapsed = wait_until_stable(
        capture, max_seconds=effective_max, poll_interval=poll_interval,
    )
    _credit_settle(elapsed)
    return elapsed


def wait_for_networkidle(
    page: Any,
    *,
    max_seconds: float,
) -> float:
    """CDP/Playwright-backed settle.

    Wraps ``page.wait_for_load_state("networkidle", timeout=...)``. Returns
    seconds waited (capped at ``max_seconds``). On any error — page closed,
    page navigating, playwright unavailable — falls back to a plain sleep
    of ``max_seconds`` rather than skipping the settle entirely.
    """
    if max_seconds <= 0:
        return 0.0
    start = time.monotonic()
    if page is None:
        time.sleep(max_seconds)
        elapsed = time.monotonic() - start
        _credit_settle(elapsed)
        return elapsed
    try:
        page.wait_for_load_state(
            "networkidle", timeout=int(max_seconds * 1000),
        )
    except Exception as exc:
        logger.debug("adaptive-settle: networkidle wait raised %s", exc)
        remaining = max_seconds - (time.monotonic() - start)
        if remaining > 0:
            time.sleep(remaining)
    elapsed = time.monotonic() - start
    _credit_settle(elapsed)
    return elapsed


def _credit_settle(elapsed: float) -> None:
    """Best-effort: credit the ``settle`` bucket on the runner's
    TimeMeter (epic #362). No-op when no dispatch context is published
    or the import fails — bookkeeping must not break the runtime.
    """
    if elapsed <= 0:
        return
    try:
        from . import time_meter as _tm
        _tm.record_to_current("settle", elapsed)
    except Exception as exc:  # noqa: BLE001 — observability, never fatal
        logger.debug("adaptive-settle: time_meter credit failed: %s", exc)
