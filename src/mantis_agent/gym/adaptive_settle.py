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
        return time.monotonic() - start
    try:
        page.wait_for_load_state(
            "networkidle", timeout=int(max_seconds * 1000),
        )
    except Exception as exc:
        logger.debug("adaptive-settle: networkidle wait raised %s", exc)
        remaining = max_seconds - (time.monotonic() - start)
        if remaining > 0:
            time.sleep(remaining)
    return time.monotonic() - start
