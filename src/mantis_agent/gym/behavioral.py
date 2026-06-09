"""Humanlike behavioral signals — pre-click curve + jittered settle (#824).

Modern bot-detection scoring (DataDome, PerimeterX, Cloudflare Turnstile
when escalation is close) reads behavioral telemetry: mouse trajectory,
click cadence, inter-event timing. Today every ``xdotool`` click
teleports the cursor and fires a click in immediate succession — a
machine-perfect tell that drives the score upward.

This module ships two small primitives:

1. :func:`bezier_waypoints` — Bezier curve between current cursor and
   target. Returns 2–4 intermediate points so the click handler can
   issue ``xdotool mousemove`` at each waypoint with a tiny delay.
2. :func:`jittered_settle` — turn a fixed settle time into a
   ``base + uniform(-0.3, +0.6)`` range so step-to-step cadence isn't
   machine-perfect.

Both honor the ``MANTIS_BEHAVIORAL_JITTER`` env var (default ``"1"``;
set to ``"0"`` to restore deterministic behavior for CI / replay).

CUA provenance: these are **action-only** modifications of how we
*emit* xdotool / sleep commands. They don't derive grounding from
DOM state or otherwise breach screenshot-grounded purity (see
``feedback_cua_no_dom_access.md``).
"""

from __future__ import annotations

import os
import random


def is_enabled() -> bool:
    """Whether behavioral jitter is active.

    Default ``True`` — opt out with ``MANTIS_BEHAVIORAL_JITTER=0`` (or
    ``false`` / ``no``). Reads env on every call so tests can flip the
    flag without reloading the module.
    """
    raw = os.environ.get("MANTIS_BEHAVIORAL_JITTER", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def bezier_waypoints(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    steps: int = 3,
    curvature: float = 0.25,
) -> list[tuple[int, int]]:
    """Sample ``steps`` waypoints along a cubic Bezier from ``start`` to ``end``.

    Returns ONLY the intermediate waypoints — neither endpoint is
    included, so the caller picks whether to start from the current
    cursor position and where to land. Empty list when ``start == end``
    or when ``steps <= 0``.

    The curve's two control points are offset perpendicular to the
    direct line by ``curvature * distance``, with sign and magnitude
    randomized so successive calls don't trace the same arc.

    ``steps`` is clamped to ``[0, 8]`` so a runaway caller can't issue
    50 ``xdotool`` calls per click.
    """
    sx, sy = start
    ex, ey = end
    if (sx, sy) == (ex, ey) or steps <= 0:
        return []
    steps = max(0, min(int(steps), 8))
    if steps == 0:
        return []

    # Perpendicular unit vector to the start→end direction.
    dx, dy = ex - sx, ey - sy
    distance = max(1.0, (dx * dx + dy * dy) ** 0.5)
    px, py = -dy / distance, dx / distance

    # Two control points 1/3 and 2/3 along the line + a perpendicular
    # offset. Offset magnitude jittered so the path differs between calls.
    jitter_a = random.uniform(-curvature, curvature) * distance
    jitter_b = random.uniform(-curvature, curvature) * distance
    cx1, cy1 = sx + dx / 3 + px * jitter_a, sy + dy / 3 + py * jitter_a
    cx2, cy2 = sx + 2 * dx / 3 + px * jitter_b, sy + 2 * dy / 3 + py * jitter_b

    points: list[tuple[int, int]] = []
    # t in (0, 1), excluding endpoints — caller already knows where it
    # started and where it's landing.
    for i in range(1, steps + 1):
        t = i / (steps + 1)
        mt = 1 - t
        # Cubic Bezier interpolation.
        x = (
            mt * mt * mt * sx
            + 3 * mt * mt * t * cx1
            + 3 * mt * t * t * cx2
            + t * t * t * ex
        )
        y = (
            mt * mt * mt * sy
            + 3 * mt * mt * t * cy1
            + 3 * mt * t * t * cy2
            + t * t * t * ey
        )
        points.append((int(round(x)), int(round(y))))
    return points


def waypoint_delay() -> float:
    """Inter-waypoint delay for the pre-click mouse curve.

    Returns ``uniform(0.005, 0.012)`` seconds — short enough that the
    full curve adds ≤ 50 ms to a click, long enough to register as
    multi-frame movement in mousemove timestamps.
    """
    return random.uniform(0.005, 0.012)


def jittered_settle(base: float) -> float:
    """Add ``uniform(-0.3, +0.6)`` jitter to a settle time.

    Real users don't navigate / scroll / click on a machine-perfect
    cadence. Returns the original ``base`` unchanged when jitter is
    disabled (``MANTIS_BEHAVIORAL_JITTER=0``).

    Negative jitter caps at ``base * 0.5`` so a small base value
    (e.g. 0.5 s) never collapses to near-zero.
    """
    if not is_enabled():
        return base
    if base <= 0:
        return base
    delta = random.uniform(-0.3, 0.6)
    floor = base * 0.5
    return max(floor, base + delta)


def jittered_wait(base: float, *, spread: float = 1.5) -> float:
    """Same as :func:`jittered_settle` but for longer waits.

    Used for ``wait_after_load_seconds`` on navigate steps where the
    natural variation is larger than per-step settle. Returns
    ``uniform(base - 0.5, base + spread)`` (clamped to ``base * 0.5``).
    """
    if not is_enabled():
        return base
    if base <= 0:
        return base
    delta = random.uniform(-0.5, spread)
    floor = base * 0.5
    return max(floor, base + delta)


__all__ = [
    "bezier_waypoints",
    "is_enabled",
    "jittered_settle",
    "jittered_wait",
    "waypoint_delay",
]
