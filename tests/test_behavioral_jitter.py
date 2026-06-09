"""Tests for the humanlike behavioral signals module (#824).

Two primitives:

- ``bezier_waypoints(start, end, steps)`` — Bezier path between cursor
  and click target. Returns intermediate waypoints only.
- ``jittered_settle(base)`` / ``jittered_wait(base)`` — randomized
  delays so step-to-step cadence isn't machine-perfect.

Both honor ``MANTIS_BEHAVIORAL_JITTER`` (default ``"1"``).
"""

from __future__ import annotations

import math
import random

import pytest

from mantis_agent.gym.behavioral import (
    bezier_waypoints,
    is_enabled,
    jittered_settle,
    jittered_wait,
    waypoint_delay,
)


# ── is_enabled / env gating ────────────────────────────────────────


def test_is_enabled_defaults_true(monkeypatch):
    monkeypatch.delenv("MANTIS_BEHAVIORAL_JITTER", raising=False)
    assert is_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", "Off"])
def test_is_enabled_falsy_values(monkeypatch, value):
    monkeypatch.setenv("MANTIS_BEHAVIORAL_JITTER", value)
    assert is_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "anything"])
def test_is_enabled_truthy_values(monkeypatch, value):
    """Anything other than the explicit falsy set is truthy — opt-in
    behavior is default-on, so unknown values lean toward enabled."""
    monkeypatch.setenv("MANTIS_BEHAVIORAL_JITTER", value)
    assert is_enabled() is True


# ── bezier_waypoints ───────────────────────────────────────────────


def test_bezier_waypoints_returns_requested_count():
    random.seed(0)
    pts = bezier_waypoints((100, 100), (500, 300), steps=3)
    assert len(pts) == 3


def test_bezier_waypoints_excludes_endpoints():
    """The returned list never contains the start or end — caller
    already knows where the cursor is and where it's landing."""
    random.seed(0)
    pts = bezier_waypoints((100, 100), (500, 300), steps=4)
    assert (100, 100) not in pts
    assert (500, 300) not in pts


def test_bezier_waypoints_empty_when_start_equals_end():
    """Zero-distance move → no intermediate path."""
    assert bezier_waypoints((50, 50), (50, 50), steps=3) == []


def test_bezier_waypoints_empty_when_steps_zero():
    assert bezier_waypoints((100, 100), (500, 300), steps=0) == []


def test_bezier_waypoints_clamps_runaway_step_counts():
    """Defensive: 50 steps would issue 50 xdotool calls per click."""
    pts = bezier_waypoints((100, 100), (500, 300), steps=50)
    assert len(pts) <= 8


def test_bezier_waypoints_progress_roughly_monotonic():
    """The Bezier curve adds perpendicular jitter, but along the
    primary axis (start→end direction) waypoints should advance toward
    the target. With curvature=0 the path becomes a straight line."""
    pts = bezier_waypoints((0, 0), (1000, 0), steps=4, curvature=0)
    xs = [p[0] for p in pts]
    assert xs == sorted(xs)  # monotonically increasing along x-axis
    assert all(0 < x < 1000 for x in xs)


def test_bezier_waypoints_path_actually_curves():
    """With curvature > 0, the path should not be a straight line —
    at least one waypoint must deviate perpendicularly from the
    straight start→end line."""
    random.seed(0)
    pts = bezier_waypoints((0, 0), (1000, 0), steps=4, curvature=0.25)
    # Straight line at y=0; perpendicular deviation is just |y|.
    max_dev = max(abs(y) for _, y in pts)
    assert max_dev > 5, f"path should curve, max y-deviation was {max_dev}"


def test_bezier_waypoints_jitters_between_calls():
    """Successive calls should NOT trace the same arc (the perpendicular
    offset is randomized). This guards against a constant-seed regression
    sneaking back in."""
    random.seed(0)
    a = bezier_waypoints((0, 0), (1000, 1000), steps=4)
    b = bezier_waypoints((0, 0), (1000, 1000), steps=4)
    # The two arcs will differ in at least one waypoint.
    assert a != b


# ── waypoint_delay ─────────────────────────────────────────────────


def test_waypoint_delay_in_human_range():
    """Inter-waypoint delay should land in the documented range."""
    for _ in range(50):
        d = waypoint_delay()
        assert 0.005 <= d <= 0.012


# ── jittered_settle ────────────────────────────────────────────────


def test_jittered_settle_returns_base_when_disabled(monkeypatch):
    monkeypatch.setenv("MANTIS_BEHAVIORAL_JITTER", "0")
    assert jittered_settle(2.0) == 2.0


def test_jittered_settle_returns_base_for_zero(monkeypatch):
    monkeypatch.delenv("MANTIS_BEHAVIORAL_JITTER", raising=False)
    assert jittered_settle(0.0) == 0.0


def test_jittered_settle_in_documented_range(monkeypatch):
    """Documented: ``uniform(-0.3, +0.6)`` around base, floor at base/2."""
    monkeypatch.delenv("MANTIS_BEHAVIORAL_JITTER", raising=False)
    base = 2.0
    samples = [jittered_settle(base) for _ in range(200)]
    assert min(samples) >= base * 0.5 - 1e-6
    assert max(samples) <= base + 0.6 + 1e-6
    # Mean should be near base + 0.15 (midpoint of (-0.3, +0.6)).
    assert math.isclose(sum(samples) / len(samples), base + 0.15, abs_tol=0.15)


def test_jittered_settle_floor_holds_for_tiny_base(monkeypatch):
    """Negative jitter capped so a 0.5 s settle never collapses to 0.2 s."""
    monkeypatch.delenv("MANTIS_BEHAVIORAL_JITTER", raising=False)
    base = 0.5
    for _ in range(100):
        assert jittered_settle(base) >= base * 0.5


# ── jittered_wait ──────────────────────────────────────────────────


def test_jittered_wait_returns_base_when_disabled(monkeypatch):
    monkeypatch.setenv("MANTIS_BEHAVIORAL_JITTER", "0")
    assert jittered_wait(6.0) == 6.0


def test_jittered_wait_in_documented_range(monkeypatch):
    """Default spread is 1.5 — uniform(-0.5, +1.5)."""
    monkeypatch.delenv("MANTIS_BEHAVIORAL_JITTER", raising=False)
    base = 6.0
    samples = [jittered_wait(base) for _ in range(200)]
    assert min(samples) >= base * 0.5 - 1e-6
    assert max(samples) <= base + 1.5 + 1e-6
