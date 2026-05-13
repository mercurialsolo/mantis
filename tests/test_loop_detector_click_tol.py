"""Tests for #296 — adaptive ``click_tol_px`` by screen DPI + element class.

The fixed ``click_tol_px = 8`` in ``LoopDetector`` is wrong on 4K (too
tight — legitimate retries near the same target get flagged as drift
loops) and on phone-class viewports (too loose — small drifts go
unflagged). Element semantics also matter: a "Submit" button covers
~150 × 50 px so 1.5× tolerance is fine; a navigation link is one word
wide so 0.5× catches drift onto the next link.

Coverage:
- ``compute_click_tol_px`` scaling table on common viewports.
- Element-class multipliers via reasoning text (``button``, ``link``,
  ``dropdown``, …).
- ``MANTIS_ADAPTIVE_CLICK_TOL=disabled`` falls back to the floor.
- ``GymRunner`` wires the env's ``screen_size`` into the detector.
"""

from __future__ import annotations

import pytest

from mantis_agent.actions import Action, ActionType
from mantis_agent.loop_detector import (
    LoopDetector,
    adaptive_click_tol_enabled,
    compute_click_tol_px,
)


# ── compute_click_tol_px ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "viewport, expected",
    [
        ((1280, 800), 8),    # default Mantis viewport — floor
        ((1366, 768), 8),    # laptop — floor
        ((1920, 1080), 9),   # FHD desktop
        ((2560, 1440), 12),  # QHD
        ((3840, 2160), 18),  # 4K
        ((7680, 4320), 35),  # 8K
    ],
)
def test_compute_click_tol_scales_with_diagonal(
    viewport: tuple[int, int],
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MANTIS_ADAPTIVE_CLICK_TOL", raising=False)
    assert compute_click_tol_px(viewport) == expected


def test_compute_click_tol_zero_dimensions_returns_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: a zeroed viewport (env not yet initialized) doesn't crash."""
    monkeypatch.delenv("MANTIS_ADAPTIVE_CLICK_TOL", raising=False)
    assert compute_click_tol_px((0, 0)) == 8
    assert compute_click_tol_px((1920, 0)) == 8


def test_compute_click_tol_disabled_returns_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANTIS_ADAPTIVE_CLICK_TOL", "disabled")
    assert compute_click_tol_px((3840, 2160)) == 8


def test_compute_click_tol_custom_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANTIS_ADAPTIVE_CLICK_TOL", raising=False)
    assert compute_click_tol_px((1280, 800), floor=12) == 12
    assert compute_click_tol_px((3840, 2160), floor=12) == 18


# ── element-class multipliers ────────────────────────────────────────


def _click(x: int, y: int, reasoning: str = "") -> Action:
    return Action(ActionType.CLICK, {"x": x, "y": y}, reasoning=reasoning)


def test_button_multiplier_widens_drift_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two clicks 11 px apart on a Submit button (1.5× → ~12 px tol)
    are NOT a drift loop; without the multiplier they would be."""
    monkeypatch.delenv("MANTIS_ADAPTIVE_CLICK_TOL", raising=False)
    d = LoopDetector(click_tol_px=8)  # base 8, button 1.5× → 12
    for _ in range(3):
        d.record(_click(100, 100, reasoning="Click the Submit button"))
        d.record(_click(111, 109, reasoning="Click the Submit button"))
    assert d.is_drift_loop(window=5) is True
    # Now try 13 px apart — outside the 12 px button tolerance.
    d2 = LoopDetector(click_tol_px=8)
    for _ in range(3):
        d2.record(_click(100, 100, reasoning="Click the Submit button"))
        d2.record(_click(113, 109, reasoning="Click the Submit button"))
    assert d2.is_drift_loop(window=5) is False


def test_link_multiplier_tightens_drift_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two clicks 5 px apart on a link (0.5× → 4 px tol) ARE a drift
    loop only if within the tightened tolerance."""
    monkeypatch.delenv("MANTIS_ADAPTIVE_CLICK_TOL", raising=False)
    d = LoopDetector(click_tol_px=8)
    # 5 px drift > 4 px link tolerance → NOT a drift loop
    for _ in range(3):
        d.record(_click(100, 100, reasoning="Click the link to next page"))
        d.record(_click(105, 100, reasoning="Click the link to next page"))
    assert d.is_drift_loop(window=5) is False
    # 3 px drift ≤ 4 px link tolerance → drift loop
    d2 = LoopDetector(click_tol_px=8)
    for _ in range(3):
        d2.record(_click(100, 100, reasoning="Click the link to next page"))
        d2.record(_click(103, 100, reasoning="Click the link to next page"))
    assert d2.is_drift_loop(window=5) is True


def test_no_classification_uses_base_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reasoning without a class keyword falls back to ``click_tol_px``."""
    monkeypatch.delenv("MANTIS_ADAPTIVE_CLICK_TOL", raising=False)
    d = LoopDetector(click_tol_px=8)
    for _ in range(3):
        d.record(_click(100, 100, reasoning="click somewhere"))
        d.record(_click(107, 107, reasoning="click somewhere"))
    assert d.is_drift_loop(window=5) is True
    d2 = LoopDetector(click_tol_px=8)
    for _ in range(3):
        d2.record(_click(100, 100, reasoning="click somewhere"))
        d2.record(_click(109, 109, reasoning="click somewhere"))
    assert d2.is_drift_loop(window=5) is False


def test_class_multiplier_disabled_falls_back_to_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the toggle is off, button text doesn't widen tolerance."""
    monkeypatch.setenv("MANTIS_ADAPTIVE_CLICK_TOL", "disabled")
    d = LoopDetector(click_tol_px=8)
    # 10 px drift would fit a 12 px button tolerance, but toggle is off.
    for _ in range(3):
        d.record(_click(100, 100, reasoning="Click the Submit button"))
        d.record(_click(110, 100, reasoning="Click the Submit button"))
    # 10 > 8 → not a drift loop with raw 8 px tolerance.
    assert d.is_drift_loop(window=5) is False


def test_compound_keyword_matched_before_substring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The keyword tuple is ordered so multi-word phrases match first.
    ``submit button`` and the bare ``button`` both map to 1.5× today,
    but the ordering protects against future divergence (different
    multipliers for ``submit button`` vs generic ``button``).
    """
    monkeypatch.delenv("MANTIS_ADAPTIVE_CLICK_TOL", raising=False)
    d = LoopDetector(click_tol_px=8)
    # Both "submit button" and bare "button" → 1.5× → 12 px tolerance.
    for _ in range(3):
        d.record(_click(100, 100, reasoning="Click the submit button"))
        d.record(_click(111, 100, reasoning="Click the submit button"))
    assert d.is_drift_loop(window=5) is True


# ── env-var toggle ───────────────────────────────────────────────────


def test_adaptive_click_tol_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANTIS_ADAPTIVE_CLICK_TOL", raising=False)
    assert adaptive_click_tol_enabled() is True


@pytest.mark.parametrize(
    "value", ["disabled", "0", "false", "FALSE", "off", "no"]
)
def test_adaptive_click_tol_disabled_when_off(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANTIS_ADAPTIVE_CLICK_TOL", value)
    assert adaptive_click_tol_enabled() is False


# ── GymRunner wiring ─────────────────────────────────────────────────


def test_gym_runner_uses_env_screen_size_for_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GymRunner`` should construct its detector with a tolerance
    derived from ``env.screen_size``, not the hardcoded 8."""
    monkeypatch.delenv("MANTIS_ADAPTIVE_CLICK_TOL", raising=False)

    from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
    from mantis_agent.gym.runner import GymRunner
    from PIL import Image

    class _Env(GymEnvironment):
        @property
        def screen_size(self) -> tuple[int, int]:
            return (3840, 2160)  # 4K

        def reset(self, task: str, **kw):
            return GymObservation(screenshot=Image.new("RGB", (3840, 2160)))

        def step(self, action):
            return GymResult(self.reset(""), 0.0, False, {})

        def close(self) -> None:
            pass

    class _Brain:
        def think(self, frames, task, action_history=None, screen_size=(3840, 2160)):
            class R:
                action = Action(ActionType.DONE, {"success": True})
                thinking = ""
                predicted_outcome = ""
            return R()

    runner = GymRunner(_Brain(), _Env(), max_steps=1)
    assert runner._loop_detector.click_tol_px == 18  # 4K diagonal scale


def test_gym_runner_handles_env_without_screen_size_attribute() -> None:
    """Defensive: an env that raises on ``screen_size`` falls back to 8."""
    from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
    from mantis_agent.gym.runner import GymRunner
    from PIL import Image

    class _BadEnv(GymEnvironment):
        @property
        def screen_size(self):
            raise RuntimeError("not initialized yet")

        def reset(self, task: str, **kw):
            return GymObservation(screenshot=Image.new("RGB", (100, 100)))

        def step(self, action):
            return GymResult(self.reset(""), 0.0, False, {})

        def close(self) -> None:
            pass

    class _Brain:
        def think(self, frames, task, action_history=None, screen_size=(100, 100)):
            class R:
                action = Action(ActionType.DONE, {"success": True})
                thinking = ""
                predicted_outcome = ""
            return R()

    runner = GymRunner(_Brain(), _BadEnv(), max_steps=1)
    assert runner._loop_detector.click_tol_px == 8
