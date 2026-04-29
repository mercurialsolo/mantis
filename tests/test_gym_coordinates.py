"""Coordinate-space invariants for GymEnvironment dispatch.

Pins the contract documented in docs/reference/coordinate-spaces.md so the
1.5x click-offset class of bug (closed #25) cannot regress.
"""

from __future__ import annotations

import pytest

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.xdotool_env import XdotoolGymEnv, scale_brain_to_display


# ── scale_brain_to_display: pure helper ────────────────────────────────────


def test_scale_identity_when_sizes_match():
    assert scale_brain_to_display(640, 360, (1280, 720), (1280, 720)) == (640, 360)
    assert scale_brain_to_display(0, 0, (1280, 720), (1280, 720)) == (0, 0)
    assert scale_brain_to_display(1279, 719, (1280, 720), (1280, 720)) == (1279, 719)


def test_scale_uniform_upscale_brain_smaller_than_display():
    # Brain saw 768x432; display is 1280x720 (5/3 scale on both axes).
    x, y = scale_brain_to_display(640, 360, (768, 432), (1280, 720))
    # 640 * 1280/768 ≈ 1066.667 → 1067 ; 360 * 720/432 = 600
    assert (x, y) == (1067, 600)


def test_scale_uniform_downscale_brain_larger_than_display():
    # 1920x1080 brain on 1280x720 display (2/3 scale).
    x, y = scale_brain_to_display(960, 540, (1920, 1080), (1280, 720))
    assert (x, y) == (640, 360)


def test_scale_asymmetric_aspect_ratio():
    # Letterboxed brain image (1280x720) on a 1280x800 display.
    # X scale 1.0; Y scale 800/720 = 1.111…
    x, y = scale_brain_to_display(640, 360, (1280, 720), (1280, 800))
    assert x == 640
    # 360 * 800 / 720 = 400 exactly
    assert y == 400


def test_scale_rejects_zero_or_negative_sizes():
    with pytest.raises(ValueError):
        scale_brain_to_display(0, 0, (0, 720), (1280, 720))
    with pytest.raises(ValueError):
        scale_brain_to_display(0, 0, (1280, 720), (-1, 720))


# ── XdotoolGymEnv: passthrough invariant ────────────────────────────────────


class _XdotoolRecorder(XdotoolGymEnv):
    """XdotoolGymEnv with the xdotool subprocess shimmed out.

    Records every call so the test can assert on the dispatched display-space
    coordinates without needing a real Xvfb instance.
    """

    def __init__(self, viewport: tuple[int, int]):
        super().__init__(viewport=viewport, display=":dummy")
        self.calls: list[tuple[str, ...]] = []

    def _xdotool(self, *args: str) -> None:
        self.calls.append(tuple(args))

    def _xdotool_type(self, text: str) -> None:
        self.calls.append(("type", text))


def test_xdotool_env_dispatches_click_in_display_pixels():
    """Brain image == display size: click coords pass through untouched."""
    env = _XdotoolRecorder(viewport=(1280, 720))
    env.step(Action(ActionType.CLICK, {"x": 640, "y": 360}))

    move = next(c for c in env.calls if c[0] == "mousemove")
    assert move == ("mousemove", "640", "360")
    assert ("click", "1") in env.calls


def test_xdotool_env_clamps_out_of_bounds_clicks_to_viewport():
    """A bad brain emitting (10000, 10000) shouldn't crash — clamp instead."""
    env = _XdotoolRecorder(viewport=(1280, 720))
    env.step(Action(ActionType.CLICK, {"x": 10000, "y": 10000}))

    move = next(c for c in env.calls if c[0] == "mousemove")
    assert move == ("mousemove", "1279", "719")  # viewport - 1


def test_xdotool_env_screen_size_matches_viewport():
    """The promise of screen_size: it's the dispatch-side display size."""
    env = _XdotoolRecorder(viewport=(1920, 1080))
    assert env.screen_size == (1920, 1080)
