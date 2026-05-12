"""Tests for #292 — brain_opencua coordinate dispatch and scroll-offset.

The bug as filed in #292 only manifests when the env captures full-page
screenshots; today every Mantis env captures viewport-only (xdotool grabs
the Xvfb framebuffer; Playwright's `page.screenshot()` defaults to
`full_page=False`). These tests:

1. Lock the current viewport contract — `scroll_offset=(0,0)` produces
   the same coords as before the change (no regression).
2. Verify the math is correct under a hypothetical full-page contract —
   when `scroll_offset` is non-zero, the dispatched screen pixel falls
   inside the viewport regardless of where the click target lives in
   document space.
3. Cover the long-page lu.ma reproduction described in the issue: a
   half-scrolled page where the model targets the 3rd visible card.
"""

from __future__ import annotations

import pytest

from mantis_agent.actions import ActionType
from mantis_agent.brain_opencua import (
    OpenCUABrain,
    _model_coords_to_screen,
    _smart_resize,
)


# ── Viewport contract (current default) ────────────────────────────────


def test_scroll_offset_defaults_to_zero_preserves_legacy_math() -> None:
    """No scroll arg → identical coords to the pre-#292 implementation.

    Anchored on the smart-resize math: model coord at the resized-space
    centre maps back to the screen-space centre, with no offset applied.
    """
    screen_w, screen_h = 1280, 720
    rh, rw = _smart_resize(screen_h, screen_w)
    # Model emits the centre of the resized image.
    mx, my = rw // 2, rh // 2
    sx, sy = _model_coords_to_screen(mx, my, screen_w, screen_h)
    # Should land within 1 pixel of the screen centre.
    assert abs(sx - screen_w // 2) <= 1
    assert abs(sy - screen_h // 2) <= 1


def test_scroll_offset_zero_round_trips_corners() -> None:
    """The (0,0) → (0,0) and bottom-right → bottom-right invariants hold."""
    screen_w, screen_h = 1280, 720
    rh, rw = _smart_resize(screen_h, screen_w)

    sx, sy = _model_coords_to_screen(0, 0, screen_w, screen_h)
    assert (sx, sy) == (0, 0)

    sx, sy = _model_coords_to_screen(rw - 1, rh - 1, screen_w, screen_h)
    assert sx == screen_w - 1 or sx == screen_w  # rounding tolerance
    assert sy == screen_h - 1 or sy == screen_h


# ── Full-page contract (defensive correctness for future) ──────────────


def test_scroll_offset_subtracted_when_page_scrolled() -> None:
    """Hypothetical full-page mode: model emits document-space coords,
    runner passes scrollY; dispatch must land inside the viewport.

    Setup: viewport is 1280x720, page is scrolled down 500px. Model
    emits a coord that, in document space, is at y=600 (so 100px below
    viewport top). After scroll-offset subtraction the dispatched
    screen y should be 100, not 600.
    """
    screen_w, screen_h = 1280, 720
    rh, rw = _smart_resize(screen_h, screen_w)

    # Model coord that maps to (640, 600) in screen space pre-offset.
    mx = int(640 / screen_w * rw)
    my = int(600 / screen_h * rh)

    # Without scroll: lands at ~(640, 600).
    sx, sy = _model_coords_to_screen(mx, my, screen_w, screen_h)
    assert abs(sy - 600) <= 2

    # With scroll_offset=(0, 500): lands at ~(640, 100) — inside viewport.
    sx, sy = _model_coords_to_screen(
        mx, my, screen_w, screen_h, scroll_offset=(0, 500),
    )
    assert abs(sy - 100) <= 2
    # X is unchanged (no horizontal scroll).
    assert abs(sx - 640) <= 1


def test_scroll_offset_x_subtracted() -> None:
    """Horizontal scroll subtracts from x."""
    screen_w, screen_h = 1280, 720
    rh, rw = _smart_resize(screen_h, screen_w)
    mx = int(800 / screen_w * rw)
    my = int(360 / screen_h * rh)
    sx, _ = _model_coords_to_screen(
        mx, my, screen_w, screen_h, scroll_offset=(200, 0),
    )
    assert abs(sx - 600) <= 1


def test_scroll_offset_lu_ma_long_page_reproduction() -> None:
    """Issue #292 reproduction: half-scrolled lu.ma, 3rd visible card.

    Without scroll-offset compensation, a click on the 3rd visible card
    (which the model sees at viewport y≈400 but lives at document y≈900
    after a 500px scroll) gets dispatched to screen y≈900 — well below
    the viewport, hitting nothing or a different card.
    """
    screen_w, screen_h = 1280, 720
    rh, rw = _smart_resize(screen_h, screen_w)

    # Model sees the card at viewport y=400 (which is doc y=900 after scroll).
    # Under full-page contract, model emits doc coord (640, 900).
    mx = int(640 / screen_w * rw)
    my = int(900 / screen_h * rh)

    # Pre-fix behaviour (no scroll): dispatch lands at y≈900 — off-screen.
    _, sy_no_scroll = _model_coords_to_screen(mx, my, screen_w, screen_h)
    assert sy_no_scroll > screen_h, (
        f"reproduction precondition: y={sy_no_scroll} should exceed "
        f"viewport height {screen_h}"
    )

    # Post-fix: with scroll_offset=(0, 500), lands at y≈400 — inside viewport.
    _, sy_fixed = _model_coords_to_screen(
        mx, my, screen_w, screen_h, scroll_offset=(0, 500),
    )
    assert 0 <= sy_fixed <= screen_h
    assert abs(sy_fixed - 400) <= 5


# ── _parse_pyautogui / _parse_json_action thread the offset ────────────


def _fake_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 0}}


@pytest.fixture
def brain() -> OpenCUABrain:
    return OpenCUABrain(base_url="http://localhost:8000/v1")


def test_parse_pyautogui_threads_scroll_offset(brain: OpenCUABrain) -> None:
    screen = (1280, 720)
    rh, rw = _smart_resize(screen[1], screen[0])
    mx = int(640 / 1280 * rw)
    my = int(900 / 720 * rh)

    # No scroll: y ends up off-viewport.
    a_no = brain._parse_pyautogui(f"pyautogui.click({mx}, {my})", screen)
    assert a_no is not None and a_no.action_type == ActionType.CLICK
    assert a_no.params["y"] > 720

    # With scroll: y is subtracted before dispatch.
    a_scroll = brain._parse_pyautogui(
        f"pyautogui.click({mx}, {my})", screen, scroll_offset=(0, 500),
    )
    assert a_scroll is not None and a_scroll.action_type == ActionType.CLICK
    assert 0 <= a_scroll.params["y"] <= 720


def test_parse_json_action_threads_scroll_offset(brain: OpenCUABrain) -> None:
    screen = (1280, 720)
    rh, rw = _smart_resize(screen[1], screen[0])
    mx = int(640 / 1280 * rw)
    my = int(900 / 720 * rh)

    body = f'{{"action": "click", "x": {mx}, "y": {my}}}'
    a_no = brain._parse_json_action(body, screen)
    assert a_no is not None
    assert a_no.params["y"] > 720

    a_scroll = brain._parse_json_action(body, screen, scroll_offset=(0, 500))
    assert a_scroll is not None
    assert 0 <= a_scroll.params["y"] <= 720


def test_parse_response_threads_scroll_offset_end_to_end(
    brain: OpenCUABrain,
) -> None:
    """`think → _parse_response → _parse_pyautogui` propagates the offset."""
    screen = (1280, 720)
    rh, rw = _smart_resize(screen[1], screen[0])
    mx = int(640 / 1280 * rw)
    my = int(900 / 720 * rh)

    data = _fake_response(f"pyautogui.click({mx}, {my})")
    no_scroll = brain._parse_response(data, screen)
    assert no_scroll.action.params["y"] > 720

    scrolled = brain._parse_response(data, screen, scroll_offset=(0, 500))
    assert 0 <= scrolled.action.params["y"] <= 720


# ── Sanity: signature defaults preserve all existing call sites ────────


def test_model_coords_to_screen_default_signature_unchanged() -> None:
    """Callers that omit ``scroll_offset`` get identical legacy behaviour."""
    legacy = _model_coords_to_screen(100, 200, 1280, 720)
    new = _model_coords_to_screen(100, 200, 1280, 720, scroll_offset=(0, 0))
    assert legacy == new
