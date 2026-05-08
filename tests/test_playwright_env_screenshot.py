"""Tests for PlaywrightGymEnv.screenshot() — added so step handlers can
drive the playwright env from ``mantis plan run``.

Surfaced when the staff-crm smoke run failed on the first
``env.screenshot()`` call: ``ClaudeGuidedClickHandler`` and
``ClaudeGuidedFormHandler`` both call ``env.screenshot()`` mid-step
to grab a fresh frame after Page_Down scrolls, between submit and
verify, etc. ``XdotoolGymEnv`` had this method; ``PlaywrightGymEnv``
did not, so the handler-dispatched path was env-specific without a
clear error.

Tests cover:
- pre-reset raises a clear ``RuntimeError`` (no silent ``AttributeError``)
- post-reset returns a PIL Image with the expected dimensions
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest
from PIL import Image


def test_screenshot_before_reset_raises_runtime_error() -> None:
    """A handler that calls env.screenshot() before reset() must get an
    actionable error, not an AttributeError on None._page."""
    from mantis_agent.gym.playwright_env import PlaywrightGymEnv

    env = PlaywrightGymEnv()
    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        env.screenshot()


def test_screenshot_returns_pil_image_after_reset_with_mocked_page() -> None:
    """Once a page is bound, screenshot() should return a PIL Image
    decoded from the page's PNG bytes."""
    from mantis_agent.gym.playwright_env import PlaywrightGymEnv

    # Build a tiny 4x3 red PNG and feed it through the mocked page.
    src = Image.new("RGB", (4, 3), color=(255, 0, 0))
    buf = io.BytesIO()
    src.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    page = MagicMock()
    page.screenshot.return_value = png_bytes

    env = PlaywrightGymEnv()
    env._page = page  # bypass real reset — we're testing the screenshot API only

    out = env.screenshot()

    assert isinstance(out, Image.Image)
    assert out.size == (4, 3)
    page.screenshot.assert_called_once_with(type="png")
