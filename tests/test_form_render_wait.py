"""Tests for the form handler's render-wait primitives.

The staff-crm smoke from #215's CLI surfaced a real failure mode: the
first ``env.screenshot()`` after a navigate sometimes captured a
completely blank/white page (the SPA hadn't painted yet), causing
``ClaudeExtractor.find_form_target`` to correctly return ``not_found``
and the runner to halt a required step that would have succeeded a
few seconds later. Three smoke runs confirmed the race — debug PNGs
at /tmp/mantis_debug showed pure-white 1280x720 frames at the failed
attempts and fully-rendered login forms at later attempts.

Generic fix, no plan vocabulary baked in: detect a blank screenshot
via a cheap pixel-count heuristic and re-screenshot until the page
has visible content or the deadline expires.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from mantis_agent.gym.step_handlers.form import (
    _is_blank_screenshot,
    _wait_for_rendered_screenshot,
)


# ── _is_blank_screenshot ─────────────────────────────────────────────────


def test_blank_detect_returns_true_for_pure_white_image() -> None:
    img = Image.new("RGB", (1280, 720), color=(255, 255, 255))
    assert _is_blank_screenshot(img) is True


def test_blank_detect_returns_true_for_off_white_at_threshold() -> None:
    """A nearly-white page (e.g. light grey background, no content) should
    still be flagged as blank — the heuristic uses near-white (>=250)."""
    img = Image.new("RGB", (1280, 720), color=(252, 252, 252))
    assert _is_blank_screenshot(img) is True


def test_blank_detect_returns_false_for_solid_dark_image() -> None:
    """A blue / dark login form is not blank, even though pixels are
    uniform colour — uniformity isn't the signal, near-whiteness is."""
    img = Image.new("RGB", (1280, 720), color=(40, 50, 130))
    assert _is_blank_screenshot(img) is False


def test_blank_detect_returns_false_for_real_form_screenshot() -> None:
    """A page with even sparse content (10% non-white) is NOT blank.
    Synthesises a roughly-rendered page: white background with a
    coloured strip at the top (header)."""
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    # Paint 15 rows (15%) with a coloured header — plenty of content
    # for the threshold.
    for y in range(15):
        for x in range(100):
            img.putpixel((x, y), (50, 100, 200))
    assert _is_blank_screenshot(img) is False


@pytest.mark.parametrize(
    "fraction_dark,threshold,expected",
    [
        # Far-from-boundary cases — robust to NEAREST resize quantisation.
        (0.00, 0.99, True),    # all white, conservative threshold
        (0.50, 0.99, False),   # half dark, well below threshold
        (0.50, 0.90, False),   # half dark, looser threshold still rejects
        (0.05, 0.90, True),    # 5% dark, 90% threshold permits
        (0.20, 0.90, False),   # 20% dark, exceeds 10% non-white budget
    ],
)
def test_blank_detect_respects_threshold(
    fraction_dark: float, threshold: float, expected: bool,
) -> None:
    """The threshold is tunable; default 0.99 is conservative.

    Tests stay clear of the exact resize boundary (e.g. 0.99/0.99) —
    NEAREST quantisation when downscaling 100×100 → 64×64 means a
    single source row maps to ``round(64/100) = 1`` dest row, so the
    realised fraction at the boundary depends on the source size in
    a way that's not worth pinning here.
    """
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    n_dark_rows = int(round(fraction_dark * 100))
    for y in range(n_dark_rows):
        for x in range(100):
            img.putpixel((x, y), (10, 10, 10))
    assert _is_blank_screenshot(img, threshold=threshold) is expected


def test_blank_detect_returns_false_on_conversion_error() -> None:
    """Defensive: a non-Image input must not crash the helper. Returns
    False so the caller proceeds with the existing (possibly broken)
    screenshot rather than entering an unbounded retry."""
    assert _is_blank_screenshot("not-an-image") is False  # type: ignore[arg-type]


# ── _wait_for_rendered_screenshot ───────────────────────────────────────


def _white_image() -> Image.Image:
    return Image.new("RGB", (640, 360), color=(255, 255, 255))


def _rendered_image() -> Image.Image:
    return Image.new("RGB", (640, 360), color=(40, 50, 130))


def test_wait_returns_immediately_when_first_screenshot_is_rendered(
    monkeypatch,
) -> None:
    """Common case: the page is already painted. No sleeps, single
    screenshot call."""
    monkeypatch.setenv("MANTIS_ADAPTIVE_SETTLE", "disabled")  # exercise the legacy blank-retry fallback
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.time.sleep",
        lambda s: sleep_calls.append(s),
    )

    env = MagicMock()
    env.screenshot.return_value = _rendered_image()

    out = _wait_for_rendered_screenshot(env, max_retries=5, poll_seconds=2)

    assert env.screenshot.call_count == 1
    assert sleep_calls == []
    assert _is_blank_screenshot(out) is False


def test_wait_retries_when_first_screenshots_are_blank(monkeypatch) -> None:
    """Two blank captures, then a rendered one — the helper should
    return the rendered frame and have slept twice."""
    monkeypatch.setenv("MANTIS_ADAPTIVE_SETTLE", "disabled")  # exercise the legacy blank-retry fallback
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.time.sleep",
        lambda s: sleep_calls.append(s),
    )

    env = MagicMock()
    env.screenshot.side_effect = [
        _white_image(),
        _white_image(),
        _rendered_image(),
    ]

    out = _wait_for_rendered_screenshot(env, max_retries=5, poll_seconds=2)

    assert env.screenshot.call_count == 3
    assert sleep_calls == [2, 2]  # two retries, both polling 2s
    assert _is_blank_screenshot(out) is False


def test_wait_gives_up_after_max_retries_and_returns_last_screenshot(monkeypatch) -> None:
    """If the page never renders, the helper must NOT loop forever.
    It returns the last (still-blank) screenshot after ``max_retries``
    so the caller can decide what to do — usually let find_form_target
    return not_found and the existing retry chain kick in."""
    monkeypatch.setenv("MANTIS_ADAPTIVE_SETTLE", "disabled")  # exercise the legacy blank-retry fallback
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.time.sleep",
        lambda s: sleep_calls.append(s),
    )

    env = MagicMock()
    env.screenshot.return_value = _white_image()

    out = _wait_for_rendered_screenshot(env, max_retries=3, poll_seconds=2)

    # 1 initial capture + 3 retries = 4 screenshot calls; 3 sleeps.
    assert env.screenshot.call_count == 4
    assert sleep_calls == [2, 2, 2]
    assert _is_blank_screenshot(out) is True


def test_wait_with_zero_retries_returns_first_screenshot(monkeypatch) -> None:
    """Defensive: max_retries=0 must not loop. Returns the first
    screenshot regardless of whether it's blank — caller cap of 0
    means 'don't retry, return what you got'."""
    monkeypatch.setenv("MANTIS_ADAPTIVE_SETTLE", "disabled")  # exercise the legacy blank-retry fallback
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.time.sleep",
        lambda s: sleep_calls.append(s),
    )

    env = MagicMock()
    env.screenshot.return_value = _white_image()

    out = _wait_for_rendered_screenshot(env, max_retries=0)

    assert env.screenshot.call_count == 1
    assert sleep_calls == []
    assert _is_blank_screenshot(out) is True


# ── Integration: form handler tolerates a blank-then-rendered sequence ───


def test_fill_field_tolerates_blank_first_screenshot(monkeypatch) -> None:
    """End-to-end on the form handler: env returns a blank screenshot
    twice, then the real login form. The handler should NOT report
    form_target_not_found — the render-wait kicks in, the third
    screenshot is rendered, find_form_target succeeds."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform", lambda a, b: 0.0,
    )

    from mantis_agent.gym.step_context import StepContext
    from mantis_agent.gym.step_handlers.form import ClaudeGuidedFormHandler
    from mantis_agent.plan_decomposer import MicroIntent

    runner = MagicMock()
    runner.costs = {"claude_extract": 0, "gpu_steps": 0, "gpu_seconds": 0}
    runner._url_history = ["", ""]
    runner._best_effort_current_url = MagicMock(return_value="")
    runner._adaptive_submit_settle = MagicMock(return_value=0.5)
    runner._safe_screenshot = MagicMock(return_value=_rendered_image())
    runner._dump_debug_screenshot = MagicMock()

    env = MagicMock()
    env.screenshot.side_effect = [
        _white_image(),
        _white_image(),
        _rendered_image(),
    ]
    extractor = MagicMock()
    extractor.find_form_target.return_value = {
        "x": 691, "y": 284, "action": "type",
        "value": "alice", "label": "User ID input field",
    }

    ctx = StepContext(
        env=env, brain=None, extractor=extractor, grounding=None,
        cost_meter=None, dynamic_verifier=None, scanner=None,
        site_config=None, tool_channel=None, extraction_cache=None,
        state={"index": 1},
    )

    step = MicroIntent(
        intent="Enter alice in the user ID field",
        type="fill_field",
        params={"label": "user ID", "value": "alice"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    # Render-wait absorbed the two blank screenshots, third was real.
    assert env.screenshot.call_count == 3
    assert result.success is True
    assert result.data == "fill:user ID"
