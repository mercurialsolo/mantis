"""Tests for #116 — drift + state-loop detection."""

from __future__ import annotations

import pytest

from mantis_agent.actions import Action, ActionType
from mantis_agent.loop_detector import LoopDetector, phash_64


def _click(x: int, y: int) -> Action:
    return Action(ActionType.CLICK, {"x": x, "y": y})


def _key(keys: str) -> Action:
    return Action(ActionType.KEY_PRESS, {"keys": keys})


def _scroll(direction: str = "down", amount: int = 3) -> Action:
    return Action(ActionType.SCROLL, {"direction": direction, "amount": amount})


# ── Repeat (legacy byte-equal) ───────────────────────────────────────────


def test_repeat_loop_byte_equal() -> None:
    d = LoopDetector()
    for _ in range(5):
        d.record(_click(100, 100), url="https://x.test/a")
    assert d.is_repeat_loop(window=5) is True


def test_repeat_loop_below_window_returns_false() -> None:
    d = LoopDetector()
    for _ in range(2):
        d.record(_click(100, 100))
    assert d.is_repeat_loop(window=5) is False


def test_repeat_loop_state_changing_scroll_is_not_a_loop() -> None:
    """A scroll-down on a long page where each press changes hash is progress."""
    d = LoopDetector()
    # Different frame hashes simulated via different URL anchors → state changes
    for i in range(5):
        d.record(_key("Page_Down"), url=f"https://x.test/p#scroll-{i}")
    assert d.is_repeat_loop(window=5) is False


# ── Drift loop (coordinate near-equal) ───────────────────────────────────


def test_drift_loop_within_tolerance_fires() -> None:
    d = LoopDetector(click_tol_px=8)
    coords = [(487, 312), (489, 310), (491, 308), (488, 311), (490, 309)]
    for x, y in coords:
        d.record(_click(x, y))
    assert d.is_drift_loop(window=5) is True
    # is_any_loop also fires.
    assert d.is_any_loop(window=5) is True


def test_drift_loop_beyond_tolerance_does_not_fire() -> None:
    d = LoopDetector(click_tol_px=8)
    coords = [(487, 312), (489, 310), (491, 308), (488, 311), (520, 400)]
    for x, y in coords:
        d.record(_click(x, y))
    assert d.is_drift_loop(window=5) is False


def test_drift_loop_only_for_click_like() -> None:
    d = LoopDetector()
    for _ in range(5):
        d.record(_key("Tab"))
    # KEY_PRESS doesn't qualify for drift — repeat catches it instead.
    assert d.is_drift_loop(window=5) is False


# ── State loop (frozen URL+frame across diverse actions) ─────────────────


def test_state_loop_frozen_url_frozen_hash() -> None:
    d = LoopDetector()
    diverse = [
        _click(100, 100),
        _click(200, 200),
        _key("Escape"),
        _scroll(),
        _click(300, 300),
    ]
    for a in diverse:
        d.record(a, url="https://x.test/stuck", frame=None)
        # Manually inject identical frame_hash via the public surface:
        d._samples[-1].frame_hash = "deadbeefdeadbeef"
    assert d.is_state_loop(window=5) is True


def test_state_loop_url_changes_breaks() -> None:
    d = LoopDetector()
    for i, a in enumerate([_click(1, 1)] * 5):
        d.record(a, url=f"https://x.test/p{i}")
    # Five identical clicks → byte-equal repeat fires…
    assert d.is_repeat_loop(window=5) is True
    # …but state loop does NOT fire because URL kept changing.
    assert d.is_state_loop(window=5) is False


def test_state_loop_no_signal_returns_false() -> None:
    d = LoopDetector()
    for _ in range(5):
        d.record(_click(100, 100))  # no url, no frame
    assert d.is_state_loop(window=5) is False


# ── Reset / window edge cases ────────────────────────────────────────────


def test_reset_clears_history() -> None:
    d = LoopDetector()
    for _ in range(5):
        d.record(_click(1, 1))
    assert d.is_repeat_loop(window=5) is True
    d.reset()
    assert d.is_repeat_loop(window=5) is False


def test_window_one_or_zero_never_fires() -> None:
    d = LoopDetector()
    d.record(_click(1, 1))
    assert d.is_repeat_loop(window=1) is False
    assert d.is_repeat_loop(window=0) is False


# ── Perceptual hash sanity ───────────────────────────────────────────────


def test_phash_identical_images_match() -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    img = Image.new("RGB", (32, 32), (128, 64, 200))
    h1 = phash_64(img)
    h2 = phash_64(img.copy())
    assert h1 == h2
    assert len(h1) == 17  # 16 hex of dHash + 1 of brightness bucket


def test_phash_different_images_differ() -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    a = Image.new("RGB", (32, 32), (10, 10, 10))
    b = Image.new("RGB", (32, 32), (250, 250, 250))
    assert phash_64(a) != phash_64(b)
