"""Tests for #298 — adaptive loop-detector windows.

The fixed default windows (soft=3, hard=8) miss two classes of work:

- **Pagination / drilldown**: same-coordinate clicks at "Next" advance
  the page; the runner used to soft-nudge spuriously. Adaptive expansion
  defers the nudge when state moved across the window.
- **Stuck loops on micro-changing pages**: brain emits identical clicks
  on a captcha that re-renders pixels but doesn't really change state.
  Adaptive shrinking trips earlier on the low-diversity / no-state
  signature.

The toggle ``MANTIS_LOOP_ADAPTIVE=disabled`` falls back to fixed-window
``is_any_loop`` so the ablation harness can run an A/B without redeploys.
"""

from __future__ import annotations

import pytest

from mantis_agent.actions import Action, ActionType
from mantis_agent.loop_detector import LoopDetector, adaptive_loop_enabled


def _click(x: int, y: int) -> Action:
    return Action(ActionType.CLICK, {"x": x, "y": y})


def _key(keys: str) -> Action:
    return Action(ActionType.KEY_PRESS, {"keys": keys})


def _scroll(direction: str = "down") -> Action:
    return Action(ActionType.SCROLL, {"direction": direction, "amount": 3})


# ── pattern_diversity ───────────────────────────────────────────────


def test_pattern_diversity_all_same_action_is_low() -> None:
    d = LoopDetector()
    for _ in range(5):
        d.record(_click(100, 100))
    # 5 identical bucketed clicks → 1 unique signature / 5 = 0.2
    assert d.pattern_diversity(window=5) == pytest.approx(1 / 5)


def test_pattern_diversity_all_different_is_one() -> None:
    d = LoopDetector()
    actions = [
        _click(50, 50),
        _click(200, 200),
        _key("Tab"),
        _scroll("down"),
        Action(ActionType.TYPE, {"text": "hello"}),
    ]
    for a in actions:
        d.record(a)
    assert d.pattern_diversity(window=5) == 1.0


def test_pattern_diversity_buckets_drift_clicks_low() -> None:
    """Drift clicks within ~8 px collapse to a small number of bucket
    signatures (boundary-crossing means it's not always 1, but it's
    always low — the metric is approximate, not exact)."""
    d = LoopDetector()
    coords = [(487, 312), (489, 310), (491, 308), (488, 311), (490, 309)]
    for x, y in coords:
        d.record(_click(x, y))
    diversity = d.pattern_diversity(window=5)
    # ≤ 0.5: drift always lands in a small subset of buckets even when it
    # straddles boundaries. Compare against 5 distinct widely-separated
    # clicks (diversity == 1.0) to confirm drift collapses.
    assert diversity <= 0.5


def test_pattern_diversity_short_buffer_is_zero() -> None:
    d = LoopDetector()
    d.record(_click(1, 1))
    assert d.pattern_diversity(window=5) == 0.0


# ── state_progressed ─────────────────────────────────────────────────


def test_state_progressed_on_url_change() -> None:
    d = LoopDetector()
    for i in range(5):
        d.record(_click(1, 1), url=f"https://x.test/p{i}")
    assert d.state_progressed(window=5) is True


def test_state_progressed_on_frozen_state() -> None:
    d = LoopDetector()
    for _ in range(5):
        d.record(_click(1, 1), url="https://x.test/p")
    assert d.state_progressed(window=5) is False


# ── adaptive_window ──────────────────────────────────────────────────


def test_adaptive_window_extends_on_progressing_state() -> None:
    """Pagination case: identical click coords, but URL advances → extend."""
    d = LoopDetector()
    for i in range(3):
        d.record(_click(800, 600), url=f"https://x.test/page-{i}")
    # base soft=3, max_extension=2 → effective window 5.
    assert d.adaptive_window(base_window=3) == 5


def test_adaptive_window_extends_on_high_diversity() -> None:
    d = LoopDetector()
    for a in [_click(1, 1), _key("Tab"), _scroll(), _click(50, 50), _key("Return")]:
        d.record(a)
    # diversity 1.0 ≥ 0.6 → extend.
    assert d.adaptive_window(base_window=3) == 5


def test_adaptive_window_tightens_on_low_diversity_frozen_state() -> None:
    """Captcha-trap case: identical clicks, no state movement → tighten."""
    d = LoopDetector()
    for _ in range(8):
        d.record(_click(100, 100), url="https://x.test/stuck")
    # diversity ≤ 0.25 AND not progressed → shrink hard window 8 → 7.
    assert d.adaptive_window(base_window=8) == 7


def test_adaptive_window_floor_respected() -> None:
    """Tightening must not drive the window below ``floor``."""
    d = LoopDetector()
    for _ in range(3):
        d.record(_click(1, 1), url="https://x.test/")
    # base 3, floor default 2 → tighten by 1 → 2 (floor).
    assert d.adaptive_window(base_window=3, max_extension=2) >= 2


def test_adaptive_window_unchanged_on_moderate_diversity() -> None:
    """No clear signal in either direction → don't move the window."""
    d = LoopDetector()
    actions = [_click(1, 1), _click(100, 100), _click(1, 1)]
    for a in actions:
        d.record(a, url="https://x.test/")
    # 2/3 diversity = 0.66 → triggers extension. Use a more moderate case.
    d2 = LoopDetector()
    moderate = [_click(1, 1), _click(1, 1), _click(50, 50)]  # 2/3 = 0.66
    for a in moderate:
        d2.record(a, url="https://x.test/")
    # Even 2/3 ≈ 0.66 ≥ 0.6 triggers extend; tweak threshold by recording
    # 3/4 unique-actions across a 4-window:
    d3 = LoopDetector()
    for a in [_click(1, 1), _click(1, 1), _click(50, 50), _click(100, 100)]:
        d3.record(a, url="https://x.test/")
    # 3 uniques / 4 = 0.75 → extend. Build a moderate diversity 0.5 window:
    d4 = LoopDetector()
    for a in [_click(1, 1), _click(1, 1), _click(50, 50), _click(50, 50)]:
        d4.record(a, url="https://x.test/")
    # diversity 2/4 = 0.5 < 0.6, > 0.25, not progressed → window unchanged.
    assert d4.adaptive_window(base_window=4) == 4


# ── is_any_loop_adaptive end-to-end ──────────────────────────────────


def test_pagination_does_not_fire_adaptive_loop() -> None:
    """Five identical "Next" clicks with state advancing → not a loop."""
    d = LoopDetector()
    for i in range(5):
        d.record(_click(800, 600), url=f"https://x.test/page-{i}")
    # Non-adaptive: extends soft=3 → effective 5; state-loop fails (URLs vary)
    # repeat-loop fires on identical clicks but adaptive widens past soft.
    # Use the soft window to confirm:
    assert d.is_any_loop(window=3) is True  # legacy: false positive
    assert d.is_any_loop_adaptive(base_window=3) is False  # fixed by #298


def test_real_repeat_still_fires_adaptive() -> None:
    """Five identical clicks on a frozen page → still a loop."""
    d = LoopDetector()
    for _ in range(5):
        d.record(_click(800, 600), url="https://x.test/p")
    assert d.is_any_loop_adaptive(base_window=3) is True


def test_diverse_actions_on_progressing_state_no_loop() -> None:
    d = LoopDetector()
    actions = [_click(1, 1), _key("Tab"), _scroll(), _key("Return"), _click(50, 50)]
    for i, a in enumerate(actions):
        d.record(a, url=f"https://x.test/p{i}")
    assert d.is_any_loop_adaptive(base_window=3) is False


def test_diverse_actions_on_frozen_state_still_loop() -> None:
    """State-loop signal must keep firing under adaptive widening."""
    d = LoopDetector()
    actions = [_click(1, 1), _key("Tab"), _scroll(), _key("Return"), _click(50, 50)]
    for a in actions:
        d.record(a, url="https://x.test/stuck")
        d._samples[-1].frame_hash = "deadbeefdeadbeef"
    # State-loop fires regardless of window size since URLs are all equal.
    assert d.is_any_loop_adaptive(base_window=3) is True


# ── env-var toggle ───────────────────────────────────────────────────


def test_adaptive_loop_enabled_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANTIS_LOOP_ADAPTIVE", raising=False)
    assert adaptive_loop_enabled() is True


@pytest.mark.parametrize(
    "value", ["disabled", "0", "false", "FALSE", "off", "no", " disabled "]
)
def test_adaptive_loop_disabled_when_off(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANTIS_LOOP_ADAPTIVE", value)
    assert adaptive_loop_enabled() is False


@pytest.mark.parametrize("value", ["enabled", "1", "true", "on", "yes", ""])
def test_adaptive_loop_enabled_for_truthy_values(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANTIS_LOOP_ADAPTIVE", value)
    assert adaptive_loop_enabled() is True
