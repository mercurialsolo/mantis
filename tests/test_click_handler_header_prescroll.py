"""Tests for issue #600 — pre-scroll past the page header before scan.

Without the prescroll, viewport stage 0 lands at Home + 0 Page_Downs.
On a typical results page that puts a sticky header + search bar +
promo band in the visible area and the first row of cards peeks in
only as photo crops at the bottom edge — no titles, prices, or any
text. ``find_all_listings`` correctly returns ``title="unknown"`` for
all cards, the runner falls back to coord placeholders, and PR #597's
already-clicked prefilter has nothing semantic to dedupe against.
The brain then re-clicks the same screen coordinates next iteration,
hitting the listing-dedup gate and halting with ``duplicate_listing``.

These tests pin: the ``HEADER_PRESCROLL_PAGE_DOWNS`` constant is set
to 1, both the scan-time scroll and the click-time scroll honor it
(so the brain clicks at the same Y the scan reported), and the
``_set_scroll_state`` page_downs argument reflects the prescroll-aware
position (so checkpoint resume restores the correct scroll).

The same prescroll is applied in the probe-area fallback scroll
(line ~703 in click.py) so any retry path scrolling is consistent.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.actions import ActionType
from mantis_agent.gym.step_handlers.click import (
    HEADER_PRESCROLL_PAGE_DOWNS,
    ClaudeGuidedClickHandler,
)
from mantis_agent.plan_decomposer import MicroIntent

from test_click_handler import _ctx, _FakeRunner  # type: ignore[import-not-found]


def _step() -> MicroIntent:
    return MicroIntent(intent="Click next listing", type="click", section="extraction")


def _page_down_count(env: MagicMock) -> int:
    """Count Page_Down keypresses sent to the env."""
    return sum(
        1
        for c in env.step.call_args_list
        if getattr(c.args[0], "action_type", None) == ActionType.KEY_PRESS
        and c.args[0].params.get("keys") == "Page_Down"
    )


def test_constant_is_one() -> None:
    """The prescroll defaults to 1 Page_Down — enough to push the
    sticky header off the top so the first card row is fully visible
    on a 720-pixel viewport (header band ≈ 500–600 px on typical
    results pages)."""
    assert HEADER_PRESCROLL_PAGE_DOWNS == 1


def test_stage_0_scan_fires_one_page_down(monkeypatch) -> None:
    """At viewport stage 0, scan-time scroll should send exactly
    HEADER_PRESCROLL_PAGE_DOWNS Page_Down keypresses (not zero).
    Without this fix, stage 0 sent 0 Page_Downs and the scan saw the
    page header instead of the cards."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    runner._max_viewport_stages = 1  # stop after one scan
    extractor = MagicMock()
    extractor.find_all_listings.return_value = []  # empty → page_exhausted, no click path
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    ClaudeGuidedClickHandler(runner).execute(_step(), ctx)

    # Stage 0 + prescroll of 1 = 1 Page_Down keypress on the scan path.
    assert _page_down_count(env) == HEADER_PRESCROLL_PAGE_DOWNS


def test_stage_2_scan_fires_prescroll_plus_two_page_downs(monkeypatch) -> None:
    """Stage N's scan-time scroll sends HEADER_PRESCROLL + N Page_Downs.
    Verifies the prescroll stacks on top of the viewport-stage advance,
    not in place of it."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    runner._max_viewport_stages = 3
    runner._viewport_stage = 2  # already advanced past stages 0 and 1
    extractor = MagicMock()
    extractor.find_all_listings.return_value = []
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    ClaudeGuidedClickHandler(runner).execute(_step(), ctx)

    # Stage 2 fires once (the only remaining stage), so 1 scan call =
    # HEADER_PRESCROLL + 2 Page_Downs.
    assert _page_down_count(env) == HEADER_PRESCROLL_PAGE_DOWNS + 2


def test_scroll_state_records_prescroll_position(monkeypatch) -> None:
    """``_set_scroll_state`` should record page_downs = PRESCROLL + stage
    so checkpoint persistence / verifier diagnostics see the actual
    scroll position, not the logical viewport stage."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    runner._max_viewport_stages = 1
    runner._viewport_stage = 0
    extractor = MagicMock()
    extractor.find_all_listings.return_value = []
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    ClaudeGuidedClickHandler(runner).execute(_step(), ctx)

    # Exactly one _set_scroll_state call from the scan path.
    scan_calls = [c for c in runner.scroll_state_calls if c.get("context") == "results_scan"]
    assert len(scan_calls) == 1
    assert scan_calls[0]["page_downs"] == HEADER_PRESCROLL_PAGE_DOWNS
    # The logical viewport stage stays 0 — the prescroll is an
    # absolute offset, not a re-staging.
    assert scan_calls[0]["viewport_stage"] == 0


def test_multiple_stages_scroll_loop_count_matches(monkeypatch) -> None:
    """Across stages 0 and 1 (empty scans), the total Page_Down count
    is (PRESCROLL + 0) + (PRESCROLL + 1) — each stage starts from
    Home and counts from scratch, not cumulative."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    runner._max_viewport_stages = 2
    extractor = MagicMock()
    extractor.find_all_listings.return_value = []  # both stages empty
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    ClaudeGuidedClickHandler(runner).execute(_step(), ctx)

    expected = (HEADER_PRESCROLL_PAGE_DOWNS + 0) + (HEADER_PRESCROLL_PAGE_DOWNS + 1)
    assert _page_down_count(env) == expected
