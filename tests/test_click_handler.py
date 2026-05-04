"""ClaudeGuidedClickHandler unit tests — Phase 2 of EPIC #161.

The handler is the thickest piece of runner logic Phase 2 has lifted
so far (369 LOC). These tests exercise the four exits that don't
require a real browser navigation:

- viewport scan returns ``("blocked", ...)`` twice → ``page_blocked``
- viewport scan returns ``("error", ...)`` → ``scan_error``
- all viewport stages exhausted → ``page_exhausted``
- handler.step_type matches the dispatch key

The successful-click path (click → grounding → verify → detail-page
URL) is exercised end-to-end on Modal as part of every Phase 2 PR
verification, since faking 3-attempt verify + middle-click + 5-point
probe-area fallback in unit tests would test the mock more than the
handler. The fast paths above are what unit tests are good at.

No Xvfb, no GymRunner, no live brain — exactly what EPIC #161 promised.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from mantis_agent.gym.listings_scanner import ListingsScanner
from mantis_agent.gym.step_context import StepContext
from mantis_agent.gym.step_handlers.click import ClaudeGuidedClickHandler
from mantis_agent.plan_decomposer import MicroIntent


class _FakeRunner:
    """Minimal back-reference. Mirrors the runner attrs the handler reads.

    Property delegates exposed by MicroPlanRunner are not present on this
    fake; the handler addresses scanner state via the runner attribute
    names (``_seen_urls`` / ``_extracted_titles`` / etc.) so the fake
    backs those onto a real :class:`ListingsScanner` and forwards reads
    / writes to it. This is the same pattern the production runner uses.
    """

    def __init__(self) -> None:
        self.scanner = ListingsScanner()
        self.costs: dict[str, float] = {
            "claude_extract": 0,
            "claude_grounding": 0,
            "gpu_steps": 0,
            "gpu_seconds": 0,
            "proxy_mb": 0.0,
        }
        self._current_page = 1
        self._last_known_url = ""
        self._last_extracted: dict[str, Any] = {}
        self._last_click_title = ""
        self._opened_detail_in_new_tab = False
        self._page_listing_count = 0
        # Side-effect-tracker for runner methods the handler calls.
        self.scroll_state_calls: list[dict] = []

    # Scanner property delegates (subset — only the fields the click
    # handler reads/writes):
    @property
    def _page_listings(self): return self.scanner.page_listings
    @_page_listings.setter
    def _page_listings(self, v): self.scanner.page_listings = v
    @property
    def _page_listing_index(self): return self.scanner.page_listing_index
    @_page_listing_index.setter
    def _page_listing_index(self, v): self.scanner.page_listing_index = v
    @property
    def _viewport_stage(self): return self.scanner.viewport_stage
    @_viewport_stage.setter
    def _viewport_stage(self, v): self.scanner.viewport_stage = v
    @property
    def _max_viewport_stages(self): return self.scanner.max_viewport_stages
    @_max_viewport_stages.setter
    def _max_viewport_stages(self, v): self.scanner.max_viewport_stages = v
    @property
    def _results_base_url(self): return self.scanner.results_base_url
    @_results_base_url.setter
    def _results_base_url(self, v): self.scanner.results_base_url = v
    @property
    def _extracted_titles(self): return self.scanner.extracted_titles
    @_extracted_titles.setter
    def _extracted_titles(self, v): self.scanner.extracted_titles = v

    @property
    def _listings_on_page(self): return self._page_listing_count
    @_listings_on_page.setter
    def _listings_on_page(self, v): self._page_listing_count = v

    # Runner methods the handler calls:
    def _set_scroll_state(self, **kwargs) -> None:
        self.scroll_state_calls.append(kwargs)

    def _current_results_page_url(self) -> str:
        return self.scanner.results_base_url

    def _read_current_url(self, screenshot=None) -> str:
        return ""  # default: not on detail page


def _ctx(runner: _FakeRunner, *, env=None, extractor=None) -> StepContext:
    return StepContext(
        env=env or MagicMock(),
        brain=None,
        extractor=extractor or MagicMock(),
        grounding=None,
        cost_meter=None,
        dynamic_verifier=MagicMock(),
        scanner=runner.scanner,
        site_config=MagicMock(),
        tool_channel=None,
        extraction_cache=None,
        state={"index": 4},
    )


def _step() -> MicroIntent:
    return MicroIntent(intent="Click next listing", type="click", section="extraction")


def test_step_type_property():
    handler = ClaudeGuidedClickHandler(_FakeRunner())
    assert handler.step_type == "click"


def test_blocked_page_returns_page_blocked_after_retry(monkeypatch):
    """find_all_listings returns blocked twice → 12s rescan, then page_blocked."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    extractor = MagicMock()
    # Both initial scan and the post-12s rescan return blocked.
    extractor.find_all_listings.side_effect = [
        ("blocked", "anti-bot challenge"),
        ("blocked", "still blocked"),
    ]
    ctx = _ctx(runner, extractor=extractor)

    result = ClaudeGuidedClickHandler(runner).execute(_step(), ctx)

    assert result.success is False
    assert result.data == "page_blocked"
    assert result.step_index == 4
    # First scan + rescan: 2 claude_extract calls billed
    assert runner.costs["claude_extract"] == 2
    # Verifier saw the blocked outcome
    ctx.dynamic_verifier.record_viewport_scan.assert_called()
    last_call = ctx.dynamic_verifier.record_viewport_scan.call_args
    assert last_call.kwargs["status"] == "blocked"


def test_scan_error_returns_scan_error(monkeypatch):
    """find_all_listings returns error tuple → scan_error, no rescan."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_all_listings.return_value = ("error", "API parse failure")
    ctx = _ctx(runner, extractor=extractor)

    result = ClaudeGuidedClickHandler(runner).execute(_step(), ctx)

    assert result.success is False
    assert result.data == "scan_error"
    assert runner.costs["claude_extract"] == 1
    last_call = ctx.dynamic_verifier.record_viewport_scan.call_args
    assert last_call.kwargs["status"] == "error"


def test_all_viewports_exhausted_returns_page_exhausted(monkeypatch):
    """Every viewport scan returns 0 cards → page_exhausted after _max_viewport_stages."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    runner._max_viewport_stages = 2  # shrink for fast test
    extractor = MagicMock()
    extractor.find_all_listings.return_value = []  # empty scan
    ctx = _ctx(runner, extractor=extractor)

    result = ClaudeGuidedClickHandler(runner).execute(_step(), ctx)

    assert result.success is False
    assert result.data == "page_exhausted"
    # Two scans (one per viewport stage) — both empty
    assert runner.costs["claude_extract"] == 2
    ctx.dynamic_verifier.record_page_exhausted.assert_called_once()
    assert runner._viewport_stage == 2  # advanced past both stages


def test_handler_filters_already_extracted_titles(monkeypatch):
    """Cards whose titles are in _extracted_titles get filtered out before clicking."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    runner._extracted_titles = ["Listing A", "Listing B"]  # already done
    runner._max_viewport_stages = 1
    extractor = MagicMock()
    # Returns a list directly (not a tuple) — non-blocked path
    extractor.find_all_listings.return_value = [
        (100, 200, "Listing A"),  # already extracted → skip
        (100, 250, "Listing B"),  # already extracted → skip
        (100, 300, "Listing C"),  # new
    ]
    env = MagicMock()
    env.screen_size = (1280, 800)
    # _read_current_url returns empty → click won't verify, but we just
    # care that ONLY "Listing C" gets selected from the cache.
    ctx = _ctx(runner, env=env, extractor=extractor)
    runner._read_current_url = lambda *a, **kw: ""  # never on detail page

    handler = ClaudeGuidedClickHandler(runner)
    handler.execute(_step(), ctx)

    # The cache should have been seeded with just "Listing C"
    # (or empty + page_exhausted if filter worked the right way).
    # Either way: Listing A / B are NOT clicked.
    last_click_title = runner._last_click_title or ""
    assert "Listing A" not in last_click_title
    assert "Listing B" not in last_click_title
