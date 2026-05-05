"""ClaudeStepHandler unit tests — Phase 2 of EPIC #161.

The biggest concentration of extraction logic on the runner: the
``_execute_claude_step`` body (200 LOC, two step types) plus the
``_extract_listing_data_deep`` subroutine (137 LOC, multi-viewport
deep extract). Tests pin the high-value paths:

- extract_url: success, dedup against ``_seen_urls``, missing URL
  returns failure
- extract_data cache hit short-circuits the deep-extract Claude call
  (the cost-savings path added in PR #166)
- extract_data dedup branch when the deep-extract returns a URL we've
  already seen — no duplicate lead emitted
- extract_data viable lead writes to cache when ``cache_write`` is
  enabled
- extract_data dealer / incomplete rejection paths return the right
  ``REJECTED_*`` data string
- non-Claude-only step types return failure (safety net)
- step_type property

The deep-extract subroutine itself is a black box for these tests —
the handler's correctness depends on what comes back from
``extractor.extract_multi``, which is mocked. End-to-end deep-extract
behavior is exercised on every Modal verification.

No Xvfb, no GymRunner, no real ClaudeExtractor.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from mantis_agent.gym.listings_scanner import ListingsScanner
from mantis_agent.gym.step_context import StepContext
from mantis_agent.gym.step_handlers.claude_step import ClaudeStepHandler
from mantis_agent.plan_decomposer import MicroIntent


class _FakeRunner:
    """Minimal back-reference. Tests poke ``costs`` / ``scanner`` /
    ``_last_known_url`` / ``_last_extracted`` and rely on ``_lead_key``
    / ``_lead_has_phone`` / ``_current_item_label`` being callable.

    Phase 4 of EPIC #161: dedup state lives on a real ListingsScanner
    instance now (was a bare ``_seen_urls`` set on the runner).
    """

    def __init__(self) -> None:
        self.costs: dict[str, float] = {"claude_extract": 0}
        self.scanner = ListingsScanner()
        self._current_page = 1
        self._last_known_url = ""
        self._last_extracted: dict[str, Any] = {}
        self.scroll_state_calls: list[dict] = []

    @property
    def _seen_urls(self) -> set[str]:
        """Backward-compat shim mirroring the runner's property delegate."""
        return self.scanner.seen_urls

    def _lead_key(self, summary: str) -> str:
        return summary[:32]

    def _lead_has_phone(self, summary: str) -> bool:
        return "phone" in summary.lower()

    def _current_item_label(self, data: Any = None) -> str:
        return "fake-label"

    def _set_scroll_state(self, **kwargs) -> None:
        self.scroll_state_calls.append(kwargs)

    def _best_effort_current_url(self) -> str:
        return self._url_for_peek

    _url_for_peek: str = ""


def _ctx(runner: _FakeRunner, *, env=None, extractor=None, cache=None) -> StepContext:
    return StepContext(
        env=env or MagicMock(),
        brain=None,
        extractor=extractor,
        grounding=None,
        cost_meter=None,
        dynamic_verifier=MagicMock(),
        scanner=None,
        site_config=None,
        tool_channel=None,
        extraction_cache=cache,
        state={"index": 7},
    )


# ── extract_url ─────────────────────────────────────────────────────


def test_extract_url_returns_url_payload():
    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.extract.return_value = MagicMock(url="https://example.com/listing/42")
    ctx = _ctx(runner, extractor=extractor)

    step = MicroIntent(intent="Read URL", type="extract_url", claude_only=True)
    result = ClaudeStepHandler(runner).execute(step, ctx)

    assert result.success is True
    assert result.data == "URL:https://example.com/listing/42"
    assert "https://example.com/listing/42" in runner._seen_urls
    assert runner._last_known_url == "https://example.com/listing/42"


def test_extract_url_dedup_returns_DUPLICATE_marker():
    runner = _FakeRunner()
    runner._seen_urls.add("https://example.com/seen")
    extractor = MagicMock()
    extractor.extract.return_value = MagicMock(url="https://example.com/seen")
    ctx = _ctx(runner, extractor=extractor)

    step = MicroIntent(intent="Read URL", type="extract_url", claude_only=True)
    result = ClaudeStepHandler(runner).execute(step, ctx)

    assert result.success is False
    assert result.data == "DUPLICATE|https://example.com/seen"
    ctx.dynamic_verifier.record_item_completed.assert_called_once()
    assert ctx.dynamic_verifier.record_item_completed.call_args.kwargs["reason"] == "duplicate_url_skipped"


def test_extract_url_returns_failure_when_url_empty():
    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.extract.return_value = MagicMock(url="")
    ctx = _ctx(runner, extractor=extractor)

    step = MicroIntent(intent="Read URL", type="extract_url", claude_only=True)
    result = ClaudeStepHandler(runner).execute(step, ctx)

    assert result.success is False
    assert result.data == ""


# ── extract_data: cache short-circuit ───────────────────────────────


def test_extract_data_cache_hit_skips_deep_extract():
    runner = _FakeRunner()
    runner._url_for_peek = "https://example.com/cached-event"
    cache = MagicMock()
    cache.read_enabled = True
    cached_entry = MagicMock(
        summary="VIABLE | Title: Cached event",
        item_label="Cached event",
    )
    cache.get.return_value = cached_entry
    extractor = MagicMock()
    ctx = _ctx(runner, extractor=extractor, cache=cache)

    step = MicroIntent(intent="Extract event data", type="extract_data", claude_only=True)
    result = ClaudeStepHandler(runner).execute(step, ctx)

    assert result.success is True
    assert result.data == "VIABLE | Title: Cached event"
    # Critical: the deep-extract / Claude pipeline never ran.
    extractor.extract_multi.assert_not_called()
    extractor.extract.assert_not_called()
    cache.get.assert_called_once_with("https://example.com/cached-event")
    assert "https://example.com/cached-event" in runner._seen_urls
    last_record = ctx.dynamic_verifier.record_item_completed.call_args
    assert last_record.kwargs["reason"] == "cache_hit"


# ── extract_data: viable + write-through to cache ───────────────────


def test_extract_data_viable_writes_to_cache_when_write_enabled(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.claude_step.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_listing_content_control.return_value = None  # no expand controls
    viable = MagicMock(
        url="https://example.com/listing/777",
        is_viable=lambda: True,
        dealer_reason=lambda: None,
        missing_required_reason=lambda: None,
        to_summary=lambda: "VIABLE | year:2024 | make:Acme",
        extracted_fields={"year": "2024", "make": "Acme"},
    )
    extractor.extract_multi.return_value = viable
    cache = MagicMock()
    cache.read_enabled = False  # only write
    cache.get.return_value = None
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor, cache=cache)

    step = MicroIntent(intent="Extract listing", type="extract_data", claude_only=True)
    result = ClaudeStepHandler(runner).execute(step, ctx)

    assert result.success is True
    assert "VIABLE | year:2024" in result.data
    cache.put.assert_called_once()
    args, kwargs = cache.put.call_args
    assert args[0] == "https://example.com/listing/777"
    assert "VIABLE | year:2024" in args[1]
    assert kwargs["fields"] == {"year": "2024", "make": "Acme"}


# ── extract_data: dedup against _seen_urls ──────────────────────────


def test_extract_data_post_extract_dedup_returns_duplicate(monkeypatch):
    """Deep extract returns a URL we've already seen → DUPLICATE marker."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.claude_step.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    runner._seen_urls.add("https://example.com/already-seen")
    extractor = MagicMock()
    extractor.find_listing_content_control.return_value = None
    extractor.extract_multi.return_value = MagicMock(
        url="https://example.com/already-seen",
        is_viable=lambda: True,
        dealer_reason=lambda: None,
        missing_required_reason=lambda: None,
        to_summary=lambda: "VIABLE | seen",
        extracted_fields={},
    )
    ctx = _ctx(runner, env=MagicMock(), extractor=extractor)

    step = MicroIntent(intent="Extract", type="extract_data", claude_only=True)
    result = ClaudeStepHandler(runner).execute(step, ctx)

    assert result.success is False
    assert result.data.startswith("DUPLICATE|https://example.com/already-seen")


# ── extract_data: dealer / incomplete rejection ─────────────────────


def test_extract_data_rejects_dealer_listing(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.claude_step.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    # When extract_multi returns a non-viable result, the deep-extract
    # subroutine falls back to extractor.extract(fallback_shot). Both
    # need to return the same dealer-flagged result so the handler
    # enters the dealer-rejection branch consistently.
    dealer_data = MagicMock(
        url="https://example.com/dealer/123",
        is_viable=lambda: False,
        dealer_reason=lambda: "seller looks like dealer: AcmeAuto",
        missing_required_reason=lambda: None,
        to_summary=lambda: "DEALER | year:2020 | make:Acme | seller:AcmeAuto",
    )
    extractor = MagicMock()
    extractor.find_listing_content_control.return_value = None
    extractor.extract_multi.return_value = dealer_data
    extractor.extract.return_value = dealer_data
    ctx = _ctx(runner, env=MagicMock(), extractor=extractor)

    step = MicroIntent(intent="Extract", type="extract_data", claude_only=True)
    result = ClaudeStepHandler(runner).execute(step, ctx)

    assert result.success is False
    assert result.data.startswith("REJECTED_DEALER|seller looks like dealer: AcmeAuto|")


def test_extract_data_rejects_incomplete_listing(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.claude_step.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    incomplete_data = MagicMock(
        url="https://example.com/incomplete/9",
        is_viable=lambda: False,
        dealer_reason=lambda: None,
        missing_required_reason=lambda: "missing required field(s): year",
        to_summary=lambda: "INCOMPLETE | make:Acme",
    )
    extractor = MagicMock()
    extractor.find_listing_content_control.return_value = None
    extractor.extract_multi.return_value = incomplete_data
    extractor.extract.return_value = incomplete_data
    ctx = _ctx(runner, env=MagicMock(), extractor=extractor)

    step = MicroIntent(intent="Extract", type="extract_data", claude_only=True)
    result = ClaudeStepHandler(runner).execute(step, ctx)

    assert result.success is False
    assert result.data.startswith("REJECTED_INCOMPLETE|missing required field(s): year|")


# ── safety nets ─────────────────────────────────────────────────────


def test_extractor_missing_returns_failure():
    runner = _FakeRunner()
    ctx = _ctx(runner, extractor=None)
    step = MicroIntent(intent="X", type="extract_url", claude_only=True)
    result = ClaudeStepHandler(runner).execute(step, ctx)
    assert result.success is False


def test_unknown_step_type_returns_failure(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.claude_step.time.sleep", lambda *_: None)
    runner = _FakeRunner()
    extractor = MagicMock()
    ctx = _ctx(runner, extractor=extractor)
    step = MicroIntent(intent="X", type="some_other_type", claude_only=True)
    result = ClaudeStepHandler(runner).execute(step, ctx)
    assert result.success is False


def test_step_type_property():
    handler = ClaudeStepHandler(_FakeRunner())
    assert handler.step_type == "extract_url"
