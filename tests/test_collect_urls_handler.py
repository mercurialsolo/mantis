"""CollectUrlsHandler unit tests (#615).

Exercises the four exits that don't require a real browser:

  - extractor returns ``("blocked",)`` signal → empty harvest, success=False
  - find_all_listings returns cards → CDP href lookup runs per card
  - CDP unavailable → empty hrefs, success=False, stash is empty list
  - duplicate hrefs across cards are deduped before stashing

The CDP JS payload is mocked at the env boundary — same pattern the
click-handler tests use.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from mantis_agent.gym.step_context import StepContext
from mantis_agent.gym.step_handlers.collect_urls import CollectUrlsHandler
from mantis_agent.plan_decomposer import MicroIntent


class _FakeRunner:
    def __init__(self) -> None:
        self.costs: dict[str, float] = {"claude_extract": 0}
        self._collected_urls: list[str] = []
        self._last_known_url = ""


def _ctx(env, extractor) -> StepContext:
    return StepContext(
        env=env,
        brain=None,
        extractor=extractor,
        grounding=None,
        cost_meter=None,
        dynamic_verifier=MagicMock(),
        scanner=MagicMock(),
        site_config=MagicMock(),
        tool_channel=None,
        extraction_cache=None,
        state={"index": 3},
    )


def _step() -> MicroIntent:
    return MicroIntent(
        intent="Collect listing URLs", type="collect_urls", section="extraction",
    )


def test_step_type_property() -> None:
    assert CollectUrlsHandler(_FakeRunner()).step_type == "collect_urls"


def test_blocked_scan_returns_empty_harvest(monkeypatch) -> None:
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.adaptive_content_settle",
        lambda *_, **__: None,
    )
    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_all_listings.return_value = ("blocked", "anti-bot wall")

    env = MagicMock()
    env.screenshot.return_value = MagicMock()

    result = CollectUrlsHandler(runner).execute(_step(), _ctx(env, extractor))

    assert result.success is False
    assert result.data == "scan_signal:blocked"
    assert runner._collected_urls == []


def test_cards_resolve_to_hrefs_via_cdp(monkeypatch) -> None:
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.adaptive_content_settle",
        lambda *_, **__: None,
    )
    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_all_listings.return_value = [
        (100, 200, "Boat 1"),
        (100, 400, "Boat 2"),
        (100, 600, "Boat 3"),
    ]

    env = MagicMock()
    env.screenshot.return_value = MagicMock()
    # CDP returns the href for each elementFromPoint call. The handler
    # builds a unique JS string per (sx, sy) so the mock just maps by
    # call order.
    env.cdp_evaluate.side_effect = [
        "https://example.com/boat/1/",
        "https://example.com/boat/2/",
        "https://example.com/boat/3/",
    ]

    result = CollectUrlsHandler(runner).execute(_step(), _ctx(env, extractor))

    assert result.success is True
    assert runner._collected_urls == [
        "https://example.com/boat/1/",
        "https://example.com/boat/2/",
        "https://example.com/boat/3/",
    ]
    assert result.data == "urls:3/3"
    assert runner.costs["claude_extract"] == 1


def test_cdp_unavailable_returns_empty(monkeypatch) -> None:
    """No ``cdp_evaluate`` attribute on env → handler resolves nothing
    and exits gracefully."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.adaptive_content_settle",
        lambda *_, **__: None,
    )
    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_all_listings.return_value = [(50, 60, "card")]

    class _NoCdpEnv:
        def screenshot(self) -> Any:
            return object()
        # No ``cdp_evaluate`` attribute at all.

    env = _NoCdpEnv()
    result = CollectUrlsHandler(runner).execute(_step(), _ctx(env, extractor))

    assert result.success is False
    assert runner._collected_urls == []


def test_duplicate_hrefs_dedup(monkeypatch) -> None:
    """Lazy-rendered results pages occasionally render the same card
    twice with slightly different (x, y). The URL is the primary key."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.adaptive_content_settle",
        lambda *_, **__: None,
    )
    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_all_listings.return_value = [
        (100, 200, "Boat 1"),
        (100, 210, "Boat 1 again"),  # duplicate href
        (100, 400, "Boat 2"),
    ]

    env = MagicMock()
    env.screenshot.return_value = MagicMock()
    env.cdp_evaluate.side_effect = [
        "https://example.com/boat/1/",
        "https://example.com/boat/1/",  # same URL
        "https://example.com/boat/2/",
    ]

    result = CollectUrlsHandler(runner).execute(_step(), _ctx(env, extractor))

    assert runner._collected_urls == [
        "https://example.com/boat/1/",
        "https://example.com/boat/2/",
    ]
    # Coverage tracking counts cards, not hrefs: 3 cards → 2 unique URLs
    # = the data string still reflects the unique fraction.
    assert result.data == "urls:2/3"


def test_cards_without_anchor_silently_drop(monkeypatch) -> None:
    """``find_all_listings`` may locate a card whose centroid doesn't
    overlap an <a> (e.g. a thumbnail-only area). Handler drops it from
    the partition; coverage drops below the 80% threshold and a
    WARNING is emitted."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.adaptive_content_settle",
        lambda *_, **__: None,
    )
    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_all_listings.return_value = [
        (100, 200, "Boat 1"),
        (100, 400, "Boat 2"),
    ]

    env = MagicMock()
    env.screenshot.return_value = MagicMock()
    env.cdp_evaluate.side_effect = [
        "https://example.com/boat/1/",
        None,  # no anchor above this point
    ]

    result = CollectUrlsHandler(runner).execute(_step(), _ctx(env, extractor))

    assert runner._collected_urls == ["https://example.com/boat/1/"]
    assert result.data == "urls:1/2"
