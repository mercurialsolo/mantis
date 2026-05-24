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


def _step(max_viewport_stages: int = 1) -> MicroIntent:
    """Build a collect_urls MicroIntent.

    Default ``max_viewport_stages=1`` pins the multi-viewport loop
    (#638) to a single iteration so existing single-pass tests still
    pin a single ``find_all_listings`` call + one set of CDP returns.
    Tests that exercise the multi-viewport scan pass a larger value.
    """
    return MicroIntent(
        intent="Collect listing URLs", type="collect_urls", section="extraction",
        hints={"max_viewport_stages": max_viewport_stages},
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


# ── #638: multi-viewport scan ──────────────────────────────────────────


def test_multi_viewport_accumulates_urls_across_stages(monkeypatch) -> None:
    """Three viewport stages, each scans 3 distinct cards → 9 unique URLs.
    Verifies the handler scrolls + rescans + accumulates across stages."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.adaptive_content_settle",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.time.sleep",
        lambda *_: None,
    )
    runner = _FakeRunner()
    extractor = MagicMock()
    # Stage 0, 1, 2 each return 3 distinct cards.
    extractor.find_all_listings.side_effect = [
        [(100, 200, f"S0-{i}") for i in range(3)],
        [(100, 200, f"S1-{i}") for i in range(3)],
        [(100, 200, f"S2-{i}") for i in range(3)],
    ]
    env = MagicMock()
    env.screenshot.return_value = MagicMock()
    # 9 distinct URLs (3 cards × 3 stages).
    env.cdp_evaluate.side_effect = [
        f"https://example.com/boat/{i}/" for i in range(9)
    ]

    result = CollectUrlsHandler(runner).execute(
        _step(max_viewport_stages=3), _ctx(env, extractor),
    )

    assert result.success is True
    assert len(runner._collected_urls) == 9
    assert result.data == "urls:9/9"
    # 3 stages = 3 Claude scan calls (one per find_all_listings invocation).
    assert runner.costs["claude_extract"] == 3
    # Page_Down keypress fired at least once (between stage 0→1 and 1→2).
    assert env.step.call_count >= 1


def test_multi_viewport_dedups_overlap_between_stages(monkeypatch) -> None:
    """PAGE_DOWNS_PER_STAGE < 1 viewport so adjacent stages overlap —
    the same card appears in stages N and N+1. The seen-URL set
    collapses the redundancy."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.adaptive_content_settle",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.time.sleep",
        lambda *_: None,
    )
    runner = _FakeRunner()
    extractor = MagicMock()
    # Both stages return overlapping cards (boat/1 + boat/2 in both,
    # plus a new card in each).
    extractor.find_all_listings.side_effect = [
        [(100, 100, "B1"), (100, 200, "B2"), (100, 300, "B3")],
        [(100, 100, "B2"), (100, 200, "B3"), (100, 300, "B4")],
    ]
    env = MagicMock()
    env.screenshot.return_value = MagicMock()
    env.cdp_evaluate.side_effect = [
        "https://example.com/boat/1/", "https://example.com/boat/2/",
        "https://example.com/boat/3/",  # stage 0
        "https://example.com/boat/2/", "https://example.com/boat/3/",
        "https://example.com/boat/4/",  # stage 1 — 2 dup, 1 new
    ]

    CollectUrlsHandler(runner).execute(
        _step(max_viewport_stages=2), _ctx(env, extractor),
    )

    # 4 unique URLs (1, 2, 3, 4) from 6 raw cards across stages.
    assert len(runner._collected_urls) == 4
    assert set(runner._collected_urls) == {
        "https://example.com/boat/1/", "https://example.com/boat/2/",
        "https://example.com/boat/3/", "https://example.com/boat/4/",
    }


def test_multi_viewport_early_exit_on_two_empty_stages(monkeypatch) -> None:
    """Two consecutive stages adding 0 new URLs triggers early exit
    without burning the rest of the budget. Caps wasted Claude calls
    on short results pages."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.adaptive_content_settle",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.time.sleep",
        lambda *_: None,
    )
    runner = _FakeRunner()
    extractor = MagicMock()
    # Stage 0 returns 2 cards; stages 1+2 return same 2 cards (all dups);
    # stages 3+ never reached because early exit fires after 2 empty.
    same_two = [(100, 100, "B1"), (100, 200, "B2")]
    extractor.find_all_listings.side_effect = [
        same_two, same_two, same_two,
        # Unreached:
        [(100, 100, "BX")], [(100, 100, "BY")],
    ]
    env = MagicMock()
    env.screenshot.return_value = MagicMock()
    # 6 cdp calls for the 3 stages we DO execute; nothing after.
    env.cdp_evaluate.side_effect = [
        "https://example.com/boat/1/", "https://example.com/boat/2/",
    ] * 3

    CollectUrlsHandler(runner).execute(
        _step(max_viewport_stages=5), _ctx(env, extractor),
    )

    assert len(runner._collected_urls) == 2
    # 3 Claude calls (not 5) — early exit at stage 2 after 2 zero-new stages.
    assert runner.costs["claude_extract"] == 3


def test_multi_viewport_stage0_blocked_aborts(monkeypatch) -> None:
    """find_all_listings returning ('blocked',) at stage 0 means the
    whole page is gone — abort with skip envelope, don't waste budget
    on subsequent stages."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.adaptive_content_settle",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.time.sleep",
        lambda *_: None,
    )
    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_all_listings.return_value = ("blocked", "cf challenge")
    env = MagicMock()
    env.screenshot.return_value = MagicMock()

    result = CollectUrlsHandler(runner).execute(
        _step(max_viewport_stages=4), _ctx(env, extractor),
    )

    assert result.success is False
    assert result.skip is True
    assert result.skip_reason == "collect_urls_signal_blocked"
    # ONE Claude call before bailing — not 4.
    assert runner.costs["claude_extract"] == 1


# ── #638 axis 2: multi-page accumulation ──────────────────────────────


def test_second_invocation_appends_to_existing_collected_urls(monkeypatch) -> None:
    """When the runner already has URLs from a prior collect_urls call
    (Phase-1 multi-page: navigate(page-1) → collect → navigate(page-2) →
    collect), the second invocation appends to the existing list rather
    than replacing it."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.adaptive_content_settle",
        lambda *_, **__: None,
    )
    runner = _FakeRunner()
    # Simulate page-1 already harvested.
    runner._collected_urls = [
        "https://example.com/boat/1/",
        "https://example.com/boat/2/",
    ]

    extractor = MagicMock()
    # Page-2 scan returns 2 new cards.
    extractor.find_all_listings.return_value = [
        (100, 200, "Boat 3"),
        (100, 400, "Boat 4"),
    ]
    env = MagicMock()
    env.screenshot.return_value = MagicMock()
    env.cdp_evaluate.side_effect = [
        "https://example.com/boat/3/",
        "https://example.com/boat/4/",
    ]

    result = CollectUrlsHandler(runner).execute(_step(), _ctx(env, extractor))

    assert result.success is True
    # Order preserved: page-1 URLs first, page-2 URLs appended.
    assert runner._collected_urls == [
        "https://example.com/boat/1/",
        "https://example.com/boat/2/",
        "https://example.com/boat/3/",
        "https://example.com/boat/4/",
    ]
    # data string reports the CUMULATIVE count (4) over THIS-CALL cards (2).
    assert result.data == "urls:4/2"


def test_second_invocation_dedups_against_prior_urls(monkeypatch) -> None:
    """Featured / sponsored listings frequently appear on every paginated
    page. The cross-call dedup catches them via the seeded ``seen`` set."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.adaptive_content_settle",
        lambda *_, **__: None,
    )
    runner = _FakeRunner()
    runner._collected_urls = [
        "https://example.com/boat/1/",  # featured — will re-appear on page-2
        "https://example.com/boat/2/",
    ]
    extractor = MagicMock()
    extractor.find_all_listings.return_value = [
        (100, 200, "Featured"),
        (100, 400, "New"),
    ]
    env = MagicMock()
    env.screenshot.return_value = MagicMock()
    env.cdp_evaluate.side_effect = [
        "https://example.com/boat/1/",  # duplicate from page-1
        "https://example.com/boat/9/",  # genuinely new
    ]

    CollectUrlsHandler(runner).execute(_step(), _ctx(env, extractor))

    assert runner._collected_urls == [
        "https://example.com/boat/1/",
        "https://example.com/boat/2/",
        "https://example.com/boat/9/",
    ]


def test_max_viewport_stages_hint_overrides_default(monkeypatch) -> None:
    """``step.hints['max_viewport_stages']`` is the operator knob —
    pin it to 2 and verify the loop respects it even when more cards
    could be found at later stages."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.adaptive_content_settle",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.collect_urls.time.sleep",
        lambda *_: None,
    )
    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_all_listings.side_effect = [
        [(100, 100, "B1")],
        [(100, 200, "B2")],
        # Stage 3+ never reached because hint capped at 2.
        [(100, 300, "B3")],
        [(100, 400, "B4")],
    ]
    env = MagicMock()
    env.screenshot.return_value = MagicMock()
    env.cdp_evaluate.side_effect = [
        "https://example.com/boat/1/",
        "https://example.com/boat/2/",
    ]

    CollectUrlsHandler(runner).execute(
        _step(max_viewport_stages=2), _ctx(env, extractor),
    )

    assert len(runner._collected_urls) == 2
    assert runner.costs["claude_extract"] == 2
