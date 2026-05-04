"""ListingsScanner — state container for results-page scanning.

Phase 1.2 of EPIC #161 — verifies the dataclass defaults match the
pre-refactor literals in ``MicroPlanRunner.__init__`` so a checkpoint
round-trip behaves identically. Phase 4 will add behavior tests as
methods are pulled in from ``run()``.
"""

from __future__ import annotations

from mantis_agent.gym.listings_scanner import ListingsScanner


def test_default_state_matches_pre_refactor_init():
    """Defaults must match what MicroPlanRunner.__init__ used to set."""
    s = ListingsScanner()
    assert s.seen_urls == set()
    assert s.extracted_titles == []
    assert s.page_listings == []
    assert s.page_listing_index == 0
    assert s.viewport_stage == 0
    assert s.max_viewport_stages == 6
    assert s.results_base_url == ""
    assert s.required_filter_tokens == ()


def test_each_instance_gets_independent_collections():
    """Defaults use ``field(default_factory=...)``; instances must not share."""
    a = ListingsScanner()
    b = ListingsScanner()
    a.seen_urls.add("http://x")
    a.extracted_titles.append("title")
    a.page_listings.append((10, 20, "card"))
    assert b.seen_urls == set()
    assert b.extracted_titles == []
    assert b.page_listings == []


def test_state_mutates_in_place():
    """Operations on the underlying containers persist (no setter required)."""
    s = ListingsScanner()
    s.seen_urls.add("http://a")
    s.extracted_titles.append("Listing A")
    s.page_listings.append((100, 200, "Listing A"))
    s.page_listing_index += 1
    s.viewport_stage = 2
    s.results_base_url = "https://example.com/search"
    s.required_filter_tokens = ("seller-private",)

    assert s.seen_urls == {"http://a"}
    assert s.extracted_titles == ["Listing A"]
    assert s.page_listings == [(100, 200, "Listing A")]
    assert s.page_listing_index == 1
    assert s.viewport_stage == 2
    assert s.results_base_url == "https://example.com/search"
    assert s.required_filter_tokens == ("seller-private",)
