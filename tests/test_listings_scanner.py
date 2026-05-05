"""ListingsScanner — state container + behavior for results-page scanning.

Phase 1.2 of EPIC #161 verified the dataclass defaults match the
pre-refactor literals; Phase 4 added the behavior methods (is_duplicate,
mark_seen, on_page_change, scroll_directive_for, advance_attempted).
This test file covers both.
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
    assert s.listings_attempted == 0


# ── Phase 4 behavior methods ───────────────────────────────────────


def test_is_duplicate_returns_true_for_seen_url():
    s = ListingsScanner()
    s.seen_urls.add("https://example.com/listing/1")
    assert s.is_duplicate("https://example.com/listing/1") is True


def test_is_duplicate_returns_false_for_unseen_url():
    s = ListingsScanner()
    s.seen_urls.add("https://example.com/listing/1")
    assert s.is_duplicate("https://example.com/listing/2") is False


def test_is_duplicate_returns_false_for_empty_url():
    """Empty URL must NOT be flagged duplicate even when seen_urls is non-empty."""
    s = ListingsScanner()
    s.seen_urls.add("https://example.com/x")
    assert s.is_duplicate("") is False
    assert s.is_duplicate(None) is False  # type: ignore[arg-type]


def test_mark_seen_adds_url_to_set():
    s = ListingsScanner()
    s.mark_seen("https://example.com/abc")
    assert "https://example.com/abc" in s.seen_urls
    s.mark_seen("https://example.com/xyz")
    assert s.seen_urls == {"https://example.com/abc", "https://example.com/xyz"}


def test_mark_seen_is_noop_for_empty_url():
    """Empty URL must NOT pollute the seen-set with falsy entries."""
    s = ListingsScanner()
    s.mark_seen("")
    s.mark_seen(None)  # type: ignore[arg-type]
    assert s.seen_urls == set()


def test_on_page_change_resets_per_page_state():
    """Paginate-success pulls the trigger; everything per-page resets to 0."""
    s = ListingsScanner()
    s.seen_urls.add("https://example.com/x")  # cross-page → keep
    s.required_filter_tokens = ("private",)   # cross-page → keep
    s.results_base_url = "https://example.com/search"  # cross-page → keep
    s.extracted_titles.extend(["A", "B", "C"])
    s.page_listings.extend([(1, 2, "x"), (3, 4, "y")])
    s.page_listing_index = 2
    s.viewport_stage = 3
    s.listings_attempted = 7

    s.on_page_change()

    # Per-page state cleared.
    assert s.extracted_titles == []
    assert s.page_listings == []
    assert s.page_listing_index == 0
    assert s.viewport_stage == 0
    assert s.listings_attempted == 0
    # Cross-page state preserved.
    assert s.seen_urls == {"https://example.com/x"}
    assert s.required_filter_tokens == ("private",)
    assert s.results_base_url == "https://example.com/search"


def test_advance_attempted_increments_listings_counter():
    s = ListingsScanner()
    assert s.listings_attempted == 0
    s.advance_attempted()
    assert s.listings_attempted == 1
    s.advance_attempted()
    s.advance_attempted()
    assert s.listings_attempted == 3


def test_scroll_directive_for_zero_returns_none():
    assert ListingsScanner.scroll_directive_for(0) is None
    assert ListingsScanner.scroll_directive_for(-1) is None


def test_scroll_directive_for_positive_returns_intent_string():
    s = ListingsScanner.scroll_directive_for(5)
    assert s == (
        "Scroll down past the first 5 listings. "
        "Then click the next listing title text below a photo."
    )


def test_scroll_directive_for_is_stateless():
    """Static method — doesn't read or mutate any scanner state."""
    s1 = ListingsScanner.scroll_directive_for(2)
    s2 = ListingsScanner.scroll_directive_for(2)
    assert s1 == s2


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
