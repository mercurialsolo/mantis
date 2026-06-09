"""Tests for pagination URL filter (#833).

User feedback on HN extraction:

> Even with instructions like "do not click story links / More", it
> still clicked/paginated and returned URLs like
> ``news.ycombinator.com/?p=2`` or ``newest?next=...`` instead of
> actual story URLs.

The filter rejects extract_url results whose URL is structurally a
pagination URL. Conservative vocabulary — false-positive cost (skip a
real URL with a ``?p=`` query) is low; false-negative cost (emit a
pagination URL as a "story") is high.
"""

from __future__ import annotations

import pytest

from mantis_agent.extraction.pagination_filter import (
    is_allowed_pagination,
    is_pagination_url,
)


# ── Pagination URLs we want filtered ──────────────────────────────


@pytest.mark.parametrize("url", [
    # The exact HN-feedback shapes:
    "https://news.ycombinator.com/?p=2",
    "https://news.ycombinator.com/newest?next=12345",
    "https://news.ycombinator.com/news?p=3",
    # Common query-param patterns:
    "https://example.com/?page=4",
    "https://example.com/?pg=2",
    "https://example.com/search?start=20",
    "https://example.com/?offset=40",
    "https://example.com/listings?skip=10",
    "https://example.com/?after=cursor123",
    "https://example.com/?before=cursor456",
    "https://example.com/feed?since=2026-01-01",
    # Path-segment patterns:
    "https://example.com/page/3",
    "https://example.com/page/3/",
    "https://example.com/p/2",
])
def test_pagination_urls_detected(url):
    assert is_pagination_url(url), f"should detect pagination: {url}"


# ── Non-pagination URLs that must pass through ────────────────────


@pytest.mark.parametrize("url", [
    # Real HN story URLs:
    "https://news.ycombinator.com/item?id=12345",
    "https://news.ycombinator.com/from?site=anthropic.com",
    # Real story / detail URLs on other sites:
    "https://github.com/owner/repo/issues/123",
    "https://anthropic.com/news/claude-fable-5",
    "https://en.wikipedia.org/wiki/Page_title",
    # Pure fragment changes are not pagination:
    "https://example.com/article#comments",
    "https://example.com/article#section-2",
    # Empty / malformed:
    "",
    "not-a-url",
    "/just/a/path",
    "?p=2",  # No scheme + netloc — not a complete URL
])
def test_non_pagination_urls_pass(url):
    assert not is_pagination_url(url), f"should NOT detect pagination: {url}"


# ── Plan-author override ──────────────────────────────────────────


def test_allow_pagination_opt_in():
    """Sitemap-style scrapers that genuinely want pagination URLs
    set ``allow_pagination_urls: true`` on the extract block."""
    assert is_allowed_pagination({"allow_pagination_urls": True}) is True


@pytest.mark.parametrize("block", [
    None,
    {},
    {"allow_pagination_urls": False},
    {"some_other_field": True},
    "not a dict",  # defensive: filter never crashes
    42,
])
def test_allow_pagination_default_false(block):
    assert is_allowed_pagination(block) is False


# ── Defensive: never crash ────────────────────────────────────────


def test_filter_handles_garbage_input():
    """The filter must never propagate a parse error — caller is in
    a hot extract_url path and a crash means a missed step."""
    for url in [None, "", "   ", "\x00", "javascript:alert(1)"]:
        try:
            result = is_pagination_url(url or "")
        except Exception:  # noqa: BLE001
            pytest.fail(f"filter raised on: {url!r}")
        assert isinstance(result, bool)
