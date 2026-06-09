"""Pagination URL filter — keep ``extract_url`` from emitting ``?p=2`` (#833).

User feedback on HN extraction:

> Even with instructions like "do not click story links / More", it
> still clicked/paginated and returned URLs like
> ``news.ycombinator.com/?p=2`` or ``newest?next=...`` instead of
> actual story URLs.

Root cause has two parts. The decomposer-emits-click side is #831; this
module handles the extractor-emits-pagination-URL side. When the URL
extracted is structurally a pagination URL (query params like ``?p=2``,
``?page=3``, ``/page/4/``, ``?start=20``, ``?next=...``), it's almost
never the URL the caller wanted — they wanted a *story / item / detail*
URL.

The filter is **vocabulary-based regex**, intentionally conservative:
better to skip a real "page=" param that happens to mean something
else than to silently leak pagination URLs into the leads list. Add
new patterns when feedback surfaces a missed shape.

Plan-author override: a step's ``extract`` block can set
``allow_pagination_urls: true`` to opt back in (e.g. for sitemap-style
scrapers that genuinely want the pagination URLs).
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse


# Compiled patterns. Each one represents a "this URL is structurally
# pagination" signal. Tested against the parsed path + query separately
# so a path-shaped pattern (``/page/3``) doesn't accidentally match a
# query-shaped one.
_PATH_PATTERNS = tuple(
    re.compile(p) for p in (
        # Path-segment pagination — ``/page/3``, ``/page/3/``, etc.
        # Deliberately NOT matching a bare trailing ``/3`` because
        # that's the shape real detail URLs take on many sites
        # (``/issues/123``, ``/items/45``). Require the literal
        # ``page``/``p`` segment to disambiguate.
        r"/page/\d+/?$",
        r"/p/\d+/?$",
    )
)

# Query keys whose presence means "pagination". Values don't matter —
# the key alone is the signal. Conservative set: don't add ``id``,
# ``slug`` etc. without a stronger signal.
_PAGINATION_QUERY_KEYS = frozenset({
    "p", "page", "pg", "start", "offset", "skip",
    "next", "after", "before", "since",
})

# Common HN-style pagination paths that don't carry a clear page number
# in the URL but always represent "the next batch of items".
_HN_STYLE_PAGINATION_PATHS = frozenset({
    "/newest",
    "/news",
    "/active",
    "/front",
})


def is_pagination_url(url: str) -> bool:
    """Return True when ``url`` is structurally a pagination URL.

    Implementation:
    - Pure-fragment changes (``#comments``) are NOT pagination.
    - Path-segment ``/page/3`` or ``/page/3/`` patterns match.
    - Any documented query key (``p``, ``page``, ``start``, ``next``…)
      with a non-empty value matches.
    - HN-style ``/newest?next=...`` with a ``next`` query param matches.

    Returns False for clearly-non-pagination URLs (a real story URL,
    a github issue URL, etc.) and for empty / malformed input.
    """
    if not url:
        return False
    try:
        parsed = urlparse(url.strip())
    except Exception:  # noqa: BLE001 — never propagate a parse failure
        return False
    if not parsed.scheme or not parsed.netloc:
        # Bare paths / fragments aren't useful URLs to report regardless,
        # but they're not "pagination" — caller decides what to do.
        return False
    path = parsed.path or "/"

    # Query-key match.
    if parsed.query:
        try:
            qs = parse_qs(parsed.query, keep_blank_values=False)
        except Exception:  # noqa: BLE001
            qs = {}
        for key in qs:
            if key.lower() in _PAGINATION_QUERY_KEYS:
                return True

    # Path-segment match.
    for pattern in _PATH_PATTERNS:
        if pattern.search(path):
            return True

    return False


def is_allowed_pagination(extract_block: dict | None) -> bool:
    """Return True when the step's ``extract`` block opts into
    pagination URL emission via ``allow_pagination_urls: true``.

    Defensive: empty / missing / non-dict input → False.
    """
    if not isinstance(extract_block, dict):
        return False
    return bool(extract_block.get("allow_pagination_urls", False))


__all__ = [
    "is_allowed_pagination",
    "is_pagination_url",
]
