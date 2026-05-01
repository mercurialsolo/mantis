"""Generic boolean / spam helpers used by the extractor and result types.

The marketplace-listing dealer/private classification lives under
``mantis_agent.recipes.marketplace_listings._data`` after PR #102; the
helpers here lazy-load those constants only on the legacy path that does
not supply an explicit ``ExtractionSchema``.
"""

from __future__ import annotations


def parse_bool(value: object) -> bool:
    """Parse API booleans that may arrive as strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _legacy_dealer_indicators() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Lazy-load the marketplace_listings recipe constants.

    Cached on first call. Only exists so the legacy ``dealer_reason`` /
    ``find_all`` paths (callers that don't supply an ``ExtractionSchema``)
    keep working without re-introducing a vertical-specific dependency at
    import time.
    """
    cached = getattr(_legacy_dealer_indicators, "_cached", None)
    if cached is None:
        from ..recipes.marketplace_listings import _data as _md
        cached = (_md.DEALER_TEXT_INDICATORS, _md.DEALER_SELLER_INDICATORS)
        _legacy_dealer_indicators._cached = cached
    return cached


def contains_dealer_text(text: str) -> bool:
    """True when ``text`` contains a known dealer / non-private signal."""
    text_lower = text.lower()
    text_indicators, _ = _legacy_dealer_indicators()
    return any(indicator in text_lower for indicator in text_indicators)


def seller_looks_like_dealer(seller: str) -> bool:
    """True when ``seller`` reads like a dealership / brokerage name."""
    seller_lower = seller.lower()
    _, seller_indicators = _legacy_dealer_indicators()
    return any(indicator in seller_lower for indicator in seller_indicators)


# Single-underscore aliases retained for one minor release. The original
# ``mantis_agent.extraction._contains_dealer_text`` / ``_seller_looks_like_dealer``
# / ``_parse_bool`` were "internal" by convention but external code does
# read them in places. The package ``__init__`` re-exports all three.
_parse_bool = parse_bool
_contains_dealer_text = contains_dealer_text
_seller_looks_like_dealer = seller_looks_like_dealer
