"""SiteConfig overlay for the marketplace_listings recipe.

Carries URL/pagination patterns, gate-prompt scaffolding, and URL-filter
encoding rules for boat-trader-style listing sites. Sibling to
``schema.py``: where the schema exports ``SCHEMA: ExtractionSchema``,
this module exports ``SITE_CONFIG: SiteConfig``. Both are picked up by
the symmetric loaders in ``recipes/__init__.py``.

Per issue #224 this is an overlay, not a primary config — it
extends the probe-derived default with patterns the probe can't
infer from a single landing screenshot (the detail-URL slug shape is
the canonical example).

Per issue #464 the ``filter_url_strategies`` mapping also carries the
BoatTrader URL-encoding rules that used to be hardcoded inside
``graph/enhancer.py:_try_build_filtered_url``. Moving them here lets
non-marketplace plans use the graph enhancer without picking up
boattrader-shaped URL guesses.
"""

from __future__ import annotations

from dataclasses import replace

from ...site_config import SiteConfig


# Maps objective-text keywords (lowercased) to URL path segments. A
# ``{value}`` placeholder is substituted with extracted numeric / city
# / state values; keyword-only entries are boolean filters.
_FILTER_URL_STRATEGIES: dict[str, str] = {
    # Private-seller / by-owner toggle
    "private seller": "by-owner",
    "by owner": "by-owner",
    "by-owner": "by-owner",
    # Numeric / extracted-value segments
    "zip": "zip-{value}",
    "zip code": "zip-{value}",
    "price": "price-{value}",
    "min price": "price-{value}",
    "minimum price": "price-{value}",
    "city": "city-{value}",
    "state": "state-{value}",
    "location": "city-{value}",
}


SITE_CONFIG: SiteConfig = replace(
    SiteConfig.default_boattrader(),
    filter_url_strategies=_FILTER_URL_STRATEGIES,
)
