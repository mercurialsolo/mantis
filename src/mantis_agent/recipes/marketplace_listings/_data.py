"""Vertical-specific constants for marketplace_listings.

Lives under the recipe so adding a new vertical does not require editing
the core. ``schema.py`` is the public surface; this module is its data
backing and is not part of the recipe contract.
"""

from __future__ import annotations

from typing import Any

# Dealer / non-private signals scraped from listing card text.
DEALER_TEXT_INDICATORS: tuple[str, ...] = (
    "dealername-",
    "dealer website",
    "view dealer website",
    "more from this dealer",
    "request a price",
    "condition-new",
    "certified dealer",
    "sponsored",
    "advertisement",
    "boatsgroup",
    "marinemax",
)

# Tokens that suggest the seller is a dealership rather than a private party.
DEALER_SELLER_INDICATORS: tuple[str, ...] = (
    "marine",
    "marinemax",
    "yacht",
    "boats",
    "brokerage",
    "sales",
    "dealer",
    "center",
    "inc",
    "llc",
)

# Lead-form / inquiry-form labels we should never click — they hand off to
# the dealer rather than reveal the seller's contact info.
FORBIDDEN_CONTROLS: tuple[str, ...] = (
    "Contact Seller",
    "Request Info",
    "Email Seller",
    "Get Pre-Qualified",
    "loan",
    "financing",
)

# Reveal controls we DO click to surface a phone number / additional details.
ALLOWED_CONTROLS: tuple[str, ...] = (
    "Show more",
    "Read more",
    "See more",
    "Show phone",
    "View phone",
    "Call",
)

# #584: non-listing cards that look like listings to Claude vision.
# Plumbed into ``ClaudeExtractor._get_find_listings_prompt`` as an
# EXCLUDE list so find_all_listings returns only organic results
# (not marketing/financing/promo cards). Without this, a click step
# can target ``/boat-loans/`` instead of an actual boat detail page
# → wrong-page extract → halt_reason cycle.
LISTING_CARD_EXCLUSIONS: tuple[str, ...] = (
    "Get Pre-Qualified financing CTAs (loan-rate prompts, monthly-payment calculators)",
    "Boat Loans cards / 'Get Started' loan promos",
    "Get Insurance / insurance-quote CTAs",
    "Live Video Tour overlays (the overlay is not a listing — the card under it is)",
    "Sponsored / Promoted boats labelled 'Sponsored' or 'Promoted'",
    "Newsletter signup, email-alert, or saved-search promo cards",
    "Auction or trade-in CTA tiles",
)


def fields() -> list[dict[str, Any]]:
    """Field descriptors for the marketplace listing schema."""
    return [
        {"name": "year", "type": "str", "required": True, "example": "2018"},
        {"name": "make", "type": "str", "required": True, "example": "Sea Ray"},
        {"name": "model", "type": "str", "required": False, "example": "240 Sundeck"},
        {"name": "price", "type": "str", "required": False, "example": "$42,500"},
        {"name": "phone", "type": "str", "required": False, "example": "305-555-1234"},
        {"name": "url", "type": "str", "required": False, "example": "boattrader.com/boat/..."},
        {"name": "seller", "type": "str", "required": False, "example": "John Smith"},
    ]


REQUIRED_FIELDS: tuple[str, ...] = ("year", "make")
# Issue #236: looser required-field contract for SEARCH_TILE
# extraction. Tiles often render year/make only in alt-text or
# JSON-LD (invisible to vision-mode extraction); the strict
# ``REQUIRED_FIELDS`` set was rejecting every tile row, halting
# the run with zero leads. Tile mode now needs only ``url`` so
# the runner can keep the row to drive a follow-up navigation
# into the detail page where ``REQUIRED_FIELDS`` is enforced.
TILE_REQUIRED_FIELDS: tuple[str, ...] = ("url",)
# Informational hint: which fields the search tile is expected to
# surface. The runner uses this to skip re-reading these on the
# detail page (carry from tile → enrich detail). url, title,
# price near-universally render on marketplace tile cards.
TILE_CARRY_FIELDS: tuple[str, ...] = ("url", "title", "price")
# "boat listing" is preserved verbatim from the legacy
# ExtractionSchema.default_boattrader() default. Renaming it would be a
# behavior change for callers reading entity_name in their own prompts.
ENTITY_NAME = "boat listing"
SPAM_LABEL = "dealer"

# Issue #246: recipe-rejection → host-facing intent.
#
# - ``dealer``: terminal-for-this-row. The detail page truly is a
#   dealer listing (storefront banner, "Contact Dealer" CTAs, dealer
#   seller name). No future read can turn it into a private-seller
#   lead. Host should mark the URL as processed-but-skipped and
#   advance to the next listing — *not* retry the extraction. Live
#   reproducer: BoatTrader plan d0693cd9 looped 6× on a single
#   dealer-flagged URL across 90 minutes before this annotation.
# - ``incomplete_required``: the search tile didn't surface
#   year/make in rendered text, but the detail page may. Host
#   should follow up with a deeper read; ``extract_more`` keeps
#   the row in the extraction loop rather than dropping it.
REJECTION_INTENTS: dict[str, str] = {
    "dealer": "skip",
    "incomplete_required": "extract_more",
}
