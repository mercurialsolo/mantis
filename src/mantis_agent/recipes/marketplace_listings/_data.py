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
# "boat listing" is preserved verbatim from the legacy
# ExtractionSchema.default_boattrader() default. Renaming it would be a
# behavior change for callers reading entity_name in their own prompts.
ENTITY_NAME = "boat listing"
SPAM_LABEL = "dealer"
