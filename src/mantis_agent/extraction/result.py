"""Structured result of one extraction call.

``ExtractionResult`` is the dataclass that ``ClaudeExtractor.extract()``
returns. Has two modes:

- **Schema-driven** — ``_schema`` is set; ``extracted_fields`` is the
  primary data store; spam / viability checks consult the schema.
- **Legacy** — ``_schema`` is ``None``; named fields (``year`` / ``make`` /
  ``model`` / etc.) are populated; spam / viability checks fall back to
  the marketplace-listing heuristics in :mod:`.spam`.

Kept as its own module so callers that just need to construct or
inspect a result can import it without paying for the full extractor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import ExtractionContext, ExtractionSchema


@dataclass
class ExtractionResult:
    """Structured data extracted from a listing screenshot.

    Named fields (year, make, model, etc.) are kept for backward compatibility.
    When an ExtractionSchema is set, the generic ``fields`` dict is the primary
    data store and all viability/spam checks use the schema configuration.
    """

    # Existing named fields (backward compat with BoatTrader)
    year: str = ""
    make: str = ""
    model: str = ""
    price: str = ""
    phone: str = ""
    url: str = ""
    seller: str = ""
    is_dealer: bool = False
    raw_response: str = ""
    confidence: float = 0.0

    # Generic field storage — populated when schema is set
    extracted_fields: dict[str, str] = field(default_factory=dict)
    _schema: ExtractionSchema | None = field(default=None, repr=False)

    def dealer_reason(self) -> str:
        """Return a reason if this looks like spam/dealer inventory.

        Spam classification is RECIPE-GATED: only plans whose
        extraction was configured with an :class:`ExtractionSchema`
        (typically via a recipe like ``marketplace_listings``) opt
        into spam detection. Plans without a schema get ``""`` — no
        spam check — so the framework doesn't leak vertical-specific
        keyword heuristics (``"dealer"``, ``"sponsored"``, marketplace-
        seller phrases) into unrelated domains.

        Pre-2026-05-17 behavior had a legacy fallback that ran
        BoatTrader's ``contains_dealer_text`` / ``seller_looks_like_dealer``
        on EVERY un-schema'd ExtractionResult. The staff-crm-long step 9
        halt surfaced the leak: a CRM lead-detail page's text tripped the
        boattrader heuristics and the whole step rejected as
        ``REJECTED_DEALER``. Per ``feedback_no_plan_specific_framework``,
        framework primitives must not bake in plan/vertical specifics.
        """
        if self._schema is None:
            return ""
        if self.is_dealer:
            return f"extractor marked as {self._schema.spam_label}"
        if self._schema.seller_looks_like_spam(self.seller or self.extracted_fields.get("seller", "")):
            seller = self.seller or self.extracted_fields.get("seller", "")
            return f"seller looks like {self._schema.spam_label}: {seller}"
        text = f"{self.url} {self.raw_response} " + " ".join(self.extracted_fields.values())
        if self._schema.contains_spam_text(text):
            return f"{self._schema.spam_label} indicator in listing text"
        return ""

    def is_private_seller(self) -> bool:
        """Not spam/dealer."""
        return not self.dealer_reason()

    def has_phone(self) -> bool:
        """Require an actually visible phone number."""
        phone_val = self.phone or self.extracted_fields.get("phone", "")
        phone = phone_val.strip().lower()
        if phone in {"", "none", "n/a", "na", "unknown", "not visible", "not shown"}:
            return False
        digits = re.sub(r"\D", "", phone)
        return len(digits) >= 10

    def missing_required_reason(
        self,
        context: ExtractionContext = ExtractionContext.UNKNOWN,
    ) -> str:
        """Return why this extraction is not a usable lead.

        Issue #236: ``context`` selects which required-field contract
        to enforce. ``DETAIL_PAGE`` uses ``schema.required_fields``
        (canonical strict set). ``SEARCH_TILE`` uses
        ``schema.tile_required_fields`` when set — typically just
        ``["url"]`` so the runner keeps the row to drive a follow-up
        navigate-into-detail. ``UNKNOWN`` (default) preserves legacy
        behavior: enforces ``required_fields`` regardless of context.

        When the schema doesn't define ``tile_required_fields`` (empty
        list), ``SEARCH_TILE`` falls back to ``required_fields`` —
        recipes that don't opt into the split see no behavior change.
        """
        if self._schema:
            if (
                context == ExtractionContext.SEARCH_TILE
                and self._schema.tile_required_fields
            ):
                fields_to_check = self._schema.tile_required_fields
            else:
                fields_to_check = self._schema.required_fields
            missing = [
                name for name in fields_to_check
                if not self.extracted_fields.get(name)
            ]
            return f"missing required field(s): {', '.join(missing)}" if missing else ""
        # Legacy
        missing = []
        if not self.year:
            missing.append("year")
        if not self.make:
            missing.append("make")
        return f"missing required field(s): {', '.join(missing)}" if missing else ""

    def to_summary(self) -> str:
        """Format as the VIABLE summary string."""
        if self._schema and self.extracted_fields:
            parts = []
            for f in self._schema.fields:
                name = f["name"]
                val = self.extracted_fields.get(name, "")
                if val:
                    parts.append(f"{name.replace('_', ' ').title()}: {val}")
                elif name == "phone":
                    parts.append("Phone: none")
            return "VIABLE | " + " | ".join(parts) if parts else ""
        # Legacy BoatTrader
        parts = []
        if self.year:
            parts.append(f"Year: {self.year}")
        if self.make:
            parts.append(f"Make: {self.make}")
        if self.model:
            parts.append(f"Model: {self.model}")
        if self.price:
            parts.append(f"Price: {self.price}")
        if self.phone:
            parts.append(f"Phone: {self.phone}")
        else:
            parts.append("Phone: none")
        if self.url:
            parts.append(f"URL: {self.url}")
        if self.seller:
            parts.append(f"Seller: {self.seller}")
        return "VIABLE | " + " | ".join(parts) if parts else ""

    def is_viable(self) -> bool:
        """Has enough data to be a useful lead (not spam, required fields present)."""
        if self._schema:
            has_required = all(
                self.extracted_fields.get(name)
                for name in self._schema.required_fields
            )
            return has_required and self.is_private_seller()
        # Legacy
        return bool(self.year and self.make and self.is_private_seller())
