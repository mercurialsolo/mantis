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

from .schema import ExtractionSchema
from .spam import contains_dealer_text, seller_looks_like_dealer


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
        """Return a reason if this looks like spam/dealer inventory."""
        if self._schema:
            if self.is_dealer:
                return f"extractor marked as {self._schema.spam_label}"
            if self._schema.seller_looks_like_spam(self.seller or self.extracted_fields.get("seller", "")):
                seller = self.seller or self.extracted_fields.get("seller", "")
                return f"seller looks like {self._schema.spam_label}: {seller}"
            text = f"{self.url} {self.raw_response} " + " ".join(self.extracted_fields.values())
            if self._schema.contains_spam_text(text):
                return f"{self._schema.spam_label} indicator in listing text"
            return ""
        # Legacy BoatTrader path
        if self.is_dealer:
            return "extractor marked listing as dealer"
        if seller_looks_like_dealer(self.seller):
            return f"seller looks like dealer: {self.seller}"
        if contains_dealer_text(f"{self.url} {self.price} {self.raw_response}"):
            return "dealer/sponsored indicator in listing text"
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

    def missing_required_reason(self) -> str:
        """Return why this extraction is not a usable lead."""
        if self._schema:
            missing = [
                name for name in self._schema.required_fields
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
