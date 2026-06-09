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
from typing import Any, ClassVar

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

    # #579: Claude's tool_use schema is permissive (#558) — domain
    # fields are optional, so the brain may return literal "<UNKNOWN>"
    # / "none" / "" when it can't read a field off the screenshot.
    # Treat all of these as "missing" for required-field checks so a
    # detail-page extract that landed on a marketing CTA (every field
    # ``<UNKNOWN>``) doesn't get accepted as VIABLE. Without this, the
    # boattrader-style failure produces a "1 lead" output where the
    # only "lead" is junk pointing at ``/boat-loans/``.
    _UNKNOWN_PLACEHOLDERS: ClassVar[frozenset[str]] = frozenset({
        "", "unknown", "<unknown>", "none", "n/a", "na",
        "not visible", "not shown", "not available", "tbd",
    })

    @classmethod
    def _is_unknown(cls, value: Any) -> bool:
        """True when the value is empty or one of the documented
        unknown-placeholder strings (case + whitespace insensitive)."""
        return str(value or "").strip().lower() in cls._UNKNOWN_PLACEHOLDERS

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
        navigate-into-detail. ``UNKNOWN`` (default) enforces
        ``required_fields`` regardless of context.

        When the schema doesn't define ``tile_required_fields`` (empty
        list), ``SEARCH_TILE`` falls back to ``required_fields`` —
        recipes that don't opt into the split see no behavior change.

        When ``_schema`` is None (no recipe / no schema configured),
        returns ``no_schema_configured`` so the failure mode is honest
        rather than the historical ``missing required field(s): year,
        make`` lie that fired on every non-marketplace plan. The
        ``# Legacy`` boattrader fallback the year/make check used to
        live in was the exact contamination pattern flagged in
        ``feedback_legacy_fallback_smell.md`` (#785 follow-up).
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
                if self._is_unknown(self.extracted_fields.get(name))
            ]
            return f"missing required field(s): {', '.join(missing)}" if missing else ""
        return "no_schema_configured"

    def to_summary(self) -> str:
        """Format as the VIABLE summary string.

        Schema-bound results emit one ``Field: value`` clause per
        declared field, joined with ``|``.

        Non-marketplace plans with no schema (post-#815 validator rip)
        previously fell through to a BoatTrader-shape default
        (``VIABLE | Year: ... | Make: ...``) — that was the original
        complaint surfaced by user feedback on HN extraction. Now the
        fall-through emits whatever raw fields the result has, with
        explicit names — no Year/Make/Model invention.
        """
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

        # No-schema fall-through. Emit ONLY what's actually populated
        # — no marketplace-shape invention. Pre-fix this synthesised
        # ``Year: <empty>`` / ``Make: <empty>`` for any extracted
        # result regardless of domain, which is how an HN extraction
        # ended up emitting ``VIABLE | Year | Make | Model | ...``.
        parts: list[str] = []
        # Iterate raw fields populated on this result. The list is
        # restricted to the legacy ExtractionResult dataclass fields so
        # we don't accidentally emit ``_schema`` etc.
        for label, val in (
            ("Year", self.year),
            ("Make", self.make),
            ("Model", self.model),
            ("Price", self.price),
            ("Phone", self.phone),
            ("URL", self.url),
            ("Seller", self.seller),
        ):
            if val:
                parts.append(f"{label}: {val}")
        if not parts:
            # No schema AND nothing usable extracted → don't emit a
            # misleading ``VIABLE | ...`` line at all.
            return ""
        return "VIABLE | " + " | ".join(parts)

    def is_viable(self) -> bool:
        """Has enough data to be a useful lead (not spam, required fields present).

        #579: ``<UNKNOWN>`` / ``none`` / empty placeholders all count
        as missing for the required-field check, so a detail-page
        extract that landed on a marketing CTA (every field returned
        as a placeholder) is rejected as non-viable.

        Without an ``ExtractionSchema`` configured, returns ``False``:
        a result has no schema contract to satisfy, so it can't be
        certified viable. The historical ``# Legacy`` path silently
        treated ``year + make + private_seller`` as the universal
        viability contract — boattrader-shape baggage that leaked
        into every non-marketplace plan (#785 follow-up;
        ``feedback_legacy_fallback_smell.md``).
        """
        if self._schema:
            has_required = all(
                not self._is_unknown(self.extracted_fields.get(name))
                for name in self._schema.required_fields
            )
            return has_required and self.is_private_seller()
        return False
