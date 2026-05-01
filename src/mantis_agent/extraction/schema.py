"""Domain-agnostic extraction schema.

Describes what fields to pull, how to detect spam, and what counts as a
viable extraction. Concrete vertical schemas (boat-listing, job-posting,
real-estate) live under ``mantis_agent.recipes.<name>.schema`` and use
``ExtractionSchema`` as their carrier dataclass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractionSchema:
    """Describes what to extract, how to detect spam, and what viability means.

    When passed to ``ClaudeExtractor``, overrides the hardcoded defaults
    with dynamic prompts generated from these fields.

    Use ``ExtractionSchema.from_objective(spec)`` to build from an
    ``ObjectiveSpec``, or import a recipe's ``SCHEMA`` directly via
    ``mantis_agent.recipes.load_schema(name)``.
    """

    entity_name: str = "listing"  # "boat listing", "job posting", "property"
    fields: list[dict[str, Any]] = field(default_factory=list)  # OutputField-like dicts
    required_fields: list[str] = field(default_factory=list)  # field names for viability
    spam_indicators: list[str] = field(default_factory=list)
    spam_seller_indicators: list[str] = field(default_factory=list)
    spam_label: str = "dealer/spam"  # what to call spam (e.g. "dealer", "recruiter")
    forbidden_controls: list[str] = field(default_factory=list)  # "Contact Seller", etc.
    allowed_controls: list[str] = field(default_factory=list)  # "Show more", "Show phone"

    @classmethod
    def from_objective(cls, objective: Any) -> ExtractionSchema:
        """Build from an ObjectiveSpec.

        All domain-specific signals (spam indicators, allowed reveal
        controls, forbidden lead-form labels) come from the objective. No
        hardcoded application-specific defaults are injected — callers that
        want them must specify them on the ObjectiveSpec.
        """
        fields = [
            {"name": f.name, "type": f.type, "required": f.required, "example": f.example}
            for f in getattr(objective, "output_schema", [])
        ]
        required = [f["name"] for f in fields if f.get("required", True)]
        forbidden = list(getattr(objective, "forbidden_actions", []))
        allowed = list(getattr(objective, "allowed_reveal_actions", []))
        spam_text = list(getattr(objective, "spam_text_indicators", []))
        spam_seller = list(getattr(objective, "spam_seller_indicators", []))
        spam_label = str(getattr(objective, "spam_label", "") or "non-organic")

        return cls(
            entity_name=getattr(objective, "target_entity", "item") or "item",
            fields=fields or cls._default_fields(),
            required_fields=required or ["url"],
            spam_indicators=spam_text,
            spam_seller_indicators=spam_seller,
            spam_label=spam_label,
            forbidden_controls=forbidden,
            allowed_controls=allowed,
        )

    @classmethod
    def default_boattrader(cls) -> ExtractionSchema:
        """Deprecated alias for ``recipes.marketplace_listings.schema.SCHEMA``.

        Kept for one minor release so existing callers (tests, training
        configs, deployed plans) keep working. New callers should import
        ``mantis_agent.recipes.marketplace_listings.schema.SCHEMA`` or
        resolve a recipe by name via ``mantis_agent.recipes.load_schema``.
        """
        import warnings

        warnings.warn(
            "ExtractionSchema.default_boattrader() is deprecated; import "
            "mantis_agent.recipes.marketplace_listings.schema.SCHEMA "
            "or call mantis_agent.recipes.load_schema('marketplace_listings').",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..recipes.marketplace_listings.schema import SCHEMA

        return SCHEMA

    @staticmethod
    def _default_fields() -> list[dict[str, Any]]:
        return [
            {"name": "url", "type": "str", "required": True, "example": ""},
            {"name": "title", "type": "str", "required": False, "example": ""},
            {"name": "price", "type": "str", "required": False, "example": ""},
            {"name": "phone", "type": "str", "required": False, "example": ""},
            {"name": "seller", "type": "str", "required": False, "example": ""},
        ]

    def field_names(self) -> list[str]:
        return [f["name"] for f in self.fields]

    def json_template(self) -> str:
        """JSON template string for the extraction prompt."""
        obj = {}
        for f in self.fields:
            obj[f["name"]] = ""
        obj["is_spam"] = False
        return json.dumps(obj)

    def field_descriptions(self) -> str:
        """Numbered field list for extraction prompts."""
        lines = []
        for i, f in enumerate(self.fields, 1):
            example = f" (e.g. {f['example']})" if f.get("example") else ""
            required = " [REQUIRED]" if f.get("required") else ""
            lines.append(f"{i}. {f['name']}: {f.get('type', 'str')}{example}{required}")
        lines.append(f"{len(self.fields) + 1}. is_spam: Is this a {self.spam_label} listing? (true/false)")
        return "\n".join(lines)

    def contains_spam_text(self, text: str) -> bool:
        text_lower = text.lower()
        return any(ind in text_lower for ind in self.spam_indicators)

    def seller_looks_like_spam(self, seller: str) -> bool:
        seller_lower = seller.lower()
        return any(ind in seller_lower for ind in self.spam_seller_indicators)
