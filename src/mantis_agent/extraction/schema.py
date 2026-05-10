"""Domain-agnostic extraction schema.

Describes what fields to pull, how to detect spam, and what counts as a
viable extraction. Concrete vertical schemas (boat-listing, job-posting,
real-estate) live under ``mantis_agent.recipes.<name>.schema`` and use
``ExtractionSchema`` as their carrier dataclass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExtractionContext(str, Enum):
    """Where the extractor is reading from — drives which required-field
    contract to enforce.

    Issue #236: ``required_fields`` enforced uniformly is the dominant
    failure mode on listings sites. Search-result tiles often render
    canonical fields (year, make, ...) only in image alt-text or
    JSON-LD that vision-mode extraction can't see; rejecting every
    tile-mode row that lacks them halts the run before any leads are
    captured. Tagging context lets the schema apply a strict contract
    on the canonical detail page and a looser one on tiles.

    The string-enum shape lets callers pass either the enum or the
    raw string (matching the JSON-serialisable plan-step shape).
    """

    SEARCH_TILE = "search_tile"   # reading a row from a list of cards
    DETAIL_PAGE = "detail_page"   # reading the canonical entity page
    UNKNOWN = "unknown"           # default — preserves current behavior


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
    # Issue #236: per-context required-field contracts. ``required_fields``
    # is the strict DETAIL_PAGE contract (canonical entity fields the
    # detail page must surface). ``tile_required_fields`` is the looser
    # SEARCH_TILE contract — typically just ``["url"]`` so the runner can
    # keep the row to drive a follow-up navigate-into-detail. Empty list
    # ``[]`` opts out of the split (uses ``required_fields`` for both
    # contexts), preserving existing recipe behavior.
    tile_required_fields: list[str] = field(default_factory=list)
    # Informational hint: which fields the search tile is expected to
    # surface. The runner uses this to decide what NOT to re-read on
    # the detail page (carry from tile → enrich detail). Recipe-side
    # only; framework primitives don't enforce it. Empty list = no
    # carry (today's behavior).
    tile_carry_fields: list[str] = field(default_factory=list)
    spam_indicators: list[str] = field(default_factory=list)
    spam_seller_indicators: list[str] = field(default_factory=list)
    spam_label: str = "dealer/spam"  # what to call spam (e.g. "dealer", "recruiter")
    forbidden_controls: list[str] = field(default_factory=list)  # "Contact Seller", etc.
    allowed_controls: list[str] = field(default_factory=list)  # "Show more", "Show phone"
    # Issue #246: recipe-rejection → host-facing intent map. Keys are
    # canonical rejection-reason tokens (``"dealer"``,
    # ``"incomplete_required"``, ``"parse_error"``…) that the runner
    # tags every rejection with. Values are intent strings the host
    # can branch on:
    #   ``"skip"``         — terminal-for-this-row; host should mark the
    #                        listing as processed-but-skipped and advance
    #                        to the next, not retry
    #   ``"extract_more"`` — caller could enrich on the detail page;
    #                        runner stays in extraction loop
    #   ``"retry"``        — transient; safe to retry the same step
    # Empty default preserves today's behavior (no skip-semantic
    # surfaces; every rejection looks like a generic step failure).
    # Recipes opt in by setting their canonical rejections explicitly.
    rejection_intents: dict[str, str] = field(default_factory=dict)

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
            tile_required_fields=list(getattr(objective, "tile_required_fields", []) or []),
            tile_carry_fields=list(getattr(objective, "tile_carry_fields", []) or []),
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

    def overlay(self, other: ExtractionSchema | None) -> ExtractionSchema:
        """Merge a recipe overlay into ``self`` (the derived base).

        Issue #224 Phase 1: ``self`` is the derive-first schema produced
        from plan text via :meth:`from_objective`; ``other`` is the
        optional production-hardened recipe carrying empirical tokens
        (dealer indicators, controls vocabulary) accumulated from real
        runs. The merge rules favour the derived shape for *what* to
        extract, the recipe for *how* to defend against site-specific
        noise:

        - **List fields** (``spam_indicators``, ``spam_seller_indicators``,
          ``forbidden_controls``, ``allowed_controls``) — overlay UNION
          (recipe extends the derived list, deduped, derived ordering
          preserved). The recipe was authored by accreting empirical
          tokens; dropping derived tokens would discard the plan-text
          signal.
        - **Scalar fields** (``entity_name``, ``spam_label``) — recipe
          wins when it sets a non-default value, otherwise derived
          stays. Default sentinels: ``entity_name == "listing"``,
          ``spam_label == "dealer/spam"``.
        - **Schema body** (``fields``, ``required_fields``) — derived
          wins. The plan text is the source of truth for *what* is
          being extracted; recipes don't override that. Recipes that
          want to override must do it via the explicit
          ``ExtractionSchema(...)`` constructor before overlay.

        Passing ``other=None`` is a no-op — returns ``self`` unchanged
        so callers can write::

            schema = ExtractionSchema.from_objective(spec).overlay(
                recipes.load_schema(name) if name else None
            )

        without an outer guard.
        """
        if other is None:
            return self

        def _union(base: list, ext: list) -> list:
            seen: set = set()
            merged: list = []
            for item in (*base, *ext):
                if item in seen:
                    continue
                seen.add(item)
                merged.append(item)
            return merged

        # Sentinel-aware scalar pick: keep derived once it has departed
        # from the dataclass default; otherwise take recipe's value.
        # This makes "default" mean "no opinion" so the recipe can fill
        # it in, but keeps a derived non-default value sticky against
        # the recipe.
        entity = (
            self.entity_name
            if self.entity_name and self.entity_name != "listing"
            else (other.entity_name or self.entity_name)
        )
        spam_lbl = (
            self.spam_label
            if self.spam_label and self.spam_label != "dealer/spam"
            else (other.spam_label or self.spam_label)
        )
        # Tile-context fields (#236): derived schema body wins for
        # ``tile_required_fields`` (the plan text owns the tile contract
        # the same way it owns ``required_fields``). For
        # ``tile_carry_fields`` — informational only — recipe extends
        # derived (recipes accumulate empirical knowledge of which
        # tile fields actually render across the marketplace's
        # listing variants).
        # ``rejection_intents``: derived keys win on conflict
        # (operator-authored override of recipe defaults), recipe
        # extends with new keys (recipes accumulate empirical
        # vertical knowledge of which rejections are terminal vs
        # retryable). Same shape as the dict-merge primitive used
        # for tile_carry_fields list union, just dict-flavoured.
        merged_intents = dict(other.rejection_intents)
        merged_intents.update(self.rejection_intents)

        return ExtractionSchema(
            entity_name=entity,
            fields=self.fields,
            required_fields=self.required_fields,
            tile_required_fields=(
                self.tile_required_fields
                if self.tile_required_fields
                else other.tile_required_fields
            ),
            tile_carry_fields=_union(
                self.tile_carry_fields, other.tile_carry_fields,
            ),
            spam_indicators=_union(self.spam_indicators, other.spam_indicators),
            spam_seller_indicators=_union(
                self.spam_seller_indicators, other.spam_seller_indicators,
            ),
            spam_label=spam_lbl,
            forbidden_controls=_union(self.forbidden_controls, other.forbidden_controls),
            allowed_controls=_union(self.allowed_controls, other.allowed_controls),
            rejection_intents=merged_intents,
        )
