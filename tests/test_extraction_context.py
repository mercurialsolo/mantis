"""Tests for issue #236 — tile vs detail-page extraction context.

The dominant failure mode on listings sites: ``required_fields``
enforced uniformly rejects every search-tile row when the tile only
surfaces ``url`` / ``title`` / ``price`` (canonical fields like
``year``/``make`` live in alt-text or JSON-LD, invisible to vision
extraction). The recipe-level fix tags context so the strict contract
applies only on detail-page extractions; tile-mode falls back to a
looser ``tile_required_fields`` set.

These tests pin:

- :class:`ExtractionContext` enum surface
- :meth:`ExtractionResult.missing_required_reason` — context-aware
  field selection, fallback to required_fields when tile contract
  unset, legacy UNKNOWN preserves prior behavior
- :class:`ExtractionSchema` overlay merges ``tile_required_fields``
  / ``tile_carry_fields`` cleanly
- ``marketplace_listings`` recipe ships the new fields
- ``_resolve_extraction_context`` helper picks the right enum
  from a runner / SiteConfig
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent import recipes
from mantis_agent.extraction import (
    ExtractionContext,
    ExtractionResult,
    ExtractionSchema,
)
from mantis_agent.gym.step_handlers.claude_step import _resolve_extraction_context
from mantis_agent.site_config import SiteConfig


# ── ExtractionContext enum ───────────────────────────────────────────────


def test_extraction_context_enum_values() -> None:
    """The string-enum shape lets callers pass either the enum or the
    raw string — important when the context flows through JSON-shaped
    plan steps."""
    assert ExtractionContext.SEARCH_TILE == "search_tile"
    assert ExtractionContext.DETAIL_PAGE == "detail_page"
    assert ExtractionContext.UNKNOWN == "unknown"


# ── missing_required_reason context routing ─────────────────────────────


def _row(**kwargs: str) -> ExtractionResult:
    schema = kwargs.pop("schema", None)
    fields = kwargs
    result = ExtractionResult(extracted_fields=fields, _schema=schema)
    return result


def test_detail_page_enforces_required_fields() -> None:
    """Detail-page context uses the strict ``required_fields`` set —
    today's behavior. Missing year on a detail page → rejected."""
    schema = ExtractionSchema(
        required_fields=["year", "make"],
        tile_required_fields=["url"],
    )
    row = _row(schema=schema, url="https://x.test/boat/123", make="Sea Ray")
    reason = row.missing_required_reason(ExtractionContext.DETAIL_PAGE)
    assert "year" in reason


def test_search_tile_enforces_tile_required_fields_when_set() -> None:
    """Tile context uses ``tile_required_fields`` when defined — a row
    with ``url`` only passes even though ``year``/``make`` are missing."""
    schema = ExtractionSchema(
        required_fields=["year", "make"],
        tile_required_fields=["url"],
    )
    row = _row(schema=schema, url="https://x.test/boats/?page=1")
    reason = row.missing_required_reason(ExtractionContext.SEARCH_TILE)
    assert reason == ""


def test_search_tile_falls_back_to_required_when_tile_unset() -> None:
    """Recipes that don't opt into the split (no ``tile_required_fields``)
    enforce ``required_fields`` for both contexts — preserves legacy
    behavior."""
    schema = ExtractionSchema(
        required_fields=["year", "make"],
        # tile_required_fields=[] — empty (default)
    )
    row = _row(schema=schema, url="https://x.test/boats/", title="Sea Ray")
    reason = row.missing_required_reason(ExtractionContext.SEARCH_TILE)
    # No url-only escape — falls back to required_fields strict check.
    assert "year" in reason and "make" in reason


def test_unknown_context_enforces_required_fields() -> None:
    """``UNKNOWN`` (default for legacy callers) preserves today's
    behavior — strict ``required_fields`` regardless of page kind."""
    schema = ExtractionSchema(
        required_fields=["year", "make"],
        tile_required_fields=["url"],
    )
    row = _row(schema=schema, url="https://x.test/boat/")
    reason = row.missing_required_reason(ExtractionContext.UNKNOWN)
    assert "year" in reason and "make" in reason


def test_default_argument_is_unknown_for_backwards_compat() -> None:
    """Existing callers that don't pass a context get UNKNOWN — same
    behavior they had before #236."""
    schema = ExtractionSchema(
        required_fields=["year", "make"],
        tile_required_fields=["url"],
    )
    row = _row(schema=schema, url="https://x.test/boat/")
    # No context arg — default UNKNOWN — strict enforcement.
    reason = row.missing_required_reason()
    assert "year" in reason


def test_tile_search_passes_with_url_only() -> None:
    """End-to-end tile case: row has just ``url`` from the listing
    card; tile contract requires only ``url``; gate passes."""
    schema = ExtractionSchema(
        required_fields=["year", "make", "price"],
        tile_required_fields=["url"],
    )
    row = _row(schema=schema, url="https://boats.test/listing/123")
    assert row.missing_required_reason(ExtractionContext.SEARCH_TILE) == ""


# ── ExtractionSchema overlay merges new fields ──────────────────────────


def test_overlay_recipe_extends_tile_carry_fields() -> None:
    """``tile_carry_fields`` is informational accumulation — recipe
    extends derived (deduped union)."""
    derived = ExtractionSchema(
        entity_name="property listing",
        tile_carry_fields=["url", "title"],
    )
    recipe = ExtractionSchema(
        tile_carry_fields=["url", "price", "address"],
    )
    merged = derived.overlay(recipe)
    assert merged.tile_carry_fields == ["url", "title", "price", "address"]


def test_overlay_keeps_derived_tile_required_fields_when_set() -> None:
    """Like ``required_fields``, derived owns ``tile_required_fields``
    once it has a value — recipe doesn't clobber."""
    derived = ExtractionSchema(tile_required_fields=["url", "title"])
    recipe = ExtractionSchema(tile_required_fields=["url"])
    merged = derived.overlay(recipe)
    assert merged.tile_required_fields == ["url", "title"]


def test_overlay_recipe_fills_in_when_derived_tile_required_empty() -> None:
    """When derived has no opinion on tile contract, recipe fills it in."""
    derived = ExtractionSchema()  # no tile_required_fields
    recipe = ExtractionSchema(tile_required_fields=["url"])
    merged = derived.overlay(recipe)
    assert merged.tile_required_fields == ["url"]


# ── marketplace_listings recipe ships the new fields ────────────────────


def test_marketplace_listings_has_tile_required_fields() -> None:
    schema = recipes.load_schema("marketplace_listings")
    assert schema.tile_required_fields == ["url"]
    # Strict required_fields preserved for detail-page mode.
    assert "year" in schema.required_fields
    assert "make" in schema.required_fields


def test_marketplace_listings_has_tile_carry_fields() -> None:
    schema = recipes.load_schema("marketplace_listings")
    assert "url" in schema.tile_carry_fields
    assert "title" in schema.tile_carry_fields
    assert "price" in schema.tile_carry_fields


def test_marketplace_recipe_tile_extraction_does_not_reject_url_only_row() -> None:
    """End-to-end against the real recipe: a tile row with just url
    must NOT be rejected (this is exactly the scenario that halted
    the BoatTrader → PopYachts plan run before this fix)."""
    schema = recipes.load_schema("marketplace_listings")
    row = _row(schema=schema, url="https://boattrader.com/boat/abc")
    assert row.missing_required_reason(ExtractionContext.SEARCH_TILE) == ""


def test_marketplace_recipe_detail_extraction_still_strict() -> None:
    """Same recipe, detail-page context: strict canonical fields
    required. A row missing year is rejected — today's behavior
    preserved for the detail-page case."""
    schema = recipes.load_schema("marketplace_listings")
    row = _row(
        schema=schema, url="https://boattrader.com/boat/abc",
        make="Sea Ray", model="240 Sundeck",
    )
    reason = row.missing_required_reason(ExtractionContext.DETAIL_PAGE)
    assert "year" in reason


# ── _resolve_extraction_context helper ──────────────────────────────────


def _runner_with_site(site_config: SiteConfig | None, last_url: str = "") -> MagicMock:
    runner = MagicMock()
    runner.site_config = site_config
    runner._last_known_url = last_url
    return runner


def test_resolve_context_returns_unknown_when_no_site_config() -> None:
    runner = _runner_with_site(None)
    data = MagicMock(url="https://x.test/")
    assert _resolve_extraction_context(runner, data) == ExtractionContext.UNKNOWN


def test_resolve_context_classifies_detail_page_from_url() -> None:
    """SiteConfig.is_detail_page returns True → DETAIL_PAGE."""
    site = SiteConfig(
        domain="boattrader.com",
        detail_page_pattern=r"/boat/[\w-]+",
    )
    runner = _runner_with_site(site)
    data = MagicMock(url="https://boattrader.com/boat/2020-sea-ray-240/")
    assert _resolve_extraction_context(runner, data) == ExtractionContext.DETAIL_PAGE


def test_resolve_context_classifies_search_tile_from_url() -> None:
    """SiteConfig.is_results_page returns True → SEARCH_TILE."""
    site = SiteConfig(
        domain="boattrader.com",
        results_page_pattern=r"/boats/",
    )
    runner = _runner_with_site(site)
    data = MagicMock(url="https://boattrader.com/boats/by-owner/")
    assert _resolve_extraction_context(runner, data) == ExtractionContext.SEARCH_TILE


def test_resolve_context_falls_back_to_runner_url_when_data_url_missing() -> None:
    """Some extraction paths produce a result without a URL — fall
    back to ``runner._last_known_url`` so the classification still
    fires."""
    site = SiteConfig(
        domain="boattrader.com",
        detail_page_pattern=r"/boat/[\w-]+",
    )
    runner = _runner_with_site(
        site, last_url="https://boattrader.com/boat/2020-x/",
    )
    # data has empty url; should use runner's _last_known_url.
    data = MagicMock(url="")
    assert _resolve_extraction_context(runner, data) == ExtractionContext.DETAIL_PAGE


def test_resolve_context_returns_unknown_on_neither_match() -> None:
    """URL doesn't match either pattern (e.g. login page, dashboard)
    — UNKNOWN, falls back to legacy strict enforcement."""
    site = SiteConfig(
        domain="boattrader.com",
        detail_page_pattern=r"/boat/[\w-]+",
        results_page_pattern=r"/boats/",
    )
    runner = _runner_with_site(site)
    data = MagicMock(url="https://boattrader.com/login")
    assert _resolve_extraction_context(runner, data) == ExtractionContext.UNKNOWN


def test_resolve_context_swallows_site_config_exceptions() -> None:
    """Defensive: a malformed site config that raises on URL check
    must not break the extraction path. Fall through to UNKNOWN."""
    site = MagicMock()
    site.is_detail_page.side_effect = ValueError("bad regex")
    site.is_results_page.side_effect = ValueError("bad regex")
    runner = _runner_with_site(site, last_url="https://x.test/")
    data = MagicMock(url="https://x.test/")
    assert _resolve_extraction_context(runner, data) == ExtractionContext.UNKNOWN
