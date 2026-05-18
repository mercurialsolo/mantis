"""Tests for issue #224 Phase 1 — overlay primitives + load_site_config.

Three additions exercised here:

1. :meth:`ExtractionSchema.overlay` — derived schema (from plan text)
   gets recipe overlay merged in. Recipe extends the spam / control
   lists, derived schema body wins.
2. :meth:`SiteConfig.overlay` — probe-derived URL patterns get recipe
   overlay merged in. Recipe wins on URL/pagination patterns; derived
   gate prompt preserved.
3. :func:`recipes.load_site_config` — symmetric to ``load_schema``;
   returns ``None`` when a recipe has no ``site_config.py`` (unlike
   ``load_schema`` which raises for a missing schema).

These primitives are additive — they don't change behavior for
existing callers that don't invoke them. The full migration to
derive-first wiring lands in Phase 2/3/4 (see issue #224).
"""

from __future__ import annotations

import pytest

from mantis_agent import recipes
from mantis_agent.extraction import ExtractionSchema
from mantis_agent.graph.objective import ObjectiveSpec, OutputField
from mantis_agent.site_config import SiteConfig


# ── ExtractionSchema.overlay ─────────────────────────────────────────────


def _derived_property_schema() -> ExtractionSchema:
    """A representative derived-from-plan schema (no spam tokens — those
    are the empirical accumulations the recipe overlay supplies)."""
    return ExtractionSchema.from_objective(
        ObjectiveSpec(
            raw_text="Find houses for sale and capture address + price",
            domains=["zillow.com"],
            target_entity="property listing",
            forbidden_actions=["Contact Agent"],
            allowed_reveal_actions=["Show more"],
            output_schema=[
                OutputField(name="address", required=True, example="123 Main St"),
                OutputField(name="price", required=True, example="$450,000"),
            ],
        )
    )


def test_overlay_with_none_is_noop() -> None:
    """Callers do ``derived.overlay(recipes.load_site_config(name))`` and
    pass ``None`` when no recipe is named. Must round-trip cleanly."""
    derived = _derived_property_schema()
    result = derived.overlay(None)
    assert result is derived


def test_overlay_unions_spam_indicators() -> None:
    """Recipe extends the derived spam list — derived order preserved
    and recipe items appended deduplicated. The derived schema can be
    empty (as it usually is — the plan text rarely names dealer
    tokens) but must NOT replace the recipe's accreted vocabulary."""
    derived = ExtractionSchema(
        entity_name="property listing",
        spam_indicators=["sponsored"],
    )
    recipe = ExtractionSchema(
        spam_indicators=["sponsored", "dealer", "broker"],
    )
    result = derived.overlay(recipe)
    assert result.spam_indicators == ["sponsored", "dealer", "broker"]


def test_overlay_unions_seller_indicators() -> None:
    derived = ExtractionSchema(spam_seller_indicators=[])
    recipe = ExtractionSchema(spam_seller_indicators=["llc", "inc"])
    result = derived.overlay(recipe)
    assert result.spam_seller_indicators == ["llc", "inc"]


def test_overlay_unions_forbidden_and_allowed_controls() -> None:
    """Lead-form / reveal vocabularies — recipe extends derived. The
    plan text often enumerates these (``Do NOT click X``); the recipe
    accretes synonyms found in production runs."""
    derived = ExtractionSchema(
        forbidden_controls=["Contact Seller"],
        allowed_controls=["Show more"],
    )
    recipe = ExtractionSchema(
        forbidden_controls=["Request Info", "Contact Seller"],  # dup
        allowed_controls=["Show phone", "Show email"],
    )
    result = derived.overlay(recipe)
    assert result.forbidden_controls == [
        "Contact Seller",
        "Request Info",
    ]
    assert result.allowed_controls == [
        "Show more",
        "Show phone",
        "Show email",
    ]


def test_overlay_preserves_derived_fields_and_required() -> None:
    """The plan text is the source of truth for *what* is being
    extracted. Recipe-side ``fields`` / ``required_fields`` MUST NOT
    silently override the derived shape — that would re-introduce the
    boattrader-specific ``year, make`` regression #224 calls out."""
    derived = _derived_property_schema()
    recipe = ExtractionSchema(
        fields=[{"name": "year", "type": "str", "required": True, "example": "2020"}],
        required_fields=["year", "make"],
    )
    result = derived.overlay(recipe)
    assert [f["name"] for f in result.fields] == ["address", "price"]
    assert result.required_fields == ["address", "price"]


def test_overlay_recipe_entity_name_overrides_default() -> None:
    """Default ``entity_name == 'listing'`` is the sentinel for "no
    derive opinion"; recipe wins in that case."""
    derived = ExtractionSchema()  # entity_name="listing" (the default)
    recipe = ExtractionSchema(entity_name="boat listing")
    result = derived.overlay(recipe)
    assert result.entity_name == "boat listing"


def test_overlay_keeps_derived_entity_name_when_set() -> None:
    """Once the derive step has named the entity (anything but the
    default sentinel), the recipe MUST NOT clobber it."""
    derived = ExtractionSchema(entity_name="property listing")
    recipe = ExtractionSchema(entity_name="boat listing")
    result = derived.overlay(recipe)
    assert result.entity_name == "property listing"


def test_overlay_recipe_spam_label_overrides_default() -> None:
    derived = ExtractionSchema()  # spam_label="dealer/spam"
    recipe = ExtractionSchema(spam_label="recruiter")
    result = derived.overlay(recipe)
    assert result.spam_label == "recruiter"


def test_overlay_keeps_derived_spam_label_when_set() -> None:
    derived = ExtractionSchema(spam_label="recruiter")
    recipe = ExtractionSchema(spam_label="dealer")
    result = derived.overlay(recipe)
    assert result.spam_label == "recruiter"


def test_load_site_config_staff_crm_exposes_filter_url_strategies() -> None:
    """Smoke: the staff_crm recipe ships a SiteConfig with the
    keyword → query-string filter mapping that GraphLearner /
    PlanEnhancer consume to emit direct navigates for filter steps
    (workaround for fixtures whose sidebar onclick is non-functional
    or whose dropdown options Holo3 can't visually ground)."""
    cfg = recipes.load_site_config("staff_crm")
    assert cfg is not None
    # Status filters — every named LEAD VIEWS sidebar entry maps to a
    # status=<value> URL fragment.
    assert cfg.filter_url_strategies.get("contacted") == "status=Contacted"
    assert cfg.filter_url_strategies.get("qualified") == "status=Qualified"
    # Priority filters from the BY PRIORITY sidebar / dropdown.
    assert cfg.filter_url_strategies.get("high") == "priority=High"
    assert cfg.filter_url_strategies.get("critical") == "priority=Critical"
    # URL patterns for the lead surface.
    assert cfg.detail_page_pattern == r"/leads/\d+"


def test_overlay_with_marketplace_listings_recipe() -> None:
    """Smoke: end-to-end overlay from a derive-flavoured base into the
    actual ``marketplace_listings`` recipe. Confirms the recipe's
    accreted vocabulary (``marinemax``, ``brokerage``, …) reaches the
    final schema, while the derived schema body is preserved."""
    derived = _derived_property_schema()
    overlay = recipes.load_schema("marketplace_listings")
    merged = derived.overlay(overlay)

    # Derived shape preserved.
    assert merged.entity_name == "property listing"
    assert merged.required_fields == ["address", "price"]

    # Recipe vocabulary extended in.
    assert "marinemax" in merged.spam_indicators
    assert "brokerage" in merged.spam_seller_indicators


# ── SiteConfig.overlay ───────────────────────────────────────────────────


def test_site_overlay_with_none_is_noop() -> None:
    base = SiteConfig(domain="example.com", detail_page_pattern=r"/listing/\d+")
    assert base.overlay(None) is base


def test_site_overlay_recipe_wins_on_url_patterns() -> None:
    """Probe sees the landing URL but can only guess at the detail-URL
    slug shape; the recipe knows it for sure."""
    base = SiteConfig(domain="boattrader.com", results_page_pattern=r"/boats/")
    overlay = SiteConfig(detail_page_pattern=r"/boat/[\w-]+")
    result = base.overlay(overlay)
    assert result.domain == "boattrader.com"
    assert result.detail_page_pattern == r"/boat/[\w-]+"
    assert result.results_page_pattern == r"/boats/"


def test_site_overlay_pagination_recipe_overrides_when_set() -> None:
    base = SiteConfig(pagination_format="page={n}", pagination_type="query_param")
    overlay = SiteConfig(
        pagination_format="/page-{n}/",
        pagination_type="path_suffix",
        pagination_strip_pattern=r"/page-\d+/?$",
    )
    result = base.overlay(overlay)
    assert result.pagination_format == "/page-{n}/"
    assert result.pagination_type == "path_suffix"
    assert result.pagination_strip_pattern == r"/page-\d+/?$"


def test_site_overlay_empty_recipe_field_keeps_probe_value() -> None:
    """Recipe's empty string means "no opinion" — the probe-derived
    value stays."""
    base = SiteConfig(
        detail_page_pattern=r"/probe-detail/\d+",
        pagination_format="page={n}",
    )
    overlay = SiteConfig()  # all defaults / empty
    result = base.overlay(overlay)
    assert result.detail_page_pattern == r"/probe-detail/\d+"
    assert result.pagination_format == "page={n}"


def test_site_overlay_gate_prompt_derived_wins() -> None:
    """Gate prompts come from plan text via the decompose stage —
    recipe should not silently override unless explicitly set."""
    base = SiteConfig(gate_verify_prompt="Plan-derived gate prompt")
    overlay = SiteConfig()  # empty gate prompt — no opinion
    result = base.overlay(overlay)
    assert result.gate_verify_prompt == "Plan-derived gate prompt"


def test_site_overlay_gate_prompt_recipe_overrides_when_set() -> None:
    base = SiteConfig()  # empty gate
    overlay = SiteConfig(gate_verify_prompt="Recipe-tuned gate prompt")
    result = base.overlay(overlay)
    assert result.gate_verify_prompt == "Recipe-tuned gate prompt"


def test_site_overlay_prefer_som_or_combines() -> None:
    """``prefer_som_grounding`` is OR-style — either the probe inferred
    a SoM-friendly DOM or the recipe declared one; both are evidence
    the runner can promote SoM grounding."""
    base = SiteConfig(prefer_som_grounding=False)
    overlay = SiteConfig(prefer_som_grounding=True)
    assert base.overlay(overlay).prefer_som_grounding is True

    base2 = SiteConfig(prefer_som_grounding=True)
    overlay2 = SiteConfig(prefer_som_grounding=False)
    assert base2.overlay(overlay2).prefer_som_grounding is True


def test_site_overlay_independent_grounding_recipe_wins_when_set() -> None:
    base = SiteConfig(require_independent_grounding=("listing-card-title",))
    overlay = SiteConfig(require_independent_grounding=("recipe-card",))
    assert base.overlay(overlay).require_independent_grounding == ("recipe-card",)


def test_site_overlay_independent_grounding_keeps_base_when_recipe_empty() -> None:
    base = SiteConfig(require_independent_grounding=("base-tag",))
    overlay = SiteConfig()  # empty tuple
    assert base.overlay(overlay).require_independent_grounding == ("base-tag",)


def test_site_overlay_with_marketplace_listings() -> None:
    """End-to-end: probe-derived base + recipe overlay produces the
    expected merged config for boat-trader-style sites."""
    base = SiteConfig(domain="boattrader.com")  # only domain known from probe
    overlay = recipes.load_site_config("marketplace_listings")
    assert overlay is not None  # recipe ships SITE_CONFIG
    merged = base.overlay(overlay)
    assert merged.detail_page_pattern == r"/boat/[\w-]+"
    assert merged.pagination_format == "/page-{n}/"
    assert merged.filtered_results_url


# ── recipes.load_site_config ─────────────────────────────────────────────


def test_load_site_config_returns_config_for_marketplace_listings() -> None:
    config = recipes.load_site_config("marketplace_listings")
    assert config is not None
    assert isinstance(config, SiteConfig)
    assert config.detail_page_pattern == r"/boat/[\w-]+"


def test_load_site_config_returns_none_when_recipe_lacks_site_config(
    tmp_path, monkeypatch,
) -> None:
    """A recipe with a schema but no site_config.py must yield None
    (no exception). Construct an ad-hoc importable package on disk
    that has only ``schema.py`` to avoid touching the real recipes
    directory."""
    import sys

    pkg_root = tmp_path / "ad_hoc_recipes"
    pkg_root.mkdir()
    (pkg_root / "__init__.py").write_text("")
    sub = pkg_root / "halfway"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    (sub / "schema.py").write_text(
        "from mantis_agent.extraction import ExtractionSchema\n"
        "SCHEMA = ExtractionSchema(entity_name='halfway')\n"
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    # Import the ad-hoc package once so its name resolves in importlib.
    import importlib
    importlib.import_module("ad_hoc_recipes")
    importlib.import_module("ad_hoc_recipes.halfway")

    # Patch ``recipes.__name__`` lookup by re-pointing load_site_config
    # at our ad-hoc tree. Simpler: temporarily swap recipes.__name__.
    monkeypatch.setattr(recipes, "__name__", "ad_hoc_recipes")
    try:
        result = recipes.load_site_config("halfway")
    finally:
        # Cleanup imported modules so subsequent tests get a fresh state.
        for k in list(sys.modules):
            if k.startswith("ad_hoc_recipes"):
                sys.modules.pop(k, None)
    assert result is None


def test_load_site_config_raises_for_unknown_recipe() -> None:
    """Recipe directory truly missing — re-raise rather than swallow."""
    with pytest.raises(ModuleNotFoundError):
        recipes.load_site_config("definitely_not_a_real_recipe_name_xyz")
