"""Integration tests proving the generic extraction pipeline works for non-BoatTrader sites.

Tests the full pipeline offline (no browser, no API calls):
  ObjectiveSpec → ExtractionSchema → WorkflowGraph → GraphCompiler → MicroPlan

Validates that each site produces structurally correct plans with
appropriate section tags, gate flags, loop targets, and prompts.
"""

from mantis_agent.extraction import ClaudeExtractor, ExtractionResult, ExtractionSchema
from mantis_agent.graph import (
    GraphCompiler,
    ObjectiveSpec,
    OutputField,
    PhaseRole,
    RepeatMode,
    WorkflowGraph,
)
from mantis_agent.graph.learner import GraphLearner
from mantis_agent.graph.probe import ProbeResult
from mantis_agent.site_config import SiteConfig


def _build_pipeline(objective_text, domain, target_entity, output_fields, start_url=""):
    """Build the full pipeline for a given site and return (spec, schema, plan, config)."""
    spec = ObjectiveSpec(
        raw_text=objective_text,
        domains=[domain],
        start_url=start_url or f"https://www.{domain}/",
        target_entity=target_entity,
        output_schema=[OutputField(**f) for f in output_fields],
    )
    schema = ExtractionSchema.from_objective(spec)
    learner = GraphLearner()
    graph = learner._default_skeleton(spec, ProbeResult(estimated_listings_per_page=20))
    compiler = GraphCompiler()
    plan = compiler.compile(graph)
    return spec, schema, plan, graph


# ── Zillow (Real Estate) ──


def test_zillow_pipeline():
    spec, schema, plan, graph = _build_pipeline(
        "Find houses for sale in Miami FL under $500,000 with 3+ bedrooms",
        "zillow.com",
        "property listing",
        [
            {"name": "address", "type": "str", "required": True, "example": "123 Main St"},
            {"name": "price", "type": "str", "required": True, "example": "$450,000"},
            {"name": "beds", "type": "str", "required": False, "example": "3"},
            {"name": "baths", "type": "str", "required": False, "example": "2"},
            {"name": "sqft", "type": "str", "required": False, "example": "1,500"},
        ],
        start_url="https://www.zillow.com/homes/miami-fl/",
    )

    # Schema is correct
    assert schema.entity_name == "property listing"
    assert schema.required_fields == ["address", "price"]

    # Plan has correct structure
    assert len(plan.steps) >= 10
    setup = [s for s in plan.steps if s.section == "setup"]
    extraction = [s for s in plan.steps if s.section == "extraction"]
    pagination = [s for s in plan.steps if s.section == "pagination"]
    assert len(setup) >= 2
    assert len(extraction) >= 5
    assert len(pagination) >= 2

    # Gate exists
    gates = [s for s in plan.steps if s.gate]
    assert len(gates) == 1
    assert "property listing" in gates[0].verify.lower() or "filtered" in gates[0].verify.lower()

    # Loops exist
    loops = [s for s in plan.steps if s.type == "loop"]
    assert len(loops) == 2

    # Prompts are generic, not BoatTrader
    extractor = ClaudeExtractor(schema=schema)
    prompt = extractor._get_extract_prompt()
    assert "property listing" in prompt
    assert "address" in prompt
    assert "boat" not in prompt.lower()
    assert "boattrader" not in prompt.lower()


# ── Indeed (Job Board) ──


def test_indeed_pipeline():
    spec, schema, plan, graph = _build_pipeline(
        "Find senior software engineer jobs in San Francisco paying over $150k",
        "indeed.com",
        "job posting",
        [
            {"name": "title", "type": "str", "required": True, "example": "Senior Software Engineer"},
            {"name": "company", "type": "str", "required": True, "example": "Acme Corp"},
            {"name": "salary", "type": "str", "required": False, "example": "$150,000 - $200,000"},
            {"name": "location", "type": "str", "required": False, "example": "San Francisco, CA"},
        ],
        start_url="https://www.indeed.com/jobs?q=senior+software+engineer&l=San+Francisco",
    )

    assert schema.entity_name == "job posting"
    assert schema.required_fields == ["title", "company"]
    assert len(plan.steps) >= 10

    # Prompts say "job posting" not "boat"
    extractor = ClaudeExtractor(schema=schema)
    prompt = extractor._get_extract_prompt()
    assert "job posting" in prompt
    assert "title" in prompt
    assert "company" in prompt

    # find_all_listings prompt is generic
    listings_prompt = extractor._get_find_listings_prompt()
    assert "job posting" in listings_prompt
    assert "boat" not in listings_prompt.lower()


# ── Yelp (Local Business) ──


def test_yelp_pipeline():
    spec, schema, plan, graph = _build_pipeline(
        "Find Italian restaurants in NYC with 4+ stars",
        "yelp.com",
        "restaurant listing",
        [
            {"name": "name", "type": "str", "required": True, "example": "Tony's Pizzeria"},
            {"name": "rating", "type": "str", "required": False, "example": "4.5"},
            {"name": "price_range", "type": "str", "required": False, "example": "$$"},
            {"name": "phone", "type": "str", "required": False, "example": "(212) 555-1234"},
            {"name": "address", "type": "str", "required": False, "example": "123 Broadway, NY"},
        ],
        start_url="https://www.yelp.com/search?find_desc=italian&find_loc=New+York",
    )

    assert schema.entity_name == "restaurant listing"
    assert schema.required_fields == ["name"]
    assert len(plan.steps) >= 10

    extractor = ClaudeExtractor(schema=schema)
    prompt = extractor._get_extract_prompt()
    assert "restaurant listing" in prompt
    assert "name" in prompt
    assert "rating" in prompt


# ── Craigslist (Classifieds) ──


def test_craigslist_pipeline():
    spec, schema, plan, graph = _build_pipeline(
        "Find apartments for rent in SF under $3000/month",
        "craigslist.org",
        "apartment listing",
        [
            {"name": "title", "type": "str", "required": True, "example": "Sunny 1BR in Mission"},
            {"name": "price", "type": "str", "required": True, "example": "$2,500"},
            {"name": "location", "type": "str", "required": False, "example": "Mission District"},
            {"name": "beds", "type": "str", "required": False, "example": "1"},
        ],
        start_url="https://sfbay.craigslist.org/search/apa",
    )

    assert schema.entity_name == "apartment listing"
    assert schema.required_fields == ["title", "price"]

    # ExtractionResult with schema works for apartment data
    result = ExtractionResult(
        _schema=schema,
        extracted_fields={"title": "Sunny 1BR", "price": "$2,500", "location": "Mission"},
    )
    assert result.is_viable()
    assert "Title: Sunny 1BR" in result.to_summary()


# ── SiteConfig for different sites ──


def test_site_config_per_domain():
    """Each domain gets appropriate URL patterns."""
    bt = SiteConfig.default_boattrader()
    assert bt.is_detail_page("https://boattrader.com/boat/2020-sea-ray-240/")
    assert not bt.is_detail_page("https://boattrader.com/boats/by-owner/")

    zillow = SiteConfig(
        domain="zillow.com",
        detail_page_pattern=r"/homes/\d+_zpid",
        results_page_pattern=r"/homes/",
    )
    assert zillow.is_detail_page("https://zillow.com/homes/12345_zpid/")
    assert zillow.is_results_page("https://zillow.com/homes/miami-fl/")

    indeed = SiteConfig(
        domain="indeed.com",
        detail_page_pattern=r"/viewjob",
        results_page_pattern=r"/jobs\?",
    )
    assert indeed.is_detail_page("https://indeed.com/viewjob?jk=abc123")
    assert indeed.is_results_page("https://indeed.com/jobs?q=engineer")


# ── Backward compatibility ──


def test_boattrader_pipeline_unchanged():
    """BoatTrader pipeline still works identically."""
    spec, schema, plan, graph = _build_pipeline(
        "Search BoatTrader for private seller boats near Miami",
        "boattrader.com",
        "boat listing",
        [
            {"name": "year", "type": "str", "required": True, "example": "2020"},
            {"name": "make", "type": "str", "required": True, "example": "Sea Ray"},
            {"name": "model", "type": "str", "required": False, "example": "240 Sundeck"},
            {"name": "price", "type": "str", "required": False, "example": "$42,500"},
            {"name": "phone", "type": "str", "required": False, "example": "305-555-1234"},
        ],
        start_url="https://www.boattrader.com/boats/by-owner/",
    )

    assert schema.entity_name == "boat listing"
    assert schema.required_fields == ["year", "make"]
    assert len(plan.steps) == 11  # same as extract_url_filtered.json

    # Default (no schema) extractor still uses BoatTrader prompts
    legacy = ClaudeExtractor()
    assert "boat listing" in legacy._get_extract_prompt().lower()


def test_no_schema_default_extractor():
    """ClaudeExtractor() with no args is backward compatible."""
    ext = ClaudeExtractor()
    assert ext.schema is None
    prompt = ext._get_extract_prompt()
    assert "boat listing" in prompt.lower()
    multi = ext._get_multi_extract_prompt()
    assert "BoatTrader" in multi
