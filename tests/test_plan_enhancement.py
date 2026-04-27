"""Tests for the plan enhancement pipeline."""

from mantis_agent.graph import GraphCompiler, PlanValidator
from mantis_agent.graph.enhancer import PlanEnhancer
from mantis_agent.graph.section_decomposer import SectionDecomposer, ExecutionSection
from mantis_agent.graph.objective import ObjectiveSpec, OutputField
from mantis_agent.graph.probe import ProbeResult
from mantis_agent.graph.graph import WorkflowGraph, PhaseRole, RepeatMode
from mantis_agent.verification.playbook import Playbook


def _boattrader_probe() -> ProbeResult:
    return ProbeResult(
        url="https://www.boattrader.com/boats/",
        domain="boattrader.com",
        page_type="search_results",
        filters_detected=[
            {"name": "Seller Type", "options": ["Private Seller"], "location": "URL path: by-owner"},
        ],
        listing_container={"description": "cards with photo + title + price"},
        pagination_controls={"type": "numbered"},
        estimated_listings_per_page=25,
        detail_page_pattern={
            "expandable_sections": ["Description", "More Details"],
            "expand_controls": ["Show", "Read more"],
            "phone_location": "In Description section",
        },
    )


def _build_enhanced_plan(objective_text, domain, filters, probe=None):
    spec = ObjectiveSpec(
        raw_text=objective_text,
        domains=[domain],
        start_url=f"https://www.{domain}/",
        target_entity="listing",
        required_filters=filters,
    )
    probe = probe or ProbeResult(domain=domain)
    enhancer = PlanEnhancer()
    enhancement = enhancer._enhance_heuristic(spec, probe)
    phases, edges = enhancer.build_enhanced_phases(spec, probe, enhancement)
    graph = WorkflowGraph(
        objective=spec, phases=phases, edges=edges,
        playbook=Playbook(domain=domain, listings_per_page=20),
        domain=domain, objective_hash=spec.objective_hash,
    )
    compiler = GraphCompiler()
    plan = compiler.compile(graph)
    return spec, plan, phases


# ── PlanEnhancer ──


def test_enhancer_heuristic_detects_sidebar_filters():
    spec = ObjectiveSpec(raw_text="test", required_filters=["private seller", "zip"])
    probe = ProbeResult(
        filters_detected=[
            {"name": "Seller", "options": ["Private Seller"], "location": "sidebar"},
        ],
    )
    enhancer = PlanEnhancer()
    result = enhancer._enhance_heuristic(spec, probe)
    strategies = result["filter_strategy"]
    seller_strat = [s for s in strategies if s["filter"] == "private seller"]
    assert seller_strat[0]["method"] == "sidebar"


def test_enhancer_builds_concrete_phases():
    spec = ObjectiveSpec(
        raw_text="test", domains=["example.com"],
        start_url="https://example.com/search",
        target_entity="job posting",
        required_filters=["remote"],
    )
    probe = ProbeResult(domain="example.com")
    enhancer = PlanEnhancer()
    enhancement = enhancer._enhance_heuristic(spec, probe)
    phases, edges = enhancer.build_enhanced_phases(spec, probe, enhancement)

    assert "navigate" in phases
    assert "verify_scope" in phases
    assert phases["verify_scope"].gate is True
    assert "job posting" in phases["verify_scope"].intent_template


def test_enhancer_url_filters_skip_sidebar_steps():
    """When all filters are URL-encoded, no sidebar filter steps are created."""
    spec = ObjectiveSpec(
        raw_text="test", domains=["example.com"],
        start_url="https://example.com/results?type=private",
        required_filters=["private"],
    )
    probe = ProbeResult(domain="example.com")
    enhancer = PlanEnhancer()
    # Simulate: the enhancer API returns url-method filters
    enhancement = {
        "navigation_url": "https://example.com/results?type=private",
        "filter_strategy": [{"filter": "private", "method": "url", "detail": "in URL query"}],
        "card_description": "",
        "detail_scrolls": 3,
        "expandable_sections": [],
        "expand_controls": [],
        "phone_location": "",
        "pagination_method": "button_click",
        "pagination_detail": "",
    }
    phases, edges = enhancer.build_enhanced_phases(spec, probe, enhancement)

    # Should have navigate with filtered URL, no filter_N steps
    assert "filtered results" in phases["navigate"].intent_template
    filter_steps = [pid for pid in phases if pid.startswith("filter_")]
    assert len(filter_steps) == 0


def test_enhancer_detail_page_knowledge():
    spec = ObjectiveSpec(raw_text="test", required_filters=[])
    probe = _boattrader_probe()
    enhancer = PlanEnhancer()
    enhancement = enhancer._enhance_heuristic(spec, probe)

    assert enhancement["expandable_sections"] == ["Description", "More Details"]
    assert enhancement["phone_location"] == "In Description section"


# ── SectionDecomposer ──


def test_section_decomposer_creates_sections():
    _, plan, phases = _build_enhanced_plan("test", "example.com", ["filter1", "filter2"])
    decomposer = SectionDecomposer()
    sections = decomposer.decompose(phases)

    section_names = [s.name for s in sections]
    assert "setup" in section_names
    assert "gate" in section_names
    assert "extraction" in section_names
    assert "pagination" in section_names


def test_section_dependencies():
    _, plan, phases = _build_enhanced_plan("test", "example.com", ["filter1"])
    decomposer = SectionDecomposer()
    sections = decomposer.decompose(phases)

    gate = [s for s in sections if s.name == "gate"][0]
    extraction = [s for s in sections if s.name == "extraction"][0]
    pagination = [s for s in sections if s.name == "pagination"][0]

    # Gate depends on setup
    assert any("setup" in d for d in gate.depends_on)
    # Extraction depends on gate
    assert "gate" in extraction.depends_on
    # Pagination depends on extraction
    assert "extraction" in pagination.depends_on


def test_extraction_section_is_loop():
    _, plan, phases = _build_enhanced_plan("test", "example.com", [])
    decomposer = SectionDecomposer()
    sections = decomposer.decompose(phases)

    extraction = [s for s in sections if s.name == "extraction"][0]
    assert extraction.is_loop is True


# ── Full pipeline validation ──


def test_boattrader_enhanced_plan_validates_clean():
    text = open("plans/boattrader/extract_only.txt").read()
    spec, plan, phases = _build_enhanced_plan(
        text, "boattrader.com",
        ["private seller", "by owner", "zip", "price", "location"],
        probe=_boattrader_probe(),
    )
    validator = PlanValidator()
    issues = validator.validate(plan, objective=spec)
    assert len(issues) == 0, f"Expected 0 issues, got: {[i.code for i in issues]}"


def test_enhanced_plan_has_correct_structure():
    text = open("plans/boattrader/extract_only.txt").read()
    spec, plan, _ = _build_enhanced_plan(
        text, "boattrader.com",
        ["private seller", "by owner", "zip", "price", "location"],
        probe=_boattrader_probe(),
    )

    # Has navigate
    assert plan.steps[0].type == "navigate"
    # Has gate
    gates = [s for s in plan.steps if s.gate]
    assert len(gates) == 1
    # Has extraction loop
    extraction_loops = [s for s in plan.steps if s.type == "loop" and s.section == "extraction"]
    assert len(extraction_loops) == 1
    # Has pagination
    paginate_steps = [s for s in plan.steps if s.type == "paginate"]
    assert len(paginate_steps) == 1
    # Has navigate_back
    back_steps = [s for s in plan.steps if s.type == "navigate_back"]
    assert len(back_steps) == 1


def test_enhanced_plan_paginate_is_paginate_type():
    """Regression: paginate step must have type=paginate, not type=navigate."""
    _, plan, _ = _build_enhanced_plan("test", "example.com", [])
    paginate_steps = [s for s in plan.steps if s.section == "pagination" and s.type != "loop"]
    assert len(paginate_steps) == 1
    assert paginate_steps[0].type == "paginate"
