"""Tests for the graph learning package."""

import json
import tempfile
from pathlib import Path

from mantis_agent.graph import (
    CompletionCondition,
    GraphCompiler,
    GraphStore,
    ObjectiveSpec,
    OutputField,
    PhaseEdge,
    PhaseNode,
    PhaseRole,
    Postcondition,
    Precondition,
    RepeatMode,
    WorkflowGraph,
)
from mantis_agent.graph.learner import GraphLearner
from mantis_agent.graph.probe import ProbeResult
from mantis_agent.verification.playbook import Playbook


# ── ObjectiveSpec ──


def test_objective_heuristic_parse():
    spec = ObjectiveSpec._parse_heuristic(
        "Search BoatTrader.com for private seller boats near Miami FL zip 33101 "
        "over $35,000. Extract year, make, model, price, phone."
    )
    assert "BoatTrader.com" in spec.domains
    assert "private seller" in spec.required_filters
    assert "zip" in spec.required_filters
    assert "price" in spec.required_filters
    assert len(spec.objective_hash) == 64


def test_objective_cache_key():
    spec = ObjectiveSpec(raw_text="test", domains=["example.com"])
    key = spec.cache_key()
    assert key.startswith("example.com_")
    assert len(key) > 15


def test_objective_hash_deterministic():
    spec1 = ObjectiveSpec(raw_text="search for boats")
    spec2 = ObjectiveSpec(raw_text="search for boats")
    assert spec1.objective_hash == spec2.objective_hash


def test_objective_hash_normalized():
    spec1 = ObjectiveSpec(raw_text="search  for  boats")
    spec2 = ObjectiveSpec(raw_text="Search For Boats")
    assert spec1.objective_hash == spec2.objective_hash


def test_objective_serialization_roundtrip():
    spec = ObjectiveSpec(
        raw_text="test",
        domains=["example.com"],
        start_url="https://example.com",
        target_entity="listing",
        required_filters=["private seller"],
        forbidden_actions=["Contact"],
        allowed_reveal_actions=["Show more"],
        output_schema=[OutputField(name="price", type="int", required=True, example="50000")],
        completion=CompletionCondition(type="count", max_items=100),
    )
    data = spec.to_dict()
    restored = ObjectiveSpec.from_dict(data)
    assert restored.domains == ["example.com"]
    assert restored.target_entity == "listing"
    assert len(restored.output_schema) == 1
    assert restored.output_schema[0].name == "price"
    assert restored.completion.type == "count"
    assert restored.completion.max_items == 100


# ── WorkflowGraph ──


def _make_simple_graph() -> WorkflowGraph:
    """Build a minimal 3-phase graph for testing."""
    spec = ObjectiveSpec(raw_text="test", domains=["example.com"])
    return WorkflowGraph(
        objective=spec,
        domain="example.com",
        objective_hash=spec.objective_hash,
        phases={
            "navigate": PhaseNode(
                id="navigate", role=PhaseRole.SETUP,
                intent_template="Navigate to https://example.com",
                budget=3, required=True,
            ),
            "verify": PhaseNode(
                id="verify", role=PhaseRole.GATE,
                intent_template="Verify page loaded",
                claude_only=True, gate=True,
            ),
            "extract": PhaseNode(
                id="extract", role=PhaseRole.EXTRACTION,
                intent_template="Extract data",
                claude_only=True,
            ),
        },
        edges=[
            PhaseEdge(source="navigate", target="verify"),
            PhaseEdge(source="verify", target="extract"),
        ],
    )


def test_topological_order():
    graph = _make_simple_graph()
    order = graph.topological_order()
    assert order == ["navigate", "verify", "extract"]


def test_graph_serialization_roundtrip():
    graph = _make_simple_graph()
    graph.playbook = Playbook(domain="example.com", listings_per_page=25)
    data = graph.to_dict()
    restored = WorkflowGraph.from_dict(data)
    assert len(restored.phases) == 3
    assert len(restored.edges) == 2
    assert restored.domain == "example.com"
    assert restored.playbook.listings_per_page == 25
    assert restored.topological_order() == ["navigate", "verify", "extract"]


def test_phase_node_flags():
    graph = _make_simple_graph()
    assert graph.phases["navigate"].required is True
    assert graph.phases["verify"].gate is True
    assert graph.phases["verify"].claude_only is True
    assert graph.phases["extract"].claude_only is True


# ── GraphStore ──


def test_graph_store_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(base_path=tmpdir)
        graph = _make_simple_graph()
        store.save(graph)
        assert store.exists("example.com", graph.objective_hash)

        loaded = store.load("example.com", graph.objective_hash)
        assert loaded is not None
        assert len(loaded.phases) == 3
        assert loaded.domain == "example.com"


def test_graph_store_load_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(base_path=tmpdir)
        assert store.load("nonexistent.com", "abc123") is None
        assert not store.exists("nonexistent.com", "abc123")


# ── GraphCompiler ──


def test_compiler_simple_graph():
    """Simple ONCE-only graph compiles to linear MicroPlan."""
    graph = _make_simple_graph()
    compiler = GraphCompiler()
    plan = compiler.compile(graph)
    assert len(plan.steps) == 3
    assert plan.steps[0].type == "navigate"
    assert plan.steps[0].section == "setup"
    assert plan.steps[0].required is True
    assert plan.steps[1].gate is True
    assert plan.steps[2].claude_only is True


def test_compiler_gate_verify_prompt_from_postconditions():
    """Gate step.verify should come from postconditions verify_prompt or description."""
    spec = ObjectiveSpec(raw_text="test", domains=["example.com"])
    graph = WorkflowGraph(
        objective=spec,
        phases={
            "setup": PhaseNode(
                id="setup", role=PhaseRole.SETUP,
                intent_template="Apply filters", required=True,
            ),
            "gate": PhaseNode(
                id="gate", role=PhaseRole.GATE,
                intent_template="Verify filters",
                gate=True, claude_only=True,
                postconditions=[Postcondition(
                    description="Page shows filtered results",
                    verify_prompt="Check that the page heading says 'filtered' and result count < 1000",
                )],
            ),
        },
        edges=[PhaseEdge(source="setup", target="gate")],
    )
    compiler = GraphCompiler()
    plan = compiler.compile(graph)

    gate_step = [s for s in plan.steps if s.gate][0]
    # Should prefer verify_prompt over description
    assert gate_step.verify == "Check that the page heading says 'filtered' and result count < 1000"


def test_compiler_gate_falls_back_to_description():
    """When verify_prompt is empty, gate step.verify uses description."""
    spec = ObjectiveSpec(raw_text="test", domains=["example.com"])
    graph = WorkflowGraph(
        objective=spec,
        phases={
            "gate": PhaseNode(
                id="gate", role=PhaseRole.GATE,
                intent_template="Verify", gate=True, claude_only=True,
                postconditions=[Postcondition(description="Filters are applied")],
            ),
        },
        edges=[],
    )
    compiler = GraphCompiler()
    plan = compiler.compile(graph)
    assert plan.steps[0].verify == "Filters are applied"


def test_compiler_default_skeleton_produces_valid_microplan():
    """Default listing-extraction skeleton compiles to correct MicroPlan."""
    spec = ObjectiveSpec._parse_heuristic("Search BoatTrader.com for private seller boats")
    learner = GraphLearner()
    graph = learner._default_skeleton(spec, ProbeResult(estimated_listings_per_page=25))
    compiler = GraphCompiler()
    plan = compiler.compile(graph)

    # Should produce the canonical structure: navigate, filter, gate, click, url, scroll, extract, back, loop, paginate, loop
    assert len(plan.steps) >= 10

    # Check section tags
    setup_steps = [s for s in plan.steps if s.section == "setup"]
    extraction_steps = [s for s in plan.steps if s.section == "extraction"]
    pagination_steps = [s for s in plan.steps if s.section == "pagination"]
    assert len(setup_steps) >= 2  # navigate + filter + gate
    assert len(extraction_steps) >= 5  # click + url + scroll + extract + back + loop
    assert len(pagination_steps) >= 2  # paginate + loop

    # Check gate
    gates = [s for s in plan.steps if s.gate]
    assert len(gates) == 1

    # Check loops
    loops = [s for s in plan.steps if s.type == "loop"]
    assert len(loops) == 2  # extraction loop + pagination loop
    for loop in loops:
        assert loop.loop_target >= 0
        assert loop.loop_count > 0

    # Extraction loop should target the click step
    extraction_loop = [s for s in loops if s.section == "extraction"][0]
    click_steps = [i for i, s in enumerate(plan.steps) if s.type == "click"]
    assert extraction_loop.loop_target == click_steps[0]


def test_compiler_respects_max_items():
    """Compiler uses objective.completion.max_items for loop count."""
    spec = ObjectiveSpec(
        raw_text="test",
        domains=["example.com"],
        completion=CompletionCondition(type="count", max_items=10),
    )
    learner = GraphLearner()
    graph = learner._default_skeleton(spec, ProbeResult(estimated_listings_per_page=25))
    compiler = GraphCompiler()
    plan = compiler.compile(graph)

    extraction_loops = [s for s in plan.steps if s.type == "loop" and s.section == "extraction"]
    assert extraction_loops[0].loop_count == 10


# ── ProbeResult ──


def test_probe_result_serialization():
    probe = ProbeResult(
        url="https://example.com",
        domain="example.com",
        page_type="search_results",
        filters_detected=[{"name": "Category", "options": ["A", "B"]}],
        estimated_listings_per_page=25,
    )
    data = probe.to_dict()
    restored = ProbeResult.from_dict(data)
    assert restored.page_type == "search_results"
    assert restored.estimated_listings_per_page == 25
    assert len(restored.filters_detected) == 1


# ── Default skeleton ──


def test_default_skeleton_has_all_phases():
    spec = ObjectiveSpec(raw_text="test", domains=["example.com"])
    learner = GraphLearner()
    graph = learner._default_skeleton(spec, ProbeResult())

    roles = {p.role for p in graph.phases.values()}
    assert PhaseRole.SETUP in roles
    assert PhaseRole.GATE in roles
    assert PhaseRole.ADMISSION in roles
    assert PhaseRole.EXTRACTION in roles
    assert PhaseRole.RETURN in roles
    assert PhaseRole.PAGINATION in roles


def test_default_skeleton_has_for_each_phases():
    spec = ObjectiveSpec(raw_text="test", domains=["example.com"])
    learner = GraphLearner()
    graph = learner._default_skeleton(spec, ProbeResult())

    for_each = [p for p in graph.phases.values() if p.repeat == RepeatMode.FOR_EACH]
    assert len(for_each) >= 4  # admit, url, scroll, extract, return
    for phase in for_each:
        assert phase.source_phase == "discover_candidates"


def test_default_skeleton_pagination_is_until_exhausted():
    spec = ObjectiveSpec(raw_text="test", domains=["example.com"])
    learner = GraphLearner()
    graph = learner._default_skeleton(spec, ProbeResult())

    paginate = graph.phases.get("paginate")
    assert paginate is not None
    assert paginate.repeat == RepeatMode.UNTIL_EXHAUSTED
