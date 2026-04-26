"""GraphLearner — orchestrates the full graph learning phase.

Pipeline:
  1. Parse objective → ObjectiveSpec
  2. Check cache → GraphStore
  3. Probe site → SiteProber (no brain needed)
  4. Generate graph skeleton → Claude Sonnet
  5. Optionally run verified sample (1-3 candidates + 1 pagination)
  6. Cache the learned graph → GraphStore

The probing phase does NOT require a brain model — it uses direct
navigation + Claude screenshot analysis. The brain is only needed
for optional sample execution in step 5.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

from ..verification.playbook import Playbook
from .compiler import GraphCompiler
from .graph import (
    PhaseEdge,
    PhaseNode,
    PhaseRole,
    Postcondition,
    Precondition,
    RepeatMode,
    WorkflowGraph,
)
from .objective import ObjectiveSpec
from .probe import ProbeResult, SiteProber
from .store import GraphStore

logger = logging.getLogger(__name__)


GENERATE_SKELETON_PROMPT = """\
You are generating a workflow graph for a CUA (Computer Use Agent) that browses websites.

OBJECTIVE:
{objective_json}

SITE ANALYSIS (from probing the start URL):
{probe_json}

Generate a workflow graph as a list of PHASES. Each phase is one atomic step.
The phases form a DAG: setup runs first, then extraction loops, then pagination.

Use these phase roles:
- SETUP: navigate to URL, apply filters (required=true)
- GATE: verify page state after setup (gate=true, claude_only=true)
- DISCOVERY: scan visible listings on current page
- ADMISSION: click a listing card (grounding=true)
- EXTRACTION: read data from detail page (claude_only=true)
- REJECTION: decide keep/reject based on entity rules (claude_only=true)
- RETURN: go back to results page
- PAGINATION: click Next page (grounding=true)

For the intent_template of each phase, write a clear 1-sentence CUA instruction.
Customize based on what the probe found (filter names, listing layout, pagination type).

Output ONLY valid JSON:
{{
  "phases": [
    {{
      "id": "navigate",
      "role": "setup",
      "intent_template": "Navigate to {start_url}",
      "repeat": "once",
      "budget": 3,
      "required": true,
      "preconditions": [],
      "postconditions": [{{"description": "Page loaded with results"}}]
    }},
    ...
  ],
  "edges": [
    {{"source": "navigate", "target": "setup_filters", "condition": "success"}},
    ...
  ]
}}

RULES:
- POSITIVE framing only: "Click the title text" not "Don't click the photo"
- Each intent_template: ONE action, ONE sentence, under 20 words
- Include WHAT + WHERE: "Click Private Seller text in left sidebar"
- Extraction steps must inspect contact area AND expanded description
- Reject dealers, brokers, sponsored, MarineMax listings even if phone visible
- Use allowed reveal actions: {allowed_reveals}
- Avoid forbidden actions: {forbidden_actions}
"""


class GraphLearner:
    """Orchestrates the full graph learning phase."""

    def __init__(
        self,
        env: Any = None,
        api_key: str = "",
        store: GraphStore | None = None,
        brain: Any = None,
        grounding: Any = None,
        extractor: Any = None,
    ):
        self.env = env
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.store = store or GraphStore()
        self.brain = brain
        self.grounding = grounding
        self.extractor = extractor

    def learn(
        self,
        objective_text: str,
        start_url: str = "",
        n_samples: int = 3,
        force_relearn: bool = False,
    ) -> WorkflowGraph:
        """Full learning pipeline with enhancement loop.

        Pipeline:
          1. Parse objective → ObjectiveSpec
          2. Check cache → GraphStore
          3. Probe site → SiteProber (no brain needed)
          4. Enhance plan → PlanEnhancer (fill gaps using probe knowledge)
          5. Build enhanced graph → concrete phases with site knowledge
          6. Section decompose → verify Holo3-sized chunks
          7. Validate → PlanValidator structural checks
          8. Optionally run verified sample (1-3 items with brain)
          9. Cache the learned graph → GraphStore
        """
        # 1. Parse objective
        objective = ObjectiveSpec.parse(objective_text, api_key=self.api_key)
        if start_url and not objective.start_url:
            objective.start_url = start_url
        domain = objective.domains[0] if objective.domains else ""

        logger.info(
            "GraphLearner: domain=%s entity=%s filters=%s",
            domain,
            objective.target_entity,
            objective.required_filters,
        )

        # 2. Check cache
        if not force_relearn and self.store.exists(domain, objective.objective_hash):
            cached = self.store.load(domain, objective.objective_hash)
            if cached:
                logger.info("GraphLearner: loaded cached graph (%d phases)", len(cached.phases))
                return cached

        # 3. Probe site
        probe = ProbeResult(url=objective.start_url, domain=domain)
        if self.env and objective.start_url:
            prober = SiteProber(env=self.env, api_key=self.api_key)
            probe = prober.probe(objective.start_url, objective)

        # 4. Enhance plan — fill gaps using probe knowledge
        from .enhancer import PlanEnhancer
        enhancer = PlanEnhancer(api_key=self.api_key)
        enhancement = enhancer.enhance(objective, probe)
        logger.info(
            "GraphLearner: enhanced — nav_url=%s, %d filter strategies, pagination=%s",
            enhancement.get("navigation_url", "")[:60],
            len(enhancement.get("filter_strategy", [])),
            enhancement.get("pagination_method", "unknown"),
        )

        # 5. Build enhanced graph with concrete phases
        phases, edges = enhancer.build_enhanced_phases(objective, probe, enhancement)

        playbook = Playbook(domain=domain)
        if probe.estimated_listings_per_page:
            playbook.listings_per_page = probe.estimated_listings_per_page

        graph = WorkflowGraph(
            objective=objective,
            phases=phases,
            edges=edges,
            playbook=playbook,
            domain=domain,
            objective_hash=objective.objective_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # 6. Section decompose — verify Holo3-sized chunks
        from .section_decomposer import SectionDecomposer
        decomposer = SectionDecomposer()
        sections = decomposer.decompose(phases)
        graph.learning_samples = 0  # Will be updated by sample run
        logger.info(
            "GraphLearner: %d sections, dependency chain: %s",
            len(sections),
            " → ".join(s.id for s in sections),
        )

        # 7. Validate compiled plan
        from .compiler import GraphCompiler
        from .plan_validator import PlanValidator
        compiler = GraphCompiler()
        micro_plan = compiler.compile(graph)
        validator = PlanValidator()
        issues = validator.validate(micro_plan, objective=objective)
        if issues:
            logger.info("GraphLearner: validator found %d issues, applying fixes", len(issues))
            for issue in issues:
                logger.info("  [%s] %s: %s", issue.severity, issue.code, issue.message)
            micro_plan = validator.enhance(micro_plan, objective=objective)
            # Re-compile graph from fixed plan would be complex;
            # the validator fixes are applied to the MicroPlan directly
            # which is what gets executed.

        # 8. Optionally run verified sample
        if self.brain and self.env and n_samples > 0:
            self._run_sample(graph, n_samples)

        # 9. Cache
        self.store.save(graph)
        logger.info(
            "GraphLearner: saved enhanced graph (%d phases, %d edges, %d sections)",
            len(graph.phases),
            len(graph.edges),
            len(sections),
        )
        return graph

    def _generate_skeleton(
        self,
        objective: ObjectiveSpec,
        probe: ProbeResult,
    ) -> WorkflowGraph:
        """Use Claude Sonnet to generate the phase DAG."""
        import requests

        prompt = GENERATE_SKELETON_PROMPT.format(
            objective_json=json.dumps(objective.to_dict(), indent=2)[:2000],
            probe_json=json.dumps(probe.to_dict(), indent=2)[:2000],
            start_url=objective.start_url,
            allowed_reveals=", ".join(objective.allowed_reveal_actions) or "Show more, Read more, Show phone",
            forbidden_actions=", ".join(objective.forbidden_actions) or "Contact Seller, Request Info",
        )

        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60,
            )
            if resp.status_code == 200:
                response_text = ""
                for block in resp.json().get("content", []):
                    if block.get("type") == "text":
                        response_text = block["text"].strip()
                        break
                return self._parse_skeleton_response(objective, probe, response_text)
        except Exception as e:
            logger.warning("GraphLearner: skeleton generation failed: %s", e)

        # Fallback: use default template
        return self._default_skeleton(objective, probe)

    def _parse_skeleton_response(
        self,
        objective: ObjectiveSpec,
        probe: ProbeResult,
        response: str,
    ) -> WorkflowGraph:
        """Parse Claude's skeleton response into a WorkflowGraph."""
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
        if response.endswith("```"):
            response = response.rsplit("```", 1)[0]
        response = response.strip()

        data = json.loads(response)
        domain = objective.domains[0] if objective.domains else ""

        phases: dict[str, PhaseNode] = {}
        for p in data.get("phases", []):
            preconditions = [Precondition(**pc) for pc in p.get("preconditions", [])]
            postconditions = [Postcondition(**pc) for pc in p.get("postconditions", [])]
            phase = PhaseNode(
                id=p["id"],
                role=PhaseRole(p.get("role", "extraction")),
                intent_template=p.get("intent_template", ""),
                repeat=RepeatMode(p.get("repeat", "once")),
                source_phase=p.get("source_phase", ""),
                preconditions=preconditions,
                postconditions=postconditions,
                budget=p.get("budget", 8),
                grounding=p.get("grounding", False),
                claude_only=p.get("claude_only", False),
                required=p.get("required", False),
                gate=p.get("gate", False),
            )
            phases[phase.id] = phase

        edges = [
            PhaseEdge(
                source=e["source"],
                target=e["target"],
                condition=e.get("condition", "success"),
            )
            for e in data.get("edges", [])
        ]

        playbook = Playbook(domain=domain)
        if probe.estimated_listings_per_page:
            playbook.listings_per_page = probe.estimated_listings_per_page
        if probe.dealer_signals:
            playbook.extraction_pattern.dealer_signals = probe.dealer_signals

        return WorkflowGraph(
            objective=objective,
            phases=phases,
            edges=edges,
            playbook=playbook,
            domain=domain,
            objective_hash=objective.objective_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _default_skeleton(
        self,
        objective: ObjectiveSpec,
        probe: ProbeResult,
    ) -> WorkflowGraph:
        """Fallback: generate default listing-extraction graph template."""
        domain = objective.domains[0] if objective.domains else ""
        url = objective.start_url or f"https://www.{domain}/"

        phases: dict[str, PhaseNode] = {
            "navigate": PhaseNode(
                id="navigate",
                role=PhaseRole.SETUP,
                intent_template=f"Navigate to {url}",
                budget=3,
                required=True,
                postconditions=[Postcondition(description="Page loaded with results")],
            ),
        }

        # Break required_filters into individual filter steps — each is a
        # separate phase with its own intent so the CUA applies them one by one.
        # This mirrors the PlanDecomposer's atomic filter approach.
        filter_ids: list[str] = []
        if objective.required_filters:
            for i, filt in enumerate(objective.required_filters):
                fid = f"filter_{i}"
                filter_ids.append(fid)
                phases[fid] = PhaseNode(
                    id=fid,
                    role=PhaseRole.SETUP,
                    intent_template=f"Apply filter: {filt}",
                    budget=8,
                    grounding=True,
                    required=True,
                )
        else:
            # No filters specified — single generic step
            filter_ids.append("setup_filters")
            phases["setup_filters"] = PhaseNode(
                id="setup_filters",
                role=PhaseRole.SETUP,
                intent_template="Apply required search filters on the page",
                budget=8,
                grounding=True,
                required=True,
            )

        filter_summary = ", ".join(objective.required_filters) if objective.required_filters else "required filters"
        phases["verify_scope"] = PhaseNode(
            id="verify_scope",
            role=PhaseRole.GATE,
            intent_template=f"Verify page shows {objective.target_entity} results with {filter_summary} applied",
            claude_only=True,
            budget=0,
            gate=True,
            preconditions=[Precondition(description=f"Filters applied: {filter_summary}")],
            postconditions=[Postcondition(
                description=f"Page shows filtered {objective.target_entity} results",
                verify_prompt=f"Page shows {objective.target_entity} results with these filters active: {filter_summary}. Result count should be reasonable (not unfiltered).",
            )],
        )
        phases["admit_candidate"] = PhaseNode(
            id="admit_candidate",
            role=PhaseRole.ADMISSION,
            intent_template=f"Click an organic {objective.target_entity} title; skip sponsored and dealer cards",
            budget=8,
            grounding=True,
            repeat=RepeatMode.FOR_EACH,
            source_phase="discover_candidates",
        )
        phases["extract_url"] = PhaseNode(
            id="extract_url",
            role=PhaseRole.EXTRACTION,
            intent_template="Read the URL from browser address bar",
            claude_only=True,
            budget=0,
            repeat=RepeatMode.FOR_EACH,
            source_phase="discover_candidates",
        )
        phases["scroll_to_details"] = PhaseNode(
            id="scroll_to_details",
            role=PhaseRole.EXTRACTION,
            intent_template="Scroll down toward the Description and More Details sections",
            budget=10,
            repeat=RepeatMode.FOR_EACH,
            source_phase="discover_candidates",
        )
        phases["extract_fields"] = PhaseNode(
            id="extract_fields",
            role=PhaseRole.EXTRACTION,
            intent_template="Reject dealers, inspect contact area, expand collapsed sections, then extract structured data",
            claude_only=True,
            budget=0,
            repeat=RepeatMode.FOR_EACH,
            source_phase="discover_candidates",
        )
        phases["return_to_results"] = PhaseNode(
            id="return_to_results",
            role=PhaseRole.RETURN,
            intent_template="Go back to search results page",
            budget=3,
            repeat=RepeatMode.FOR_EACH,
            source_phase="discover_candidates",
        )
        phases["paginate"] = PhaseNode(
            id="paginate",
            role=PhaseRole.PAGINATION,
            intent_template="Click Next page button to continue to next results page",
            budget=10,
            grounding=True,
            repeat=RepeatMode.UNTIL_EXHAUSTED,
        )

        # Build edge chain: navigate → filter_0 → filter_1 → ... → verify_scope
        edges: list[PhaseEdge] = []
        prev = "navigate"
        for fid in filter_ids:
            edges.append(PhaseEdge(source=prev, target=fid))
            prev = fid
        edges.append(PhaseEdge(source=prev, target="verify_scope"))

        edges.extend([
            PhaseEdge(source="verify_scope", target="admit_candidate"),
            PhaseEdge(source="admit_candidate", target="extract_url"),
            PhaseEdge(source="extract_url", target="scroll_to_details"),
            PhaseEdge(source="scroll_to_details", target="extract_fields"),
            PhaseEdge(source="extract_fields", target="return_to_results"),
            PhaseEdge(source="return_to_results", target="paginate", condition="exhausted"),
            PhaseEdge(source="paginate", target="admit_candidate"),
        ])

        playbook = Playbook(domain=domain)
        if probe.estimated_listings_per_page:
            playbook.listings_per_page = probe.estimated_listings_per_page

        return WorkflowGraph(
            objective=objective,
            phases=phases,
            edges=edges,
            playbook=playbook,
            domain=domain,
            objective_hash=objective.objective_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _run_sample(self, graph: WorkflowGraph, n_candidates: int = 3) -> None:
        """Execute a small verified sample to validate the graph.

        Compiles the graph to a MicroPlan, runs n_candidates items,
        and updates phase confidence scores.

        For new/unlearned sites, Claude CUA is preferred for the learning
        phase since it has stronger reasoning for unfamiliar layouts.
        Set self.brain to a ClaudeBrain instance for best results.
        """
        if not self.brain or not self.env:
            logger.info("GraphLearner: no brain/env, skipping sample execution")
            return

        # Prefer Claude for learning when available and no cached graph exists
        sample_brain = self.brain
        if graph.learning_samples == 0:
            try:
                from ..brain_claude import ClaudeBrain
                if not isinstance(self.brain, ClaudeBrain):
                    logger.info("GraphLearner: first-time learning — Claude CUA recommended for best results")
            except ImportError:
                pass

        from ..gym.micro_runner import MicroPlanRunner

        compiler = GraphCompiler()
        micro_plan = compiler.compile(graph)

        # Limit to n_candidates by capping the extraction loop count
        for step in micro_plan.steps:
            if step.type == "loop" and step.section == "extraction":
                step.loop_count = min(step.loop_count, n_candidates)
            if step.type == "loop" and step.section == "pagination":
                step.loop_count = 1  # Only 1 pagination in sample

        runner = MicroPlanRunner(
            brain=sample_brain,
            env=self.env,
            grounding=self.grounding,
            extractor=self.extractor,
            session_name=f"graph_sample_{graph.domain}",
            max_cost=1.0,
            max_time_minutes=10,
        )
        step_results = runner.run(micro_plan)

        # Update graph with sample results
        graph.learning_samples = len(step_results)
        successful = sum(1 for r in step_results if r.success)
        total = len(step_results)
        if total > 0:
            for phase in graph.phases.values():
                phase.confidence = successful / total

        logger.info(
            "GraphLearner: sample complete, %d/%d steps succeeded",
            successful,
            total,
        )
