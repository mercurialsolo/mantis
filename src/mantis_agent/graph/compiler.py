"""GraphCompiler — compile a WorkflowGraph into a MicroPlan.

Walks the phase DAG in topological order and emits MicroIntent steps:
  - ONCE phases → single MicroIntent
  - FOR_EACH phases → sequence + loop step with loop_target
  - UNTIL_EXHAUSTED phases → sequence + loop step with high loop_count
  - Gate phases → MicroIntent with gate=True
  - Preconditions → verify steps (claude_only=True)

Output is a standard MicroPlan — zero changes to MicroPlanRunner.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from ..plan_decomposer import MicroIntent, MicroPlan
from .graph import PhaseNode, PhaseRole, RepeatMode, WorkflowGraph

logger = logging.getLogger(__name__)

# Map PhaseRole to MicroIntent section name
_ROLE_TO_SECTION = {
    PhaseRole.SETUP: "setup",
    PhaseRole.GATE: "setup",
    PhaseRole.DISCOVERY: "extraction",
    PhaseRole.ADMISSION: "extraction",
    PhaseRole.EXTRACTION: "extraction",
    PhaseRole.REJECTION: "extraction",
    PhaseRole.RETURN: "extraction",
    PhaseRole.PAGINATION: "pagination",
    PhaseRole.FINALIZE: "extraction",
}

# Map PhaseRole to MicroIntent type
_ROLE_TO_TYPE = {
    PhaseRole.SETUP: "filter",
    PhaseRole.GATE: "extract_data",
    PhaseRole.DISCOVERY: "scroll",
    PhaseRole.ADMISSION: "click",
    PhaseRole.EXTRACTION: "extract_data",
    PhaseRole.REJECTION: "extract_data",
    PhaseRole.RETURN: "navigate_back",
    PhaseRole.PAGINATION: "paginate",
    PhaseRole.FINALIZE: "extract_data",
}


class GraphCompiler:
    """Compile a WorkflowGraph into a MicroPlan for MicroPlanRunner."""

    def compile(self, graph: WorkflowGraph) -> MicroPlan:
        """Generate a MicroPlan from the learned WorkflowGraph."""
        order = graph.topological_order()
        steps: list[MicroIntent] = []

        # Track where FOR_EACH loops start (for loop_target)
        loop_body_start: dict[str, int] = {}  # source_phase → step index
        loop_body_phases: dict[str, list[str]] = {}  # source_phase → [phase_ids in body]

        # Group FOR_EACH phases by their source_phase (scan ALL phases, not just topo order)
        for pid, phase in graph.phases.items():
            if phase.repeat == RepeatMode.FOR_EACH and phase.source_phase:
                loop_body_phases.setdefault(phase.source_phase, []).append(pid)

        # First pass: emit navigation/setup phases (ONCE, before any loops)
        for pid in order:
            phase = graph.phases[pid]
            if phase.repeat != RepeatMode.ONCE:
                continue
            if phase.role in (PhaseRole.PAGINATION, PhaseRole.FINALIZE):
                continue  # handled after loops
            # Skip discovery phases that feed a FOR_EACH — they go inside the loop context
            if pid in loop_body_phases:
                continue

            intent = self._emit_phase(phase)
            steps.append(intent)

        # Second pass: emit FOR_EACH loop bodies
        for source_pid, body_pids in loop_body_phases.items():
            # The discovery phase that feeds the loop
            if source_pid in graph.phases:
                discovery_phase = graph.phases[source_pid]
                steps.append(self._emit_phase(discovery_phase))

            # Mark where the loop body starts (first admission/click step)
            loop_start_idx = len(steps)
            loop_body_start[source_pid] = loop_start_idx

            for body_pid in body_pids:
                phase = graph.phases[body_pid]
                steps.append(self._emit_phase(phase))

            # Emit extraction loop step
            max_items = graph.playbook.listings_per_page * 2 if graph.playbook.listings_per_page > 0 else 50
            if graph.objective.completion.max_items > 0:
                max_items = graph.objective.completion.max_items
            steps.append(
                MicroIntent(
                    intent="Loop back to click next listing title",
                    type="loop",
                    loop_target=loop_start_idx,
                    loop_count=max_items,
                    section="extraction",
                )
            )

        # Third pass: emit pagination (UNTIL_EXHAUSTED) — scan all phases
        for pid, phase in graph.phases.items():
            if phase.repeat == RepeatMode.UNTIL_EXHAUSTED:
                steps.append(self._emit_phase(phase))

                # Pagination loop: jump back to discovery phase
                # Find the discovery step index
                discovery_target = loop_body_start.get(
                    next(iter(loop_body_phases), ""),
                    max(len(steps) - 2, 0),
                )
                max_pages = graph.objective.completion.max_pages or 50
                steps.append(
                    MicroIntent(
                        intent="Loop back to process listings on new page",
                        type="loop",
                        loop_target=discovery_target,
                        loop_count=max_pages,
                        section="pagination",
                    )
                )

        # Build MicroPlan
        domain = graph.domain or (graph.objective.domains[0] if graph.objective.domains else "unknown")
        plan = MicroPlan(steps=steps, domain=domain)

        # Compute plan hash
        steps_json = json.dumps(
            [
                {
                    "intent": s.intent,
                    "type": s.type,
                    "section": s.section,
                    "gate": s.gate,
                    "required": s.required,
                }
                for s in steps
            ],
            sort_keys=True,
        )
        graph.plan_hash = hashlib.sha256(steps_json.encode()).hexdigest()

        logger.info(
            "Compiled WorkflowGraph → MicroPlan: %d phases → %d steps",
            len(graph.phases),
            len(steps),
        )
        return plan

    def _emit_phase(self, phase: PhaseNode) -> MicroIntent:
        """Convert a single PhaseNode into a MicroIntent."""
        step_type = _ROLE_TO_TYPE.get(phase.role, "click")
        section = _ROLE_TO_SECTION.get(phase.role, "extraction")

        # Override type based on intent template content
        intent = phase.intent_template
        if intent.lower().startswith("navigate to"):
            step_type = "navigate"
            section = "setup"
        elif "read the url" in intent.lower() or "address bar" in intent.lower():
            step_type = "extract_url"
        elif "scroll" in intent.lower():
            step_type = "scroll"
        elif "go back" in intent.lower():
            step_type = "navigate_back"

        return MicroIntent(
            intent=intent,
            type=step_type,
            budget=phase.budget,
            grounding=phase.grounding,
            claude_only=phase.claude_only,
            section=section,
            required=phase.required,
            gate=phase.gate,
            reverse=phase.recovery_action,
            verify=phase.postconditions[0].description if phase.postconditions else "",
        )
