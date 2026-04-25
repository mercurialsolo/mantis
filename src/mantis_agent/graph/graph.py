"""WorkflowGraph — DAG of phases with cyclic state machines for loops.

The graph represents a browsing workflow as:
  - Acyclic phase dependencies (setup must finish before extraction)
  - FOR_EACH loops (process each discovered candidate)
  - UNTIL_EXHAUSTED loops (paginate until no more pages)

Each PhaseNode has pre/postconditions that can be verified via Claude screenshots.
The GraphCompiler converts this into a flat MicroPlan for MicroPlanRunner execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..verification.playbook import Playbook
from .objective import ObjectiveSpec


class RepeatMode(Enum):
    """How a phase repeats during execution."""

    ONCE = "once"  # Execute once
    FOR_EACH = "for_each"  # Repeat for each item from source_phase
    UNTIL_EXHAUSTED = "until_exhausted"  # Repeat until postcondition "exhausted"


class PhaseRole(Enum):
    """The role of a phase in the workflow."""

    SETUP = "setup"
    GATE = "gate"
    DISCOVERY = "discovery"
    ADMISSION = "admission"
    EXTRACTION = "extraction"
    REJECTION = "rejection"
    RETURN = "return"
    PAGINATION = "pagination"
    FINALIZE = "finalize"


@dataclass
class Precondition:
    """What must be true before a phase can execute."""

    description: str  # "On results page with private seller filter applied"
    verify_prompt: str = ""  # Claude prompt to check via screenshot
    required_url_pattern: str = ""  # Regex for URL validation


@dataclass
class Postcondition:
    """What should be true after a phase executes successfully."""

    description: str  # "Detail page loaded, URL contains /boat/"
    verify_prompt: str = ""  # Claude prompt to verify via screenshot
    success_signal: str = ""  # "URL changed", "heading changed"
    failure_signal: str = ""  # "popup appeared", "404"


@dataclass
class PhaseNode:
    """A single phase in the workflow graph."""

    id: str  # "setup_filters", "discover_candidates", etc.
    role: PhaseRole
    intent_template: str  # MicroIntent-compatible instruction

    # Repeat semantics
    repeat: RepeatMode = RepeatMode.ONCE
    source_phase: str = ""  # For FOR_EACH: which phase provides items

    # Pre/postconditions
    preconditions: list[Precondition] = field(default_factory=list)
    postconditions: list[Postcondition] = field(default_factory=list)

    # MicroIntent parameters
    budget: int = 8
    grounding: bool = False
    claude_only: bool = False
    required: bool = False
    gate: bool = False

    # Learned knowledge
    known_traps: list[str] = field(default_factory=list)
    recovery_action: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role.value,
            "intent_template": self.intent_template,
            "repeat": self.repeat.value,
            "source_phase": self.source_phase,
            "preconditions": [
                {"description": p.description, "verify_prompt": p.verify_prompt, "required_url_pattern": p.required_url_pattern}
                for p in self.preconditions
            ],
            "postconditions": [
                {"description": p.description, "verify_prompt": p.verify_prompt, "success_signal": p.success_signal, "failure_signal": p.failure_signal}
                for p in self.postconditions
            ],
            "budget": self.budget,
            "grounding": self.grounding,
            "claude_only": self.claude_only,
            "required": self.required,
            "gate": self.gate,
            "known_traps": self.known_traps,
            "recovery_action": self.recovery_action,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseNode:
        preconditions = [
            Precondition(**p) for p in data.get("preconditions", [])
        ]
        postconditions = [
            Postcondition(**p) for p in data.get("postconditions", [])
        ]
        return cls(
            id=data["id"],
            role=PhaseRole(data.get("role", "extraction")),
            intent_template=data.get("intent_template", ""),
            repeat=RepeatMode(data.get("repeat", "once")),
            source_phase=data.get("source_phase", ""),
            preconditions=preconditions,
            postconditions=postconditions,
            budget=data.get("budget", 8),
            grounding=data.get("grounding", False),
            claude_only=data.get("claude_only", False),
            required=data.get("required", False),
            gate=data.get("gate", False),
            known_traps=data.get("known_traps", []),
            recovery_action=data.get("recovery_action", ""),
            confidence=data.get("confidence", 0.0),
        )


@dataclass
class PhaseEdge:
    """A directed edge between phases."""

    source: str  # PhaseNode.id
    target: str  # PhaseNode.id
    condition: str = "success"  # "success", "failure", "exhausted", "always"

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target, "condition": self.condition}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> PhaseEdge:
        return cls(source=data["source"], target=data["target"], condition=data.get("condition", "success"))


@dataclass
class WorkflowGraph:
    """A DAG of phases with cyclic loops for browsing workflows.

    The graph is acyclic at the phase level: setup → gate → discovery →
    extraction → pagination → finalize. Item traversal loops (FOR_EACH)
    and pagination loops (UNTIL_EXHAUSTED) are encoded in PhaseNode.repeat,
    not as cycles in the edge list.
    """

    objective: ObjectiveSpec
    phases: dict[str, PhaseNode] = field(default_factory=dict)
    edges: list[PhaseEdge] = field(default_factory=list)
    playbook: Playbook = field(default_factory=lambda: Playbook(domain=""))
    domain: str = ""
    objective_hash: str = ""
    plan_hash: str = ""
    created_at: str = ""
    learning_samples: int = 0
    version: int = 1

    def topological_order(self) -> list[str]:
        """Return phase IDs in topological order.

        FOR_EACH/UNTIL_EXHAUSTED back-edges are not followed — they
        represent loops within phases, not phase dependencies.
        """
        adj: dict[str, list[str]] = {pid: [] for pid in self.phases}
        in_degree: dict[str, int] = {pid: 0 for pid in self.phases}
        for edge in self.edges:
            if edge.source in adj and edge.target in in_degree:
                adj[edge.source].append(edge.target)
                in_degree[edge.target] += 1

        queue = [pid for pid, deg in in_degree.items() if deg == 0]
        order: list[str] = []
        while queue:
            queue.sort()  # deterministic order for same in-degree
            node = queue.pop(0)
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return order

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "domain": self.domain,
            "objective_hash": self.objective_hash,
            "plan_hash": self.plan_hash,
            "created_at": self.created_at,
            "learning_samples": self.learning_samples,
            "objective": self.objective.to_dict(),
            "phases": {pid: phase.to_dict() for pid, phase in self.phases.items()},
            "edges": [e.to_dict() for e in self.edges],
            "playbook": self.playbook.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowGraph:
        objective = ObjectiveSpec.from_dict(data.get("objective", {}))
        phases = {
            pid: PhaseNode.from_dict(pdata)
            for pid, pdata in data.get("phases", {}).items()
        }
        edges = [PhaseEdge.from_dict(e) for e in data.get("edges", [])]
        playbook_data = data.get("playbook", {})
        playbook = Playbook.from_dict(playbook_data) if playbook_data else Playbook(domain="")
        return cls(
            objective=objective,
            phases=phases,
            edges=edges,
            playbook=playbook,
            domain=data.get("domain", ""),
            objective_hash=data.get("objective_hash", ""),
            plan_hash=data.get("plan_hash", ""),
            created_at=data.get("created_at", ""),
            learning_samples=data.get("learning_samples", 0),
            version=data.get("version", 1),
        )
