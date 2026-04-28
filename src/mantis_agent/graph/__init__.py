"""Graph learning for CUA workflows.

Adds an explicit dependency graph between plan decomposition and execution:
  ObjectiveSpec → SiteProber → WorkflowGraph → GraphCompiler → MicroPlan

The graph is a DAG of phases (setup, gate, discovery, extraction, pagination)
with cyclic state machines for item loops (FOR_EACH) and pagination
(UNTIL_EXHAUSTED). Dependencies are acyclic at the phase level.
"""

from .objective import CompletionCondition, ObjectiveSpec, OutputField
from .graph import (
    PhaseEdge,
    PhaseNode,
    PhaseRole,
    Precondition,
    Postcondition,
    RepeatMode,
    WorkflowGraph,
)
from .store import GraphStore
from .compiler import GraphCompiler
from .learner import GraphLearner
from .plan_validator import PlanValidator
from .enhancer import PlanEnhancer
from .section_decomposer import SectionDecomposer, ExecutionSection

__all__ = [
    "CompletionCondition",
    "ExecutionSection",
    "GraphCompiler",
    "GraphLearner",
    "GraphStore",
    "ObjectiveSpec",
    "OutputField",
    "PhaseEdge",
    "PhaseNode",
    "PhaseRole",
    "PlanEnhancer",
    "PlanValidator",
    "Postcondition",
    "Precondition",
    "RepeatMode",
    "SectionDecomposer",
    "WorkflowGraph",
]
