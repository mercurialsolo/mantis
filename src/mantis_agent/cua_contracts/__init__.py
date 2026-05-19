"""Versioned CUA contracts (#476).

Typed, schema-validated data types for the CUA design's source-of-truth
event stream. These contracts are the substrate for:

- :issue:`478` — canonical per-step trajectory events emitted from the
  runner. Each emitted event is a :class:`TrajectoryEvent`.
- :issue:`479` — grounding traces persisted per dispatched step. The
  :class:`ActionResult.grounding_trace` field carries them.
- :issue:`480` — mandatory verifier verdict before advancing.
  :class:`TrajectoryEvent.verdict` is the typed verdict; events without
  one fail validation.
- :issue:`477` — action ontology + reversibility classifier. Lives in
  the sibling :mod:`reversibility` module and is referenced by
  :class:`Step.reversibility`.
- :issue:`487`–:issue:`489` — model / prompt / sandbox version stamps.
  :class:`TrajectoryEvent.versions` is the slot.

**Why a new module instead of extending api_schemas / actions / checkpoint?**

The existing types — ``PredictRequest`` (api_schemas), ``MicroIntent``
(plan_decomposer), ``Action`` (actions), ``StepResult`` (checkpoint) —
were built up incrementally and each carries assumptions about its
caller. The CUA design's reliability story needs *one* canonical
trajectory record that doesn't drift with the runner refactors and
that downstream consumers (shadow routing, model registry, eval) can
target stably.

The first version (``SCHEMA_VERSION = 1``) maps cleanly onto today's
runtime via adapter functions in :mod:`adapters`. Existing surfaces
keep working unchanged; the canonical event stream is emitted *in
parallel* until the new contracts have proven themselves.

Public surface:

- Data types: :class:`TaskSpec`, :class:`Plan`, :class:`Step`,
  :class:`Observation`, :class:`ActionResult`, :class:`Verdict`,
  :class:`TrajectoryEvent`.
- Enums: :class:`ReversibilityClass`, :class:`VerdictKind`.
- Validation: :func:`validate_task_spec`, :func:`validate_step`,
  :func:`validate_verdict`, :func:`validate_trajectory_event`.
- Adapters: :func:`step_from_micro_intent`,
  :func:`action_result_from_action`.
- Constants: :data:`SCHEMA_VERSION`.
"""

from __future__ import annotations

from .adapters import (
    DEFAULT_RETRY_BUDGET,
    action_result_from_action,
    classify_legacy_reversibility,
    decide_recovery,
    observation_from_screenshot_ref,
    step_from_micro_intent,
    verdict_from_step_result,
)
from .emit import JSONL_FILENAME, TrajectoryEmitter
from .ontology import (
    ActionTyped,
    classify_action,
    is_irreversible,
    validate_action_type,
)
from .lifecycle import (
    PURE_PHASES,
    SIDE_EFFECTFUL_PHASES,
    Activity,
    LifecyclePhase,
)
from .observation_store import (
    DiskObservationStore,
    InMemoryObservationStore,
    ObservationStore,
    RedactionPolicy,
    StoredObservation,
    identity_redaction,
)
from .serving import (
    ModelCallResult,
    ModelServingFacade,
    PassthroughFacade,
    Role,
    RoutingMode,
    stamp_runtime_versions,
)
from .versions import VERSION_KEYS, collect_versions
from .types import (
    ActionResult,
    GroundingTrace,
    Observation,
    Plan,
    RecoveryDecision,
    ReversibilityClass,
    SCHEMA_VERSION,
    Step,
    TaskSpec,
    TrajectoryEvent,
    Verdict,
    VerdictKind,
)
from .validation import (
    ContractValidationError,
    validate_step,
    validate_task_spec,
    validate_trajectory_event,
    validate_verdict,
)

__all__ = [
    "ActionResult",
    "ActionTyped",
    "Activity",
    "ContractValidationError",
    "DEFAULT_RETRY_BUDGET",
    "DiskObservationStore",
    "GroundingTrace",
    "InMemoryObservationStore",
    "JSONL_FILENAME",
    "LifecyclePhase",
    "ModelCallResult",
    "ModelServingFacade",
    "Observation",
    "ObservationStore",
    "PURE_PHASES",
    "PassthroughFacade",
    "Plan",
    "RecoveryDecision",
    "RedactionPolicy",
    "ReversibilityClass",
    "Role",
    "RoutingMode",
    "SCHEMA_VERSION",
    "SIDE_EFFECTFUL_PHASES",
    "Step",
    "StoredObservation",
    "TaskSpec",
    "TrajectoryEmitter",
    "TrajectoryEvent",
    "VERSION_KEYS",
    "Verdict",
    "VerdictKind",
    "action_result_from_action",
    "classify_action",
    "classify_legacy_reversibility",
    "collect_versions",
    "decide_recovery",
    "identity_redaction",
    "is_irreversible",
    "observation_from_screenshot_ref",
    "stamp_runtime_versions",
    "step_from_micro_intent",
    "validate_action_type",
    "validate_step",
    "validate_task_spec",
    "validate_trajectory_event",
    "validate_verdict",
    "verdict_from_step_result",
]
