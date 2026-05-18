"""Versioned data types for the CUA trajectory contract (#476).

The dataclasses in this module are the on-the-wire / on-disk shape of
the canonical trajectory event stream. They are deliberately kept
**provider-neutral** (no Anthropic / Holo3 / Modal specifics leak
into a field name) and **independent of the runner internals** (no
imports from :mod:`mantis_agent.gym`) so they can be loaded by
downstream consumers — shadow router, model registry, eval — without
pulling the full execution stack.

Versioning rules:

* Every persisted artifact carries ``schema_version`` and pins to
  :data:`SCHEMA_VERSION`. A future bump means a writer change AND a
  reader migration; the validator below enforces the pin.
* Field additions are backward compatible as long as defaults are
  supplied; field removals or rename require a schema bump.
* ``Observation.screenshot_ref`` is a **string reference** (URI / path
  / opaque content-addressed key), never a base64 blob. Inline blobs
  would balloon the event size and break the "events are cheap to
  emit, store, and replay" contract — the validator rejects them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


SCHEMA_VERSION: int = 1
"""Bump on incompatible changes. Keep an `add-only` field discipline
between bumps — see module docstring."""


class ReversibilityClass(str, Enum):
    """Coarse risk class for a single action (#477 lives this fully).

    The intended consumer is the step-safety preview gate (#481): an
    irreversible action gets a cross-check screenshot pass before
    dispatch; reversible ones don't.

    The vocabulary is intentionally small. Mantis has historically
    classified by step ``type``; the move to an action ontology will
    refine this list (drag / file_upload / tab_open all need their
    own class). For v1 we settle on three classes that capture the
    decisions we actually make today.
    """

    READ_ONLY = "read_only"      # scroll, screenshot, extract — no side effects
    REVERSIBLE = "reversible"    # click, type, key — can be undone by alt+Left / esc / backspace
    IRREVERSIBLE = "irreversible"  # submit, send, buy, delete, confirm — no programmatic undo


class VerdictKind(str, Enum):
    """Outcome of post-action verification (#480).

    Drives the runner's next action: ``ok`` → advance; ``recoverable``
    → retry / replan; ``non_recoverable`` → terminate or rollback.
    Always present on a TrajectoryEvent; the validator rejects events
    with ``verdict=None``.
    """

    OK = "ok"
    RECOVERABLE = "recoverable"
    NON_RECOVERABLE = "non_recoverable"


@dataclass(frozen=True)
class Observation:
    """A reference to the screenshot + minimal context the brain saw
    before emitting an action.

    Stored by reference (``screenshot_ref``) — never inline. The
    reference can be:

    * a relative path (``runs/abc/step_3.png``) on the host that owns
      the artifact store;
    * an s3:// URL (or any URI scheme the host understands);
    * a content-addressed hash (``sha256:...``) for dedup-by-content
      storage.

    ``url`` and ``viewport`` are cheap to inline and immediately
    useful for grepping events without round-tripping to storage,
    so they ride alongside.
    """

    schema_version: int = SCHEMA_VERSION
    screenshot_ref: str = ""
    url: str = ""
    viewport: tuple[int, int] = (0, 0)
    captured_at: float = 0.0  # epoch seconds; 0 = unset


@dataclass(frozen=True)
class ActionResult:
    """The action that was dispatched + the dispatcher's local outcome.

    ``grounding_trace`` is the structured payload from :issue:`479`:
    how the target was localized, model/prompt version, confidence,
    selected dispatch strategy, confirmation evidence. v1 keeps it as
    a free-form dict so the grounding side can iterate without a
    schema bump; v2 will pin the shape.
    """

    schema_version: int = SCHEMA_VERSION
    action_type: str = ""                       # canonical ontology name (#477)
    params: dict[str, Any] = field(default_factory=dict)
    grounding_trace: dict[str, Any] = field(default_factory=dict)
    dispatched: bool = False                    # did the dispatcher commit the action?
    dispatch_error: str = ""                    # empty when dispatched=True


@dataclass(frozen=True)
class Verdict:
    """Typed post-action verifier verdict (#480).

    A step does not advance the cursor until a Verdict is recorded.
    ``reason`` is a short machine code (``no_state_change``,
    ``wrong_target``, ``brain_loop_exhausted``, ``cf_challenge``,
    ``unknown``, ...) — keep it grep-able. ``evidence`` is human-
    readable prose suitable for surfacing in result.json.
    """

    schema_version: int = SCHEMA_VERSION
    kind: VerdictKind = VerdictKind.OK
    reason: str = ""
    evidence: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class Step:
    """One typed step in a Plan.

    Maps onto today's :class:`~mantis_agent.plan_decomposer.MicroIntent`
    via :func:`~.adapters.step_from_micro_intent`. The new fields
    relative to MicroIntent are:

    * ``action_type`` — the canonical ontology name (vs MicroIntent's
      free-form ``type`` string).
    * ``reversibility`` — explicit class, not inferred from ``type``.
    * ``expected_outcome`` — what the verifier should check
      (MicroIntent's ``verify`` field, renamed for clarity).
    * ``confidence`` — planner's self-rated confidence; downstream
      can route low-confidence steps through extra preview/verify
      hops without re-asking the planner.
    """

    schema_version: int = SCHEMA_VERSION
    intent: str = ""
    action_type: str = ""
    reversibility: ReversibilityClass = ReversibilityClass.REVERSIBLE
    expected_outcome: str = ""
    confidence: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    hints: dict[str, Any] = field(default_factory=dict)
    required: bool = False


@dataclass(frozen=True)
class Plan:
    """Ordered list of typed Steps + the source plan text that produced it."""

    schema_version: int = SCHEMA_VERSION
    steps: tuple[Step, ...] = field(default_factory=tuple)
    source_plan: str = ""
    domain: str = ""


@dataclass(frozen=True)
class TaskSpec:
    """Top-level task envelope. The input to a CUA run.

    ``reversibility_policy`` is the **task-level** posture, not a
    per-step class. Values:

    * ``"prompt_on_irreversible"`` — pause + ask before any
      :class:`ReversibilityClass.IRREVERSIBLE` step (the default for
      production write-action tasks).
    * ``"halt_on_irreversible"`` — refuse irreversible steps outright
      (read-only research / scrape tasks).
    * ``"auto"`` — dispatch without prompting (CI smoke tests, sim envs).
    """

    schema_version: int = SCHEMA_VERSION
    task_id: str = ""
    goal: str = ""
    reversibility_policy: str = "prompt_on_irreversible"
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryEvent:
    """Canonical per-step event (#478).

    Emitted exactly once per completed step, after the verdict is
    recorded and before the cursor advances. The validator pins:

    * non-zero ``schema_version`` + ``run_id`` + ``step_index`` —
      otherwise the event isn't addressable.
    * an :class:`Observation` with a string reference (no inline
      base64).
    * an :class:`ActionResult` and :class:`Verdict` (not None).
    * a ``versions`` dict — the slot where #487 / #488 model / prompt
      / browser / sandbox stamps land. v1 doesn't pin its shape; it
      just requires the key exists so consumers don't get None.

    ``committed`` records the runner's decision to advance off this
    step. Idempotent re-emits set ``committed=True`` only once per
    (run_id, step_index) tuple — the dedup happens upstream in the
    emit path (#478), not in the event itself.
    """

    schema_version: int = SCHEMA_VERSION
    run_id: str = ""
    step_index: int = -1
    step: Step | None = None
    observation: Observation | None = None
    action_result: ActionResult | None = None
    verdict: Verdict | None = None
    versions: dict[str, str] = field(default_factory=dict)
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    committed: bool = False
    emitted_at: float = 0.0  # epoch seconds
