"""Schema validators for CUA contracts (#476).

Each validator raises :class:`ContractValidationError` on the first
issue it finds. The validator surface is small on purpose — these
checks run on the hot path (every event emit), so they must stay
cheap. Anything that requires DOM / network / model access belongs
in a separate layer.

Validators reject the canonical failure modes the acceptance
criteria call out:

* missing or wrong-version ``schema_version``;
* missing :class:`~.types.Verdict` on a :class:`~.types.TrajectoryEvent`;
* inline observation blobs (``screenshot_ref`` that decodes as base64
  rather than being a string reference);
* missing ``run_id`` / unset ``step_index`` — without them the event
  isn't deterministically addressable, which breaks idempotent emit.
"""

from __future__ import annotations

from .types import (
    ActionResult,
    Observation,
    SCHEMA_VERSION,
    Step,
    TaskSpec,
    TrajectoryEvent,
    Verdict,
)


# Heuristic that catches an inline base64 PNG masquerading as a
# ``screenshot_ref``. A real reference is a path / URI / hash —
# typically < 256 chars and contains a path separator or scheme.
# Inline base64 PNGs are kilobytes and start with the standard PNG
# magic when decoded. We don't decode here (cost on every emit); the
# length cap + magic-prefix check is enough to keep honest callers
# honest and to bounce callers who confused ``Observation`` with the
# old screenshot-in-payload pattern.
_INLINE_BLOB_LEN_THRESHOLD = 1024
_INLINE_PNG_BASE64_PREFIX = "iVBORw0KGgo"  # base64-encoded "\x89PNG\r\n\x1a\n"


class ContractValidationError(ValueError):
    """Raised when a contract instance fails schema validation."""


def _require_current_version(name: str, observed: int) -> None:
    if observed != SCHEMA_VERSION:
        raise ContractValidationError(
            f"{name}: schema_version={observed!r} does not match "
            f"current SCHEMA_VERSION={SCHEMA_VERSION}. A bump requires "
            f"a coordinated writer+reader migration; this validator "
            f"refuses to silently downgrade."
        )


def validate_task_spec(spec: TaskSpec) -> None:
    """Reject TaskSpecs that can't drive a deterministic run."""
    _require_current_version("TaskSpec", spec.schema_version)
    if not spec.task_id:
        raise ContractValidationError("TaskSpec.task_id is required")
    if not spec.goal:
        raise ContractValidationError("TaskSpec.goal is required")
    allowed_policies = {"prompt_on_irreversible", "halt_on_irreversible", "auto"}
    if spec.reversibility_policy not in allowed_policies:
        raise ContractValidationError(
            f"TaskSpec.reversibility_policy={spec.reversibility_policy!r}; "
            f"must be one of {sorted(allowed_policies)}"
        )


def validate_step(step: Step) -> None:
    """Reject Steps that lack the fields the dispatcher needs."""
    _require_current_version("Step", step.schema_version)
    if not step.intent:
        raise ContractValidationError("Step.intent is required")
    if not step.action_type:
        raise ContractValidationError("Step.action_type is required")


def validate_verdict(verdict: Verdict) -> None:
    """Reject Verdicts missing the explanation a recoverable verdict
    needs to drive the next runner decision."""
    _require_current_version("Verdict", verdict.schema_version)
    # OK verdicts can omit reason+evidence (happy path); failure verdicts
    # must surface a reason so the recovery policy / IntentRewriter has
    # something to key on. Pre-#480, demotions sometimes landed as
    # ``data="...:no_state_change"`` with no structured field — drove
    # the unknown-classification debt the failure_class taxonomy work
    # has been chipping at.
    if verdict.kind.value != "ok" and not verdict.reason:
        raise ContractValidationError(
            f"Verdict.reason is required when kind={verdict.kind.value}; "
            f"recovery policy can't route without a code"
        )


def _looks_inlined(ref: str) -> bool:
    """Heuristic: does ``screenshot_ref`` look like an inline base64 blob?"""
    if len(ref) < _INLINE_BLOB_LEN_THRESHOLD:
        return False
    return ref.startswith(_INLINE_PNG_BASE64_PREFIX)


def _validate_observation(obs: Observation) -> None:
    _require_current_version("Observation", obs.schema_version)
    if not obs.screenshot_ref:
        raise ContractValidationError(
            "Observation.screenshot_ref is required — events store "
            "references, not blobs"
        )
    if _looks_inlined(obs.screenshot_ref):
        raise ContractValidationError(
            "Observation.screenshot_ref looks like an inline base64 PNG "
            "blob; events must reference screenshots by path / URI / "
            "content hash, not embed them"
        )


def _validate_action_result(result: ActionResult) -> None:
    _require_current_version("ActionResult", result.schema_version)
    if not result.action_type:
        raise ContractValidationError("ActionResult.action_type is required")
    if not result.dispatched and not result.dispatch_error:
        raise ContractValidationError(
            "ActionResult: when dispatched=False, dispatch_error must "
            "carry the failure reason"
        )


def validate_trajectory_event(event: TrajectoryEvent) -> None:
    """Reject TrajectoryEvents that can't be reconstructed / replayed.

    Acceptance-criteria checks (per :issue:`476`):

    * non-default ``schema_version`` matches ``SCHEMA_VERSION``;
    * a deterministic addressable key — ``run_id`` + ``step_index >= 0``;
    * an :class:`Observation` whose ``screenshot_ref`` is a string
      reference (not an inline base64 blob);
    * an :class:`ActionResult` and a :class:`Verdict` (neither None);
    * a ``versions`` dict — slot for model / prompt / browser stamps
      added by :issue:`488` (presence required, contents free-form in v1).
    """
    _require_current_version("TrajectoryEvent", event.schema_version)
    if not event.run_id:
        raise ContractValidationError("TrajectoryEvent.run_id is required")
    if event.step_index < 0:
        raise ContractValidationError(
            f"TrajectoryEvent.step_index={event.step_index!r}; must be >= 0"
        )
    if event.step is None:
        raise ContractValidationError("TrajectoryEvent.step is required")
    validate_step(event.step)
    if event.observation is None:
        raise ContractValidationError("TrajectoryEvent.observation is required")
    _validate_observation(event.observation)
    if event.action_result is None:
        raise ContractValidationError("TrajectoryEvent.action_result is required")
    _validate_action_result(event.action_result)
    if event.verdict is None:
        raise ContractValidationError(
            "TrajectoryEvent.verdict is required — runner must record "
            "a verdict before emitting the event (#480)"
        )
    validate_verdict(event.verdict)
    if not isinstance(event.versions, dict):
        raise ContractValidationError(
            "TrajectoryEvent.versions must be a dict (slot for model / "
            "prompt / browser stamps — #488)"
        )
    _validate_versions(event.versions)


# Keys the validator REQUIRES on every committed event (#488). Both
# are static facts about the writer's contract / ontology shape;
# they're populated by :func:`~.versions.collect_versions` at
# emitter-construction time so a writer that goes through the
# normal path always has them.
#
# Model / prompt / runtime stamps are optional in v1 — they
# populate incrementally as handlers + deploy scripts thread their
# stamps through. A future bump can promote some of those to
# required once population is universal.
_REQUIRED_VERSION_KEYS: frozenset[str] = frozenset({
    "action_ontology",
    "contracts_schema",
})


def _validate_versions(versions: dict) -> None:
    """Enforce the minimum required set + shape on ``versions``.

    Required keys (per :issue:`488` acceptance: "Missing required
    version fields fail validation"):

    * ``action_ontology`` — bump signal for the closed action enum.
    * ``contracts_schema`` — bump signal for the cua_contracts
      package itself.

    All other keys (model + prompt stamps, runtime stamps) are
    optional in v1 — populated where available. Values must be
    non-empty strings; an empty string is "we tried but couldn't
    determine" and is rejected so the event stream stays
    interpretable.
    """
    missing = _REQUIRED_VERSION_KEYS - set(versions)
    if missing:
        raise ContractValidationError(
            f"TrajectoryEvent.versions missing required keys: "
            f"{sorted(missing)}; the canonical writer must stamp "
            f"these on every event (#488)"
        )
    for key, value in versions.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ContractValidationError(
                f"TrajectoryEvent.versions entries must be str→str; "
                f"got {key!r}={value!r}"
            )
        if not value.strip():
            raise ContractValidationError(
                f"TrajectoryEvent.versions[{key!r}] is empty — drop "
                f"unset keys rather than emit empty strings, so "
                f"readers can distinguish 'not stamped' from "
                f"'stamped as blank'"
            )
