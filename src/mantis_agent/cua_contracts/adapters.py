"""Adapter shims between the new contracts and the existing surface
(:mod:`mantis_agent.plan_decomposer`, :mod:`mantis_agent.actions`).

These adapters let the runner emit canonical events without
refactoring every handler in one go. The compatibility direction is
**existing → new**: take a legacy ``MicroIntent`` / ``Action`` /
``StepResult`` and project it into the typed contract. The reverse
direction (new contract back to legacy types) is intentionally not
implemented — once a callsite is migrated to emit/consume the new
types, it shouldn't fall back.

The adapters live in their own module so :mod:`types` stays free of
imports from the legacy execution layer. Downstream consumers of the
contract types (shadow router, eval, registry) don't pay the import
cost of the runner stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .types import (
    ActionResult,
    Observation,
    RecoveryDecision,
    ReversibilityClass,
    SCHEMA_VERSION,
    Step,
    Verdict,
    VerdictKind,
)


# Default retry budget per step. Mirrors the existing runner's
# ``REQUIRED step failed after 2 retries — HALTING`` behaviour: the
# first attempt + 2 retries = 3 total attempts before a required step
# terminates. Non-required steps share the same budget for v1; future
# work can route per step-type budgets here.
DEFAULT_RETRY_BUDGET: int = 3

if TYPE_CHECKING:
    from ..actions import Action
    from ..gym.checkpoint import StepResult
    from ..plan_decomposer import MicroIntent


# Failure classes that genuinely can't be recovered from without
# operator intervention — re-running them would just churn budget.
# Anything not in this set defaults to RECOVERABLE so the retry /
# recovery / replan ladder gets a shot before terminating.
#
# Empty ``failure_class`` on a failed step also defaults to
# RECOVERABLE — the safer choice when classification was missing.
# #477's action ontology will refine this; for v1 the set stays
# small and grep-able so an operator auditing a HALT can find the
# entry quickly.
_NON_RECOVERABLE_FAILURE_CLASSES: frozenset[str] = frozenset({
    "cf_challenge",      # Cloudflare bot challenge — needs new IP / browser
    "http_4xx",          # 401 / 403 / 404 — credential or routing problem
    "extractor_error",   # extractor itself crashed, not a content miss
    "budget_exceeded",   # already exhausted the per-step budget
})


def classify_legacy_reversibility(step_type: str) -> ReversibilityClass:
    """Project a legacy ``MicroIntent.type`` string onto the
    reversibility class registered in :mod:`.ontology`.

    Public helper — also useful from tests / safety gates that
    operate on the legacy string vocabulary.

    Failure mode: unknown step types default to
    :class:`ReversibilityClass.REVERSIBLE` rather than raising.
    Rationale — the adapter is a back-compat shim and the runner
    has long tolerated unrecognised plan step types (the executor
    falls through to the Holo3 step handler). Failing closed here
    would regress that tolerance. Callers that want strict
    validation should use :func:`.ontology.classify_action` directly,
    which raises :class:`ContractValidationError` on unknown.
    """
    from .ontology import _REVERSIBILITY_MAP, ActionTyped
    try:
        return _REVERSIBILITY_MAP[ActionTyped(step_type)]
    except (ValueError, KeyError):
        return ReversibilityClass.REVERSIBLE


def step_from_micro_intent(mi: "MicroIntent") -> Step:
    """Project a legacy :class:`MicroIntent` onto the typed :class:`Step`.

    No information loss in this direction — ``MicroIntent`` is wider
    (carries ``gate`` / ``claude_only`` / ``budget`` / ``loop_*``
    fields the typed Step deliberately omits). The dropped fields are
    runtime-routing concerns, not contract concerns; they belong on
    the runner-internal :class:`MicroIntent`, not the on-the-wire
    :class:`Step` consumers see.

    Field map:

    ============================ ==============================
    ``MicroIntent`` field         ``Step`` field
    ============================ ==============================
    ``intent``                    ``intent``
    ``type``                      ``action_type``
    ``verify``                    ``expected_outcome``
    ``required``                  ``required``
    ``params``                    ``params``
    ``hints``                     ``hints``
    (derived from ``type``)       ``reversibility``
    (planner doesn't emit it)     ``confidence`` = 0.0
    ============================ ==============================
    """
    return Step(
        schema_version=SCHEMA_VERSION,
        intent=str(getattr(mi, "intent", "") or ""),
        action_type=str(getattr(mi, "type", "") or ""),
        reversibility=classify_legacy_reversibility(
            str(getattr(mi, "type", "") or ""),
        ),
        expected_outcome=str(getattr(mi, "verify", "") or ""),
        confidence=0.0,
        params=dict(getattr(mi, "params", {}) or {}),
        hints=dict(getattr(mi, "hints", {}) or {}),
        required=bool(getattr(mi, "required", False)),
    )


def action_result_from_action(
    action: "Action | None",
    *,
    dispatched: bool,
    dispatch_error: str = "",
    grounding_trace: dict[str, Any] | None = None,
    snapshot_id: str = "",
) -> ActionResult:
    """Project a legacy :class:`Action` (+ dispatcher outcome) onto
    :class:`ActionResult`.

    ``action=None`` is the deterministic-handler case (navigate /
    paginate / gate — handlers that don't synthesise an
    :class:`Action`). The adapter still produces a valid ActionResult
    with ``action_type=""`` so the canonical event can be emitted;
    the validator will reject events whose ``action_type`` is empty
    AND ``dispatched=False`` AND no ``dispatch_error`` — that
    combination means "nothing happened and nobody knows why", which
    isn't useful to record.

    ``snapshot_id`` (#484) carries the sandbox snapshot taken before
    this action dispatched. Empty when no snapshot was gated
    (reversible actions, no sandbox runtime wired). The runtime
    owns the format; readers treat as opaque.
    """
    if action is None:
        return ActionResult(
            schema_version=SCHEMA_VERSION,
            action_type="",
            params={},
            grounding_trace=grounding_trace or {},
            dispatched=dispatched,
            dispatch_error=dispatch_error,
            snapshot_id=snapshot_id,
        )
    action_type_value = getattr(action.action_type, "value", str(action.action_type))
    return ActionResult(
        schema_version=SCHEMA_VERSION,
        action_type=str(action_type_value),
        params=dict(action.params or {}),
        grounding_trace=grounding_trace or {},
        dispatched=dispatched,
        dispatch_error=dispatch_error,
        snapshot_id=snapshot_id,
    )


def verdict_from_step_result(r: "StepResult") -> Verdict:
    """Project a legacy :class:`~mantis_agent.gym.checkpoint.StepResult`
    onto :class:`Verdict` (#478, #480).

    Until #480 lands a typed verdict at the source, the runner records
    outcome as a tuple of ``(success: bool, failure_class: str,
    data: str)``. This adapter is the projection the canonical event
    emitter uses on every step:

    * ``success=True`` → :class:`VerdictKind.OK`. ``reason`` is empty
      (happy path needs no recovery code); ``evidence`` carries the
      runner's ``data`` string (often a brief "extracted 7 leads" /
      "navigated to /detail/123" note).
    * ``success=False`` + ``failure_class`` in
      :data:`_NON_RECOVERABLE_FAILURE_CLASSES` →
      :class:`VerdictKind.NON_RECOVERABLE`. Recovery loops on these
      classes burn budget without converging.
    * ``success=False`` everything else → :class:`VerdictKind.RECOVERABLE`.
      ``reason`` falls back to ``"unknown"`` when no class was
      stamped so the validator accepts the verdict (it requires a
      non-empty reason on failure verdicts).
    """
    if r.success:
        return Verdict(
            schema_version=SCHEMA_VERSION,
            kind=VerdictKind.OK,
            reason="",
            evidence=str(getattr(r, "data", "") or ""),
            confidence=1.0,
        )
    failure_class = str(getattr(r, "failure_class", "") or "")
    kind = (
        VerdictKind.NON_RECOVERABLE
        if failure_class in _NON_RECOVERABLE_FAILURE_CLASSES
        else VerdictKind.RECOVERABLE
    )
    return Verdict(
        schema_version=SCHEMA_VERSION,
        kind=kind,
        reason=failure_class or "unknown",
        evidence=str(getattr(r, "data", "") or ""),
        confidence=0.5 if kind == VerdictKind.RECOVERABLE else 0.9,
    )


def decide_recovery(
    verdict: Verdict,
    *,
    attempt_index: int = 0,
    required: bool = False,
    retry_budget: int = DEFAULT_RETRY_BUDGET,
) -> RecoveryDecision:
    """Project a typed verdict (+ runtime context) onto a typed
    :class:`RecoveryDecision` (#483).

    Mapping rules:

    * ``VerdictKind.OK`` → :class:`RecoveryDecision.ADVANCE`. Happy
      path — cursor moves to the next step regardless of attempt
      count.
    * ``VerdictKind.NON_RECOVERABLE`` →
      :class:`RecoveryDecision.TERMINATE`. The runner can't usefully
      retry these (cf_challenge, http_4xx, extractor_error,
      budget_exceeded — per
      :data:`_NON_RECOVERABLE_FAILURE_CLASSES`).
    * ``VerdictKind.RECOVERABLE`` + attempts still in budget →
      :class:`RecoveryDecision.RETRY`. The IntentRewriter /
      agentic_recovery / preview-gate hint layers get another shot.
    * ``VerdictKind.RECOVERABLE`` + attempts exhausted + ``required``
      → :class:`RecoveryDecision.TERMINATE`. Mirrors the existing
      "REQUIRED step failed after N retries — HALTING" behaviour.
    * ``VerdictKind.RECOVERABLE`` + attempts exhausted + not
      required → :class:`RecoveryDecision.ADVANCE`. The existing
      runner skips past non-required failures rather than halting;
      the typed decision matches that.

    The pure-function adapter doesn't yet drive control flow — it
    stamps a typed decision the runner / metrics / dashboards can
    read. Existing retry / halt branches keep operating on
    ``StepResult.success`` + ``failure_class``; future PRs migrate
    them to read this field.

    ``attempt_index`` is 0-based (first attempt = 0). The budget
    check is ``attempts_so_far >= retry_budget - 1`` because the
    current call is itself the ``attempt_index + 1``-th attempt.
    """
    if verdict.kind is VerdictKind.OK:
        return RecoveryDecision.ADVANCE
    if verdict.kind is VerdictKind.NON_RECOVERABLE:
        return RecoveryDecision.TERMINATE
    # RECOVERABLE — branch on retry budget + required.
    attempts_so_far = max(0, int(attempt_index)) + 1
    if attempts_so_far < max(1, int(retry_budget)):
        return RecoveryDecision.RETRY
    # Budget exhausted.
    return RecoveryDecision.TERMINATE if required else RecoveryDecision.ADVANCE


def observation_from_screenshot_ref(
    screenshot_ref: str,
    *,
    url: str = "",
    viewport: tuple[int, int] = (0, 0),
    captured_at: float = 0.0,
) -> Observation:
    """Convenience constructor for the runner emit-path.

    Centralises the contract that an Observation is *always* a
    reference, never a blob. Callers tempted to pass base64 here get
    rejected at validation time — by then the screenshot is already
    serialised and the cost is sunk. Building observations through
    this helper keeps the discipline in the writer instead of relying
    on the reader to catch it.
    """
    return Observation(
        schema_version=SCHEMA_VERSION,
        screenshot_ref=screenshot_ref,
        url=url,
        viewport=viewport,
        captured_at=captured_at,
    )
