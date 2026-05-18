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
    ReversibilityClass,
    SCHEMA_VERSION,
    Step,
)

if TYPE_CHECKING:
    from ..actions import Action
    from ..plan_decomposer import MicroIntent


# Mapping from legacy MicroIntent.type strings to a coarse
# reversibility class. The list of step types lives in
# ``MicroIntent``'s docstring; we preserve the runtime behaviour the
# step-recovery / preview-gate code already implements implicitly.
# Anything not listed defaults to REVERSIBLE so unknown step types
# fail safe rather than fail open (the preview gate will run, not
# the auto-dispatch path). #477 will refine this with a proper
# action ontology.
_LEGACY_TYPE_TO_REVERSIBILITY: dict[str, ReversibilityClass] = {
    # Read-only — no side effects.
    "scroll": ReversibilityClass.READ_ONLY,
    "extract_data": ReversibilityClass.READ_ONLY,
    "extract_url": ReversibilityClass.READ_ONLY,
    # Reversible by alt+Left / Escape / Backspace.
    "click": ReversibilityClass.REVERSIBLE,
    "navigate": ReversibilityClass.REVERSIBLE,
    "navigate_back": ReversibilityClass.REVERSIBLE,
    "fill_field": ReversibilityClass.REVERSIBLE,
    "select_option": ReversibilityClass.REVERSIBLE,
    "right_click": ReversibilityClass.REVERSIBLE,
    "filter": ReversibilityClass.REVERSIBLE,
    "paginate": ReversibilityClass.REVERSIBLE,
    "loop": ReversibilityClass.READ_ONLY,
    # Irreversible — pre-dispatch preview gate should always run.
    # ``submit`` is the canonical write action; the rest are aliases
    # plans use for the same semantic (#477 will collapse them).
    "submit": ReversibilityClass.IRREVERSIBLE,
}


def classify_legacy_reversibility(step_type: str) -> ReversibilityClass:
    """Public helper — also useful from tests / safety gates."""
    return _LEGACY_TYPE_TO_REVERSIBILITY.get(
        step_type, ReversibilityClass.REVERSIBLE,
    )


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
    """
    if action is None:
        return ActionResult(
            schema_version=SCHEMA_VERSION,
            action_type="",
            params={},
            grounding_trace=grounding_trace or {},
            dispatched=dispatched,
            dispatch_error=dispatch_error,
        )
    action_type_value = getattr(action.action_type, "value", str(action.action_type))
    return ActionResult(
        schema_version=SCHEMA_VERSION,
        action_type=str(action_type_value),
        params=dict(action.params or {}),
        grounding_trace=grounding_trace or {},
        dispatched=dispatched,
        dispatch_error=dispatch_error,
    )


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
