"""Action ontology + reversibility classifier (#477).

The canonical action vocabulary the CUA contracts pin against.
Centralising it here means:

* :issue:`476` typed Step's ``action_type`` field has a closed,
  validated vocabulary;
* :issue:`478` canonical event emitter and :issue:`479` grounding
  trace consumers can dispatch by action class without re-deriving
  semantics from prose;
* :issue:`481` preview gate can ask one question ("is this
  IRREVERSIBLE?") and route deterministically;
* future planners that emit raw strings get a clear rejection at
  validation time instead of producing silently-dispatched
  free-form actions.

The vocabulary is the *union* of two pre-existing surfaces:

* :class:`~mantis_agent.actions.ActionType` — the brain-emitted
  vocabulary (click / type_text / scroll / wait / done / ...).
* :class:`~mantis_agent.plan_decomposer.MicroIntent`'s ``type`` —
  the plan-step vocabulary (navigate / fill_field / submit /
  extract_data / paginate / loop / ...).

The brain vocabulary describes *what the actor did*; the plan
vocabulary describes *what the planner asked for*. The two
intersect (CLICK / TYPE / SCROLL / WAIT / DONE) and diverge — the
ontology unifies them. The reversibility class is plan-side: it
asks "if this step is dispatched and turns out wrong, can the
runner undo it cheaply?".

The vocabulary is intentionally narrow for v1. Aliases (e.g. some
plans use ``"click_button"`` instead of ``"submit"``) and
domain-specific verbs (``"confirm_purchase"``, ``"agree_tos"``)
are NOT separate enum values — they map onto the closest canonical
verb at the adapter boundary. The reversibility classifier is
where any domain refinement goes (e.g. a future "confirm_purchase"
keyword detector could escalate a ``submit`` to
:class:`ReversibilityClass.IRREVERSIBLE` based on label
context — but that lives in a separate layer, not the enum).
"""

from __future__ import annotations

from enum import Enum

from .types import ReversibilityClass
from .validation import ContractValidationError


class ActionTyped(str, Enum):
    """Canonical action vocabulary the CUA contracts validate against.

    String values match the existing
    :class:`~mantis_agent.actions.ActionType` enum where they
    overlap, so callers passing the legacy enum's ``.value`` keep
    working without conversion. Plan-side verbs that have no brain
    counterpart (``fill_field`` is a *plan* op that decomposes into
    a CLICK + TYPE pair at the actor) carry their plan-side name
    here so the typed Step in :mod:`.types` round-trips cleanly
    from a MicroIntent.

    Bump on new value addition: this is a closed vocabulary, but
    additive changes are backward-compatible because the validator
    only rejects *unknown* values, not unrecognised-but-defined ones.
    """

    # Pointer + keyboard primitives (brain vocabulary).
    CLICK = "click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    TYPE = "type_text"
    KEY_PRESS = "key_press"
    SCROLL = "scroll"
    DRAG = "drag"
    WAIT = "wait"
    DONE = "done"
    LAUNCH_APP = "launch_app"
    TOOL_CALL = "tool_call"

    # Plan-level verbs (MicroIntent.type).
    NAVIGATE = "navigate"
    NAVIGATE_BACK = "navigate_back"
    FILL_FIELD = "fill_field"
    SELECT_OPTION = "select_option"
    SUBMIT = "submit"
    EXTRACT_DATA = "extract_data"
    EXTRACT_URL = "extract_url"
    PAGINATE = "paginate"
    FILTER = "filter"
    LOOP = "loop"


# Reversibility classification per action type. Pinned here so the
# preview-gate / safety-gate / recovery-policy layers all read from
# the same map.
#
# Rationale per class:
#
# * READ_ONLY — pure observation actions. No env mutation possible
#   short of the screenshot capture itself. Preview gate skips these
#   entirely.
# * REVERSIBLE — pointer / keyboard / navigation actions that can be
#   undone by alt+Left / Esc / Backspace or a re-navigation. Preview
#   gate may run as advisory but doesn't block.
# * IRREVERSIBLE — actions that commit state the runner can't
#   programmatically undo: form submissions, file uploads, app
#   launches, ToS / purchase / send / delete confirmations dressed
#   as a submit. Preview gate is MANDATORY before dispatch (#481).
_REVERSIBILITY_MAP: dict[ActionTyped, ReversibilityClass] = {
    # Read-only.
    ActionTyped.SCROLL: ReversibilityClass.READ_ONLY,
    ActionTyped.WAIT: ReversibilityClass.READ_ONLY,
    ActionTyped.EXTRACT_DATA: ReversibilityClass.READ_ONLY,
    ActionTyped.EXTRACT_URL: ReversibilityClass.READ_ONLY,
    ActionTyped.LOOP: ReversibilityClass.READ_ONLY,
    # Reversible.
    ActionTyped.CLICK: ReversibilityClass.REVERSIBLE,
    ActionTyped.RIGHT_CLICK: ReversibilityClass.REVERSIBLE,
    ActionTyped.DOUBLE_CLICK: ReversibilityClass.REVERSIBLE,
    ActionTyped.TYPE: ReversibilityClass.REVERSIBLE,
    ActionTyped.KEY_PRESS: ReversibilityClass.REVERSIBLE,
    ActionTyped.DRAG: ReversibilityClass.REVERSIBLE,
    ActionTyped.NAVIGATE: ReversibilityClass.REVERSIBLE,
    ActionTyped.NAVIGATE_BACK: ReversibilityClass.REVERSIBLE,
    ActionTyped.FILL_FIELD: ReversibilityClass.REVERSIBLE,
    ActionTyped.SELECT_OPTION: ReversibilityClass.REVERSIBLE,
    ActionTyped.FILTER: ReversibilityClass.REVERSIBLE,
    ActionTyped.PAGINATE: ReversibilityClass.REVERSIBLE,
    ActionTyped.DONE: ReversibilityClass.REVERSIBLE,
    ActionTyped.TOOL_CALL: ReversibilityClass.REVERSIBLE,
    # Irreversible — preview gate mandatory.
    ActionTyped.SUBMIT: ReversibilityClass.IRREVERSIBLE,
    ActionTyped.LAUNCH_APP: ReversibilityClass.IRREVERSIBLE,
}


# Sanity-check: every enum member has a reversibility class. Caught
# at module import time so a future enum addition without an
# accompanying map entry fails immediately rather than silently
# routing through the REVERSIBLE default in production.
_uncovered = set(ActionTyped) - set(_REVERSIBILITY_MAP)
if _uncovered:
    raise RuntimeError(
        f"ActionTyped members missing from _REVERSIBILITY_MAP: "
        f"{sorted(a.value for a in _uncovered)} — add an entry "
        f"before importing this module."
    )


def validate_action_type(value: str) -> ActionTyped:
    """Parse a raw string into a typed :class:`ActionTyped`.

    Raises :class:`~.validation.ContractValidationError` when the
    string isn't a known vocabulary member. Empty / None inputs
    also raise so the validator catches them at the planner
    boundary rather than letting them pass through to the
    dispatcher.

    The error message lists the closed vocabulary so the caller
    (typically a planner / decomposer) can correct without
    spelunking.
    """
    if not value or not isinstance(value, str):
        raise ContractValidationError(
            f"action_type must be a non-empty string; got {value!r}"
        )
    try:
        return ActionTyped(value)
    except ValueError:
        known = sorted(member.value for member in ActionTyped)
        raise ContractValidationError(
            f"unknown action_type {value!r}; must be one of {known}"
        ) from None


def classify_action(action_type: str | ActionTyped) -> ReversibilityClass:
    """Return the reversibility class for an action type.

    Accepts either a raw string (which is validated against the
    ontology) or an already-parsed :class:`ActionTyped`. Unknown
    raw strings raise :class:`ContractValidationError` — the safety
    contract is "fail closed on unknown", since a default
    reversibility guess could let a hidden IRREVERSIBLE action
    through the preview gate.
    """
    if isinstance(action_type, ActionTyped):
        return _REVERSIBILITY_MAP[action_type]
    parsed = validate_action_type(action_type)
    return _REVERSIBILITY_MAP[parsed]


def is_irreversible(action_type: str | ActionTyped) -> bool:
    """Convenience predicate for the preview-gate dispatcher (#481).

    Equivalent to ``classify_action(...) ==
    ReversibilityClass.IRREVERSIBLE`` but reads more cleanly at
    the callsite. Same fail-closed contract: unknown strings raise
    rather than silently returning False.
    """
    return classify_action(action_type) is ReversibilityClass.IRREVERSIBLE
