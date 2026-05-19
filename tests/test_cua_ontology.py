"""Tests for the action ontology + reversibility classifier (#477).

Covers:

* the ``ActionTyped`` enum covers every legacy ``MicroIntent.type``
  value the runtime accepts (no silent drops);
* every enum member has a reversibility class registered (the
  import-time sanity check catches stealth additions);
* high-risk actions (submit / launch_app) classify as IRREVERSIBLE,
  pointer / keyboard / nav actions as REVERSIBLE, scroll / extract
  / loop as READ_ONLY;
* unknown action strings raise :class:`ContractValidationError`
  with a useful message (vocabulary listed);
* the legacy adapter shim (``classify_legacy_reversibility``) now
  delegates to the ontology and preserves its pre-existing
  fail-soft contract (REVERSIBLE on unknown, vs the strict path's
  raise).
"""

from __future__ import annotations

import pytest

from mantis_agent.cua_contracts import (
    ActionTyped,
    ContractValidationError,
    ReversibilityClass,
    classify_action,
    classify_legacy_reversibility,
    is_irreversible,
    validate_action_type,
)
from mantis_agent.cua_contracts.ontology import _REVERSIBILITY_MAP


# ── Enum coverage ──────────────────────────────────────────────────────


def test_every_enum_member_has_reversibility_class() -> None:
    """Sanity check the module's import-time guard would also catch —
    pinned in a test so the failure mode is explicit if someone bumps
    the enum and skips the map update."""
    assert set(_REVERSIBILITY_MAP) == set(ActionTyped)


def test_enum_covers_brain_vocabulary() -> None:
    """ActionType values from ``mantis_agent.actions.ActionType`` must
    all round-trip through the ontology — the brain emits these and
    the contracts have to accept them without a translation layer."""
    from mantis_agent.actions import ActionType
    for member in ActionType:
        # Every brain action must be in the ontology by its value.
        assert validate_action_type(member.value) is not None


def test_enum_covers_plan_step_types() -> None:
    """Every step type a MicroPlanRunner dispatches must be in the
    ontology. List mirrors MicroIntent's docstring."""
    plan_step_types = [
        "navigate", "navigate_back", "click", "right_click",
        "fill_field", "submit", "select_option",
        "extract_data", "extract_url",
        "scroll", "paginate", "filter", "loop",
    ]
    for st in plan_step_types:
        assert validate_action_type(st) is not None


# ── Reversibility classification ───────────────────────────────────────


@pytest.mark.parametrize(
    "action_type,expected",
    [
        # Read-only
        ("scroll", ReversibilityClass.READ_ONLY),
        ("wait", ReversibilityClass.READ_ONLY),
        ("extract_data", ReversibilityClass.READ_ONLY),
        ("extract_url", ReversibilityClass.READ_ONLY),
        ("loop", ReversibilityClass.READ_ONLY),
        # Reversible
        ("click", ReversibilityClass.REVERSIBLE),
        ("right_click", ReversibilityClass.REVERSIBLE),
        ("double_click", ReversibilityClass.REVERSIBLE),
        ("type_text", ReversibilityClass.REVERSIBLE),
        ("key_press", ReversibilityClass.REVERSIBLE),
        ("drag", ReversibilityClass.REVERSIBLE),
        ("navigate", ReversibilityClass.REVERSIBLE),
        ("navigate_back", ReversibilityClass.REVERSIBLE),
        ("fill_field", ReversibilityClass.REVERSIBLE),
        ("select_option", ReversibilityClass.REVERSIBLE),
        ("filter", ReversibilityClass.REVERSIBLE),
        ("paginate", ReversibilityClass.REVERSIBLE),
        ("done", ReversibilityClass.REVERSIBLE),
        ("tool_call", ReversibilityClass.REVERSIBLE),
        # Irreversible
        ("submit", ReversibilityClass.IRREVERSIBLE),
        ("launch_app", ReversibilityClass.IRREVERSIBLE),
    ],
)
def test_classify_action_pins_full_map(
    action_type: str, expected: ReversibilityClass,
) -> None:
    assert classify_action(action_type) is expected


def test_is_irreversible_predicate_matches_classify() -> None:
    assert is_irreversible("submit") is True
    assert is_irreversible("launch_app") is True
    assert is_irreversible("click") is False
    assert is_irreversible("scroll") is False


def test_classify_action_accepts_typed_enum_directly() -> None:
    """Callers that already have a typed value shouldn't re-validate
    via the string path — accept the enum directly."""
    assert classify_action(ActionTyped.SUBMIT) is ReversibilityClass.IRREVERSIBLE
    assert classify_action(ActionTyped.CLICK) is ReversibilityClass.REVERSIBLE


# ── Validation failure modes ───────────────────────────────────────────


def test_validate_action_type_rejects_unknown_with_vocabulary_list() -> None:
    """Error message must list the known vocabulary so the caller
    can correct without digging into source."""
    with pytest.raises(ContractValidationError) as exc_info:
        validate_action_type("frobnicate_lead")
    msg = str(exc_info.value)
    assert "frobnicate_lead" in msg
    assert "submit" in msg  # the listed vocabulary must contain the canonical members
    assert "click" in msg


@pytest.mark.parametrize("value", ["", None])
def test_validate_action_type_rejects_empty(value) -> None:
    with pytest.raises(ContractValidationError, match="non-empty"):
        validate_action_type(value)  # type: ignore[arg-type]


def test_classify_action_fails_closed_on_unknown() -> None:
    """Strict path: classify_action raises on unknown — preview gate
    relies on this to never silently let an irreversible action
    through under a default."""
    with pytest.raises(ContractValidationError):
        classify_action("some_new_verb_we_dont_know")


def test_is_irreversible_fails_closed_on_unknown() -> None:
    with pytest.raises(ContractValidationError):
        is_irreversible("some_new_verb_we_dont_know")


# ── Legacy adapter shim preserves fail-soft contract ───────────────────


def test_classify_legacy_reversibility_still_fail_soft_on_unknown() -> None:
    """``classify_legacy_reversibility`` is the back-compat shim —
    the existing executor falls through to Holo3 on unknown step
    types, so the shim returns REVERSIBLE rather than raising.
    Callers that want strict validation should use
    :func:`classify_action` directly."""
    assert (
        classify_legacy_reversibility("not_in_the_ontology")
        is ReversibilityClass.REVERSIBLE
    )


def test_classify_legacy_reversibility_routes_through_ontology() -> None:
    """The shim now delegates to the ontology — pinned mappings hold."""
    assert classify_legacy_reversibility("submit") is ReversibilityClass.IRREVERSIBLE
    assert classify_legacy_reversibility("scroll") is ReversibilityClass.READ_ONLY
    assert classify_legacy_reversibility("click") is ReversibilityClass.REVERSIBLE
