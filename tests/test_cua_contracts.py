"""Tests for the versioned CUA contracts (#476).

Covers:

* validator acceptance / rejection for TaskSpec, Step, Verdict, and
  TrajectoryEvent (the four types the acceptance criteria call out);
* the inline-blob rejection guard on Observation;
* the legacy-MicroIntent → Step adapter — field projection +
  reversibility classification;
* the legacy-Action → ActionResult adapter — including the
  ``action=None`` deterministic-handler case.
"""

from __future__ import annotations

import pytest

from mantis_agent.actions import Action, ActionType
from mantis_agent.cua_contracts import (
    ActionResult,
    ContractValidationError,
    Observation,
    ReversibilityClass,
    SCHEMA_VERSION,
    Step,
    TaskSpec,
    TrajectoryEvent,
    Verdict,
    VerdictKind,
    action_result_from_action,
    classify_legacy_reversibility,
    observation_from_screenshot_ref,
    step_from_micro_intent,
    validate_step,
    validate_task_spec,
    validate_trajectory_event,
    validate_verdict,
)
from mantis_agent.plan_decomposer import MicroIntent


# ── Constants pinning ──────────────────────────────────────────────────


def test_schema_version_pinned_at_one() -> None:
    """The on-the-wire schema is v1. A bump is a coordinated reader/
    writer migration — pinned here so a stealth increment fails CI."""
    assert SCHEMA_VERSION == 1


# ── TaskSpec validation ────────────────────────────────────────────────


def _valid_task_spec(**overrides) -> TaskSpec:
    base = {
        "schema_version": SCHEMA_VERSION,
        "task_id": "t_123",
        "goal": "Extract the top 10 listings",
        "reversibility_policy": "prompt_on_irreversible",
    }
    base.update(overrides)
    return TaskSpec(**base)


def test_valid_task_spec_validates() -> None:
    validate_task_spec(_valid_task_spec())


def test_task_spec_missing_task_id_rejected() -> None:
    with pytest.raises(ContractValidationError, match="task_id"):
        validate_task_spec(_valid_task_spec(task_id=""))


def test_task_spec_missing_goal_rejected() -> None:
    with pytest.raises(ContractValidationError, match="goal"):
        validate_task_spec(_valid_task_spec(goal=""))


def test_task_spec_unknown_reversibility_policy_rejected() -> None:
    with pytest.raises(ContractValidationError, match="reversibility_policy"):
        validate_task_spec(_valid_task_spec(reversibility_policy="yolo"))


def test_task_spec_schema_version_mismatch_rejected() -> None:
    """An old writer producing v0 events must NOT be silently upgraded —
    the reader has to opt in to a migration explicitly."""
    with pytest.raises(ContractValidationError, match="schema_version"):
        validate_task_spec(_valid_task_spec(schema_version=0))


# ── Step validation ────────────────────────────────────────────────────


def _valid_step(**overrides) -> Step:
    base = {
        "intent": "Click Sign Up",
        "action_type": "click",
        "reversibility": ReversibilityClass.REVERSIBLE,
    }
    base.update(overrides)
    return Step(**base)


def test_valid_step_validates() -> None:
    validate_step(_valid_step())


def test_step_missing_intent_rejected() -> None:
    with pytest.raises(ContractValidationError, match="intent"):
        validate_step(_valid_step(intent=""))


def test_step_missing_action_type_rejected() -> None:
    with pytest.raises(ContractValidationError, match="action_type"):
        validate_step(_valid_step(action_type=""))


# ── Verdict validation ────────────────────────────────────────────────


def test_ok_verdict_can_omit_reason() -> None:
    """Happy-path verdicts don't carry a recovery code — that's the
    point of the typed kind enum."""
    validate_verdict(Verdict(kind=VerdictKind.OK))


def test_failure_verdict_requires_reason() -> None:
    """Pre-#480 demotions sometimes shipped without a reason — the
    recovery policy then routed to ``unknown``, which the
    IntentRewriter and critic can't do anything with."""
    with pytest.raises(ContractValidationError, match="Verdict.reason"):
        validate_verdict(Verdict(kind=VerdictKind.RECOVERABLE, reason=""))


def test_non_recoverable_verdict_requires_reason() -> None:
    with pytest.raises(ContractValidationError, match="Verdict.reason"):
        validate_verdict(Verdict(kind=VerdictKind.NON_RECOVERABLE, reason=""))


# ── TrajectoryEvent validation ────────────────────────────────────────


def _valid_event(**overrides) -> TrajectoryEvent:
    base = {
        "run_id": "run_abc",
        "step_index": 0,
        "step": _valid_step(),
        "observation": observation_from_screenshot_ref(
            "runs/run_abc/step_0.png", url="https://example.com",
        ),
        "action_result": ActionResult(
            action_type="click", params={"x": 1, "y": 2}, dispatched=True,
        ),
        "verdict": Verdict(kind=VerdictKind.OK),
        "versions": {"planner": "claude-opus-4-7"},
    }
    base.update(overrides)
    return TrajectoryEvent(**base)


def test_valid_trajectory_event_validates() -> None:
    validate_trajectory_event(_valid_event())


def test_trajectory_event_missing_run_id_rejected() -> None:
    with pytest.raises(ContractValidationError, match="run_id"):
        validate_trajectory_event(_valid_event(run_id=""))


def test_trajectory_event_negative_step_index_rejected() -> None:
    """``step_index=-1`` is the dataclass default — without an explicit
    value the event isn't deterministically addressable."""
    with pytest.raises(ContractValidationError, match="step_index"):
        validate_trajectory_event(_valid_event(step_index=-1))


def test_trajectory_event_missing_verdict_rejected() -> None:
    """The acceptance criterion: events emit only AFTER the verdict
    lands. ``verdict=None`` means the runner advanced before
    verifying — that's the bug #480 closes structurally."""
    with pytest.raises(ContractValidationError, match="verdict is required"):
        validate_trajectory_event(_valid_event(verdict=None))


def test_trajectory_event_missing_action_result_rejected() -> None:
    with pytest.raises(ContractValidationError, match="action_result"):
        validate_trajectory_event(_valid_event(action_result=None))


def test_trajectory_event_missing_observation_rejected() -> None:
    with pytest.raises(ContractValidationError, match="observation"):
        validate_trajectory_event(_valid_event(observation=None))


def test_trajectory_event_missing_versions_dict_rejected() -> None:
    """``versions`` is the slot for #487 / #488 model + prompt stamps.
    The dict must exist even when empty so consumers don't crash on
    None."""
    with pytest.raises(ContractValidationError, match="versions"):
        validate_trajectory_event(_valid_event(versions=None))  # type: ignore[arg-type]


def test_trajectory_event_empty_versions_dict_accepted() -> None:
    """Empty versions dict is fine in v1 — the shape isn't pinned until
    the version-stamping work lands. Presence is what matters."""
    validate_trajectory_event(_valid_event(versions={}))


def test_observation_inline_base64_blob_rejected() -> None:
    """``Observation.screenshot_ref`` must be a reference. A large
    string starting with the standard PNG-base64 prefix is the
    canonical mistake the validator catches."""
    blob = "iVBORw0KGgo" + ("A" * 2000)
    obs = Observation(schema_version=SCHEMA_VERSION, screenshot_ref=blob)
    with pytest.raises(ContractValidationError, match="inline base64"):
        validate_trajectory_event(_valid_event(observation=obs))


def test_observation_missing_screenshot_ref_rejected() -> None:
    obs = Observation(schema_version=SCHEMA_VERSION, screenshot_ref="")
    with pytest.raises(ContractValidationError, match="screenshot_ref"):
        validate_trajectory_event(_valid_event(observation=obs))


def test_action_result_missing_action_type_rejected() -> None:
    """Empty action_type is allowed on the adapter for deterministic
    handlers, but the validator catches it on the way to persistence —
    by then the runner has had a chance to fill it in (or to mark
    dispatched=False with a dispatch_error)."""
    ar = ActionResult(action_type="", dispatched=True)
    with pytest.raises(ContractValidationError, match="action_type"):
        validate_trajectory_event(_valid_event(action_result=ar))


def test_action_result_undispatched_without_error_rejected() -> None:
    """``dispatched=False`` + ``dispatch_error=""`` means "nothing
    happened and nobody knows why" — useless to record."""
    ar = ActionResult(action_type="click", dispatched=False, dispatch_error="")
    with pytest.raises(ContractValidationError, match="dispatch_error"):
        validate_trajectory_event(_valid_event(action_result=ar))


# ── MicroIntent → Step adapter ────────────────────────────────────────


def test_step_from_micro_intent_projects_all_relevant_fields() -> None:
    mi = MicroIntent(
        intent="Click the Sign Up button",
        type="submit",
        verify="URL contains /signup-success",
        budget=8,
        required=True,
        params={"label": "Sign Up", "aliases": ["Register"]},
        hints={"region": "header"},
    )
    step = step_from_micro_intent(mi)
    assert step.intent == "Click the Sign Up button"
    assert step.action_type == "submit"
    assert step.expected_outcome == "URL contains /signup-success"
    assert step.required is True
    assert step.params == {"label": "Sign Up", "aliases": ["Register"]}
    assert step.hints == {"region": "header"}
    # ``submit`` is irreversible.
    assert step.reversibility == ReversibilityClass.IRREVERSIBLE
    # Planner doesn't emit confidence today — adapter defaults to 0.
    assert step.confidence == 0.0


def test_step_adapter_treats_unknown_type_as_reversible_for_safety() -> None:
    """Fail-safe: an unrecognised step type goes through the preview
    gate (REVERSIBLE) rather than skipping it (IRREVERSIBLE would
    over-trigger; READ_ONLY would skip the gate)."""
    mi = MicroIntent(intent="poke at the iframe", type="poke")
    step = step_from_micro_intent(mi)
    assert step.reversibility == ReversibilityClass.REVERSIBLE


@pytest.mark.parametrize(
    "step_type,expected",
    [
        ("scroll", ReversibilityClass.READ_ONLY),
        ("extract_data", ReversibilityClass.READ_ONLY),
        ("extract_url", ReversibilityClass.READ_ONLY),
        ("click", ReversibilityClass.REVERSIBLE),
        ("navigate", ReversibilityClass.REVERSIBLE),
        ("fill_field", ReversibilityClass.REVERSIBLE),
        ("select_option", ReversibilityClass.REVERSIBLE),
        ("submit", ReversibilityClass.IRREVERSIBLE),
    ],
)
def test_classify_legacy_reversibility_covers_canonical_types(
    step_type: str, expected: ReversibilityClass,
) -> None:
    """Pin the reversibility map for the canonical legacy step types
    — a stealth change to the map alters which steps get the preview
    gate, which is a safety-relevant decision."""
    assert classify_legacy_reversibility(step_type) is expected


# ── Action → ActionResult adapter ─────────────────────────────────────


def test_action_result_from_action_carries_type_and_params() -> None:
    action = Action(
        action_type=ActionType.CLICK,
        params={"x": 100, "y": 200, "button": "left"},
        reasoning="click sign up",
    )
    ar = action_result_from_action(action, dispatched=True)
    assert ar.action_type == "click"
    assert ar.params == {"x": 100, "y": 200, "button": "left"}
    assert ar.dispatched is True
    assert ar.dispatch_error == ""


def test_action_result_from_action_records_dispatch_failure() -> None:
    action = Action(action_type=ActionType.CLICK, params={"x": 0, "y": 0})
    ar = action_result_from_action(
        action, dispatched=False,
        dispatch_error="elementFromPoint=BUTTON (refused)",
    )
    assert ar.dispatched is False
    assert ar.dispatch_error == "elementFromPoint=BUTTON (refused)"


def test_action_result_from_none_action_is_valid_for_deterministic_handlers() -> None:
    """Deterministic handlers (navigate / paginate / gate) don't
    synthesise an Action — the adapter accepts ``None`` so they can
    still emit a canonical event. The result will fail validation if
    nobody fills in ``action_type`` later, which is the right safety
    default."""
    ar = action_result_from_action(None, dispatched=True)
    assert ar.action_type == ""
    assert ar.dispatched is True


def test_action_result_from_action_carries_grounding_trace() -> None:
    """The grounding trace dict round-trips so #479's structured
    payload can ride alongside the action without a second adapter."""
    action = Action(action_type=ActionType.CLICK, params={"x": 1, "y": 2})
    trace = {"provider": "claude", "confidence": 0.87, "selector": "som#42"}
    ar = action_result_from_action(
        action, dispatched=True, grounding_trace=trace,
    )
    assert ar.grounding_trace == trace
