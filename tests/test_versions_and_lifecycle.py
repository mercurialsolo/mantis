"""Tests for #488 (version stamps) + #486 (lifecycle substrate).

Covers:

* ``collect_versions`` always populates the static required keys
  (``action_ontology`` + ``contracts_schema``), pulls runtime
  stamps off env, and merges in per-runner model/prompt stamps.
* Validator enforces the required-set: empty dict, missing single
  key, empty value, non-string value all fail closed.
* Emitter auto-merges the static stamps so callers that don't
  pass ``versions=`` produce events that validate.
* ``LifecyclePhase`` enum + phase-class predicates pin the
  contract (5 ordered phases, 3 side-effectful, 2 pure).
* ``Activity`` protocol's ``runtime_checkable`` instance check
  works for objects that satisfy the shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mantis_agent.cua_contracts import (
    Activity,
    ContractValidationError,
    JSONL_FILENAME,
    LifecyclePhase,
    PURE_PHASES,
    SCHEMA_VERSION,
    SIDE_EFFECTFUL_PHASES,
    TrajectoryEmitter,
    VERSION_KEYS,
    collect_versions,
)
from mantis_agent.cua_contracts.validation import _validate_versions
from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.plan_decomposer import MicroIntent


# ── #488: collect_versions ─────────────────────────────────────────────


def test_collect_versions_always_includes_required_static_keys() -> None:
    """The two required keys (action_ontology + contracts_schema)
    must be present on every collect_versions() return — they're
    derived from module constants, not env, so a missing-env
    deploy still produces a validation-passing event."""
    v = collect_versions()
    assert "action_ontology" in v
    assert "contracts_schema" in v
    assert v["contracts_schema"] == f"v{SCHEMA_VERSION}"
    assert v["action_ontology"].startswith(f"v{SCHEMA_VERSION}.")


def test_collect_versions_reads_runtime_env_stamps(monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_BROWSER_IMAGE", "modal/chromium@abc123")
    monkeypatch.setenv("MANTIS_SANDBOX_RUNTIME", "modal/holo3")
    v = collect_versions()
    assert v["browser_image"] == "modal/chromium@abc123"
    assert v["sandbox_runtime"] == "modal/holo3"


def test_collect_versions_skips_missing_env_stamps(monkeypatch) -> None:
    """Missing env vars stay missing in the dict — they don't land
    as empty strings (the validator rejects empty values)."""
    monkeypatch.delenv("MANTIS_BROWSER_IMAGE", raising=False)
    monkeypatch.delenv("MANTIS_SANDBOX_RUNTIME", raising=False)
    v = collect_versions()
    assert "browser_image" not in v
    assert "sandbox_runtime" not in v


def test_collect_versions_merges_runtime_versions_from_runner() -> None:
    """Per-runner stamps that handlers stash on
    ``runner.runtime_versions`` get pulled in — that's how the
    planner / grounding / verifier model+prompt versions surface
    on every event without each handler reaching into the
    emitter."""
    class _Runner:
        runtime_versions = {
            "planner_model": "claude-opus-4-7",
            "grounding_prompt": "fft_v3",
            "actor_model": "holo3-35b-a3b",
        }

    v = collect_versions(_Runner())
    assert v["planner_model"] == "claude-opus-4-7"
    assert v["grounding_prompt"] == "fft_v3"
    assert v["actor_model"] == "holo3-35b-a3b"
    # Static stamps still present.
    assert "contracts_schema" in v


def test_collect_versions_skips_empty_runner_values() -> None:
    """Empty / non-string values from the runner get dropped — the
    validator would reject them anyway, and we want the dict to
    stay clean."""
    class _Runner:
        runtime_versions = {
            "planner_model": "",          # empty
            "actor_model": None,          # non-string
            "verifier_model": "haiku-4-5",
        }

    v = collect_versions(_Runner())
    assert "planner_model" not in v
    assert "actor_model" not in v
    assert v["verifier_model"] == "haiku-4-5"


def test_version_keys_documents_canonical_set() -> None:
    """The VERSION_KEYS tuple is the documented contract — pinned
    here so a stealth bump caused by an unintended import-order
    change fails CI."""
    assert "action_ontology" in VERSION_KEYS
    assert "contracts_schema" in VERSION_KEYS
    assert "planner_model" in VERSION_KEYS
    assert "grounding_model" in VERSION_KEYS
    assert "actor_model" in VERSION_KEYS
    assert "verifier_model" in VERSION_KEYS
    assert "browser_image" in VERSION_KEYS
    assert "sandbox_runtime" in VERSION_KEYS


# ── #488: validator required-set enforcement ──────────────────────────


def test_validate_versions_accepts_full_required_set() -> None:
    _validate_versions({
        "action_ontology": "v1.21",
        "contracts_schema": "v1",
    })


def test_validate_versions_rejects_empty_dict() -> None:
    with pytest.raises(ContractValidationError, match="missing required keys"):
        _validate_versions({})


def test_validate_versions_rejects_missing_action_ontology() -> None:
    with pytest.raises(ContractValidationError, match="action_ontology"):
        _validate_versions({"contracts_schema": "v1"})


def test_validate_versions_rejects_missing_contracts_schema() -> None:
    with pytest.raises(ContractValidationError, match="contracts_schema"):
        _validate_versions({"action_ontology": "v1.21"})


def test_validate_versions_rejects_empty_value() -> None:
    """Empty / whitespace value is a writer bug — readers can't
    distinguish 'not stamped' from 'stamped as blank'. Reject."""
    with pytest.raises(ContractValidationError, match="empty"):
        _validate_versions({
            "action_ontology": "v1.21",
            "contracts_schema": "v1",
            "planner_model": "   ",
        })


def test_validate_versions_rejects_non_string_value() -> None:
    with pytest.raises(ContractValidationError, match="must be str"):
        _validate_versions({
            "action_ontology": "v1.21",
            "contracts_schema": "v1",
            "planner_model": 42,  # type: ignore[dict-item]
        })


# ── #488: emitter auto-merges static stamps ────────────────────────────


def _intent() -> MicroIntent:
    return MicroIntent(intent="x", type="click", required=False)


def _ok_result() -> StepResult:
    return StepResult(step_index=0, intent="x", success=True, data="ok")


def test_emit_auto_merges_static_stamps_when_versions_unset(tmp_path: Path) -> None:
    """A caller that doesn't pass ``versions=`` still produces a
    validating event — the emitter merges in the static stamps
    via collect_versions()."""
    emitter = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    assert emitter.emit(_intent(), _ok_result()) is True
    record = json.loads((tmp_path / JSONL_FILENAME).read_text().strip())
    assert "action_ontology" in record["versions"]
    assert "contracts_schema" in record["versions"]


def test_emit_caller_versions_take_precedence(tmp_path: Path) -> None:
    """When the caller pins specific values, those win — even when
    they collide with the static stamps. Lets tests / shadow
    routing pin a particular contract version."""
    emitter = TrajectoryEmitter(
        run_id="r1", store_dir=str(tmp_path),
        versions={"contracts_schema": "v-test-override"},
    )
    emitter.emit(_intent(), _ok_result())
    record = json.loads((tmp_path / JSONL_FILENAME).read_text().strip())
    assert record["versions"]["contracts_schema"] == "v-test-override"


# ── #486: LifecyclePhase enum + protocol ──────────────────────────────


def test_lifecycle_phase_has_five_ordered_phases() -> None:
    """Pin the phase set + order. A bump is a contract change that
    must come with reader updates."""
    assert [p.value for p in LifecyclePhase] == [
        "plan", "observe", "act", "verify", "emit",
    ]


def test_side_effectful_phases_match_design() -> None:
    """Three phases mutate the outside world; the
    workflow/activity split routes these through the activity pool."""
    assert SIDE_EFFECTFUL_PHASES == {
        LifecyclePhase.OBSERVE,
        LifecyclePhase.ACT,
        LifecyclePhase.EMIT,
    }


def test_pure_phases_match_design() -> None:
    """Two phases are pure; they become workflow code (replayed
    deterministically by the orchestrator)."""
    assert PURE_PHASES == {LifecyclePhase.PLAN, LifecyclePhase.VERIFY}


def test_pure_and_side_effectful_phases_partition_the_enum() -> None:
    """Every phase is either pure OR side-effectful — never both,
    never neither."""
    assert PURE_PHASES.isdisjoint(SIDE_EFFECTFUL_PHASES)
    assert PURE_PHASES | SIDE_EFFECTFUL_PHASES == set(LifecyclePhase)


def test_activity_protocol_runtime_check_accepts_conforming_object() -> None:
    """runtime_checkable means an arbitrary object with the right
    shape satisfies isinstance — the future workflow orchestrator
    uses this for registration / dispatch."""

    class _Capture:
        @property
        def phase(self) -> LifecyclePhase:
            return LifecyclePhase.OBSERVE

        def execute(self, payload: dict) -> dict:
            return {"screenshot_ref": "runs/abc/step_0.png"}

    instance = _Capture()
    assert isinstance(instance, Activity)
    assert instance.phase is LifecyclePhase.OBSERVE
    out = instance.execute({"step_index": 0})
    assert out == {"screenshot_ref": "runs/abc/step_0.png"}


def test_activity_protocol_runtime_check_rejects_non_conforming_object() -> None:
    """An object without both ``phase`` AND ``execute`` is not an
    Activity. Protocol's structural typing catches this at the
    registration boundary."""

    class _NoExecute:
        @property
        def phase(self) -> LifecyclePhase:
            return LifecyclePhase.ACT

    class _NoPhase:
        def execute(self, payload: dict) -> dict:
            return {}

    assert not isinstance(_NoExecute(), Activity)
    assert not isinstance(_NoPhase(), Activity)
