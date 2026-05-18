"""Tests for the canonical trajectory emitter (#478).

Covers:

* the StepResult → Verdict adapter (success / recoverable /
  non-recoverable mapping);
* TrajectoryEmitter end-to-end — JSONL round-trip, validation
  rejection, idempotency on (run_id, step_index), and resume from
  an existing JSONL store;
* the opt-in env-var gate so the runner stays a no-op when the
  feature is disabled.

These tests deliberately don't spin up a runner. The emitter takes
plain MicroIntent + StepResult inputs and the test focuses on the
projection + persistence contract — runner integration is exercised
by the existing executor test suite which doesn't assert on emit
behaviour yet (gated by env, default off).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mantis_agent.actions import Action, ActionType
from mantis_agent.cua_contracts import (
    JSONL_FILENAME,
    SCHEMA_VERSION,
    TrajectoryEmitter,
    VerdictKind,
    verdict_from_step_result,
)
from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.plan_decomposer import MicroIntent


# ── Verdict adapter ─────────────────────────────────────────────────────


def _ok_step_result(index: int = 0, data: str = "extracted 7 leads") -> StepResult:
    return StepResult(
        step_index=index, intent="extract", success=True,
        data=data, duration=1.5,
    )


def _fail_step_result(
    *, index: int = 0, failure_class: str = "", data: str = "fill_error",
) -> StepResult:
    return StepResult(
        step_index=index, intent="fill", success=False,
        data=data, duration=2.0, failure_class=failure_class,
    )


def test_verdict_from_success_maps_to_ok() -> None:
    v = verdict_from_step_result(_ok_step_result(data="navigated to /detail/123"))
    assert v.kind is VerdictKind.OK
    assert v.reason == ""  # happy path needs no recovery code
    assert v.evidence == "navigated to /detail/123"
    assert v.confidence == 1.0


def test_verdict_from_failure_with_recoverable_class() -> None:
    """Unknown / generic failure classes route to RECOVERABLE so the
    retry / recovery / replan ladder gets a shot."""
    v = verdict_from_step_result(
        _fail_step_result(failure_class="selector_miss", data="not found"),
    )
    assert v.kind is VerdictKind.RECOVERABLE
    assert v.reason == "selector_miss"
    assert v.evidence == "not found"


def test_verdict_from_failure_with_no_failure_class_falls_back_to_unknown() -> None:
    """The validator requires a non-empty reason on failure verdicts —
    the adapter must back-fill ``unknown`` when the runner didn't
    classify the failure."""
    v = verdict_from_step_result(_fail_step_result(failure_class=""))
    assert v.kind is VerdictKind.RECOVERABLE
    assert v.reason == "unknown"


@pytest.mark.parametrize(
    "failure_class",
    ["cf_challenge", "http_4xx", "extractor_error", "budget_exceeded"],
)
def test_verdict_from_known_non_recoverable_failures(failure_class: str) -> None:
    """Pinned set of failure classes route to NON_RECOVERABLE. A
    bump to the set is intentional + grep-able from this test."""
    v = verdict_from_step_result(_fail_step_result(failure_class=failure_class))
    assert v.kind is VerdictKind.NON_RECOVERABLE
    assert v.reason == failure_class


# ── TrajectoryEmitter ──────────────────────────────────────────────────


def _intent(step_type: str = "click", intent: str = "Click Sign Up") -> MicroIntent:
    return MicroIntent(intent=intent, type=step_type, required=True)


def test_emit_writes_one_validated_jsonl_line(tmp_path: Path) -> None:
    """Happy path: emit projects + validates + appends one line."""
    emitter = TrajectoryEmitter(run_id="run_abc", store_dir=str(tmp_path))
    intent = _intent()
    result = _ok_step_result(index=0)

    assert emitter.emit(intent, result) is True

    jsonl = (tmp_path / JSONL_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(jsonl) == 1
    record = json.loads(jsonl[0])
    assert record["run_id"] == "run_abc"
    assert record["step_index"] == 0
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["verdict"]["kind"] == "ok"
    assert record["step"]["action_type"] == "click"
    assert record["observation"]["screenshot_ref"].startswith("placeholder://")
    assert record["committed"] is True


def test_emit_idempotent_on_same_step_index(tmp_path: Path) -> None:
    """Acceptance criterion: every completed step emits exactly one
    canonical event, even across retries."""
    emitter = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    intent = _intent()
    r0 = _ok_step_result(index=0)

    assert emitter.emit(intent, r0) is True
    # Same key — must not append a second line.
    assert emitter.emit(intent, r0) is False
    # Different StepResult instance, same step_index — also a no-op.
    r0_retry = _ok_step_result(index=0, data="rerun-evidence")
    assert emitter.emit(intent, r0_retry) is False

    lines = (tmp_path / JSONL_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_emit_resume_loads_existing_indices(tmp_path: Path) -> None:
    """Acceptance criterion: idempotent across restarts. A new
    emitter pointed at an existing JSONL must pick up the prior
    indices and refuse to re-emit them."""
    first = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    intent = _intent()
    first.emit(intent, _ok_step_result(index=0))
    first.emit(intent, _ok_step_result(index=1))
    assert first.emitted_indices() == {0, 1}

    # Simulate a process restart — fresh emitter, same store_dir.
    resumed = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    assert resumed.emitted_indices() == {0, 1}
    # Trying to re-emit index 0 is a no-op.
    assert resumed.emit(intent, _ok_step_result(index=0)) is False
    # Index 2 is fresh — appends normally.
    assert resumed.emit(intent, _ok_step_result(index=2)) is True

    lines = (tmp_path / JSONL_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3


def test_emit_tolerates_partial_trailing_line(tmp_path: Path) -> None:
    """A crashed writer can leave a partial line. The resume reader
    must stop at the bad line and let subsequent emits append clean
    records past it."""
    jsonl = tmp_path / JSONL_FILENAME
    valid_line = json.dumps({
        "schema_version": SCHEMA_VERSION,
        "run_id": "r1", "step_index": 0,
    })
    jsonl.write_text(valid_line + "\n{ partial-broken-json", encoding="utf-8")

    emitter = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    assert emitter.emitted_indices() == {0}
    # The corrupt trailing line stopped resume scanning early; new
    # emits append past the partial without raising.
    assert emitter.emit(_intent(), _ok_step_result(index=1)) is True


def test_emit_appends_separate_files_per_run(tmp_path: Path) -> None:
    """Each TrajectoryEmitter is per-run; two emitters pointed at
    different store_dirs must not see each other's events."""
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    em_a = TrajectoryEmitter(run_id="a", store_dir=str(dir_a))
    em_b = TrajectoryEmitter(run_id="b", store_dir=str(dir_b))

    em_a.emit(_intent(), _ok_step_result(index=0))
    em_b.emit(_intent(), _ok_step_result(index=0))

    assert (dir_a / JSONL_FILENAME).exists()
    assert (dir_b / JSONL_FILENAME).exists()
    assert (dir_a / JSONL_FILENAME).read_text().count("\n") == 1
    assert (dir_b / JSONL_FILENAME).read_text().count("\n") == 1


def test_emit_records_failure_verdict_kind(tmp_path: Path) -> None:
    """Failure steps land with the projected verdict kind so a
    downstream reader can group by recoverable / non-recoverable
    without re-reading the legacy ``failure_class``."""
    emitter = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    emitter.emit(_intent(), _fail_step_result(index=0, failure_class="cf_challenge"))
    record = json.loads((tmp_path / JSONL_FILENAME).read_text().strip())
    assert record["verdict"]["kind"] == "non_recoverable"
    assert record["verdict"]["reason"] == "cf_challenge"


def test_emit_carries_explicit_action(tmp_path: Path) -> None:
    """When the caller has a richer signal (Action object), the
    emitter uses it instead of falling through to
    ``result.last_action`` (which may be None on deterministic
    handlers)."""
    emitter = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    action = Action(action_type=ActionType.CLICK, params={"x": 10, "y": 20})
    emitter.emit(
        _intent(), _ok_step_result(),
        action=action, dispatched=True,
        grounding_trace={"provider": "claude", "confidence": 0.9},
    )
    record = json.loads((tmp_path / JSONL_FILENAME).read_text().strip())
    assert record["action_result"]["action_type"] == "click"
    assert record["action_result"]["params"] == {"x": 10, "y": 20}
    assert record["action_result"]["grounding_trace"]["provider"] == "claude"
    assert record["action_result"]["dispatched"] is True


def test_emit_carries_versions_dict(tmp_path: Path) -> None:
    """The versions slot (#487 / #488) round-trips so model / prompt
    / browser stamps surface in every event."""
    emitter = TrajectoryEmitter(
        run_id="r1", store_dir=str(tmp_path),
        versions={"planner": "claude-opus-4-7", "grounding": "claude-haiku-4-5"},
    )
    emitter.emit(_intent(), _ok_step_result())
    record = json.loads((tmp_path / JSONL_FILENAME).read_text().strip())
    assert record["versions"] == {
        "planner": "claude-opus-4-7", "grounding": "claude-haiku-4-5",
    }


def test_emit_returns_false_when_validation_fails(tmp_path: Path, caplog) -> None:
    """A malformed event (intent + action_type empty after projection)
    must be rejected by the validator and the call returns False —
    the runner keeps going, the bad event doesn't land on disk."""
    emitter = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    # Empty intent + empty type — fails Step validation.
    bad_intent = MicroIntent(intent="", type="")
    with caplog.at_level("WARNING"):
        result = emitter.emit(bad_intent, _ok_step_result())
    assert result is False
    assert any("validation failed" in r.message for r in caplog.records)
    # No JSONL written.
    assert not (tmp_path / JSONL_FILENAME).exists()


def test_emit_construct_rejects_empty_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        TrajectoryEmitter(run_id="", store_dir=str(tmp_path))


def test_emit_construct_rejects_empty_store_dir() -> None:
    with pytest.raises(ValueError, match="store_dir"):
        TrajectoryEmitter(run_id="r1", store_dir="")


# ── Opt-in env gate (run_executor wire-in) ─────────────────────────────


def test_run_executor_hook_no_op_when_env_unset(monkeypatch) -> None:
    """The runner hook is gated by ``MANTIS_CANONICAL_EVENTS_DIR``.
    With the env unset, the helper returns without touching the
    runner — keeps the default execution path free of side effects."""
    from mantis_agent.gym.run_executor import _emit_canonical_trajectory_event

    monkeypatch.delenv("MANTIS_CANONICAL_EVENTS_DIR", raising=False)

    class _Runner:
        pass

    runner = _Runner()
    # Should be a silent no-op; no emitter ever attached.
    _emit_canonical_trajectory_event(runner, _intent(), _ok_step_result())
    assert not hasattr(runner, "_trajectory_emitter")


def test_run_executor_hook_writes_when_env_set(
    monkeypatch, tmp_path: Path,
) -> None:
    """End-to-end through the run_executor helper: env var set →
    emitter lazy-created → event lands on disk."""
    from mantis_agent.gym.run_executor import _emit_canonical_trajectory_event

    monkeypatch.setenv("MANTIS_CANONICAL_EVENTS_DIR", str(tmp_path))

    class _Runner:
        run_id = "wire_in_test"

    runner = _Runner()
    _emit_canonical_trajectory_event(runner, _intent(), _ok_step_result(index=0))
    # Second step on the same runner reuses the emitter.
    _emit_canonical_trajectory_event(runner, _intent(), _ok_step_result(index=1))

    jsonl = (tmp_path / JSONL_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(jsonl) == 2
    indices = [json.loads(line)["step_index"] for line in jsonl]
    assert indices == [0, 1]


def test_run_executor_hook_idempotent_across_retry(
    monkeypatch, tmp_path: Path,
) -> None:
    """A retry / demote / recovery branch can run the executor helper
    twice on the same logical step; the second emit must be a no-op."""
    from mantis_agent.gym.run_executor import _emit_canonical_trajectory_event

    monkeypatch.setenv("MANTIS_CANONICAL_EVENTS_DIR", str(tmp_path))

    class _Runner:
        run_id = "retry_test"

    runner = _Runner()
    _emit_canonical_trajectory_event(runner, _intent(), _ok_step_result(index=0))
    _emit_canonical_trajectory_event(runner, _intent(), _fail_step_result(index=0))

    jsonl = (tmp_path / JSONL_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(jsonl) == 1  # only the first emit landed


def test_run_executor_hook_falls_back_to_default_run_id(
    monkeypatch, tmp_path: Path,
) -> None:
    """When the runner has no ``run_id`` attribute (e.g. a local CLI
    invocation), the helper synthesises a stable placeholder so
    emits still land — emitter constructor would otherwise refuse
    an empty run_id."""
    from mantis_agent.gym.run_executor import _emit_canonical_trajectory_event

    monkeypatch.setenv("MANTIS_CANONICAL_EVENTS_DIR", str(tmp_path))

    class _Runner:
        pass  # no run_id / _run_id attributes

    runner = _Runner()
    _emit_canonical_trajectory_event(runner, _intent(), _ok_step_result(index=0))

    jsonl = (tmp_path / JSONL_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(jsonl) == 1
    record = json.loads(jsonl[0])
    assert record["run_id"] == "local_run"
