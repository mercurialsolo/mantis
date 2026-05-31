"""Tests for the S1 exemplar substrate.

The substrate indexes the trace corpus the ``TraceExporter`` writes and
replays positive-labelled steps. These build real trace JSON on disk (the
exporter's schema) and assert the labeller's positive/negative split flows
through into the retrieved exemplars.
"""

from __future__ import annotations

import json
from pathlib import Path

from mantis_agent.learning.substrates import ExemplarSubstrate
from mantis_agent.learning.substrates.base import (
    Durability,
    LearningSubstrate,
    SubstrateContext,
)

PLAN_SIG = "planABC123456"


def _trace(plan_sig: str, *, run_id: str, steps: list[dict]) -> dict:
    return {
        "schema_version": 2,
        "run_id": run_id,
        "tenant_id": "",
        "plan_signature": plan_sig,
        "status": "completed",
        "steps": steps,
    }


def _positive(idx: int, intent: str = "apply used filter") -> dict:
    # success + a non-empty observed_outcome ⇒ labeller marks 'positive'.
    return {
        "step_index": idx,
        "intent": intent,
        "type": "click",
        "success": True,
        "data": "",
        "observed_outcome": "filter applied",
        "predicted_outcome": "filter should apply",
        "last_action": {"action_type": "click", "params": {"x": 1, "y": 2}},
    }


def _negative(idx: int) -> dict:
    return {
        "step_index": idx,
        "intent": "submit broken form",
        "type": "click",
        "success": False,
        "data": "",
        "observed_outcome": "",
        "predicted_outcome": "",
        "last_action": None,
    }


def _write(dir_: Path, trace: dict, *, tenant: str = "__shared__") -> Path:
    # Mirror the exporter layout: <dir>/<tenant>/<run_id>.json.
    out = dir_ / tenant / f"{trace['run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace))
    return out


def _ctx(plan_signature: str = PLAN_SIG) -> SubstrateContext:
    return SubstrateContext(
        task_id="BT01", cluster="capability",
        extras={"plan_signature": plan_signature},
    )


# ── protocol / cheap-path invariants ───────────────────────────────────


def test_conforms_to_substrate_protocol(tmp_path) -> None:
    assert isinstance(ExemplarSubstrate(tmp_path), LearningSubstrate)


def test_durability_is_session(tmp_path) -> None:
    assert ExemplarSubstrate(tmp_path).durability is Durability.SESSION


def test_cost_estimate_is_free(tmp_path) -> None:
    assert ExemplarSubstrate(tmp_path).cost_estimate(_ctx()) == 0.0


def test_missing_trace_dir_is_empty_not_error(tmp_path) -> None:
    sub = ExemplarSubstrate(tmp_path / "does-not-exist")
    res = sub.apply(_ctx())
    assert res.applied is False


# ── apply / retrieval ───────────────────────────────────────────────────


def test_apply_returns_only_positive_exemplars(tmp_path) -> None:
    _write(tmp_path, _trace(PLAN_SIG, run_id="run1",
                            steps=[_positive(0), _negative(1)]))
    sub = ExemplarSubstrate(tmp_path)

    res = sub.apply(_ctx())

    assert res.applied is True
    exemplars = res.delta_artifacts["exemplars"]
    assert len(exemplars) == 1  # the negative step is filtered out
    assert exemplars[0]["intent"] == "apply used filter"
    assert exemplars[0]["observed_outcome"] == "filter applied"
    assert exemplars[0]["source_run"] == "run1"


def test_apply_unknown_plan_is_not_applied(tmp_path) -> None:
    _write(tmp_path, _trace(PLAN_SIG, run_id="run1", steps=[_positive(0)]))
    sub = ExemplarSubstrate(tmp_path)

    res = sub.apply(_ctx(plan_signature="some-other-plan"))

    assert res.applied is False
    assert res.delta_artifacts["exemplars"] == []


def test_apply_without_plan_signature_is_not_applied(tmp_path) -> None:
    sub = ExemplarSubstrate(tmp_path)
    res = sub.apply(SubstrateContext(task_id="t", extras={}))
    assert res.applied is False
    assert "cannot retrieve" in res.notes


def test_max_exemplars_caps_the_result(tmp_path) -> None:
    steps = [_positive(i, intent=f"step {i}") for i in range(5)]
    _write(tmp_path, _trace(PLAN_SIG, run_id="run1", steps=steps))
    sub = ExemplarSubstrate(tmp_path, max_exemplars=2)

    res = sub.apply(_ctx())

    assert len(res.delta_artifacts["exemplars"]) == 2


def test_exemplars_aggregate_across_runs(tmp_path) -> None:
    _write(tmp_path, _trace(PLAN_SIG, run_id="run1", steps=[_positive(0)]))
    _write(tmp_path, _trace(PLAN_SIG, run_id="run2", steps=[_positive(0)]))
    sub = ExemplarSubstrate(tmp_path)

    res = sub.apply(_ctx())

    runs = {e["source_run"] for e in res.delta_artifacts["exemplars"]}
    assert runs == {"run1", "run2"}


def test_refresh_picks_up_new_traces(tmp_path) -> None:
    sub = ExemplarSubstrate(tmp_path)
    # Index built (lazily) against an empty dir.
    assert sub.apply(_ctx()).applied is False

    _write(tmp_path, _trace(PLAN_SIG, run_id="run1", steps=[_positive(0)]))
    # Stale cache still misses until we refresh.
    assert sub.apply(_ctx()).applied is False
    sub.refresh()
    assert sub.apply(_ctx()).applied is True


def test_corrupt_trace_is_skipped(tmp_path) -> None:
    _write(tmp_path, _trace(PLAN_SIG, run_id="good", steps=[_positive(0)]))
    bad = tmp_path / "__shared__" / "bad.json"
    bad.write_text("{ not valid json")
    sub = ExemplarSubstrate(tmp_path)

    res = sub.apply(_ctx())

    # The good trace still indexes; the corrupt file is logged + skipped.
    assert res.applied is True
    assert len(res.delta_artifacts["exemplars"]) == 1


def test_observe_is_noop(tmp_path) -> None:
    sub = ExemplarSubstrate(tmp_path)
    assert sub.observe(_ctx(), sub.apply(_ctx()), 1.0) is None
