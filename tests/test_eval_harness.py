"""Tests for #155 step 4 — eval harness.

Pin the criterion contract, the runner-error containment behavior, the
report aggregation math, and the compare-mode delta logic. Runner is
mocked so tests don't make HTTP calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


_TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(_TRAINING))

from eval_harness import (  # noqa: E402
    EvalCompare,
    EvalCriterion,
    EvalReport,
    EvalRunOutcome,
    EvalTask,
    EvalTaskResult,
    _http_runner_factory,
    compare,
    load_report,
    load_tasks,
    run_eval,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _outcome(
    task_id: str,
    *,
    success: bool = True,
    status: str = "succeeded",
    url: str = "",
    output: str = "",
    error: str = "",
) -> EvalRunOutcome:
    return EvalRunOutcome(
        task_id=task_id, success=success, status=status,
        url=url, output=output, error=error,
    )


def _task(
    task_id: str = "t1",
    *,
    criteria: list[dict[str, Any]] | None = None,
) -> EvalTask:
    return EvalTask(
        task_id=task_id,
        criteria=[EvalCriterion(**c) for c in (criteria or [])],
    )


# ── EvalCriterion ──────────────────────────────────────────────────────


def test_criterion_task_success_uses_outcome_success():
    crit = EvalCriterion(type="task_success")
    assert crit.evaluate(_outcome("t1", success=True)) is True
    assert crit.evaluate(_outcome("t1", success=False)) is False


def test_criterion_url_contains_substring_match():
    crit = EvalCriterion(type="url_contains", value="/item?id=")
    assert crit.evaluate(_outcome("t1", url="https://hn.com/item?id=42")) is True
    assert crit.evaluate(_outcome("t1", url="https://hn.com/")) is False


def test_criterion_output_contains_substring_match():
    crit = EvalCriterion(type="output_contains", value="Show HN")
    assert crit.evaluate(_outcome("t1", output="title: Show HN: foo")) is True
    assert crit.evaluate(_outcome("t1", output="something else")) is False


def test_criterion_status_eq_exact_match():
    crit = EvalCriterion(type="status_eq", value="succeeded")
    assert crit.evaluate(_outcome("t1", status="succeeded")) is True
    assert crit.evaluate(_outcome("t1", status="failed")) is False


def test_unknown_criterion_fails_closed():
    """Unknown criterion type must NOT silently pass — promotion gating
    must never let a malformed task accidentally green-light a regression."""
    crit = EvalCriterion(type="nonsense_check", value="x")
    assert crit.evaluate(_outcome("t1", success=True)) is False


# ── run_eval ───────────────────────────────────────────────────────────


def test_run_eval_empty_criteria_defaults_to_task_success():
    """A task with no criteria passes iff the runner returns success."""
    runner = lambda t: _outcome(t.task_id, success=True)  # noqa: E731
    report = run_eval(runner, [_task("t1")])
    assert report.pass_count == 1
    assert report.fail_count == 0
    assert report.pass_rate == 1.0


def test_run_eval_all_criteria_must_pass():
    runner = lambda t: _outcome(t.task_id, success=True, url="https://x.test/")  # noqa: E731
    task = _task("t1", criteria=[
        {"type": "task_success"},
        {"type": "url_contains", "value": "missing-substring"},
    ])
    report = run_eval(runner, [task])
    assert report.pass_count == 0
    assert report.fail_count == 1


def test_run_eval_runner_exception_counts_as_error_and_fail():
    def boom(_: EvalTask) -> EvalRunOutcome:
        raise RuntimeError("upstream broke")

    report = run_eval(boom, [_task("t1"), _task("t2")])
    assert report.task_count == 2
    assert report.error_count == 2
    assert report.fail_count == 2
    assert report.pass_count == 0


def test_run_eval_per_criterion_results_recorded():
    runner = lambda t: _outcome(t.task_id, success=True, url="https://hn.com/item?id=1")  # noqa: E731
    task = _task("t1", criteria=[
        {"type": "task_success"},
        {"type": "url_contains", "value": "/item?id="},
    ])
    report = run_eval(runner, [task])
    assert report.pass_count == 1
    crit_results = report.results[0].criterion_results
    assert len(crit_results) == 2
    assert all(c["passed"] for c in crit_results)


def test_run_eval_report_to_dict_round_trip(tmp_path):
    runner = lambda t: _outcome(t.task_id, success=True)  # noqa: E731
    report = run_eval(runner, [_task("t1"), _task("t2")], name="smoke")
    out = tmp_path / "report.json"
    out.write_text(json.dumps(report.to_dict()))
    loaded = load_report(out)
    assert loaded.name == "smoke"
    assert loaded.task_count == 2
    assert loaded.pass_count == 2


# ── load_tasks ─────────────────────────────────────────────────────────


def test_load_tasks_accepts_bare_array(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([
        {"task_id": "a"},
        {"task_id": "b", "criteria": [{"type": "task_success"}]},
    ]))
    tasks = load_tasks(path)
    assert [t.task_id for t in tasks] == ["a", "b"]


def test_load_tasks_accepts_wrapper_dict(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": [{"task_id": "a"}]}))
    assert len(load_tasks(path)) == 1


# ── compare ────────────────────────────────────────────────────────────


def _report(
    name: str,
    rows: list[tuple[str, bool]],
) -> EvalReport:
    """Build a stub report from (task_id, passed) tuples."""
    report = EvalReport(name=name)
    for task_id, passed in rows:
        report.results.append(EvalTaskResult(
            task_id=task_id, passed=passed, outcome=_outcome(task_id, success=passed),
        ))
    report.task_count = len(rows)
    report.pass_count = sum(1 for _, p in rows if p)
    report.fail_count = report.task_count - report.pass_count
    report.pass_rate = report.pass_count / max(report.task_count, 1)
    return report


def test_compare_aligns_by_task_id_and_counts_disagreements():
    baseline = _report("base", [("a", True), ("b", False), ("c", True)])
    candidate = _report("cand", [("a", True), ("b", True), ("c", False)])
    cmp_ = compare(baseline, candidate)
    assert cmp_.candidate_wins == 1   # b: F → T
    assert cmp_.candidate_losses == 1  # c: T → F
    assert cmp_.win_rate == 0.5
    assert "b" in cmp_.improvements
    assert "c" in cmp_.regressions


def test_compare_ignores_tasks_only_on_one_side():
    baseline = _report("base", [("a", True), ("b", False)])
    candidate = _report("cand", [("a", True), ("c", True)])  # c missing from base
    cmp_ = compare(baseline, candidate)
    assert cmp_.common_task_count == 1
    assert cmp_.candidate_wins == 0
    assert cmp_.candidate_losses == 0


def test_compare_no_disagreement_returns_neutral_win_rate():
    baseline = _report("base", [("a", True), ("b", True)])
    candidate = _report("cand", [("a", True), ("b", True)])
    cmp_ = compare(baseline, candidate)
    assert cmp_.candidate_wins == 0
    assert cmp_.candidate_losses == 0
    # No disagreements → win-rate degenerates to 0.5 (neutral, not 0).
    assert cmp_.win_rate == 0.5


def test_compare_delta_reflects_pass_rate_difference():
    baseline = _report("base", [("a", True), ("b", False)])
    candidate = _report("cand", [("a", True), ("b", True)])
    cmp_ = compare(baseline, candidate)
    assert cmp_.delta == 0.5  # candidate goes 0.5 → 1.0


def test_compare_to_dict_round_trip():
    cmp_ = EvalCompare(
        baseline_pass_rate=0.5, candidate_pass_rate=0.75, delta=0.25,
        common_task_count=4, candidate_wins=2, candidate_losses=1,
        win_rate=0.6667,
        regressions=["c"], improvements=["a", "b"],
    )
    payload = cmp_.to_dict()
    assert payload["regressions"] == ["c"]
    assert payload["delta"] == 0.25


# ── End-to-end: tasks → run → compare ─────────────────────────────────


def test_end_to_end_pipeline(tmp_path):
    """Stub two brains. The candidate is strictly better on one task and
    strictly worse on another — a realistic 'mixed result' eval."""
    tasks = [
        _task("t1", criteria=[{"type": "task_success"}]),
        _task("t2", criteria=[{"type": "task_success"}]),
    ]

    def baseline_runner(task: EvalTask) -> EvalRunOutcome:
        return _outcome(task.task_id, success=(task.task_id == "t2"))

    def candidate_runner(task: EvalTask) -> EvalRunOutcome:
        return _outcome(task.task_id, success=(task.task_id == "t1"))

    base_report = run_eval(baseline_runner, tasks)
    cand_report = run_eval(candidate_runner, tasks)
    cmp_ = compare(base_report, cand_report)
    assert cmp_.candidate_wins == 1
    assert cmp_.candidate_losses == 1
    assert cmp_.delta == 0.0  # equal pass-rates


# ── #911: LoRA challenger wiring in the HTTP runner ─────────────────────


class _FakeResp:
    status_code = 200

    def json(self) -> dict[str, Any]:
        return {"success": True, "status": "succeeded"}


def _capture_post(monkeypatch) -> list[dict[str, Any]]:
    """Patch requests.post (imported lazily inside _http_runner_factory) and
    capture the JSON bodies submitted."""
    bodies: list[dict[str, Any]] = []
    import requests

    def _fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        bodies.append(json)
        return _FakeResp()

    monkeypatch.setattr(requests, "post", _fake_post)
    return bodies


def _micro_task(task_id="t1") -> EvalTask:
    return EvalTask(task_id=task_id, micro_plan=[{"intent": "go", "type": "navigate"}])


def test_http_runner_champion_arm_has_no_adapter(monkeypatch):
    bodies = _capture_post(monkeypatch)
    runner = _http_runner_factory("https://x.modal.run")  # no lora_adapter
    runner(_micro_task())
    assert "_lora_adapter" not in bodies[0].get("task_suite", {})


def test_http_runner_challenger_arm_attaches_adapter(monkeypatch):
    bodies = _capture_post(monkeypatch)
    runner = _http_runner_factory(
        "https://x.modal.run", lora_adapter="mantis-trainer-vol:/checkpoints/sft-c3e0d799"
    )
    runner(_micro_task())
    assert bodies[0]["task_suite"]["_lora_adapter"] == "mantis-trainer-vol:/checkpoints/sft-c3e0d799"


def test_http_runner_attaches_adapter_on_plan_text_path(monkeypatch):
    bodies = _capture_post(monkeypatch)
    runner = _http_runner_factory("https://x.modal.run", lora_adapter="/data/ckpt/x")
    # plan_text task (no micro_plan) → a task_suite must still be created.
    runner(EvalTask(task_id="t2", plan_text="do the thing"))
    assert bodies[0]["task_suite"]["_lora_adapter"] == "/data/ckpt/x"


def test_http_runner_challenger_model_full_swap(monkeypatch):
    bodies = _capture_post(monkeypatch)
    runner = _http_runner_factory(
        "https://x.modal.run", challenger_model="mantis-trainer-vol:/checkpoints/sft-x/m.gguf"
    )
    runner(_micro_task())
    assert bodies[0]["task_suite"]["_challenger_model"].endswith("m.gguf")
    assert "_lora_adapter" not in bodies[0]["task_suite"]


def test_http_runner_challenger_model_precedence(monkeypatch):
    bodies = _capture_post(monkeypatch)
    runner = _http_runner_factory(
        "https://x.modal.run", lora_adapter="/a.gguf", challenger_model="/m.gguf"
    )
    runner(_micro_task())
    assert bodies[0]["task_suite"]["_challenger_model"] == "/m.gguf"
    assert "_lora_adapter" not in bodies[0]["task_suite"]


def test_http_runner_forwards_cua_model(monkeypatch):
    bodies = _capture_post(monkeypatch)
    runner = _http_runner_factory("https://x.modal.run", cua_model="fara")
    runner(_micro_task())
    assert bodies[0]["cua_model"] == "fara"
