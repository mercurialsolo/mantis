#!/usr/bin/env python3
"""Eval harness — gate model promotions on win-rate vs the current production weights (#155 step 4).

Drops into the continual-fine-tuning loop alongside the trace export +
labeller + SFT converter (steps 1–3). Produces structured pass/fail
reports for one model on a set of tasks, plus a compare-mode that
tells you whether a candidate beats the baseline by the configured
margin.

Two-stage protocol
------------------

::

    [1] eval baseline  → eval_harness.run_eval(baseline_brain, tasks) → baseline.json
    [2] eval candidate → eval_harness.run_eval(candidate_brain, tasks) → candidate.json
    [3] compare        → eval_harness.compare(baseline.json, candidate.json) → win_rate, deltas

The harness is deliberately I/O-shaped: it consumes JSON eval-task
specs, invokes a caller-supplied "runner" callable, and writes JSON
reports. That keeps it useful both for local quick checks (mocked
runner in tests) and for production-style runs on Modal A100.

EvalTask shape
--------------

::

    {
      "task_id": "hn_extract_top_3",
      "plan_text": "...",                     # OR
      "micro_plan": [...],                    # OR
      "url": "https://news.ycombinator.com",  # plus task_text
      "task_text": "Extract the top 3 stories",
      "criteria": [
        {"type": "task_success"},                            # success terminator fired
        {"type": "url_contains", "value": "/item?id="},
        {"type": "output_contains", "value": "Show HN"}
      ],
      "max_cost": 0.5,
      "max_steps": 30
    }

A task passes when *every* criterion in ``criteria`` is satisfied. An
empty criteria list is treated as ``[{"type": "task_success"}]``.

The runner callable
-------------------

The harness doesn't know how to run a plan — it delegates to a callable
the caller provides::

    def runner(task: EvalTask) -> EvalRunOutcome: ...

For SFT-cycle eval, hosts wire this to the production HTTP endpoint with
the candidate brain's weights mounted. For unit tests, hosts wire it to
a stub that returns a deterministic outcome.

CLI
---

This module also runs as a script for batch eval + compare modes:

    python -m training.eval_harness run \\
        --tasks tasks/hn_smoke.json \\
        --output reports/baseline.json \\
        --runner http://localhost:18080

    python -m training.eval_harness compare \\
        --baseline reports/baseline.json \\
        --candidate reports/candidate.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


# ── Eval task + outcome schemas ──────────────────────────────────────


@dataclass
class EvalCriterion:
    """One pass/fail check applied to a single :class:`EvalRunOutcome`.

    ``type`` is one of:
      - ``task_success``    — the runner reported terminal success
      - ``url_contains``    — final ``url`` includes ``value`` (substring)
      - ``output_contains`` — final ``output`` text includes ``value``
      - ``status_eq``       — final ``status`` == ``value``
    """

    type: str
    value: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalCriterion":
        return cls(type=payload.get("type", ""), value=str(payload.get("value", "")))

    def evaluate(self, outcome: "EvalRunOutcome") -> bool:
        if self.type == "task_success":
            return bool(outcome.success)
        if self.type == "status_eq":
            return outcome.status == self.value
        if self.type == "url_contains":
            return self.value in (outcome.url or "")
        if self.type == "output_contains":
            return self.value in (outcome.output or "")
        # Unknown criterion → fail closed so a malformed task can never
        # accidentally promote a regression.
        logger.warning("unknown EvalCriterion type %r — failing closed", self.type)
        return False


@dataclass
class EvalTask:
    """One row in an eval set. Mirrors the shape ``run_eval`` consumes."""

    task_id: str
    task_text: str = ""
    plan_text: str = ""
    micro_plan: list[dict[str, Any]] = field(default_factory=list)
    url: str = ""
    criteria: list[EvalCriterion] = field(default_factory=list)
    max_cost: float = 1.0
    max_steps: int = 30
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalTask":
        return cls(
            task_id=str(payload.get("task_id", "")),
            task_text=str(payload.get("task_text", "")),
            plan_text=str(payload.get("plan_text", "")),
            micro_plan=list(payload.get("micro_plan") or []),
            url=str(payload.get("url", "")),
            criteria=[
                EvalCriterion.from_dict(c)
                for c in (payload.get("criteria") or [])
            ],
            max_cost=float(payload.get("max_cost", 1.0)),
            max_steps=int(payload.get("max_steps", 30)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class EvalRunOutcome:
    """Result of running ONE eval task against ONE brain.

    The caller-supplied runner returns this; the harness applies criteria
    on top to compute the boolean ``passed`` field. Keeping ``success``,
    ``status``, ``url``, and ``output`` separate from ``passed`` lets a
    single runner output be checked against multiple criteria sets
    without re-running the task.
    """

    task_id: str
    success: bool = False
    status: str = ""
    url: str = ""
    output: str = ""
    cost: float = 0.0
    duration_s: float = 0.0
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalTaskResult:
    """One row in :class:`EvalReport.results` — outcome + per-criterion verdicts."""

    task_id: str
    passed: bool
    outcome: EvalRunOutcome
    criterion_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EvalReport:
    """Full report from one ``run_eval`` invocation."""

    name: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    task_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    error_count: int = 0
    pass_rate: float = 0.0
    results: list[EvalTaskResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "task_count": self.task_count,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "error_count": self.error_count,
            "pass_rate": round(self.pass_rate, 4),
            "results": [
                {
                    "task_id": r.task_id,
                    "passed": r.passed,
                    "criterion_results": r.criterion_results,
                    "outcome": asdict(r.outcome),
                }
                for r in self.results
            ],
        }


# ── Core entry: run_eval ──────────────────────────────────────────────


RunnerFn = Callable[[EvalTask], EvalRunOutcome]


def _evaluate_criteria(
    task: EvalTask, outcome: EvalRunOutcome,
) -> tuple[bool, list[dict[str, Any]]]:
    """Apply each criterion. Empty criteria list → pass on task_success."""
    crits = task.criteria or [EvalCriterion(type="task_success")]
    results: list[dict[str, Any]] = []
    all_passed = True
    for c in crits:
        ok = c.evaluate(outcome)
        results.append({"type": c.type, "value": c.value, "passed": ok})
        if not ok:
            all_passed = False
    return all_passed, results


def run_eval(
    runner: RunnerFn,
    tasks: Iterable[EvalTask],
    *,
    name: str = "eval",
) -> EvalReport:
    """Run every task through ``runner``; build a structured report.

    The runner is responsible for everything Mantis-specific (HTTP
    submit, polling, error handling). The harness only:
      - times the run
      - applies criteria on the returned outcome
      - aggregates pass/fail/error counts

    A runner exception is captured into ``EvalRunOutcome.error`` and
    counts as a fail (never a hard exit) so one broken task doesn't
    terminate the whole run.
    """
    report = EvalReport(name=name, started_at=time.time())
    for task in tasks:
        try:
            outcome = runner(task)
        except Exception as exc:  # noqa: BLE001 — never let a task crash the loop
            logger.warning("runner raised on task=%s: %s", task.task_id, exc)
            outcome = EvalRunOutcome(task_id=task.task_id, error=str(exc))
            report.error_count += 1
        passed, crit_results = _evaluate_criteria(task, outcome)
        if outcome.error:
            passed = False
        report.results.append(EvalTaskResult(
            task_id=task.task_id,
            passed=passed,
            outcome=outcome,
            criterion_results=crit_results,
        ))
        if passed:
            report.pass_count += 1
        else:
            report.fail_count += 1
    report.task_count = len(report.results)
    report.pass_rate = (
        report.pass_count / report.task_count if report.task_count else 0.0
    )
    report.ended_at = time.time()
    return report


def load_tasks(path: Path) -> list[EvalTask]:
    """Read tasks from a JSON file. Accepts ``[...]`` or ``{"tasks": [...]}``."""
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("tasks", [])
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected list of tasks or {{'tasks': [...]}}")
    return [EvalTask.from_dict(item) for item in raw]


# ── compare ───────────────────────────────────────────────────────────


@dataclass
class EvalCompare:
    """Output of :func:`compare` — baseline vs candidate diff.

    ``win_rate`` is candidate's pass-count among tasks where the two
    sides disagree, divided by the size of the disagreement set. A
    win_rate of 0.0 means the candidate lost every disagreement, 0.5
    is even, 1.0 means it won every disagreement.

    Promotion gating is the caller's policy — ``EvalCompare`` exposes
    the inputs (rates + per-task verdicts) without baking in a threshold.
    """

    baseline_pass_rate: float
    candidate_pass_rate: float
    delta: float
    common_task_count: int
    candidate_wins: int  # baseline failed, candidate passed
    candidate_losses: int  # baseline passed, candidate failed
    win_rate: float
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_pass_rate": round(self.baseline_pass_rate, 4),
            "candidate_pass_rate": round(self.candidate_pass_rate, 4),
            "delta": round(self.delta, 4),
            "common_task_count": self.common_task_count,
            "candidate_wins": self.candidate_wins,
            "candidate_losses": self.candidate_losses,
            "win_rate": round(self.win_rate, 4),
            "regressions": list(self.regressions),
            "improvements": list(self.improvements),
        }


def compare(baseline: EvalReport, candidate: EvalReport) -> EvalCompare:
    """Diff two reports. Aligns by ``task_id`` and reports per-side counts.

    Tasks that appear in only one report are ignored — promotion
    decisions should run both sides on the same eval set.
    """
    base_by_id = {r.task_id: r for r in baseline.results}
    cand_by_id = {r.task_id: r for r in candidate.results}
    common = set(base_by_id.keys()) & set(cand_by_id.keys())

    wins, losses = 0, 0
    regressions: list[str] = []
    improvements: list[str] = []
    for task_id in sorted(common):
        b = base_by_id[task_id].passed
        c = cand_by_id[task_id].passed
        if c and not b:
            wins += 1
            improvements.append(task_id)
        elif b and not c:
            losses += 1
            regressions.append(task_id)
    disagreements = wins + losses
    win_rate = wins / disagreements if disagreements else 0.5
    return EvalCompare(
        baseline_pass_rate=baseline.pass_rate,
        candidate_pass_rate=candidate.pass_rate,
        delta=candidate.pass_rate - baseline.pass_rate,
        common_task_count=len(common),
        candidate_wins=wins,
        candidate_losses=losses,
        win_rate=win_rate,
        regressions=regressions,
        improvements=improvements,
    )


def load_report(path: Path) -> EvalReport:
    payload = json.loads(path.read_text())
    report = EvalReport(
        name=str(payload.get("name", "")),
        started_at=float(payload.get("started_at", 0.0)),
        ended_at=float(payload.get("ended_at", 0.0)),
        task_count=int(payload.get("task_count", 0)),
        pass_count=int(payload.get("pass_count", 0)),
        fail_count=int(payload.get("fail_count", 0)),
        error_count=int(payload.get("error_count", 0)),
        pass_rate=float(payload.get("pass_rate", 0.0)),
    )
    for row in payload.get("results", []):
        outcome_raw = row.get("outcome", {})
        outcome = EvalRunOutcome(
            task_id=str(outcome_raw.get("task_id", row.get("task_id", ""))),
            success=bool(outcome_raw.get("success", False)),
            status=str(outcome_raw.get("status", "")),
            url=str(outcome_raw.get("url", "")),
            output=str(outcome_raw.get("output", "")),
            cost=float(outcome_raw.get("cost", 0.0)),
            duration_s=float(outcome_raw.get("duration_s", 0.0)),
            error=str(outcome_raw.get("error", "")),
            raw=dict(outcome_raw.get("raw") or {}),
        )
        report.results.append(EvalTaskResult(
            task_id=str(row.get("task_id", outcome.task_id)),
            passed=bool(row.get("passed", False)),
            outcome=outcome,
            criterion_results=list(row.get("criterion_results") or []),
        ))
    return report


# ── CLI ───────────────────────────────────────────────────────────────


def _http_runner_factory(endpoint: str, token: str = "") -> RunnerFn:
    """Build a runner that submits each task through the production
    ``/v1/predict`` endpoint and polls until terminal. Used by the
    ``run`` subcommand. Lazy: only imported when the CLI actually fires.
    """
    import requests

    def _run(task: EvalTask) -> EvalRunOutcome:
        body: dict[str, Any] = {
            "detached": False,
            "max_cost": task.max_cost,
            "max_time_minutes": max(1, task.max_steps),
        }
        if task.micro_plan:
            body["task_suite"] = {
                "session_name": task.task_id,
                "_micro_plan": task.micro_plan,
            }
        elif task.plan_text:
            body["plan_text"] = task.plan_text
        else:
            body["plan_text"] = task.task_text or task.task_id
        if task.url:
            body.setdefault("task_suite", {}).setdefault("_micro_plan", [
                {"intent": f"Navigate to {task.url}", "type": "navigate",
                 "section": "setup", "required": True},
            ])
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Mantis-Token"] = token
        t0 = time.time()
        resp = requests.post(
            f"{endpoint.rstrip('/')}/v1/predict",
            json=body, headers=headers, timeout=600,
        )
        duration = time.time() - t0
        if resp.status_code != 200:
            return EvalRunOutcome(
                task_id=task.task_id, error=f"http {resp.status_code}: {resp.text[:200]}",
                duration_s=duration,
            )
        data = resp.json()
        return EvalRunOutcome(
            task_id=task.task_id,
            success=bool(data.get("success", data.get("viable", 0) > 0)),
            status=str(data.get("status", "")),
            url=str(data.get("final_url", "")),
            output=json.dumps(data.get("leads") or data.get("result") or {}),
            cost=float((data.get("costs") or {}).get("total", 0.0)),
            duration_s=duration,
            raw=data,
        )

    return _run


def _cmd_run(args: argparse.Namespace) -> int:
    tasks = load_tasks(Path(args.tasks))
    runner = _http_runner_factory(args.runner, token=args.token)
    report = run_eval(runner, tasks, name=args.name or Path(args.tasks).stem)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    print(
        f"  {report.name}: passed {report.pass_count}/{report.task_count} "
        f"({report.pass_rate:.1%}), errors={report.error_count}"
    )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    baseline = load_report(Path(args.baseline))
    candidate = load_report(Path(args.candidate))
    cmp = compare(baseline, candidate)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(cmp.to_dict(), indent=2) + "\n")
    print(
        f"  baseline pass-rate: {cmp.baseline_pass_rate:.1%}\n"
        f"  candidate pass-rate: {cmp.candidate_pass_rate:.1%}\n"
        f"  delta: {cmp.delta:+.1%}\n"
        f"  common tasks: {cmp.common_task_count}\n"
        f"  wins: {cmp.candidate_wins}  losses: {cmp.candidate_losses}\n"
        f"  win-rate (among disagreements): {cmp.win_rate:.1%}"
    )
    if cmp.regressions:
        print(f"  regressions: {', '.join(cmp.regressions)}")
    return 1 if cmp.candidate_losses > cmp.candidate_wins else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run an eval set through an HTTP runner")
    run.add_argument("--tasks", required=True, help="Path to eval tasks JSON")
    run.add_argument("--output", required=True, help="Output report JSON")
    run.add_argument("--runner", required=True,
                     help="HTTP endpoint (e.g. https://...modal.run)")
    run.add_argument("--token", default="",
                     help="X-Mantis-Token (defaults to env MANTIS_API_TOKEN)")
    run.add_argument("--name", default="",
                     help="Report name (default: tasks file stem)")
    run.set_defaults(func=_cmd_run)

    cmp = sub.add_parser(
        "compare", help="Compare baseline + candidate reports",
    )
    cmp.add_argument("--baseline", required=True)
    cmp.add_argument("--candidate", required=True)
    cmp.add_argument(
        "--output", default="",
        help="Optional output JSON (compare summary)",
    )
    cmp.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    if not args.token and args.command == "run":
        import os
        args.token = os.environ.get("MANTIS_API_TOKEN", "")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
