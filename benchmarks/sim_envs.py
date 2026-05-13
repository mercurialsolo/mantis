"""benchmarks/sim_envs.py — batch oracle-graded eval across all plans in an env.

Walks ``plans/<env>/*.json`` and runs each plan against a freshly booted
env instance. Collects ``oracle.json`` results into a single summary
table so we can see per-task pass/fail at a glance and compare an
agent's progress over time on the same seed.

Usage::

    python -m benchmarks.sim_envs --env mantis-crm --runtime local \\
        --endpoint https://workspace--app-fn.modal.run/v1 \\
        --output-dir outputs/bench-crm-$(date +%s)

Parallelism: ``--max-workers N`` runs up to N plans concurrently. Each
worker boots its own env instance — there's no shared container in v1
(see #336 §"Isolation model"). The local backend hands out a free port
per worker; the Modal backend picks a fresh run suffix per worker so
deployed app names don't collide.

This is deliberately a thin shell over ``mantis plan run --env``: every
plan goes through the same CLI codepath, gets the same grading hook,
writes the same artifact layout. The batch runner only handles the
orchestration around N plans + the summary table.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _find_plans(env_name: str, plans_dir: Path) -> list[Path]:
    """Discover ``plans/<env>/*.json`` for the env (sorted, deterministic)."""
    target = plans_dir / env_name
    if not target.exists():
        return []
    return sorted(p for p in target.glob("*.json") if p.is_file())


def _run_one(
    plan_path: Path,
    *,
    env_name: str,
    runtime: str,
    output_dir: Path,
    endpoint: str | None,
    seed: int,
    extra_args: list[str],
) -> dict[str, Any]:
    """Invoke ``mantis plan run`` for one plan; return a summary dict.

    Subprocess so each plan gets its own clean Python process (env vars,
    network state, modal session). The CLI writes ``oracle.json`` next
    to ``result.json``; we read both back into the summary.
    """
    plan_output = output_dir / plan_path.stem
    cmd = [
        sys.executable, "-m", "mantis_agent.main", "plan", "run",
        str(plan_path),
        "--env", env_name,
        "--runtime", runtime,
        "--seed", str(seed),
        "--output-dir", str(plan_output),
    ]
    if endpoint:
        cmd += ["--endpoint", endpoint]
    cmd += extra_args

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.time() - t0

    oracle_path = plan_output / "oracle.json"
    result_path = plan_output / "result.json"

    grading: dict[str, Any] = {}
    if oracle_path.exists():
        try:
            grading = json.loads(oracle_path.read_text())
        except json.JSONDecodeError:
            grading = {"error": "oracle.json not valid JSON"}

    result_summary: dict[str, Any] = {}
    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text())
            result_summary = {
                "successes": payload.get("successes"),
                "failures": payload.get("failures"),
                "step_count": payload.get("step_count"),
            }
        except json.JSONDecodeError:
            result_summary = {"error": "result.json not valid JSON"}

    return {
        "plan": plan_path.name,
        "task_id": grading.get("task_id", plan_path.stem),
        "passed": grading.get("passed"),
        "score": grading.get("score"),
        "exit_code": proc.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "result": result_summary,
        "oracle_error": grading.get("error"),
        "stderr_tail": proc.stderr.strip().splitlines()[-5:] if proc.returncode != 0 else [],
    }


def _print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("\nNo plans run.")
        return
    passed = sum(1 for r in rows if r.get("passed") is True)
    print()
    print("=" * 72)
    print(f"sim_envs batch summary — {passed}/{len(rows)} plans passed oracle")
    print("=" * 72)
    print(f"  {'plan':35s} {'task_id':24s} {'passed':7s} {'score':>6s} {'t(s)':>6s}")
    print(f"  {'-' * 35} {'-' * 24} {'-' * 7} {'-' * 6} {'-' * 6}")
    for r in rows:
        passed_str = "—" if r.get("passed") is None else ("✓" if r["passed"] else "✗")
        score_str = "—" if r.get("score") is None else f"{r['score']:.2f}"
        print(
            f"  {r['plan'][:35]:35s} {str(r.get('task_id', ''))[:24]:24s} "
            f"{passed_str:7s} {score_str:>6s} {r.get('elapsed_seconds', 0):>6.1f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch sim-env eval (#336)")
    parser.add_argument("--env", required=True, help="Env name, e.g. 'stub' or 'mantis-crm'")
    parser.add_argument("--runtime", choices=("local", "modal", "e2b"), default="local")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--endpoint", default=None,
                        help="Brain endpoint (passed through to `mantis plan run`)")
    parser.add_argument("--plans-dir", default="plans",
                        help="Root of the plan tree (default: plans)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write per-plan dirs + bench_summary.json")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Parallel plan runs (default: 1)")
    parser.add_argument("--plan-arg", action="append", default=None,
                        metavar="ARG",
                        help="Extra arg passed through verbatim to `mantis plan run`. "
                             "Repeat for multiple. Example: --plan-arg --no-headless "
                             "--plan-arg --header --plan-arg X-Foo=bar.")
    args = parser.parse_args(argv)

    plans_dir = Path(args.plans_dir)
    plans = _find_plans(args.env, plans_dir)
    if not plans:
        print(f"error: no plans found under {plans_dir}/{args.env}/", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir or f"outputs/bench-{args.env}-{int(time.time())}")
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_args = list(args.plan_arg or [])

    print(f"sim_envs: env={args.env}  runtime={args.runtime}  plans={len(plans)}  "
          f"workers={args.max_workers}  output={output_dir}")

    rows: list[dict[str, Any]] = []
    if args.max_workers <= 1:
        for plan_path in plans:
            print(f"\n  running {plan_path.name}…")
            rows.append(_run_one(
                plan_path, env_name=args.env, runtime=args.runtime,
                output_dir=output_dir, endpoint=args.endpoint, seed=args.seed,
                extra_args=extra_args,
            ))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(
                    _run_one, plan, env_name=args.env, runtime=args.runtime,
                    output_dir=output_dir, endpoint=args.endpoint, seed=args.seed,
                    extra_args=extra_args,
                ): plan
                for plan in plans
            }
            for fut in concurrent.futures.as_completed(futures):
                plan = futures[fut]
                try:
                    rows.append(fut.result())
                    print(f"  done   {plan.name}")
                except Exception as exc:  # noqa: BLE001
                    rows.append({
                        "plan": plan.name,
                        "task_id": plan.stem,
                        "passed": None,
                        "exit_code": -1,
                        "elapsed_seconds": 0.0,
                        "oracle_error": f"worker raised: {exc!r}",
                    })

    # Stable order in the summary file regardless of completion order.
    rows.sort(key=lambda r: r["plan"])
    summary_path = output_dir / "bench_summary.json"
    summary_payload = {
        "env": args.env,
        "runtime": args.runtime,
        "seed": args.seed,
        "endpoint": args.endpoint,
        "plan_count": len(rows),
        "passed_count": sum(1 for r in rows if r.get("passed") is True),
        "rows": rows,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n",
                            encoding="utf-8")
    _print_summary(rows)
    print(f"\nsummary: {summary_path}")

    # Exit non-zero if any plan didn't pass — useful in CI.
    if summary_payload["passed_count"] < len(rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
