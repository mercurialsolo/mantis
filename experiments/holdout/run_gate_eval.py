"""#916 — holdout-eval runner: generate sim-env suites + run the promotion gate.

The trainer's gate is wired and #911 serves `base + adapter` at `/v1/predict`, but
the holdout eval couldn't actually run: holdout tasks carry only `env` + `task_id`
+ oracle (**no plan**), while `/v1/predict` rejects free-text and requires a built
`task_suite` with a `_micro_plan`. The runnable suite only existed transiently
during rollout generation. This is the missing **sim-env execution link**.

Given holdout task keys (resolved to plans via :mod:`sealed_plans`), a champion
endpoint, and an optional `--lora-adapter <ref>` challenger, this runner:

1. **generates the `task_suite`/`_micro_plan`** for each task (`build_eval_suite`,
   reusing `build_micro_suite` so the suite is properly built — a raw `{steps:…}`
   silently scores 0/0/0, see `feedback_task_suite_shape`),
2. submits to `/v1/predict` — base for the champion arm, `_lora_adapter` for the
   challenger arm (#911) — polls to terminal, oracle-grades,
3. maps each arm to a :class:`~mantis_agent.learning.promotion_gate.ArmResult`
   and runs :meth:`PromotionGate.evaluate` → a ``GateVerdict``.

Distinct ``profile_id`` per (task, arm) means every run is independent under the
per-profile rule (#912), so arms/tasks fan out in parallel (``--max-parallel``).

    python experiments/holdout/run_gate_eval.py \
        --task indeed.t01_search_save_remote \
        --task indeed.t03_employer_review_applicant \
        --task linkedin.t02_post_text_update \
        --lora-adapter mantis-trainer-vol:/checkpoints/sft-c3e0d799f432

The suite-generation + result→ArmResult mapping + tasks-JSON emission are pure
(no network) and unit-tested in ``tests/test_gate_eval_916.py``; the live submit
arm is spend-gated. ``--emit-tasks <path>`` writes an ``eval_harness``-shaped
``--tasks`` JSON (with generated ``micro_plan``s) for the equivalent path.

THIS SPENDS — 2 × N Modal GPU runs (champion + challenger per task).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sealed_plans import SANDBOXES, SEALED_TASKS  # noqa: E402
from state_key_dispatcher import Call, StateKeyDispatcher  # noqa: E402

import run_sealed_task as rst  # noqa: E402  (reuse env-resolve/submit/grade seams)

from mantis_agent.learning.promotion_gate import (  # noqa: E402
    ArmResult,
    GateVerdict,
    PromotionGate,
)
from mantis_agent.plan_decomposer import MicroPlan, PlanDecomposer  # noqa: E402
from mantis_agent.server_utils import (  # noqa: E402
    build_micro_suite,
    micro_plan_steps_to_dicts,
)

# Arm labels — kept distinct so champion/challenger never share a profile_id.
CHAMPION = "champion"
CHALLENGER = "challenger"


def _micro_plan_dicts(spec_def: dict[str, Any], env_url: str) -> list[dict[str, Any]]:
    """Resolve a holdout task's plan steps → built ``_micro_plan`` dicts.

    Substitutes the env URL into nav steps, then runs each through
    ``PlanDecomposer._build_intent`` + ``micro_plan_steps_to_dicts`` — the same
    transform the rollout/fanout suite-builder uses, so the result is a real
    runnable micro-plan rather than raw step dicts.
    """
    steps = rst._substitute(spec_def.get("steps") or [], env_url)
    plan = MicroPlan(domain=spec_def["oracle_task_id"].split(".")[0])
    for st in steps:
        plan.steps.append(PlanDecomposer._build_intent(st))
    return micro_plan_steps_to_dicts(plan.steps)


def build_eval_suite(
    spec_def: dict[str, Any],
    info: dict[str, str],
    *,
    arm: str,
    task_key: str,
    lora_adapter: str = "",
    challenger_model: str = "",
    run_nonce: str = "",
) -> dict[str, Any]:
    """Build the ``/v1/predict`` ``task_suite`` for one (task, arm).

    Champion arm serves the base (no challenger field); challenger arm sets either
    ``_challenger_model`` (#918 full merged-GGUF swap — the working path for the
    qwen3_5_moe Holo3 base) or ``_lora_adapter`` (#911 adapter, for non-MoE/vLLM
    bases). ``profile_id``/``workflow_id`` are scoped per (arm, task) so the two
    arms never collide on one Chrome profile (#912).

    #920: ``run_nonce`` (a per-invocation id) is appended to the profile so a
    *prior* gate run's stale Chrome-profile lock can't 409 this run and cost an
    arm. Each gate invocation uses fresh profiles; the env is reset per run so
    there's no login to preserve.
    """
    micro = _micro_plan_dicts(spec_def, info["url"])
    domain = spec_def["oracle_task_id"].split(".")[0]
    ident = f"gate-{arm}-{task_key}" + (f"-{run_nonce}" if run_nonce else "")
    suite = build_micro_suite(micro, domain, profile_id=ident, workflow_id=ident)
    suite["_plan_name"] = spec_def["oracle_task_id"]
    suite["_browser_extra_headers"] = rst._daytona_headers(info["preview_token"])
    suite["_proxy_disabled"] = True
    # #906: server grades this oracle at finalize so the score reflects ground truth.
    suite["_oracle_url"] = info["url"]
    suite["_oracle_task_id"] = spec_def["oracle_task_id"]
    suite["_oracle_admin_token"] = info["admin_token"]
    suite["_oracle_preview_token"] = info["preview_token"]
    if challenger_model:  # #918 — full-model swap challenger arm
        suite["_challenger_model"] = challenger_model
    elif lora_adapter:  # #911 — adapter challenger arm
        suite["_lora_adapter"] = lora_adapter
    return suite


def to_eval_tasks_json(task_keys: list[str], env_urls: dict[str, str]) -> list[dict[str, Any]]:
    """Emit an ``eval_harness``-shaped ``--tasks`` list with generated
    ``micro_plan``s, so the existing ``eval_harness run/compare`` path can consume
    holdout tasks that otherwise have no plan. ``env_urls`` maps env name → base
    URL used to substitute nav steps."""
    out: list[dict[str, Any]] = []
    for key in task_keys:
        spec = SEALED_TASKS[key]
        env_url = env_urls.get(spec["env"], "")
        out.append({
            "task_id": spec["oracle_task_id"],
            "task_text": spec.get("task_text", ""),
            "micro_plan": _micro_plan_dicts(spec, env_url),
            "metadata": {"env": spec["env"], "oracle_task_id": spec["oracle_task_id"]},
        })
    return out


def arm_from_results(results: list[dict[str, Any]], label: str = "arm") -> ArmResult:
    """Map graded per-task results → an ``ArmResult`` (scores + costs).

    ``score`` is the oracle outcome (1.0 pass / 0.0 fail) keyed by ``task_id``.
    Results missing a ``task_id`` are dropped.
    """
    scores: dict[str, float] = {}
    costs: dict[str, float] = {}
    for r in results:
        tid = r.get("task_id")
        if not tid:
            continue
        scores[tid] = 1.0 if r.get("oracle_passed") else 0.0
        if r.get("cost") is not None:
            costs[tid] = float(r["cost"])
    return ArmResult(label=label, scores=scores, costs=costs)


def _run_one(
    task_key: str, spec_def: dict[str, Any], info: dict[str, str], token: str,
    *, arm: str, lora_adapter: str, challenger_model: str, run_nonce: str,
    max_steps: int, poll_seconds: int,
) -> dict[str, Any]:
    """Submit one (task, arm) to /v1/predict, poll to terminal, oracle-grade."""
    suite = build_eval_suite(
        spec_def, info, arm=arm, task_key=task_key,
        lora_adapter=lora_adapter, challenger_model=challenger_model, run_nonce=run_nonce,
    )
    code, resp = rst._post(token, {
        "task_suite": suite, "profile_id": suite["_profile_id"],
        "workflow_id": suite["_workflow_id"], "cua_model": "holo3",
        "max_steps": max_steps, "max_cost": 2.0, "max_time_minutes": 12, "detached": True,
    })
    tid = spec_def["oracle_task_id"]
    if code != 200 or not resp.get("run_id"):
        return {"task_id": tid, "arm": arm, "submit_error": resp, "oracle_passed": False}
    run_id = resp["run_id"]

    final = None
    deadline = time.monotonic() + poll_seconds
    while time.monotonic() < deadline:
        _, s = rst._post(token, {"action": "status", "run_id": run_id})
        if s.get("status") in {"succeeded", "completed", "completed_with_failures",
                               "failed", "cancelled", "halted"}:
            final = s
            break
        time.sleep(8)

    grade = rst._grade(info["url"], tid, info["admin_token"], info["preview_token"])
    cost = float(((final or {}).get("costs") or {}).get("total", 0.0))
    return {
        "task_id": tid, "arm": arm, "run_id": run_id,
        "terminal_status": (final or {}).get("status"),
        "oracle_passed": bool(grade.get("passed")), "cost": cost,
    }


def _resolve_envs(task_keys: list[str], env: dict[str, str]) -> dict[str, dict[str, str]]:
    """Resolve every distinct env (Daytona id or direct Modal/URL), reset each
    once. Backend-agnostic via ``rst._resolve_env`` (#920)."""
    infos: dict[str, dict[str, str]] = {}
    for env_name in sorted({SEALED_TASKS[k]["env"] for k in task_keys}):
        print(f"[env] resolving '{env_name}' …")
        info = rst._resolve_env(env_name, SANDBOXES.get(env_name, ""), env)
        rst._reset_env(info["url"], info["admin_token"], info["preview_token"])
        infos[env_name] = info
    return infos


def run_gate(
    task_keys: list[str], *, lora_adapter: str = "", challenger_model: str = "",
    max_steps: int, poll_seconds: int, max_parallel: int, gate: PromotionGate,
) -> tuple[GateVerdict, list[dict[str, Any]]]:
    """Run champion + challenger arms over the holdout tasks and gate them."""
    env = rst._load_env()
    token = env.get("MANTIS_API_TOKEN", "")
    if not token:
        raise SystemExit("ERROR: MANTIS_API_TOKEN required")
    infos = _resolve_envs(task_keys, env)

    # #920: a per-invocation nonce so fresh profiles are used each run — a prior
    # run's stale Chrome-profile lock can't 409 this run and cost an arm.
    nonce = uuid.uuid4().hex[:8]

    # One job per (task, arm). Distinct profile_id per job → all independent (#912).
    jobs: list[tuple[str, str]] = []  # (task_key, arm)
    for key in task_keys:
        jobs.append((key, CHAMPION))
        jobs.append((key, CHALLENGER))

    def _mk(task_key: str, arm: str):
        spec = SEALED_TASKS[task_key]
        info = infos[spec["env"]]
        adapter = lora_adapter if arm == CHALLENGER else ""
        cmodel = challenger_model if arm == CHALLENGER else ""
        return lambda _k: _run_one(
            task_key, spec, info, token, arm=arm, lora_adapter=adapter,
            challenger_model=cmodel, run_nonce=nonce,
            max_steps=max_steps, poll_seconds=poll_seconds,
        )

    dispatcher = StateKeyDispatcher(max_parallel=max(1, max_parallel))
    results = dispatcher.run_all(
        [Call(_mk(k, arm), state_key=f"gate-{arm}-{k}-{nonce}") for (k, arm) in jobs]
    )
    dispatcher.shutdown()

    champion = arm_from_results([r for r in results if r.get("arm") == CHAMPION], CHAMPION)
    challenger = arm_from_results([r for r in results if r.get("arm") == CHALLENGER], CHALLENGER)
    return gate.evaluate(champion, challenger), results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="#916 holdout-eval runner → promotion GateVerdict")
    ap.add_argument("--task", action="append", dest="tasks", default=[],
                    help=f"holdout task key (repeatable); known: {sorted(SEALED_TASKS)}")
    ap.add_argument("--lora-adapter", default="",
                    help="#911 challenger adapter ref (--lora; non-MoE / vLLM bases)")
    ap.add_argument("--challenger-model", default="",
                    help="#918 full merged-GGUF challenger ref (-m swap; the working path "
                         "for the qwen3_5_moe Holo3 base). Mutually exclusive with --lora-adapter")
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--poll-seconds", type=int, default=600)
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--emit-tasks", default="",
                    help="write an eval_harness --tasks JSON (generated micro_plans) and exit")
    args = ap.parse_args(argv)

    keys = args.tasks or sorted(SEALED_TASKS)
    unknown = [k for k in keys if k not in SEALED_TASKS]
    if unknown:
        print(f"unknown task(s) {unknown}; known: {sorted(SEALED_TASKS)}", file=sys.stderr)
        return 2

    if args.emit_tasks:
        # Resolve env URLs (best-effort) so nav steps substitute; falls back to ""
        env = rst._load_env()
        dayt = env.get("DAYTONA_API_KEY", "")
        env_urls: dict[str, str] = {}
        if dayt:
            for name in sorted({SEALED_TASKS[k]["env"] for k in keys}):
                try:
                    env_urls[name] = rst._daytona_env(SANDBOXES[name], dayt)["url"]
                except Exception:  # noqa: BLE001 — emit with empty url if unreachable
                    env_urls[name] = ""
        Path(args.emit_tasks).write_text(json.dumps(to_eval_tasks_json(keys, env_urls), indent=2))
        print(f"wrote {len(keys)} task(s) → {args.emit_tasks}")
        return 0

    if args.lora_adapter and args.challenger_model:
        print("ERROR: pass only one of --lora-adapter / --challenger-model", file=sys.stderr)
        return 2
    if not args.lora_adapter and not args.challenger_model:
        print("⚠️  no challenger ref: champion and challenger both serve the base "
              "(a plumbing sanity run — expect delta≈0, promote=False).", file=sys.stderr)

    verdict, results = run_gate(
        keys, lora_adapter=args.lora_adapter, challenger_model=args.challenger_model,
        max_steps=args.max_steps, poll_seconds=args.poll_seconds,
        max_parallel=args.max_parallel, gate=PromotionGate(),
    )
    print("\n[gate eval result]")
    print(json.dumps({
        "lora_adapter": args.lora_adapter,
        "challenger_model": args.challenger_model,
        "results": results,
        "verdict": {
            "promote": verdict.promote, "reason": verdict.reason,
            "n_compared": verdict.n_compared, "mean_delta": verdict.mean_delta,
            "win_rate": verdict.win_rate, "prob_improvement": verdict.prob_improvement,
            "regressions": verdict.regressions,
            "champion_cost": verdict.champion_cost, "challenger_cost": verdict.challenger_cost,
        },
    }, indent=2))
    print(f"\n{'✅ PROMOTE' if verdict.promote else '❌ HOLD'} — {verdict.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
