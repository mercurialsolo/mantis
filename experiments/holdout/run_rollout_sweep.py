"""#905 — generate GRPO sibling rollout groups (shared group_id).

GRPO/DPO need *sibling groups*: N rollouts of the SAME task instance (template ×
env-seed) that share a ``group_id``, with reward variance across siblings so the
group-relative advantage is non-zero. This runner turns the stable
:class:`~mantis_agent.learning.rollout_generator.SeedSweepGenerator` specs into
real graded runs:

For each ``RolloutSpec`` (template, env_seed, group_id, sibling_index):
  1. seed the env to ``env_seed`` (``POST /__env__/seed {seed}``),
  2. submit the template's micro-plan to the Modal CUA server with
     ``_fanout_group_id = spec.group_id`` (the deployed server forwards this to
     ``micro_runner._fanout_group_id`` → Augur ``DebugSession(group_id=…)``) and
     ``_sampling_temperature > 0`` so siblings diverge (#905 server change),
  3. poll to terminal, grade via the env oracle.

Siblings of one (template, seed) share ``group_id`` but sample different
trajectories (temperature) → the reward variance GRPO standardizes over.

    python experiments/holdout/run_rollout_sweep.py indeed.t01_search_save_remote \
        --seed 42 --siblings 3 --temperature 0.7

THIS SPENDS — N sibling Modal GPU runs per (template, seed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sealed_plans import SANDBOXES, SEALED_TASKS  # noqa: E402
from state_key_dispatcher import Call, StateKeyDispatcher  # noqa: E402

import run_sealed_task as rst  # noqa: E402  (reuse env-resolve/submit/grade seams)

from mantis_agent.learning.rollout_generator import (  # noqa: E402
    RolloutSpec,
    SeedSweepGenerator,
    TaskTemplate,
)
from mantis_agent.plan_decomposer import MicroPlan, PlanDecomposer  # noqa: E402
from mantis_agent.server_utils import (  # noqa: E402
    build_micro_suite,
    micro_plan_steps_to_dicts,
)


def _seed_env(env_url: str, seed: int, admin_token: str, preview_token: str) -> tuple[int, str]:
    """Re-seed the env to a specific instance (POST /__env__/seed {seed})."""
    r = requests.post(
        f"{env_url}/__env__/seed",
        headers={"X-Env-Admin": admin_token, **rst._daytona_headers(preview_token)},
        json={"seed": seed},
        timeout=30,
    )
    return r.status_code, r.text[:120]


def _run_sibling(
    spec: Any, info: dict[str, str], token: str, temperature: float,
    *, max_steps: int, poll_seconds: int, model_judge: bool = False,
    task_text: str = "", oracle_less: bool = False,
) -> dict[str, Any]:
    tmpl = spec.template
    steps = rst._substitute(tmpl.plan_steps or [], info["url"])
    plan = MicroPlan(domain=tmpl.template_id.split(".")[0])
    for st in steps:
        plan.steps.append(PlanDecomposer._build_intent(st))
    suite = build_micro_suite(
        micro_plan_steps_to_dicts(plan.steps), plan.domain,
        profile_id=f"sweep-{spec.spec_id}", workflow_id=f"sweep-{spec.spec_id}",
    )
    suite["_plan_name"] = tmpl.oracle_task_id
    suite["_browser_extra_headers"] = rst._daytona_headers(info["preview_token"])
    suite["_proxy_disabled"] = True
    suite["_fanout_group_id"] = spec.group_id           # → Augur group_id (#905)
    suite["_sampling_temperature"] = temperature        # → sibling diversity (#905)
    # #908: drive action steps with the Holo3 policy so planner modelio +
    # per-token logprobs are captured (GRPO importance ratio requirement).
    suite["_force_holo3_grounding"] = True
    # #906: server grades this oracle at finalize + stamps the verdict on the
    # terminal step so Augur reward reflects ground truth → GRPO reward variance.
    if not oracle_less:
        suite["_oracle_url"] = info["url"]
        suite["_oracle_task_id"] = tmpl.oracle_task_id
        suite["_oracle_admin_token"] = info["admin_token"]
        suite["_oracle_preview_token"] = info["preview_token"]
    if model_judge:  # #906 extension: model-judge (primary signal on oracle-less)
        suite["_model_judge_enabled"] = True
        if oracle_less:
            suite["_task_instruction"] = task_text or tmpl.oracle_task_id.replace("_", " ")
        else:
            suite["_model_judge_force"] = True  # cross-check mode where an oracle exists
            suite["_task_instruction"] = task_text or tmpl.oracle_task_id.replace("_", " ")

    code, resp = rst._post(token, {
        "task_suite": suite, "profile_id": suite["_profile_id"],
        "workflow_id": suite["_workflow_id"], "cua_model": "holo3",
        "max_steps": max_steps, "max_cost": 2.0, "max_time_minutes": 12, "detached": True,
    })
    if code != 200 or not resp.get("run_id"):
        return {"spec_id": spec.spec_id, "group_id": spec.group_id, "submit_error": resp}
    run_id = resp["run_id"]

    last, final = None, None
    deadline = time.monotonic() + poll_seconds
    while time.monotonic() < deadline:
        _, s = rst._post(token, {"action": "status", "run_id": run_id})
        st = s.get("status")
        if st != last:
            print(f"    [{spec.spec_id}] status={st!r}")
            last = st
        if st in {"succeeded", "completed", "completed_with_failures",
                  "failed", "cancelled", "halted"}:
            final = s
            break
        time.sleep(8)

    grade = rst._grade(info["url"], tmpl.oracle_task_id, info["admin_token"], info["preview_token"])
    return {
        "spec_id": spec.spec_id, "sibling_index": spec.sibling_index,
        "group_id": spec.group_id, "run_id": run_id, "env_seed": spec.env_seed,
        "terminal_status": (final or {}).get("status"),
        "oracle_passed": bool(grade.get("passed")),
        "reward": 1.0 if grade.get("passed") else 0.0,
    }


# Augur reward spread below this (≈ step-cost "hair") is noise, not signal.
_MEANINGFUL_STD = 0.05


def _augur_group_rewards(group_id: str, env: dict[str, str]) -> dict[str, float]:
    """Fetch each sibling's Augur ``episode_return`` (process/progress-shaped)
    for the group — the reward the trainer actually standardizes. Bearer reads
    only; returns {} on any failure so the gate falls back to local outcomes."""
    base = "https://mantis-cua.ngrok-free.app/api/v1"
    dsn = env.get("AUGUR_DSN", "")
    key = env.get("AUGUR_API_KEY") or (
        dsn.split("token=", 1)[-1].split("&", 1)[0] if "token=" in dsn else ""
    )
    # Tenant from env/DSN — never hard-code a customer name in source.
    tenant = (env.get("AUGUR_TENANT") or "").strip() or (
        dsn.split("tenant=", 1)[1].split("&", 1)[0] if "tenant=" in dsn else ""
    )
    if not key:
        return {}
    h = {"Authorization": f"Bearer {key}", "ngrok-skip-browser-warning": "1"}
    out: dict[str, float] = {}
    try:
        runs = requests.get(f"{base}/runs?tenant={tenant}&group_id={group_id}&limit=50",
                            headers=h, timeout=30).json()
        for r in (runs.get("result") or runs.get("runs") or []):
            rid = r.get("run_id")
            if not rid:
                continue
            rew = requests.get(f"{base}/runs/{rid}/reward/default-v1?tenant={tenant}",
                               headers=h, timeout=30).json()
            er = rew.get("episode_return")
            if er is not None:
                out[rid] = float(er)
    except Exception:  # noqa: BLE001 — best-effort; gate falls back to local outcomes
        return {}
    return out


def _classify_group(results: list[dict[str, Any]],
                    augur_rewards: dict[str, float] | None = None) -> dict[str, Any]:
    """Group-variance gate (mantis-trainer feedback): GRPO standardizes
    ``(r − mean)/(std + eps)``, so a group with only step-cost "hair" variance
    yields ±1 noise advantages. A group is GRPO-usable when it has REAL reward
    spread — either mixed oracle outcomes OR (with #906 process/progress
    shaping) meaningful Augur ``episode_return`` variance from differing
    effort/progress even on an all-pass / all-fail group."""
    import statistics
    outcomes = [bool(r.get("oracle_passed")) for r in results if "oracle_passed" in r]
    distinct = len(set(outcomes))
    n_pass = sum(1 for o in outcomes if o)

    shaped = sorted((augur_rewards or {}).values())
    shaped_std = statistics.pstdev(shaped) if len(shaped) > 1 else 0.0
    local_std = statistics.pstdev([float(r["reward"]) for r in results if "reward" in r]) \
        if len([r for r in results if "reward" in r]) > 1 else 0.0

    mixed = distinct > 1
    shaped_signal = shaped_std >= _MEANINGFUL_STD
    usable = mixed or shaped_signal
    if usable:
        reason = ("mixed outcomes" if mixed else
                  f"shaped reward variance (episode_return std={shaped_std:.3f})")
    else:
        reason = (
            f"degenerate: all {'pass' if n_pass else 'fail'} ({n_pass}/{len(outcomes)})"
            f" and shaped std {shaped_std:.4f} < {_MEANINGFUL_STD}"
            " — GRPO advantage would be noise; exclude or re-sample"
        )
    return {"n": len(outcomes), "n_pass": n_pass, "distinct_outcomes": distinct,
            "local_reward_std": round(local_std, 4),
            "shaped_episode_return_std": round(shaped_std, 4),
            "grpo_usable": usable, "reason": reason}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="#905 GRPO sibling rollout sweep")
    ap.add_argument("task_key", help=f"template from sealed_plans: {sorted(SEALED_TASKS)}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--siblings", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--poll-seconds", type=int, default=600)
    ap.add_argument("--model-judge", action="store_true",
                    help="#906: also run the model-judge (rm_outcome + judge_ids)")
    ap.add_argument("--no-oracle", action="store_true",
                    help="#906: skip the env oracle (simulate an oracle-less task; judge is the signal)")
    ap.add_argument("--variance-seek", action="store_true",
                    help="keep adding siblings (to --max-siblings) until the group has mixed outcomes (real GRPO variance)")
    ap.add_argument("--max-siblings", type=int, default=6,
                    help="variance-seek cap on total siblings per group")
    ap.add_argument("--max-parallel", type=int, default=4,
                    help="client-side cap on simultaneous sibling runs. Siblings each "
                         "get a distinct state_key (sweep-<spec_id>), so they are safe "
                         "to fan out in parallel; 1 = sequential (old behavior)")
    args = ap.parse_args(argv)

    if args.task_key not in SEALED_TASKS:
        print(f"unknown task; known: {sorted(SEALED_TASKS)}", file=sys.stderr)
        return 2
    spec_def = SEALED_TASKS[args.task_key]
    env_name = spec_def["env"]
    env = rst._load_env()
    token, dayt = env.get("MANTIS_API_TOKEN", ""), env.get("DAYTONA_API_KEY", "")
    if not token or not dayt:
        print("ERROR: MANTIS_API_TOKEN + DAYTONA_API_KEY required", file=sys.stderr)
        return 2

    template = TaskTemplate(
        template_id=args.task_key, cluster=env_name,
        oracle_task_id=spec_def["oracle_task_id"], plan_steps=spec_def["steps"],
    )
    gen = SeedSweepGenerator(templates=[template], seeds=[args.seed],
                             siblings_per_instance=args.siblings)
    specs = list(gen.generate())
    print(f"[sweep] {len(specs)} siblings of {args.task_key} @ seed={args.seed} "
          f"group_id={specs[0].group_id!r} temp={args.temperature}")

    print(f"[env] resolving Daytona sandbox for '{env_name}' …")
    info = rst._daytona_env(SANDBOXES[env_name], dayt)
    print(f"  url={info['url']}  preview={'set' if info['preview_token'] else 'MISSING'}")
    code, body = _seed_env(info["url"], args.seed, info["admin_token"], info["preview_token"])
    print(f"[seed={args.seed}] HTTP {code} {body[:60]}")

    def _run(spec):
        print(f"  → sibling {spec.sibling_index} ({spec.spec_id})")
        return _run_sibling(spec, info, token, args.temperature,
                            max_steps=args.max_steps, poll_seconds=args.poll_seconds,
                            model_judge=args.model_judge,
                            task_text=spec_def.get("task_text", ""),
                            oracle_less=args.no_oracle)

    # Siblings each carry a distinct state_key (sweep-<spec_id> → its own Chrome
    # profile + checkpoint), so they're INDEPENDENT under Mantis's per-state-key
    # rule and safe to fan out in parallel up to --max-parallel. The dispatcher's
    # resolved key is unused here (each _run already stamps its own ids).
    dispatcher = StateKeyDispatcher(max_parallel=max(1, args.max_parallel))
    print(f"[sweep] dispatching {len(specs)} siblings with max_parallel={args.max_parallel}")
    results = dispatcher.run_all(
        [Call((lambda s: lambda _k: _run(s))(spec), state_key=f"sweep-{spec.spec_id}")
         for spec in specs]
    )

    # Variance gate (mantis-trainer feedback): a group is GRPO-usable with real
    # reward spread — mixed outcomes OR (with #906 process/progress shaping)
    # meaningful Augur episode_return variance. Keep adding siblings while
    # degenerate, up to the cap.
    gid = specs[0].group_id
    while (args.variance_seek
           and not _classify_group(results, _augur_group_rewards(gid, env))["grpo_usable"]
           and len(results) < args.max_siblings):
        idx = len(results)
        print(f"  [variance-seek] group still degenerate after {idx} siblings — adding sibling {idx}")
        extra = RolloutSpec(
            spec_id=f"{template.template_id}__seed{args.seed}__s{idx}",
            template=template, env_seed=args.seed, group_id=gid, sibling_index=idx,
        )
        # variance-seek is inherently sequential (decide, then add one), so this
        # single extra goes straight through the dispatcher and we await it.
        results.append(
            dispatcher.submit((lambda s: lambda _k: _run(s))(extra),
                              state_key=f"sweep-{extra.spec_id}").result()
        )

    dispatcher.shutdown()
    gate = _classify_group(results, _augur_group_rewards(gid, env))
    print("\n[sweep result]")
    print(json.dumps({
        "group_id": gid,
        "siblings": results,
        "rewards": [r.get("reward") for r in results if "reward" in r],
        "variance_gate": gate,
    }, indent=2))
    if not gate["grpo_usable"]:
        print(f"\n⚠️  GROUP NOT GRPO-USABLE — {gate['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
