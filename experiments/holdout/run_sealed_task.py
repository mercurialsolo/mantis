"""Run one sealed holdout task through the deployed Modal CUA server against its
live Daytona sim env, then grade it with the env oracle (#894).

Pipeline per task (recipe verified 2026-06-14):

1. Resolve the env's Daytona sandbox → ensure uvicorn → preview URL + token +
   admin token (read from inside the sandbox).
2. ``POST /__env__/reset`` (admin) so the run starts from the seeded baseline.
3. Build a micro-suite from the pre-decomposed plan in ``sealed_plans.py`` with
   ``_plan_name`` (→ Augur task_spec_id) and ``_browser_extra_headers`` carrying
   the Daytona preview token + skip-warning so the Modal browser can reach the
   sandbox. ``{env_url}`` in nav steps is substituted with the preview URL.
4. ``POST /v1/predict`` (detached) → poll to terminal.
5. Grade via ``GET /__env__/oracle?task_id=…`` (admin).

On plan-completion the producer emits a ``source:producer`` eval candidate keyed
``<domain>.<plan_name>.v1`` (see ``observability/eval_curation``); this script
prints that task_spec_id so the freeze step
(``freeze_eval_version.py``) can select the oracle-passing ones.

    python experiments/holdout/run_sealed_task.py indeed.t01_search_save_remote

THIS SPENDS — it submits a real Modal GPU run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import requests

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sealed_plans import SANDBOXES, SEALED_TASKS  # noqa: E402

from mantis_agent.plan_decomposer import MicroPlan, PlanDecomposer  # noqa: E402
from mantis_agent.server_utils import (  # noqa: E402
    build_micro_suite,
    micro_plan_steps_to_dicts,
)

ENDPOINT = "https://getmason--mantis-cua-server-api.modal.run"
PORT = 8080


def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in (REPO_ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _daytona_env(sandbox_id: str, api_key: str) -> dict[str, str]:
    """Start the sandbox if needed, ensure uvicorn, return url/token/admin."""
    from daytona import Daytona, DaytonaConfig

    d = Daytona(DaytonaConfig(api_key=api_key))
    s = d.get(sandbox_id)
    try:
        s.start(timeout=120)
    except Exception:
        pass  # already running
    # Ensure uvicorn is up (auto-relaunch on revive isn't guaranteed) + read admin token.
    # Daytona's exec proxy intermittently 504s even when the command runs — treat
    # exec failures as non-fatal and verify liveness via the preview URL instead
    # (see feedback_daytona_patch_exec_504).
    boot = (
        "python3 -c \"import urllib.request as u; "
        "print(u.urlopen('http://127.0.0.1:8080/__env__/health',timeout=4).read().decode())\" "
        "2>/dev/null || (cd /srv && nohup python -m uvicorn app.main:app "
        "--host 0.0.0.0 --port 8080 >/tmp/uv.log 2>&1 & sleep 6)"
    )

    def _exec(cmd: str) -> str:
        try:
            r = s.process.exec(cmd)
            return (getattr(r, "result", "") or getattr(r, "output", "") or "").strip()
        except Exception as e:  # noqa: BLE001 — Daytona 504s are transient
            print(f"  warn: exec failed (non-fatal): {str(e)[:80]}")
            return ""

    _exec(boot)
    admin_token = _exec("printf '%s' \"$ENV_ADMIN_TOKEN\"")
    prev = s.get_preview_link(PORT)
    return {
        "url": prev.url.rstrip("/"),
        "preview_token": getattr(prev, "token", "") or "",
        "admin_token": admin_token,
    }


def _daytona_headers(preview_token: str) -> dict[str, str]:
    h = {"X-Daytona-Skip-Preview-Warning": "true"}
    if preview_token:
        h["x-daytona-preview-token"] = preview_token
    return h


def _resolve_env(env_name: str, sandbox_ref: str, env: dict[str, str]) -> dict[str, str]:
    """Resolve a holdout env to ``{url, preview_token, admin_token}`` — backend
    agnostic (#920).

    ``sandbox_ref`` (the ``SANDBOXES[env_name]`` value) is either:
    * a **direct URL** (``https://…``) — a Modal/other-hosted sim env reached
      directly (no Daytona preview token). The ``X-Env-Admin`` token comes from
      ``<ENV>_ADMIN_TOKEN`` (e.g. ``CRM_ADMIN_TOKEN``) or the shared
      ``ENV_ADMIN_TOKEN`` in ``.env``; or
    * a **Daytona sandbox id** — resolved via :func:`_daytona_env` (boots uvicorn,
      returns the preview url + token + the in-sandbox ``ENV_ADMIN_TOKEN``).

    Raises if the env isn't wired (empty ref) so the failure is explicit, not a
    silent base-env run.
    """
    ref = (sandbox_ref or "").strip()
    if not ref:
        raise RuntimeError(
            f"env {env_name!r} is not wired in SANDBOXES (empty) — add a Daytona "
            f"sandbox id or a direct https URL before running it."
        )
    if ref.startswith("http://") or ref.startswith("https://"):
        admin = (
            env.get(f"{env_name.upper()}_ADMIN_TOKEN")
            or env.get("ENV_ADMIN_TOKEN")
            or ""
        )
        return {"url": ref.rstrip("/"), "preview_token": "", "admin_token": admin}
    return _daytona_env(ref, env.get("DAYTONA_API_KEY", ""))


def _reset_env(env_url: str, admin_token: str, preview_token: str) -> tuple[int, str]:
    r = requests.post(
        f"{env_url}/__env__/reset",
        headers={"X-Env-Admin": admin_token, **_daytona_headers(preview_token)},
        timeout=30,
    )
    return r.status_code, r.text[:200]


def _grade(env_url: str, task_id: str, admin_token: str, preview_token: str) -> dict[str, Any]:
    r = requests.get(
        f"{env_url}/__env__/oracle",
        params={"task_id": task_id},
        headers={"X-Env-Admin": admin_token, **_daytona_headers(preview_token)},
        timeout=30,
    )
    try:
        return r.json()
    except ValueError:
        return {"http": r.status_code, "raw": r.text[:300]}


def _post(token: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    r = requests.post(
        f"{ENDPOINT}/v1/predict",
        headers={"X-Mantis-Token": token, "Content-Type": "application/json"},
        data=json.dumps(body),
        timeout=90,
    )
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"raw": r.text[:300]}


def _substitute(steps: list[dict[str, Any]], env_url: str) -> list[dict[str, Any]]:
    out = json.loads(json.dumps(steps))
    for st in out:
        p = st.get("params", {})
        if isinstance(p.get("url"), str):
            p["url"] = p["url"].replace("{env_url}", env_url)
    return out


def run(task_key: str, *, max_steps: int = 25, poll_seconds: int = 600) -> int:
    if task_key not in SEALED_TASKS:
        print(f"unknown task '{task_key}'. known: {sorted(SEALED_TASKS)}", file=sys.stderr)
        return 2
    spec = SEALED_TASKS[task_key]
    env_name = spec["env"]
    env = _load_env()
    token = env.get("MANTIS_API_TOKEN", "")
    if not token:
        print("ERROR: MANTIS_API_TOKEN required in .env", file=sys.stderr)
        return 2

    print(f"[env] resolving env '{env_name}' …")
    info = _resolve_env(env_name, SANDBOXES.get(env_name, ""), env)
    print(f"  url={info['url']}")
    print(f"  preview_token={'set' if info['preview_token'] else 'MISSING'}  "
          f"admin_token={'set' if info['admin_token'] else 'MISSING'}")

    code, body = _reset_env(info["url"], info["admin_token"], info["preview_token"])
    print(f"[reset] HTTP {code} {body[:80]}")

    steps = _substitute(spec["steps"], info["url"])
    plan = MicroPlan(domain=env_name)
    for st in steps:
        plan.steps.append(PlanDecomposer._build_intent(st))
    suite = build_micro_suite(
        micro_plan_steps_to_dicts(plan.steps),
        plan.domain,
        profile_id=f"holdout-{task_key}",
        workflow_id=f"holdout-{task_key}",
    )
    suite["_plan_name"] = spec["plan_name"]
    suite["_browser_extra_headers"] = _daytona_headers(info["preview_token"])
    suite["_proxy_disabled"] = True
    # #906: let the server grade the env oracle at finalize so the verdict +
    # reward reflect ground truth AND mark_for_eval is gated on an oracle pass
    # (eval candidates become oracle-verified, not merely plan-complete).
    suite["_oracle_url"] = info["url"]
    suite["_oracle_task_id"] = spec["oracle_task_id"]
    suite["_oracle_admin_token"] = info["admin_token"]
    suite["_oracle_preview_token"] = info["preview_token"]
    expected_ts = f"{env_name}.{spec['plan_name']}.v1"
    print(f"[submit] plan_name={spec['plan_name']!r} → task_spec_id={expected_ts!r}")

    code, resp = _post(token, {
        "task_suite": suite,
        "profile_id": suite["_profile_id"],
        "workflow_id": suite["_workflow_id"],
        "cua_model": "holo3",
        "max_steps": max_steps,
        "max_cost": 2.0,
        "max_time_minutes": 12,
        "detached": True,
    })
    if code != 200 or not resp.get("run_id"):
        print(f"  FAIL submit HTTP {code}: {resp}")
        return 1
    run_id = resp["run_id"]
    print(f"  run_id={run_id}")

    print("[poll] …")
    last = None
    final = None
    deadline = time.monotonic() + poll_seconds
    while time.monotonic() < deadline:
        code, s = _post(token, {"action": "status", "run_id": run_id})
        status = s.get("status")
        if status != last:
            print(f"  status={status!r}")
            last = status
        if status in {"succeeded", "completed", "completed_with_failures",
                      "failed", "cancelled", "halted"}:
            final = s
            break
        time.sleep(8)

    fstatus = (final or {}).get("status")
    print(f"[terminal] status={fstatus!r} halt_reason={(final or {}).get('halt_reason')!r}")
    grade = _grade(info["url"], spec["oracle_task_id"], info["admin_token"], info["preview_token"])
    passed = grade.get("passed")
    print(f"[oracle] task_id={spec['oracle_task_id']} passed={passed} reasons={grade.get('reasons')}")
    print(json.dumps({
        "task_key": task_key,
        "run_id": run_id,
        "terminal_status": (final or {}).get("status"),
        "task_spec_id": expected_ts,
        "oracle_passed": bool(passed),
    }, indent=2))
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a sealed holdout task end-to-end")
    ap.add_argument("task_key", help=f"one of: {sorted(SEALED_TASKS)}")
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--poll-seconds", type=int, default=600)
    args = ap.parse_args(argv)
    return run(args.task_key, max_steps=args.max_steps, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
