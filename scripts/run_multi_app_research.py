"""Run the multi-app research plan against mantis-cua-server (#931 verify).

Submits ``plans/multi-app-research.json`` — a long chain that spans THREE
web apps (staff CRM → lu.ma → Hacker News) — to ``/v1/predict`` and polls
until terminal. Exercises the #931 fill/submit verification on the CRM
login plus cross-app navigation. No fixture seeding needed (the plan reads
"the first lead", any row works).

The CRM target + login values in the plan are ``${CRM_BASE_URL}`` /
``${CRM_USER}`` / ``${CRM_PASSWORD}`` placeholders — neither the customer CRM
host nor its credentials are committed to the repo. Supply them via env at run
time (the harness substitutes ``${VAR}`` tokens from the environment).

Usage::

    MANTIS_API_TOKEN=... CRM_BASE_URL=https://... CRM_USER=... CRM_PASSWORD=... \\
        uv run python scripts/run_multi_app_research.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mantis_agent.plan_decomposer import MicroPlan, PlanDecomposer  # noqa: E402
from mantis_agent.server_utils import (  # noqa: E402
    build_micro_suite,
    load_plan_file,
    merge_runtime,
    micro_plan_steps_to_dicts,
)

ENDPOINT = "https://getmason--mantis-cua-server-api.modal.run"
PLAN_PATH = REPO_ROOT / "plans" / "multi-app-research.json"
PROFILE_ID = f"multiapp-{int(time.time())}"
WORKFLOW_ID = f"multiapp-{int(time.time())}"


def _token() -> str:
    tok = os.environ.get("MANTIS_API_TOKEN", "")
    if not tok:
        print("ERROR: MANTIS_API_TOKEN required in env", file=sys.stderr)
        sys.exit(2)
    return tok


def _post(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    r = requests.post(
        f"{ENDPOINT}{path}",
        headers={"X-Mantis-Token": _token(), "Content-Type": "application/json"},
        data=json.dumps(body),
        timeout=120,
    )
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}


def _subst_env(obj: Any) -> Any:
    """Recursively replace ``${VAR}`` tokens with os.environ values.

    Keeps credentials out of the tracked plan: the plan ships placeholder
    tokens (``${CRM_USER}`` / ``${CRM_PASSWORD}``), the operator supplies the
    real values via env. An unset token is left verbatim so the failure is
    obvious rather than silently typing an empty string.
    """
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _subst_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_subst_env(v) for v in obj]
    return obj


def _build_suite() -> tuple[dict[str, Any], dict[str, Any]]:
    steps, plan_runtime = load_plan_file(PLAN_PATH)
    steps = [_subst_env(s) for s in steps]
    plan = MicroPlan(domain=PLAN_PATH.stem)
    for step in steps:
        plan.steps.append(PlanDecomposer._build_intent(step))
    runtime = merge_runtime(plan_runtime)
    suite = build_micro_suite(
        micro_plan_steps_to_dicts(plan.steps),
        plan.domain,
        profile_id=PROFILE_ID,
        workflow_id=WORKFLOW_ID,
        **runtime,
    )
    return suite, runtime


def main() -> int:
    print(f"endpoint:    {ENDPOINT}")
    print(f"plan:        {PLAN_PATH.name} (CRM → lu.ma → Hacker News)")
    print(f"profile_id:  {PROFILE_ID}")
    print()

    h = requests.get(f"{ENDPOINT}/v1/health", timeout=30)
    print(f"[health] HTTP {h.status_code}: {h.text[:120]}")
    if h.status_code != 200:
        print("aborting — endpoint not healthy", file=sys.stderr)
        return 3

    suite, runtime = _build_suite()
    print(f"[plan-runtime] {runtime}")
    submit_body = {
        "task_suite": suite,
        "profile_id": suite["_profile_id"],
        "workflow_id": suite["_workflow_id"],
        "cua_model": "holo3",  # run_holo3 is the executor that consumes _micro_plan
        "max_steps": 60,
        "detached": True,
        **runtime,
    }
    status, resp = _post("/v1/predict", submit_body)
    print(f"[submit] HTTP {status}  run_id={resp.get('run_id')!r}")
    if status != 200:
        print(f"  body={json.dumps(resp, indent=2)[:600]}", file=sys.stderr)
        return 4

    run_id = resp["run_id"]
    terminal = {"succeeded", "failed", "cancelled", "halted", "completed", "completed_with_failures"}
    last_status = ""
    started = time.time()
    poll_n = 0
    while True:
        poll_n += 1
        if time.time() - started > 35 * 60:
            print("TIMEOUT: 35 minutes without terminal status", file=sys.stderr)
            return 5
        _, s_resp = _post("/v1/predict", {"action": "status", "run_id": run_id})
        status_str = s_resp.get("status", "?")
        if status_str != last_status:
            print(f"[{poll_n:3d}] t={int(time.time() - started):4d}s  status={status_str}")
            last_status = status_str
        if status_str in terminal:
            print("\n=== TERMINAL STATUS ===")
            print(json.dumps(s_resp, indent=2)[:3000])
            return 0
        time.sleep(20)


if __name__ == "__main__":
    sys.exit(main())
