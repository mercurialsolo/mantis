"""Submit ``plans/luma-extract.json`` to the deployed Modal CUA server.

Companion to ``run_staff_crm_long_no_proxy.py``: same submit/poll
shape, but exercises the proxy-ON path. The plan's top-level
``runtime`` block declares ``proxy_disabled: false`` + a Miami exit,
so this script doesn't need to know anything about proxy config —
``merge_runtime()`` forwards whatever the plan declares.

Usage::

    uv run python scripts/run_luma_extract.py
"""

from __future__ import annotations

import json
import os
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
PLAN_PATH = REPO_ROOT / "plans" / "luma-extract.json"
# #572: stable PROFILE_ID by default so cookies (including any CF
# clearance) persist across runs. Override via env for a clean
# profile. WORKFLOW_ID stays timestamped per run.
PROFILE_ID = os.environ.get("MANTIS_PROFILE_ID", "luma-extract-stable")
WORKFLOW_ID = f"luma-extract-{int(time.time())}"


def _token() -> str:
    tok = os.environ.get("MANTIS_API_TOKEN", "")
    if not tok:
        print("ERROR: MANTIS_API_TOKEN required in env", file=sys.stderr)
        sys.exit(2)
    return tok


def _post(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    r = requests.post(
        f"{ENDPOINT}{path}",
        headers={
            "X-Mantis-Token": _token(),
            "Content-Type": "application/json",
        },
        data=json.dumps(body),
        timeout=120,
    )
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}


def _build_suite() -> tuple[dict[str, Any], dict[str, Any]]:
    steps, plan_runtime = load_plan_file(PLAN_PATH)
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
    print(f"plan:        {PLAN_PATH.name}")
    print(f"profile_id:  {PROFILE_ID}")
    print(f"workflow_id: {WORKFLOW_ID}")

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
        "cua_model": "holo3",
        "max_steps": 30,
        "detached": True,
        **runtime,
    }
    status, resp = _post("/v1/predict", submit_body)
    print(f"[submit] HTTP {status}")
    print(f"  run_id={resp.get('run_id')!r}")
    if status != 200:
        print(f"  body={json.dumps(resp, indent=2)[:600]}", file=sys.stderr)
        return 4

    run_id = resp["run_id"]
    terminal = {"succeeded", "failed", "cancelled", "halted"}
    last_status = ""
    started = time.time()
    poll_n = 0
    while True:
        poll_n += 1
        if time.time() - started > 15 * 60:
            print("TIMEOUT: 15 minutes without terminal status", file=sys.stderr)
            return 5
        s_status, s_resp = _post(
            "/v1/predict",
            {"action": "status", "run_id": run_id},
        )
        status_str = s_resp.get("status", "?")
        if status_str != last_status:
            print(
                f"[{poll_n:3d}] t={int(time.time() - started):4d}s  "
                f"status={status_str}"
            )
            last_status = status_str
        if status_str in terminal:
            print()
            print("=== TERMINAL STATUS ===")
            print(json.dumps(s_resp, indent=2)[:2500])
            return 0
        time.sleep(20)


if __name__ == "__main__":
    sys.exit(main())
