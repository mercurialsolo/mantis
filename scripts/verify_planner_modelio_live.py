"""Live verification for Gap 1 — Holo3 planner-layer modelio capture.

Submits a brain-driving plan (a coordinate-free ``click`` step forces
the Holo3 SoM ``brain.think`` loop, unlike navigate / host-tool / form
steps which bypass the brain). After the run terminates, the caller
checks the Augur bundle for a ``planner``-layer modelio record.

Usage:
    uv run python scripts/verify_planner_modelio_live.py \
        --token "$MANTIS_API_TOKEN"
"""

from __future__ import annotations

import argparse
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
    micro_plan_steps_to_dicts,
)

PLAN_STEPS = [
    {
        "type": "navigate",
        "intent": "Go to the login page",
        "params": {"url": "https://the-internet.herokuapp.com/login"},
        "required": True,
        "section": "setup",
        "gate": False,
    },
    {
        # Coordinate-free click → Holo3 SoM brain.think drives it.
        "type": "click",
        "intent": "Click the Login button",
        "params": {"target": "Login"},
        "required": True,
        "section": "login",
    },
]


def post(url: str, token: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    r = requests.post(
        url,
        headers={"X-Mantis-Token": token, "Content-Type": "application/json"},
        data=json.dumps(body),
        timeout=60,
    )
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--endpoint", default="https://getmason--mantis-cua-server-api.modal.run",
    )
    ap.add_argument("--token", default=os.environ.get("MANTIS_API_TOKEN", ""))
    ap.add_argument("--profile-id", default="planner-modelio")
    ap.add_argument("--workflow-id", default="planner-modelio")
    args = ap.parse_args()

    if not args.token:
        print("ERROR: --token or MANTIS_API_TOKEN required", file=sys.stderr)
        return 2

    endpoint = args.endpoint.rstrip("/")
    predict = f"{endpoint}/v1/predict"

    plan = MicroPlan(domain="planner_modelio_probe")
    for step in PLAN_STEPS:
        plan.steps.append(PlanDecomposer._build_intent(step))
    suite = build_micro_suite(
        micro_plan_steps_to_dicts(plan.steps),
        plan.domain,
        profile_id=args.profile_id,
        workflow_id=args.workflow_id,
    )

    print("[submit] holo3 brain-driving plan (navigate + coord-free click)")
    st, resp = post(
        predict, args.token,
        {
            "task_suite": suite,
            "profile_id": suite["_profile_id"],
            "workflow_id": suite["_workflow_id"],
            "cua_model": "holo3",
            "max_steps": 6,
            "max_cost": 1.0,
            "max_time_minutes": 8,
            "detached": True,
        },
    )
    print(f"  HTTP {st}: run_id={resp.get('run_id')!r}")
    if st != 200:
        print(f"  FAIL submit: {resp}")
        return 1
    run_id = resp["run_id"]

    print("[poll] waiting for terminal status ...")
    last = None
    deadline = time.monotonic() + 420
    while time.monotonic() < deadline:
        st, s = post(predict, args.token, {"action": "status", "run_id": run_id})
        status = s.get("status")
        if status != last:
            print(f"  status={status!r}")
            last = status
        if status in {"succeeded", "failed", "cancelled", "halted"}:
            print(f"  terminal status={status!r}")
            break
        time.sleep(5)

    print(f"\n  run_id={run_id}")
    print("  → inspect Augur for a planner-layer modelio record on this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
