"""End-to-end verification for the Modal HTTP endpoint (#342).

Submits the luma + staff-crm plans against the deployed `/v1/predict`,
verifies parallel dispatch on distinct ``profile_id``s, and that a
same-``profile_id`` collision returns 409. Polls one run to confirm
the action=status path works.

Usage:
    uv run python scripts/verify_modal_endpoint.py \
        --endpoint https://getmason--mantis-cua-server-api.modal.run \
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

# Bring the in-repo helpers onto sys.path.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mantis_agent.plan_decomposer import MicroPlan, PlanDecomposer  # noqa: E402
from mantis_agent.server_utils import (  # noqa: E402
    build_micro_suite,
    micro_plan_steps_to_dicts,
)


def load_plan_as_suite(plan_path: Path, profile_id: str, workflow_id: str) -> dict[str, Any]:
    raw = json.loads(plan_path.read_text())
    steps = raw["steps"] if isinstance(raw, dict) else raw
    plan = MicroPlan(domain=plan_path.stem)
    for step in steps:
        plan.steps.append(PlanDecomposer._build_intent(step))
    return build_micro_suite(
        micro_plan_steps_to_dicts(plan.steps),
        plan.domain,
        profile_id=profile_id,
        workflow_id=workflow_id,
    )


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


def submit(endpoint: str, token: str, suite: dict, model: str = "holo3") -> tuple[int, dict]:
    return post(
        f"{endpoint.rstrip('/')}/v1/predict",
        token,
        {
            "task_suite": suite,
            "profile_id": suite["_profile_id"],
            "workflow_id": suite["_workflow_id"],
            "cua_model": model,
            "max_steps": 4,  # smoke — just exercise the dispatch path, not the full run
            "max_cost": 1.0,
            "max_time_minutes": 5,
            "detached": True,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default="https://getmason--mantis-cua-server-api.modal.run",
    )
    parser.add_argument("--token", default=os.environ.get("MANTIS_API_TOKEN", ""))
    parser.add_argument("--model", default="claude")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: --token or MANTIS_API_TOKEN required", file=sys.stderr)
        return 2

    print(f"endpoint: {args.endpoint}")
    print(f"model:    {args.model}")
    print()

    # ── 1. Health check ────────────────────────────────────────────
    r = requests.get(f"{args.endpoint}/v1/health", timeout=30)
    print(f"[health] HTTP {r.status_code}: {r.text}")
    assert r.status_code == 200

    # ── 2. Submit luma + staff-crm in parallel with DISTINCT profile_ids ─
    luma_suite = load_plan_as_suite(
        REPO_ROOT / "plans" / "luma-extract.json",
        profile_id="verify-luma",
        workflow_id="verify-luma-v1",
    )
    crm_suite = load_plan_as_suite(
        REPO_ROOT / "plans" / "staff-crm.json",
        profile_id="verify-staffcrm",
        workflow_id="verify-staffcrm-v1",
    )

    print("[submit] luma → profile_id=verify-luma")
    luma_status, luma_resp = submit(args.endpoint, args.token, luma_suite, args.model)
    print(f"  HTTP {luma_status}: run_id={luma_resp.get('run_id')!r}")
    print(f"  payload.profile_id={luma_resp.get('payload', {}).get('profile_id')}")
    print(f"  payload.workflow_id={luma_resp.get('payload', {}).get('workflow_id')}")

    print("[submit] staff-crm → profile_id=verify-staffcrm")
    crm_status, crm_resp = submit(args.endpoint, args.token, crm_suite, args.model)
    print(f"  HTTP {crm_status}: run_id={crm_resp.get('run_id')!r}")
    print(f"  payload.profile_id={crm_resp.get('payload', {}).get('profile_id')}")
    print(f"  payload.workflow_id={crm_resp.get('payload', {}).get('workflow_id')}")

    assert luma_status == 200, f"luma submit failed: {luma_resp}"
    assert crm_status == 200, f"crm submit failed: {crm_resp}"
    assert luma_resp["run_id"] != crm_resp["run_id"], "run_ids must be distinct"
    # Identity round-trip — server prefixes with tenant_id__.
    assert luma_resp["payload"]["profile_id"].endswith("__verify-luma")
    assert crm_resp["payload"]["profile_id"].endswith("__verify-staffcrm")

    # ── 3. Same-profile_id submission should return 409 ───────────
    print("[submit] duplicate luma profile_id → expect 409")
    dup_status, dup_resp = submit(args.endpoint, args.token, luma_suite, args.model)
    print(f"  HTTP {dup_status}: detail={dup_resp.get('detail', dup_resp)!r}")
    assert dup_status == 409, f"expected 409, got {dup_status}: {dup_resp}"
    held = luma_resp["run_id"]
    assert held in dup_resp.get("detail", ""), (
        f"409 detail should surface the held run_id={held}; got {dup_resp}"
    )

    # ── 4. Status poll on luma ─────────────────────────────────────
    print("[status] action=status on luma run")
    s_status, s_resp = post(
        f"{args.endpoint}/v1/predict",
        args.token,
        {"action": "status", "run_id": luma_resp["run_id"]},
    )
    print(f"  HTTP {s_status}: status={s_resp.get('status')!r}, modal_call_id={s_resp.get('modal_call_id', '')[:20]}...")
    assert s_status == 200
    assert s_resp["status"] in {"queued", "running", "succeeded", "failed", "cancelled"}
    assert s_resp["profile_id"] == luma_resp["payload"]["profile_id"]
    assert s_resp["workflow_id"] == luma_resp["payload"]["workflow_id"]

    # ── 5. Cancel both so the lock releases for the next run ──────
    for run_id in (luma_resp["run_id"], crm_resp["run_id"]):
        c_status, c_resp = post(
            f"{args.endpoint}/v1/predict",
            args.token,
            {"action": "cancel", "run_id": run_id},
        )
        print(f"[cancel] {run_id}: HTTP {c_status}, status={c_resp.get('status')!r}")

    # ── 6. After cancel, same profile_id should accept a new run ──
    print("[submit] luma profile_id after cancel → expect 200 (lock released)")
    time.sleep(1)  # belt-and-braces — the cancel→release is synchronous but volume commit takes a beat
    re_status, re_resp = submit(args.endpoint, args.token, luma_suite, args.model)
    print(f"  HTTP {re_status}: run_id={re_resp.get('run_id', re_resp)!r}")
    assert re_status == 200, f"post-cancel submit failed: {re_resp}"
    assert re_resp["run_id"] != luma_resp["run_id"]

    # Tidy up.
    post(
        f"{args.endpoint}/v1/predict",
        args.token,
        {"action": "cancel", "run_id": re_resp["run_id"]},
    )

    print()
    print("✅ All verifications passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
