"""Live verification for INITIAL-path ``{{user_input}}`` substitution (#885-followup).

Unlike ``verify_request_user_input_roundtrip.py`` (which exercises the
pause/resume path), this submits a "natural" login plan with NO
``request_user_input`` pause step and an up-front ``user_input`` carried on
the task_suite. It asserts the deployed Modal CUA server:

  1. does NOT pause (no request_user_input step → runs straight through),
  2. substitutes ``{{user_input}}`` in the Username fill BEFORE typing —
     verified directly from the run's Augur modelio bundle: the grounding
     record for the username step must carry value=``tomsmith``, NOT the
     literal ``{{user_input}}`` (the pre-fix bug).

Usage:
    uv run python scripts/verify_initial_user_input_substitution.py \
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mantis_agent.plan_decomposer import MicroPlan, PlanDecomposer  # noqa: E402
from mantis_agent.server_utils import (  # noqa: E402
    build_micro_suite,
    micro_plan_steps_to_dicts,
)

USER_INPUT = "tomsmith"  # the-internet/herokuapp valid username
PASSWORD = "SuperSecretPassword!"  # literal valid password (not templated)

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
        "type": "fill_field",
        "intent": "Enter {{user_input}} in the Username field",
        "params": {"label": "Username", "value": "{{user_input}}"},
        "required": True,
        "section": "login",
    },
    {
        "type": "fill_field",
        "intent": "Enter the password in the Password field",
        "params": {"label": "Password", "value": PASSWORD},
        "required": True,
        "section": "login",
    },
    {
        "type": "click",
        "intent": "Click the Login button to submit the form",
        "params": {"label": "Login"},
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


def _augur_username_value(run_id: str) -> tuple[bool, str]:
    """Best-effort: read the run's Augur modelio and return whether the
    Username grounding record substituted the token. Returns (found, value)."""
    dsn = os.environ.get("AUGUR_DSN", "")
    # DSN: https://<host>/api/v1?token=<tok>&tenant=<tenant>
    import re
    m = re.match(r"(https?://[^/]+/api/v1)\?token=([^&]+)&tenant=([^&]+)", dsn)
    if not m:
        return False, "(AUGUR_DSN not parseable; skipping bundle check)"
    base, tok, tenant = m.group(1), m.group(2), m.group(3)
    hdr = {"ngrok-skip-browser-warning": "true", "Authorization": f"Bearer {tok}"}

    def aget(path: str) -> dict | None:
        try:
            r = requests.get(
                f"{base}/{path}{'&' if '?' in path else '?'}tenant={tenant}&token={tok}",
                headers=hdr, timeout=25,
            )
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    lst = aget(f"runs/{run_id}/modelio")
    if not lst:
        return False, "(no modelio bundle for run)"
    for rec in lst.get("records", []):
        rel = rec["relpath"].split("/")[-1]
        d = aget(f"runs/{run_id}/modelio/{rel}")
        if not d:
            continue
        inp = (d.get("response", {}).get("tool_calls") or [{}])[0].get("input", {})
        # The username fill carries the (post-substitution) value.
        val = inp.get("value")
        if val in (USER_INPUT, "{{user_input}}"):
            return True, str(val)
    return False, "(no username-fill grounding record found)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--endpoint",
        default="https://getmason--mantis-cua-server-api.modal.run",
    )
    ap.add_argument("--token", default=os.environ.get("MANTIS_API_TOKEN", ""))
    ap.add_argument("--profile-id", default="rui-natural")
    ap.add_argument("--workflow-id", default="rui-natural")
    args = ap.parse_args()

    if not args.token:
        print("ERROR: --token or MANTIS_API_TOKEN required", file=sys.stderr)
        return 2

    endpoint = args.endpoint.rstrip("/")
    predict = f"{endpoint}/v1/predict"

    plan = MicroPlan(domain="rui_natural_login")
    for step in PLAN_STEPS:
        plan.steps.append(PlanDecomposer._build_intent(step))
    suite = build_micro_suite(
        micro_plan_steps_to_dicts(plan.steps),
        plan.domain,
        profile_id=args.profile_id,
        workflow_id=args.workflow_id,
    )
    # The up-front value the runtime stages onto ``_staged_user_input`` on
    # the INITIAL path (no pause). The executor reads ``task_suite["user_input"]``.
    suite["user_input"] = USER_INPUT

    # ── 1. submit ──────────────────────────────────────────────────
    print(f"[submit] natural login plan (user_input={USER_INPUT!r}, no pause step)")
    st, resp = post(
        predict, args.token,
        {
            "task_suite": suite,
            "profile_id": suite["_profile_id"],
            "workflow_id": suite["_workflow_id"],
            "cua_model": "holo3",
            "max_steps": 10,
            "max_cost": 1.0,
            "max_time_minutes": 10,
            "detached": True,
        },
    )
    print(f"  HTTP {st}: run_id={resp.get('run_id')!r}")
    if st != 200:
        print(f"  FAIL submit: {resp}")
        return 1
    run_id = resp["run_id"]

    # ── 2. poll to terminal, asserting we NEVER pause ──────────────
    print("[poll] waiting for terminal status (must NOT pause) ...")
    final, last = None, None
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        st, s = post(predict, args.token, {"action": "status", "run_id": run_id})
        status = s.get("status")
        if status != last:
            print(f"  status={status!r}")
            last = status
        if status == "paused":
            print("  FAIL: natural plan paused — it has no request_user_input step")
            return 1
        if status in {
            "succeeded", "failed", "cancelled", "halted",
            "completed", "completed_with_failures",
        }:
            final = s
            break
        time.sleep(5)

    if final is None:
        print("  FAIL: no terminal status within window")
        return 1
    print(f"  terminal status={final.get('status')!r}")

    # ── 3. assert substitution from the result + Augur bundle ──────
    blob = json.dumps(final)
    literal_in_result = "{{user_input}}" in blob
    print(f"  literal '{{{{user_input}}}}' present in result payload: {literal_in_result}")

    found, val = _augur_username_value(run_id)
    print(f"  Augur username-fill value: {val} (record found: {found})")

    ok = True
    if literal_in_result:
        print("  FAIL: literal token leaked into the result payload")
        ok = False
    if found and val == "{{user_input}}":
        print("  FAIL: username field typed the LITERAL token — substitution did not fire")
        ok = False
    if found and val == USER_INPUT:
        print(f"  ✅ username field substituted to {USER_INPUT!r} on the initial path")
    if not found:
        print("  ⚠️  could not confirm via Augur bundle (check manually); "
              "result-payload check still applied")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
