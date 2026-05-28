"""One-shot submission for plans/boattrader_scrape_urlnav (free-text plan).

Adapted from run_staff_crm_long_no_proxy.py with two differences:

* Plan is free text → decomposed locally via PlanDecomposer before
  build_micro_suite
* Proxy ENABLED — proxy_disabled=False so the executor uses the
  PrivateProxy entrypoint configured in the Modal cua-server's
  Secret.from_dotenv (BoatTrader's Cloudflare layer blocks direct
  Modal egress)

Identity model (#572 Phase 1):

* PROFILE_ID is **stable** by default (``boattrader-urlnav-stable``)
  so the Chrome user-data-dir is reused across runs. Cookies
  (notably ``cf_clearance``) persist — the FIRST run on a fresh
  profile may hit a Cloudflare Turnstile widget; subsequent runs
  within the cookie lifetime (~30 min to several hours) reuse the
  cached clearance and skip CF entirely. Override per-run via
  ``MANTIS_PROFILE_ID`` env when you want a clean profile.
* WORKFLOW_ID is **timestamped** per run so checkpoint files don't
  collide. Each run still gets its own checkpoint, but they all
  share the same Chrome profile.

To seed CF clearance on a fresh profile, submit with
``pause_on_captcha=true`` (the default), open the viewer URL when
the run auto-pauses, solve the Turnstile widget, click Resume.
The ``cf_clearance`` cookie lands in the profile and every
subsequent run skips CF.

Usage::
    MANTIS_API_TOKEN=... uv run python scripts/run_boattrader_urlnav_with_proxy.py
    # or override the profile:
    MANTIS_PROFILE_ID=boattrader-clean uv run python scripts/run_boattrader_urlnav_with_proxy.py
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

from mantis_agent.plan_decomposer import PlanDecomposer  # noqa: E402
from mantis_agent.recipes.plan_evolution_store import (  # noqa: E402
    apply_plan_overlay,
)
from mantis_agent.server_utils import (  # noqa: E402
    build_micro_suite,
    merge_runtime,
    micro_plan_steps_to_dicts,
)


ENDPOINT = "https://getmason--mantis-cua-server-api.modal.run"
PLAN_PATH = REPO_ROOT / "plans" / "boattrader_scrape_urlnav"
# #572: stable PROFILE_ID by default so CF clearance cookies persist
# across runs. Override via env when you need a clean profile for
# debugging (e.g. testing first-run CF behavior).
PROFILE_ID = os.environ.get("MANTIS_PROFILE_ID", "boattrader-urlnav-stable")
# WORKFLOW_ID timestamped per run — each run gets its own checkpoint
# file even though they share the Chrome profile.
WORKFLOW_ID = f"boattrader-urlnav-{int(time.time())}"


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


def main() -> int:
    print(f"endpoint:    {ENDPOINT}")
    print(f"plan:        {PLAN_PATH.name} (free-text → decompose)")
    print(f"profile_id:  {PROFILE_ID}")
    print(f"workflow_id: {WORKFLOW_ID}")
    print("proxy:       ENABLED (PrivateProxy via dotenv on Modal)")
    print()

    if not PLAN_PATH.exists():
        print(f"ERROR: plan not found: {PLAN_PATH}", file=sys.stderr)
        return 2

    # 1. Health check
    h = requests.get(f"{ENDPOINT}/v1/health", timeout=30)
    print(f"[health] HTTP {h.status_code}: {h.text[:120]}")
    if h.status_code != 200:
        print("aborting — endpoint not healthy", file=sys.stderr)
        return 3

    # 2. Decompose free-text plan locally
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        # Pull from .env if not exported
        for line in (REPO_ROOT / ".env").read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY required for decompose", file=sys.stderr)
        return 4

    plan_text = PLAN_PATH.read_text()
    print(f"[decompose] {len(plan_text)} chars → PlanDecomposer")
    decomposer = PlanDecomposer(api_key=api_key, model="claude-opus-4-7")
    cache_dir = REPO_ROOT / "data" / "plan_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_template = str(cache_dir / "decomposed_{hash}.json")
    micro_plan = decomposer.decompose_text(
        plan_text, cache_path_template=cache_template,
    )
    print(f"[decompose] → {len(micro_plan.steps)} steps")

    # Plan-evolution Phase 2 (#706): apply any promoted URL rewrites
    # to the decomposed plan BEFORE building the suite. PROFILE_ID is
    # the stable scope (WORKFLOW_ID is per-run and would never
    # accumulate). When no rewrites are promoted, this is a no-op +
    # logs nothing. When rewrites apply, the warning logs print
    # `[plan-overlay] step N rewritten via <source>` lines visible in
    # Modal container output.
    _overlaid_plan, _applied = apply_plan_overlay(
        micro_plan, plan_hash=micro_plan.plan_hash, workflow_id=PROFILE_ID,
    )
    if _applied:
        print(f"[plan-overlay] applied {len(_applied)} promoted rewrites:")
        for r in _applied:
            print(
                f"  step={r.step_index} source={r.source} "
                f"({r.successful_runs} successes)"
            )
    else:
        print(
            "[plan-overlay] no promoted rewrites for "
            f"workflow_id={PROFILE_ID!r} plan_hash={micro_plan.plan_hash!r}"
        )

    # 3. Build suite with proxy ENABLED
    runtime = merge_runtime({
        "proxy_disabled": False,
        "proxy_provider": os.environ.get("PROXY_PROVIDER", "privateproxy"),
        "proxy_city": os.environ.get("PROXY_CITY", "miami"),
        "proxy_state": os.environ.get("PROXY_STATE", "florida"),
        "max_cost": 8.0,
        "max_time_minutes": 30,
    })
    suite = build_micro_suite(
        micro_plan_steps_to_dicts(micro_plan.steps),
        micro_plan.domain or PLAN_PATH.stem,
        profile_id=PROFILE_ID,
        workflow_id=WORKFLOW_ID,
        # Plan-evolution Phase 2 (#706): carry the plan_hash + scope_id
        # so the Modal executor can stamp them on the runner and the
        # mid-run recovery layer can record candidates.
        plan_hash=micro_plan.plan_hash,
        plan_evolution_scope_id=PROFILE_ID,
        **runtime,
    )

    # 4. Submit
    submit_body = {
        "task_suite": suite,
        "profile_id": suite["_profile_id"],
        "workflow_id": suite["_workflow_id"],
        "cua_model": "holo3",
        "max_steps": 80,
        "detached": True,
        **runtime,
    }
    print(f"[plan-runtime] {runtime}")
    status, resp = _post("/v1/predict", submit_body)
    print(f"[submit] HTTP {status}")
    print(f"  run_id={resp.get('run_id')!r}")
    if status != 200:
        print(f"  body={json.dumps(resp, indent=2)[:600]}", file=sys.stderr)
        return 5

    run_id = resp["run_id"]

    # 5. Poll until terminal (treat halted as terminal too)
    terminal = {"succeeded", "failed", "cancelled", "halted"}
    last_status = ""
    started = time.time()
    poll_n = 0
    while True:
        poll_n += 1
        if time.time() - started > 35 * 60:
            print("TIMEOUT: 35 minutes without terminal status", file=sys.stderr)
            return 6
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
