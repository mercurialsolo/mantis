"""/v1/cua capability re-audit — category matrix against the mantis-linkedin sim env.

The original cua-issues audit (R1-R3) ran against REAL LinkedIn/Reddit, where
anti-bot from Modal IPs dominates the signal and where the audited builds
predated the #948/#951/#952 fixes (dead brain + type-stall). This harness gets
a clean capability read on the CURRENT build by exercising the same CATEGORIES
the audit found broken — read / search-type / contenteditable-write /
multi-step — against the deterministic sim env (no anti-bot, always repaints).

It is deliberately NON-DESTRUCTIVE: the write task types into a composer but
NEVER sends; nothing logs in, submits a form, connects, or posts.

Wire status alone is misleading here (the sim env's search-results page is a
stub, so a search that TYPES PERFECTLY still `halts`). So this only records
submit + terminal status per task; the real per-category verdict — did the
brain emit a verified `type_text` / complete the click chain — comes from
``scan_cua_audit.py`` reading the trajectories by the session tag printed below.

Usage::

    MANTIS_API_TOKEN=... uv run python scripts/run_cua_audit.py
    # then: uv run modal run scripts/scan_cua_audit.py --session <SESSION printed above>
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from typing import Any

ENDPOINT = "https://getmason--mantis-cua-server-api.modal.run"
SIM = os.environ.get(
    "MANTIS_LINKEDIN_SIM_URL",
    "https://getmason--mantis-sim-env-mantis-linkedin-web.modal.run",
).rstrip("/")

# The category matrix — one task per audit category. start_url is sim-env-relative.
CATEGORIES: list[dict[str, Any]] = [
    {
        "id": "read_feed",
        "category": "read (first-frame)",
        "intent": (
            "Read the author name and the text of the single top-most post on "
            "the LinkedIn feed. Report exactly what you read. Read-only."
        ),
        "start": "/feed/",
        "max_steps": 8,
    },
    {
        "id": "read_profile",
        "category": "read (profile, needs render)",
        "intent": (
            "Read this person's name, headline, and current company from their "
            "profile page. Report all three. Read-only — do not click Connect."
        ),
        "start": "/in/aarav-yu-f41b/",
        "max_steps": 10,
    },
    {
        "id": "search_type",
        "category": "search / type-into-input (#3)",
        "intent": (
            "Click the search box at the very top of the page and type the text "
            "'machine learning'. Report that you typed it. Read-only — do not "
            "submit or navigate anywhere else."
        ),
        "start": "/feed/",
        "max_steps": 12,
    },
    {
        "id": "write_contenteditable",
        "category": "write / contenteditable (#4, W06)",
        "intent": (
            "In LinkedIn messaging, open the first conversation, click the "
            "message composer text box at the bottom, and type the text "
            "'Thanks for connecting'. Report that you typed it. CRITICAL: do "
            "NOT click Send and do NOT submit — only type into the box."
        ),
        "start": "/messaging/",
        "max_steps": 14,
    },
    {
        "id": "multistep_nav",
        "category": "multi-step click+nav (#action)",
        "intent": (
            "Starting on the feed, click the name of the author of the top post "
            "to open their profile, then read and report their headline. "
            "Read-only."
        ),
        "start": "/feed/",
        "max_steps": 14,
    },
]

TERMINAL = {"succeeded", "failed", "cancelled", "halted", "completed", "completed_with_failures"}


def _token() -> str:
    tok = os.environ.get("MANTIS_API_TOKEN", "")
    if not tok:
        print("ERROR: MANTIS_API_TOKEN required", file=sys.stderr)
        sys.exit(2)
    return tok


def _post(body: dict[str, Any]) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                f"{ENDPOINT}/v1/predict", data=json.dumps(body).encode(),
                headers={"X-Mantis-Token": _token(), "Content-Type": "application/json"},
            )
            return json.load(urllib.request.urlopen(req, timeout=90))
        except urllib.error.HTTPError as exc:
            return {"_http_error": exc.code, "_body": exc.read().decode()[:200]}
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _run_one(session: str, task: dict[str, Any]) -> dict[str, Any]:
    ts = int(time.time() * 1000)
    body = {
        "task_suite": {
            "session_name": session,
            "base_url": SIM,
            "tasks": [{
                "task_id": task["id"],
                "intent": task["intent"],
                "start_url": f"{SIM}{task['start']}",
            }],
            "_proxy_disabled": True,
            "_plan_name": session,
        },
        "profile_id": f"{session}-{task['id']}-{ts}",
        "workflow_id": f"{session}-{task['id']}-{ts}",
        "cua_model": "claude",
        "max_steps": task["max_steps"],
    }
    resp = _post(body)
    rid = resp.get("run_id")
    if not rid:
        return {"id": task["id"], "status": "SUBMIT_FAILED", "detail": resp, "run_id": None}
    t0 = time.time()
    status = "?"
    while time.time() - t0 < 6 * 60:
        s = _post({"action": "status", "run_id": rid})
        status = s.get("status", "?")
        if status in TERMINAL:
            return {
                "id": task["id"], "status": status,
                "terminal_status": s.get("terminal_status"),
                "run_id": rid, "elapsed": int(time.time() - t0),
            }
        time.sleep(8)
    return {"id": task["id"], "status": "TIMEOUT", "run_id": rid}


def main() -> int:
    session = f"cua_audit_{int(time.time())}"
    print(f"[audit] session={session}  target={SIM}")
    print(f"[audit] running {len(CATEGORIES)} category tasks...\n")

    # warm the endpoint once
    try:
        urllib.request.urlopen(f"{ENDPOINT}/v1/health", timeout=60)
    except Exception:
        pass

    results = []
    for task in CATEGORIES:
        print(f"  → {task['id']} ({task['category']}) ...", flush=True)
        r = _run_one(session, task)
        results.append({**task, **r})
        print(f"    {r['status']} / {r.get('terminal_status')}  run_id={r.get('run_id')}  ({r.get('elapsed','?')}s)")

    print("\n=== STATUS SCORECARD (wire terminal status) ===")
    for r in results:
        print(f"  {r['category']:<34} {r['status']:<10} run={r.get('run_id')}")
    print(
        f"\nNOTE: wire status is NOT the capability verdict (the sim-env search/"
        f"results pages are stubs, so a perfect type can still 'halt').\n"
        f"Run the trajectory scan for the real per-category verdict:\n"
        f"  uv run modal run scripts/scan_cua_audit.py --session {session}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
