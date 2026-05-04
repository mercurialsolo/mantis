"""End-to-end example: scrape an events page → JSON + CSV.

Usage:
    export MANTIS_ENDPOINT="https://getmason--mantis-server-api.modal.run"
    export MANTIS_API_TOKEN="<your-token>"
    python examples/extract_events.py "https://lu.ma/discover"

Pattern:
1. Submit a custom MICRO plan that does navigate → gate → extract_data.
   The extract_data step asks Claude to return JSON; we parse the raw
   response from result["steps"][i]["raw_response"] (or "response").
2. Poll until terminal.
3. Read the result, pull out the JSON Claude returned, write events.csv
   + events.json.

This avoids the marketplace-shape decomposer that plan_text uses, and
handles the lead-schema extraction path that defaults to year/make/model.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from typing import Any

import requests


ENDPOINT = os.environ["MANTIS_ENDPOINT"].rstrip("/")
TOKEN = os.environ["MANTIS_API_TOKEN"]
HEADERS = {"X-Mantis-Token": TOKEN, "Content-Type": "application/json"}


def submit(target_url: str) -> str:
    body = {
        "detached": True,
        "task_suite": {
            "session_name": "events_extract",
            "_micro_plan": [
                {
                    "intent": f"Navigate to {target_url}",
                    "type": "navigate",
                    "section": "setup",
                    "required": True,
                },
                {
                    "intent": "Verify the events page has loaded with event cards visible (not a Cloudflare challenge or error page)",
                    "type": "extract_data",
                    "claude_only": True,
                    "section": "setup",
                    "gate": True,
                    "verify": "page shows event listings with titles, dates, and locations",
                },
                {
                    "intent": (
                        "Extract the title, date_time, location, and url of the "
                        "first 10 events visible on the page. Return ONLY a JSON "
                        'array with fields: title (str), date_time (str, '
                        'human-readable), location (str), url (str). No prose, '
                        "no markdown fence — just the raw JSON array."
                    ),
                    "type": "extract_data",
                    "claude_only": True,
                    "section": "extraction",
                },
            ],
        },
        "max_cost": 1.0,
        "max_time_minutes": 6,
        "proxy_disabled": True,
    }
    r = requests.post(f"{ENDPOINT}/v1/predict", json=body, headers=HEADERS, timeout=180)
    r.raise_for_status()
    return r.json()["run_id"]


def poll(run_id: str, *, deadline_s: int = 480) -> str:
    start = time.time()
    while time.time() - start < deadline_s:
        r = requests.post(
            f"{ENDPOINT}/v1/predict",
            json={"action": "status", "run_id": run_id},
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        status = r.json().get("status", "unknown")
        print(f"  {time.strftime('%H:%M:%S')}  status={status}")
        if status in ("succeeded", "failed", "cancelled"):
            return status
        time.sleep(15)
    return "timeout"


def fetch_result(run_id: str) -> dict[str, Any]:
    r = requests.post(
        f"{ENDPOINT}/v1/predict",
        json={"action": "result", "run_id": run_id},
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def parse_events(result: dict[str, Any]) -> list[dict[str, str]]:
    """Walk result["steps"] for the last extract_data step and pull the JSON."""
    steps = result.get("steps") or result.get("per_step") or []
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        if step.get("type") != "extract_data":
            continue
        raw = (
            step.get("raw_response")
            or step.get("response")
            or step.get("extracted_text")
            or ""
        )
        if not raw:
            continue
        # Strip markdown fence if Claude added one despite instructions.
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        body = m.group(1) if m else raw
        # Also tolerate a leading text preamble before the JSON array.
        m = re.search(r"\[\s*\{.*\}\s*\]", body, re.DOTALL)
        body = m.group(0) if m else body
        try:
            data = json.loads(body)
            if isinstance(data, list):
                return [e for e in data if isinstance(e, dict)]
        except json.JSONDecodeError:
            continue
    return []


def write_outputs(events: list[dict[str, str]], stem: str) -> None:
    json_path = f"{stem}.json"
    csv_path = f"{stem}.csv"
    with open(json_path, "w") as f:
        json.dump(events, f, indent=2)
    if events:
        fields = ["title", "date_time", "location", "url"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for e in events:
                w.writerow({k: e.get(k, "") for k in fields})
    print(f"  wrote {json_path}  ({len(events)} events)")
    print(f"  wrote {csv_path}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    target = argv[1]
    stem = argv[2] if len(argv) >= 3 else "events"

    print(f"Submitting extraction plan for {target}")
    run_id = submit(target)
    print(f"  run_id={run_id}")

    print("Polling…")
    status = poll(run_id)
    print(f"  terminal status: {status}")

    result = fetch_result(run_id)
    events = parse_events(result)
    write_outputs(events, stem)

    summary = result.get("summary") or {}
    if not summary:
        # On detached runs, status response carries the summary; result has
        # the raw flat shape. Costs are at the top level.
        costs = result.get("costs") or {}
        print(f"  cost: ${costs.get('total', 0):.3f}")
    else:
        print(f"  summary: {json.dumps(summary, indent=2)[:400]}")
    return 0 if events else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
