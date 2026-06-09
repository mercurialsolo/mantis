"""Extract open issues from a GitHub repo.

Demonstrates the explicit-`micro` path: you author the step list by
hand, declare an inline `extract` block (PR #798), and opt into the
Browser-Use Plane (PR #793) for DOM-aware reads. Use this shape when
you need exact control over the plan structure — e.g. specifying gate
steps, retry semantics, or per-step hints.

Run:
    export MANTIS_API_ENDPOINT="https://your-deployment.modal.run"
    export MANTIS_API_TOKEN="mantis_..."
    python examples/github_issues.py

Cost: ~$0.30 / run.
"""

from __future__ import annotations

import os
import sys
import time

from mantis_agent.client import MantisClient, PredictRequest


REPO = "mercurialsolo/mantis"        # change to your target repo
ISSUE_COUNT = 5


def build_plan() -> dict:
    """Construct the task_suite body directly. Each step dict matches
    the MicroIntent shape; the inline `extract` block declares the
    rows the runtime should return."""
    return {
        "session_name": "github_issues",
        # Plan-level runtime declares the compute plane. Survives across
        # submissions — operators can rerun this body and the plane choice
        # comes with it.
        "runtime": {
            "compute_backend": "browser_use_plane",
            "max_cost": 0.50,
            "max_time_minutes": 6,
        },
        "_micro_plan": [
            {
                "intent": f"Navigate to https://github.com/{REPO}/issues?q=is%3Aissue+is%3Aopen",
                "type": "navigate",
                "params": {
                    "url": f"https://github.com/{REPO}/issues?q=is%3Aissue+is%3Aopen",
                    "wait_after_load_seconds": 4,
                },
                "section": "setup",
                "required": True,
                "budget": 4,
            },
            {
                "intent": (
                    f"Extract the top {ISSUE_COUNT} open issues from the issue "
                    "list. Read only what's visible on the list page; do not "
                    "click into individual issues. For each issue return "
                    "issue_number (integer), title, author, label_summary "
                    "(comma-separated label names if any are visible), "
                    "comment_count (integer), opened_age (e.g. 'opened 3 days ago')."
                ),
                "type": "extract_data",
                "params": {"claude_only": True},
                "section": "extraction",
                "required": False,
                "budget": 0,
                "claude_only": True,
                "hints": {"layout": "listings"},
                # Inline extract block — validator enforces THIS contract
                # per-step (PR #798), no recipe needed.
                "extract": {
                    "schema_name": "github_issues",
                    "entity_name": "issue",
                    "fields": [
                        {"name": "issue_number",   "type": "int", "required": True},
                        {"name": "title",          "type": "str", "required": True},
                        {"name": "author",         "type": "str", "required": False},
                        {"name": "label_summary",  "type": "str", "required": False},
                        {"name": "comment_count",  "type": "int", "required": False},
                        {"name": "opened_age",     "type": "str", "required": False},
                    ],
                    "max_items": ISSUE_COUNT,
                },
            },
        ],
    }


def main() -> int:
    if not os.environ.get("MANTIS_API_TOKEN") or not os.environ.get(
        "MANTIS_API_ENDPOINT"
    ):
        print(
            "set MANTIS_API_TOKEN and MANTIS_API_ENDPOINT in the environment",
            file=sys.stderr,
        )
        return 1

    client = MantisClient.from_env()

    print(f"[1/3] Submitting explicit micro plan against {REPO}...")
    handle = client.predict(
        PredictRequest(
            task_suite=build_plan(),
            cua_model="holo3",
            profile_id=f"gh-issues-{REPO.replace('/', '-')}",
            workflow_id=f"gh-issues-{int(time.time())}",
            max_cost=0.50,
            max_time_minutes=6,
        )
    )
    print(f"  run_id={handle.run_id!r}")

    print("[2/3] Polling for terminal state...")
    final = client.wait_for_completion(handle.run_id)
    print(f"  status={final.status!r}")

    print("[3/3] Fetching extracted rows...")
    result = client.result(handle.run_id)
    rows = (result or {}).get("rows") or []
    if not rows:
        print("  no rows returned — check action=logs for the per-step trace")
        print(f"  full result envelope: {result!r}")
        return 2

    print(f"  got {len(rows)} issues:")
    for row in rows:
        num = row.get("issue_number", "?")
        title = (row.get("title") or "")[:60]
        author = row.get("author", "?")
        comments = row.get("comment_count", 0)
        age = row.get("opened_age", "")
        print(f"    #{num:>4} {title!r:64} ({author}, {comments} comments, {age})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
