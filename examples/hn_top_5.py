"""Extract the top 5 Hacker News stories.

The simplest dev path post-#785: write the goal in prose, let the
decomposer (PR #801) emit the inline `extract` block automatically.
Browser-Use Plane (PR #793) handles dense link disambiguation that
pure-vision Holo3 struggles with on the HN row layout.

Run:
    export MANTIS_API_ENDPOINT="https://your-deployment.modal.run"
    export MANTIS_API_TOKEN="mantis_..."
    python examples/hn_top_5.py

Cost: ~$0.20 / run (decomposer ~$0.02 + Holo3 GPU + Claude extract).
"""

from __future__ import annotations

import os
import sys
import time

from mantis_agent.client import MantisClient, PredictRequest


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

    plan = (
        "Go to https://news.ycombinator.com/ and extract the top 5 stories. "
        "For each story return rank, title, story_url (the destination of the "
        "title link, only if it can be read without clicking), points, author, "
        "age (e.g. '2 hours ago'), and comments_count. Stay on the front page; "
        "do not click any links. If a story's URL isn't readable inline, leave "
        "story_url empty."
    )

    print("[1/3] Submitting plan_text — decomposer auto-emits the extract block...")
    handle = client.predict(
        PredictRequest(
            plan_text=plan,
            cua_model="holo3",
            compute_backend="browser_use_plane",   # DOM-aware for the dense list
            profile_id="hn-top5",
            workflow_id=f"hn-top5-{int(time.time())}",
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

    print(f"  got {len(rows)} rows:")
    for row in rows:
        rank = row.get("rank", "?")
        title = (row.get("title") or "")[:60]
        points = row.get("points", "?")
        author = row.get("author", "?")
        url = (row.get("story_url") or "")[:50]
        print(f"    {rank:>3}. {title!r:64} ({points} pts, {author}, {url})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
