"""Extract a single product's pricing from a marketing page.

Demonstrates the `dry_run` pattern (PR #813): preview the resolved
plan + cost estimate BEFORE submitting for real. Useful when iterating
on plan_text — you can verify the decomposer produced the steps you
expected without burning $ on a Modal cold-start.

This is the single-item shape: the plan asks for ONE row (not a list),
so no `max_items` cap on the extract block and no loop step.

Run:
    export MANTIS_API_ENDPOINT="https://your-deployment.modal.run"
    export MANTIS_API_TOKEN="mantis_..."
    python examples/pricing_page.py

Cost: ~$0.08 / real run (one navigate + one extract, no loop).
The dry-run preview costs ~$0.02 (one decomposer call).
"""

from __future__ import annotations

import json
import os
import sys
import time

from mantis_agent.client import MantisClient, PredictRequest


TARGET_URL = "https://modal.com/pricing"
PLAN_TEXT = (
    f"Go to {TARGET_URL} and extract the pricing details for the entry-level "
    "plan. Return plan_name, monthly_price_usd, included_compute, included_storage, "
    "and notable_limit_note. If the page lists multiple tiers, return only the "
    "cheapest paid tier (skip free tiers)."
)


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

    # ── Stage 1: dry-run preview. No executor spawned, no profile lock,
    # no Modal cold-start. We just want to see what the decomposer
    # produced + the estimated cost before committing.
    print("[1/4] Submitting dry_run to preview the resolved plan + cost...")
    preview = client.predict(
        PredictRequest(
            plan_text=PLAN_TEXT,
            cua_model="holo3",
            profile_id="pricing-page-prod",
            workflow_id=f"pricing-{int(time.time())}",
            dry_run=True,
            max_cost=0.30,
            max_time_minutes=4,
        )
    )

    # Dry-run returns the resolved task_suite + plan_summary + cost_estimate
    # directly (NOT a run_id + queued envelope).
    preview_dict = (
        preview if isinstance(preview, dict) else preview.model_dump()
    )
    summary = preview_dict.get("plan_summary", {})
    estimate = preview_dict.get("cost_estimate", {})

    print(f"  decomposed into {summary.get('step_count', '?')} steps:")
    for step_type, n in (summary.get("by_type") or {}).items():
        print(f"    {step_type}: {n}")
    if summary.get("with_inline_extract_schema"):
        print(
            f"  {summary['with_inline_extract_schema']} step(s) carry an "
            "inline extract block (decomposer auto-emitted per PR #801)"
        )
    print(f"  estimated cost: ${estimate.get('estimated_total_usd', '?')}")
    print(f"  compute backend: {preview_dict.get('compute_backend', '?')}")

    # Inspect the task_suite Claude produced. Comment out the dump
    # once you're confident in the shape.
    print()
    print("  resolved task_suite._micro_plan:")
    micro_plan = (preview_dict.get("task_suite", {}) or {}).get("_micro_plan", [])
    for i, step in enumerate(micro_plan):
        print(f"    [{i}] {step.get('type', '?'):15s} — {step.get('intent', '')[:80]}")
        if step.get("extract"):
            fields = ", ".join(
                f["name"] for f in step["extract"].get("fields", [])
            )
            print(f"         extract: {fields}")
    print()

    # ── Stage 2: confirm + submit for real
    if "--dry-only" in sys.argv:
        print("dry-only mode (use without --dry-only to actually run)")
        return 0

    print("[2/4] Looks reasonable — submitting for real...")
    handle = client.predict(
        PredictRequest(
            plan_text=PLAN_TEXT,
            cua_model="holo3",
            profile_id="pricing-page-prod",
            workflow_id=f"pricing-{int(time.time())}",
            max_cost=0.30,
            max_time_minutes=4,
        )
    )
    print(f"  run_id={handle.run_id!r}")

    print("[3/4] Polling for terminal state...")
    final = client.wait_for_completion(handle.run_id)
    print(f"  status={final.status!r}")

    print("[4/4] Fetching the extracted row...")
    result = client.result(handle.run_id)
    rows = (result or {}).get("rows") or []
    if not rows:
        print("  no rows returned — check action=logs for the per-step trace")
        print(f"  full result envelope: {result!r}")
        return 2

    print("  extracted:")
    print(json.dumps(rows[0], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
