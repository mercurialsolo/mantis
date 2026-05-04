"""End-of-run reporter — final cost summary, lead counts, completion block.

Phase 1 of EPIC #161 (refactor MicroPlanRunner into composable modules).
Extracts the end-of-run print block, the ``_final_costs`` dict assembly,
and the per-step progress log line out of ``MicroPlanRunner.run()`` /
``_log_progress`` so they can be unit-tested without instantiating a
runner. Pure functions: input is the data the runner already has at end
of plan, output is a dict and a side-effecting ``print``.

Behavior is preserved verbatim from the pre-extraction code path. The
runner now delegates to ``RunReporter`` in two places:

- ``_log_progress`` (after every step) — one-line cost+lead status
- end of ``run()`` — the multi-line MICRO-PLAN COMPLETE block + the
  ``_final_costs`` dict that callers (build_micro_result, etc.) read
  off the runner

Kept as a thin static helper rather than a stateful class because every
output is a pure function of (results + cost totals). When the executor
extraction in Phase 3 lands, the runner shim can pass the reporter into
``PlanExecutor`` and remove the methods entirely.
"""

from __future__ import annotations

import logging
from typing import Any

from .listing_dedup import ListingDedup

logger = logging.getLogger(__name__)


class RunReporter:
    """Pure helpers that turn a finished plan + cost totals into:

    1. The per-step progress print (cheap, called many times)
    2. The end-of-run summary print (called once)
    3. The ``_final_costs`` dict that downstream consumers read

    No runner instance required — every input is passed in. The runner
    keeps ``_final_costs`` as an attribute for backward-compat with
    callers that read it directly (e.g. result-builders); this class
    builds the dict and the runner assigns it.
    """

    @staticmethod
    def step_progress_line(
        step_index: int,
        success: bool,
        results: list[Any],
        gpu_cost: float,
        claude_cost: float,
        proxy_cost: float,
        total_cost: float,
        elapsed_seconds: float,
    ) -> str:
        """Build the single-line progress message printed after each step.

        Format matches the pre-refactor ``MicroPlanRunner._log_progress``
        verbatim — same field order, same number widths, same minute
        rounding. Tests in test_micro_runner_hooks.py compare this line
        format.
        """
        unique_leads, phone_leads = ListingDedup.lead_counts(results)
        cost_per_lead = total_cost / max(unique_leads, 1)
        cost_per_phone_lead = total_cost / max(phone_leads, 1)
        return (
            f"  [{step_index:2d}] {'OK' if success else 'FAIL'} "
            f"| {unique_leads} leads ({phone_leads} phone) | ${total_cost:.2f} total "
            f"(${cost_per_lead:.2f}/lead, ${cost_per_phone_lead:.2f}/phone lead) | "
            f"GPU ${gpu_cost:.2f} Claude ${claude_cost:.2f} Proxy ${proxy_cost:.2f} | "
            f"{elapsed_seconds/60:.0f}m"
        )

    @staticmethod
    def final_summary_lines(
        results: list[Any],
        gpu_cost: float,
        claude_cost: float,
        proxy_cost: float,
        total_cost: float,
        elapsed_seconds: float,
        gpu_steps: int,
        claude_extract_calls: int,
        claude_grounding_calls: int,
        proxy_mb: float,
    ) -> list[str]:
        """Build the multi-line MICRO-PLAN COMPLETE block.

        Returned as a list of strings (one per print call) so callers can
        log instead of print, or capture for tests, without re-deriving
        the format.
        """
        viable_count, phone_leads = ListingDedup.lead_counts(results)
        return [
            "=" * 60,
            "MICRO-PLAN COMPLETE",
            f"  Time:     {elapsed_seconds/60:.0f}m",
            f"  Steps:    {len(results)}",
            f"  Leads:    {viable_count}",
            f"  Phone:    {phone_leads}",
            (
                f"  Cost:     ${total_cost:.2f} total "
                f"(${total_cost/max(viable_count,1):.2f}/lead, "
                f"${total_cost/max(phone_leads,1):.2f}/phone lead)"
            ),
            f"    GPU:    ${gpu_cost:.2f} ({gpu_steps} steps)",
            f"    Claude: ${claude_cost:.2f} "
            f"({claude_extract_calls} extract + {claude_grounding_calls} grounding)",
            f"    Proxy:  ${proxy_cost:.2f} ({proxy_mb:.0f} MB)",
            "=" * 60,
        ]

    @staticmethod
    def final_costs_dict(
        results: list[Any],
        gpu_cost: float,
        claude_cost: float,
        proxy_cost: float,
        total_cost: float,
        final_status: str,
        checkpoint_path: str,
    ) -> dict[str, Any]:
        """Assemble the ``_final_costs`` dict that downstream consumers read.

        Key ordering and rounding match the pre-refactor literal in
        ``MicroPlanRunner.run()``. Adding a new field is a change visible
        to result-builders, so don't reshape this without updating
        ``server_utils.build_micro_result`` and the integration tests in
        the ``baseten_server`` package.
        """
        viable_count, phone_leads = ListingDedup.lead_counts(results)
        return {
            "total": round(total_cost, 3),
            "gpu": round(gpu_cost, 3),
            "claude": round(claude_cost, 3),
            "proxy": round(proxy_cost, 3),
            "leads": viable_count,
            "leads_with_phone": phone_leads,
            "per_lead": round(total_cost / max(viable_count, 1), 3),
            "per_phone_lead": round(total_cost / max(phone_leads, 1), 3),
            "status": final_status,
            "checkpoint_path": checkpoint_path,
        }
