"""Dry-run response builder (#785 DX-3 follow-up).

When a client submits `dry_run: true` on `POST /v1/predict`, the server
runs the plan-resolution pipeline (free-text decomposition, micro-suite
construction, validation) but **does not** acquire the Chrome lock or
spawn an executor. The response carries:

- `dry_run: true` — operator-visible signal that no work was started
- the resolved `task_suite` (exactly what the executor would have seen)
- a step summary keyed by type, plus the inferred `compute_backend`
- a cost estimate keyed off the resolved `cua_model` + step count

Cost estimates are rough (use the median observed during the #785
verification chain) — they tell a dev the order of magnitude before
they commit to a real run, not a contractual quote.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any


# Median per-call costs observed across the #785 verification chain.
# Update these when the pricing table or model selection changes
# materially — the goal is to give a dev an order-of-magnitude estimate
# (\"this run is ~$0.40, not $40\") not a contractual quote.
_PER_CUA_MODEL_RATES = {
    # GPU-tier executors. Per-step costs split across GPU + Claude calls.
    "holo3":     {"gpu_per_step_usd": 0.004, "claude_per_extract_usd": 0.02},
    "fara":      {"gpu_per_step_usd": 0.006, "claude_per_extract_usd": 0.02},
    "gemma4-cua":{"gpu_per_step_usd": 0.003, "claude_per_extract_usd": 0.02},
    "evocua-8b": {"gpu_per_step_usd": 0.004, "claude_per_extract_usd": 0.02},
    "evocua-32b":{"gpu_per_step_usd": 0.008, "claude_per_extract_usd": 0.02},
    "opencua-32b":{"gpu_per_step_usd": 0.008, "claude_per_extract_usd": 0.02},
    "opencua-72b":{"gpu_per_step_usd": 0.015, "claude_per_extract_usd": 0.02},
    # API-tier — no GPU, Claude does both grounding + extract.
    "claude":    {"gpu_per_step_usd": 0.0,   "claude_per_extract_usd": 0.03},
}

# Step types that trigger a Claude call (extract_data, extract_url, gates).
_CLAUDE_STEP_TYPES = frozenset({"extract_data", "extract_url"})


def _classify_compute_backend(suite: dict[str, Any]) -> str:
    """Resolve the compute backend the plan would run on.

    Plan-level `runtime.compute_backend` > submission-level field >
    global default. Mirrors `compute_backend_resolver.resolve_compute_backend`
    semantics; duplicated here to keep this module dependency-free.
    """
    runtime = suite.get("runtime")
    if isinstance(runtime, dict):
        v = runtime.get("compute_backend")
        if isinstance(v, str) and v.strip().lower() in (
            "computer_plane",
            "browser_use_plane",
        ):
            return v.strip().lower()
    return "computer_plane"


def _step_summary(steps: list[Any]) -> dict[str, Any]:
    """Counts by step type + a flag for any inline `extract` block."""
    counts: Counter[str] = Counter()
    extract_blocks = 0
    gate_count = 0
    required_count = 0
    for s in steps:
        if not isinstance(s, dict):
            continue
        counts[str(s.get("type", "unknown"))] += 1
        if isinstance(s.get("extract"), dict) and s["extract"].get("fields"):
            extract_blocks += 1
        if s.get("gate"):
            gate_count += 1
        if s.get("required"):
            required_count += 1
    return {
        "step_count": sum(counts.values()),
        "by_type": dict(counts),
        "with_inline_extract_schema": extract_blocks,
        "gate_steps": gate_count,
        "required_steps": required_count,
    }


def _estimate_cost(
    cua_model: str,
    step_count: int,
    extract_step_count: int,
    *,
    used_decomposer: bool,
) -> dict[str, Any]:
    """Rough per-run cost estimate. Telemetric only — real costs vary
    by site complexity, retry rate, and brain-budget consumption."""
    rates = _PER_CUA_MODEL_RATES.get(
        cua_model.strip().lower(),
        _PER_CUA_MODEL_RATES["holo3"],
    )
    gpu_usd = round(rates["gpu_per_step_usd"] * step_count, 4)
    claude_usd = round(
        rates["claude_per_extract_usd"] * max(extract_step_count, 1)
        + (0.015 if used_decomposer else 0.0),
        4,
    )
    # Proxy + storage are ~constant in the small range. Skip line-item.
    other_usd = 0.02
    total = round(gpu_usd + claude_usd + other_usd, 4)
    return {
        "cua_model": cua_model,
        "gpu_usd": gpu_usd,
        "claude_usd": claude_usd,
        "other_usd": other_usd,
        "estimated_total_usd": total,
        "note": (
            "Rough order-of-magnitude estimate. Real costs vary with "
            "retry rate, site complexity, and brain-budget consumption. "
            "Verify against the run's `summary.cost_breakdown` after a "
            "real submission."
        ),
    }


def build_dry_run_response(
    task_file_contents: str,
    payload: dict[str, Any],
    tenant_id: str = "",
) -> dict[str, Any]:
    """Build the JSON the /v1/predict handler returns when `dry_run: true`.

    `task_file_contents` is the canonical task_suite JSON the executor
    would have received — already through PlanDecomposer (if the plan
    was free-text) and `build_micro_suite` normalization.
    `payload` is the prepared predict payload (after `prepare_predict_payload`).
    `tenant_id` is echoed in the response for operator-side log tracing.
    """
    try:
        suite = json.loads(task_file_contents)
    except json.JSONDecodeError as exc:
        return {
            "dry_run": True,
            "error": f"task_suite is not valid JSON: {exc}",
            "tenant_id": tenant_id,
        }

    steps = suite.get("_micro_plan") or []
    if not isinstance(steps, list):
        steps = []
    summary = _step_summary(steps)

    cua_model = (payload.get("cua_model") or payload.get("model") or "holo3")
    extract_step_count = (
        summary["by_type"].get("extract_data", 0)
        + summary["by_type"].get("extract_url", 0)
    )
    used_decomposer = bool(payload.get("plan_text")) or bool(payload.get("micro_path"))

    estimate = _estimate_cost(
        cua_model,
        summary["step_count"],
        extract_step_count,
        used_decomposer=used_decomposer,
    )

    return {
        "dry_run": True,
        "tenant_id": tenant_id,
        "profile_id": payload.get("profile_id"),
        "workflow_id": payload.get("workflow_id"),
        "cua_model": cua_model,
        "compute_backend": _classify_compute_backend(suite),
        "task_suite": suite,
        "plan_summary": summary,
        "cost_estimate": estimate,
        "next_step": (
            "Resubmit the same body with `dry_run: false` (or drop the "
            "field — it defaults to false) to actually run the plan."
        ),
    }


__all__ = ["build_dry_run_response"]
