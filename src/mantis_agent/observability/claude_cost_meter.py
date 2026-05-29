"""Per-source Claude cost attribution for one run.

Companion to :mod:`time_meter` but for tokens + cost instead of wall
time. Lets operators answer "where did my $X of Claude spend go?"
after a run — separated by source (brain, grounding, extraction-
single, extraction-multi, recovery, verify) and model.

Pattern: thread-local current meter, set at the runner's entry point,
read by every Claude HTTP call site, finalised to disk at run
terminal. Best-effort throughout — never blocks the run.

Output path on the production volume::

    /data/runs/<tenant_id>/<run_id>/claude_cost_by_path.json

Schema::

    {
      "run_id": "20260529_...",
      "tenant_id": "default__ab-haiku-...",
      "totals": {"input": 12345, "output": 678, "cost_usd": 0.42},
      "by_path": {
        "brain_claude": {...},
        "grounding": {...},
        "extract_multi": {...},
        ...
      }
    }
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ── Cost model — rough per-1K-token rates in USD ─────────────────────


@dataclass(frozen=True)
class ModelRates:
    """Per-1K-token USD rates. Anthropic prices their models per
    million tokens; we store per-1K for ergonomic multiplication."""

    input: float
    output: float
    cache_read: float
    cache_creation: float


# Conservative defaults — match Anthropic's public pricing as of
# 2026-05-29. When a model name doesn't match here, falls back to the
# `_UNKNOWN` rates (set to the highest-billed model so cost estimates
# err on the cautious side).
MODEL_RATES: dict[str, ModelRates] = {
    # Rates are per-TOKEN (Anthropic publishes per-million-token rates;
    # we store divided by 1e6 so estimate_cost can multiply by raw
    # token counts). Public USD rates as of 2026-05-29.
    "claude-opus-4-7": ModelRates(
        input=15.0 / 1_000_000, output=75.0 / 1_000_000,
        cache_read=1.5 / 1_000_000, cache_creation=18.75 / 1_000_000,
    ),
    "claude-sonnet-4-6": ModelRates(
        input=3.0 / 1_000_000, output=15.0 / 1_000_000,
        cache_read=0.3 / 1_000_000, cache_creation=3.75 / 1_000_000,
    ),
    "claude-haiku-4-5": ModelRates(
        input=0.8 / 1_000_000, output=4.0 / 1_000_000,
        cache_read=0.08 / 1_000_000, cache_creation=1.0 / 1_000_000,
    ),
}
_UNKNOWN_RATES = MODEL_RATES["claude-opus-4-7"]  # conservative


def rates_for(model: str) -> ModelRates:
    """Per-1K-token rates for ``model``; falls back to Opus rates on
    unknown names so cost estimates never under-report."""
    if model in MODEL_RATES:
        return MODEL_RATES[model]
    # Handle versioned names — e.g. claude-haiku-4-5-20251001 matches
    # claude-haiku-4-5; claude-sonnet-4-6-20260201 matches sonnet-4-6.
    for known in MODEL_RATES:
        if model.startswith(known) or known.startswith(model):
            return MODEL_RATES[known]
    return _UNKNOWN_RATES


def estimate_cost(
    *, model: str, input_tokens: int, output_tokens: int,
    cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
) -> float:
    """USD cost estimate for a single response. ``input_tokens``
    excludes the cache_read portion (Anthropic reports them
    separately); the helper sums them with their respective rates."""
    r = rates_for(model)
    return (
        input_tokens * r.input
        + output_tokens * r.output
        + cache_read_tokens * r.cache_read
        + cache_creation_tokens * r.cache_creation
    )


# ── Accumulator ──────────────────────────────────────────────────────


@dataclass
class PathStats:
    """Aggregated counters for one (source, model) bucket."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


class ClaudeCostMeter:
    """Single-run accumulator. Thread-safe append; finalise once.

    Source labels (free-form strings — used as JSON keys):

    - ``brain_claude`` — Claude as the executor brain (``cua_model=claude``)
    - ``grounding`` — ClaudeGrounding refining click coords
    - ``extract_multi`` — ClaudeExtractor `_call_many` (the boattrader
      detail-page extraction hot path)
    - ``extract_single`` — ClaudeExtractor `_call` (single-screenshot
      extracts; find_listings / content_control / fields)
    - ``extract_tool`` — ClaudeExtractor `_call_with_tool_schema*`
    - ``verify`` — Haiku verifier gate
    - ``verify_escalation`` — Opus escalation on Haiku verifier fail
    - ``recovery`` — agentic_recovery's tool_use call
    - ``planner`` — PlanDecomposer's one-shot decompose call
    - other — any caller passing a custom bucket
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # buckets[(source, model)] = PathStats
        self._buckets: dict[tuple[str, str], PathStats] = {}

    def record(
        self,
        *, source: str, model: str,
        input_tokens: int = 0, output_tokens: int = 0,
        cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
    ) -> None:
        if not source or not model:
            return
        cost = estimate_cost(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )
        key = (source, model)
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = PathStats()
                self._buckets[key] = bucket
            bucket.calls += 1
            bucket.input_tokens += int(input_tokens)
            bucket.output_tokens += int(output_tokens)
            bucket.cache_read_tokens += int(cache_read_tokens)
            bucket.cache_creation_tokens += int(cache_creation_tokens)
            bucket.cost_usd += cost

    def snapshot(self) -> dict[str, Any]:
        """Read-only view of the current accumulator state."""
        with self._lock:
            by_path: dict[str, dict[str, Any]] = {}
            tot_input = tot_output = tot_read = tot_creation = 0
            tot_cost = 0.0
            for (source, model), stats in self._buckets.items():
                label = source if model == "unknown" else f"{source}::{model}"
                by_path[label] = {**stats.to_dict(), "source": source, "model": model}
                tot_input += stats.input_tokens
                tot_output += stats.output_tokens
                tot_read += stats.cache_read_tokens
                tot_creation += stats.cache_creation_tokens
                tot_cost += stats.cost_usd
            return {
                "totals": {
                    "input_tokens": tot_input,
                    "output_tokens": tot_output,
                    "cache_read_tokens": tot_read,
                    "cache_creation_tokens": tot_creation,
                    "cost_usd": round(tot_cost, 6),
                    "calls": sum(b.calls for b in self._buckets.values()),
                },
                "by_path": by_path,
            }


# ── Thread-local current meter ───────────────────────────────────────


_current: contextvars.ContextVar[ClaudeCostMeter | None] = contextvars.ContextVar(
    "mantis_claude_cost_meter", default=None,
)


def set_current_meter(meter: ClaudeCostMeter | None) -> None:
    """Bind ``meter`` as the active accumulator. Call once at runner
    entry; the call sites pick it up via :func:`current_meter`."""
    _current.set(meter)


def current_meter() -> ClaudeCostMeter | None:
    return _current.get()


def record_from_response(
    *, source: str, model: str, response_json: dict[str, Any] | None,
) -> None:
    """Convenience: pull tokens from an Anthropic response and credit
    the current meter. No-op when no meter is bound or response is
    malformed."""
    meter = _current.get()
    if meter is None or not isinstance(response_json, dict):
        return
    usage = response_json.get("usage") or {}
    if not isinstance(usage, dict):
        return
    try:
        meter.record(
            source=source, model=model,
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            cache_creation_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
        )
    except (TypeError, ValueError):
        return


# ── Persistence ──────────────────────────────────────────────────────


def _output_path(*, tenant_id: str, run_id: str) -> str:
    """Production layout: /data/runs/<tenant>/<run_id>/claude_cost_by_path.json.

    Env-overridable for tests."""
    root = os.environ.get("MANTIS_RUN_ARTIFACTS_DIR", "/data/runs")
    safe_tenant = _sanitize(tenant_id) or "default"
    safe_run = _sanitize(run_id) or "unknown"
    return os.path.join(root, safe_tenant, safe_run, "claude_cost_by_path.json")


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in (s or ""))[:120]


def finalize_to_disk(
    *, meter: ClaudeCostMeter | None = None,
    run_id: str, tenant_id: str,
    extras: dict[str, Any] | None = None,
) -> str:
    """Write the meter snapshot to disk. Returns the written path, or
    empty string when nothing was written (no meter / no run_id).

    ``extras`` is merged into the top-level JSON so callers can record
    run-outcome metadata alongside the cost breakdown — leads counted,
    steps executed, halt reason, model used. Keeps the meter purely
    about Claude spend while letting cost / outcome live in one file
    operators can pull with a single ``modal volume get``.

    Best-effort: errors are logged at DEBUG and swallowed — the run
    terminal must never fail because of cost-meter I/O.
    """
    meter = meter or current_meter()
    if meter is None or not run_id:
        return ""
    snapshot = meter.snapshot()
    if snapshot["totals"]["calls"] == 0:
        return ""
    snapshot["run_id"] = run_id
    snapshot["tenant_id"] = tenant_id
    if extras:
        # extras keys win over snapshot keys when they collide. Lets
        # callers override e.g. ``run_id`` if they have a more
        # authoritative identifier than what was passed.
        snapshot.update(extras)
    path = _output_path(tenant_id=tenant_id, run_id=run_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(snapshot, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
        logger.warning(
            "  [cost-meter] wrote %s: %d calls, $%.4f total",
            path, snapshot["totals"]["calls"],
            snapshot["totals"]["cost_usd"],
        )
        return path
    except Exception as exc:  # noqa: BLE001
        logger.debug("claude_cost_meter finalize failed: %s", exc)
        return ""
