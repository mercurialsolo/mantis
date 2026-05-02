"""Per-run cost accounting for MicroPlanRunner — extracted from
micro_runner.py (#115, step 3).

Tracks four counters per run:

* ``gpu_steps``       — number of brain-driven steps
* ``gpu_seconds``     — derived from ``gpu_steps * cost_config.gpu_seconds_per_step``
* ``claude_extract``  — Claude API calls for extraction / verification / grounding callthroughs
* ``claude_grounding``— Claude API calls for click coordinate grounding
* ``proxy_mb``        — estimated egress bandwidth in megabytes

Rates live in :class:`CostConfig` so deployments can override them via env
vars; this meter applies them and emits Prometheus inflight gauges.

Backward-compat note: ``MicroPlanRunner.costs`` still exists and points
at the **same dict** the meter owns. The 60+ scattered ``self.costs[...] += N``
mutation sites in the runner keep working unchanged because dict
subscript-assign is in-place. Subsequent splits can migrate those call
sites to ``meter.add_*()`` helpers; this PR keeps the diff minimal.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ..cost_config import CostConfig

if TYPE_CHECKING:
    from ..plan_decomposer import MicroIntent
    from .checkpoint import StepResult

logger = logging.getLogger(__name__)


_INITIAL_COSTS: dict[str, Any] = {
    "gpu_steps": 0,        # Total Holo3 steps
    "gpu_seconds": 0.0,    # Approx GPU time
    "claude_extract": 0,   # Claude extraction calls
    "claude_grounding": 0, # Claude grounding calls
    "proxy_mb": 0.0,       # Estimated proxy bandwidth
}


def make_initial_costs() -> dict[str, Any]:
    """Fresh copy of the initial counters dict. Same shape as the
    pre-#115 ``MicroPlanRunner.costs`` initializer.
    """
    return dict(_INITIAL_COSTS)


class CostMeter:
    """Owns cost counters + rate config + Prometheus inflight gauges.

    The :attr:`costs` dict is the canonical state — mutate it in-place via
    subscript assignment from anywhere in the runner. ``MicroPlanRunner.costs``
    is the *same* dict (alias) for backward compat with the runner's many
    scattered increment sites.
    """

    def __init__(
        self,
        cost_config: CostConfig | None = None,
        tenant_id: str = "",
    ) -> None:
        self.cost_config = cost_config or CostConfig.from_env()
        self.tenant_id = tenant_id
        self.costs: dict[str, Any] = make_initial_costs()
        self.run_start: float = time.time()

    # ── Pure cost computation ───────────────────────────────────────────

    def totals(self) -> tuple[float, float, float, float]:
        """Return (gpu, claude, proxy, total) USD for the current counters."""
        cfg = self.cost_config
        gpu_cost = cfg.gpu_cost(self.costs["gpu_seconds"])
        claude_cost = cfg.claude_cost(
            self.costs["claude_extract"] + self.costs["claude_grounding"]
        )
        proxy_cost = cfg.proxy_cost(self.costs["proxy_mb"])
        return gpu_cost, claude_cost, proxy_cost, gpu_cost + claude_cost + proxy_cost

    def elapsed_seconds(self) -> float:
        return time.time() - self.run_start

    # ── Per-step accounting ─────────────────────────────────────────────

    def record_step(self, step: "MicroIntent", step_result: "StepResult") -> None:
        """Apply a step's resource usage to the counters.

        Mirrors the pre-#115 ``MicroPlanRunner._record_step_costs`` exactly:
        GPU time scales with step count, Claude extraction is one call per
        ``claude_only`` step, grounding is one call per click step,
        navigation/scroll add proxy MB per the config rate.
        """
        cfg = self.cost_config
        if step_result.steps_used > 0:
            self.costs["gpu_steps"] += step_result.steps_used
            self.costs["gpu_seconds"] += step_result.steps_used * cfg.gpu_seconds_per_step
        if step.claude_only:
            self.costs["claude_extract"] += 1
        if step.grounding:
            self.costs["claude_grounding"] += step_result.steps_used  # ~1 grounding per click
        if step.type in ("click", "navigate", "paginate"):
            self.costs["proxy_mb"] += cfg.proxy_mb_per_nav
        elif step.type == "scroll":
            self.costs["proxy_mb"] += cfg.proxy_mb_per_scroll

    # ── Restore from checkpoint ─────────────────────────────────────────

    def restore(self, persisted: dict[str, Any] | None) -> None:
        """Merge persisted counters from a RunCheckpoint into the live dict."""
        if persisted:
            self.costs.update(persisted)

    def snapshot(self) -> dict[str, Any]:
        """Plain-dict copy suitable for ``RunCheckpoint.costs``."""
        return dict(self.costs)

    # ── Prometheus emission ─────────────────────────────────────────────

    def emit_inflight_gauges(
        self,
        gpu_cost: float,
        claude_cost: float,
        proxy_cost: float,
        total_cost: float,
    ) -> None:
        """Push running per-component cost to Prometheus (#122).

        Skipped when no tenant_id is set so the registry doesn't get a
        default-label series for local/script runs.
        """
        if not self.tenant_id:
            return
        try:
            from .. import metrics as _metrics
            for component, value in (
                ("gpu", gpu_cost),
                ("claude", claude_cost),
                ("proxy", proxy_cost),
                ("total", total_cost),
            ):
                _metrics.RUN_COST_USD_INFLIGHT.labels(
                    tenant_id=self.tenant_id, component=component
                ).set(value)
        except Exception as exc:  # noqa: BLE001 — metrics are observability, never fatal
            logger.debug("inflight cost gauge update failed: %s", exc)


__all__ = ["CostMeter", "make_initial_costs"]
