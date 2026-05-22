"""Per-run cost accounting for MicroPlanRunner — extracted from
micro_runner.py (#115, step 3).

Tracks four counters per run:

* ``gpu_steps``       — number of brain-driven steps
* ``gpu_seconds``     — actual brain-inference wall time from
  :class:`~.time_meter.TimeMeter`'s ``think`` bucket when a meter
  is wired (#351). Falls back to the legacy synthetic
  ``gpu_steps * cost_config.gpu_seconds_per_step`` when no meter
  is available (tests, legacy hosts).
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

import contextlib
import contextvars
import logging
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ..cost_config import CostConfig

if TYPE_CHECKING:
    from ..plan_decomposer import MicroIntent
    from .checkpoint import StepResult
    from .time_meter import TimeMeter

logger = logging.getLogger(__name__)


_INITIAL_COSTS: dict[str, Any] = {
    "gpu_steps": 0,        # Total Holo3 steps
    "gpu_seconds": 0.0,    # Approx GPU time
    "claude_extract": 0,   # Claude extraction calls (legacy call counter)
    "claude_grounding": 0, # Claude grounding calls  (legacy call counter)
    # Per-token usage from the Anthropic API responses (#514). These
    # are the canonical Claude billing surface — call-count counters
    # above stay for compatibility but stop being used in cost math
    # once tokens are populated.
    "claude_input_tokens": 0,
    "claude_output_tokens": 0,
    "claude_cached_input_tokens": 0,
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
        """Return (gpu, claude, proxy, total) USD for the current counters.

        Claude cost prefers per-token billing (#514) when token
        counters are populated by the Anthropic client. Falls back to
        the legacy per-call count × ``claude_call_usd`` when no
        tokens were recorded (tests, callers that bypass the shared
        client, hosts that haven't deployed the token-credit hook).
        """
        cfg = self.cost_config
        gpu_cost = cfg.gpu_cost(self.costs["gpu_seconds"])
        input_tokens = int(self.costs.get("claude_input_tokens", 0) or 0)
        output_tokens = int(self.costs.get("claude_output_tokens", 0) or 0)
        cached_tokens = int(self.costs.get("claude_cached_input_tokens", 0) or 0)
        if input_tokens or output_tokens:
            claude_cost = cfg.claude_token_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_tokens,
            )
        else:
            claude_cost = cfg.claude_cost(
                self.costs["claude_extract"] + self.costs["claude_grounding"]
            )
        proxy_cost = cfg.proxy_cost(self.costs["proxy_mb"])
        return gpu_cost, claude_cost, proxy_cost, gpu_cost + claude_cost + proxy_cost

    # ── Claude token accounting (#514) ──────────────────────────────────

    def record_claude_tokens(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
    ) -> None:
        """Credit the per-call token usage to the meter (#514).

        Called from the Anthropic-client wrappers right after each
        ``messages.create`` response lands. ``cached_input_tokens``
        counts INSIDE ``input_tokens`` on Anthropic responses — pass
        both fields exactly as the API reports them; the cost math
        in :meth:`CostConfig.claude_token_cost` subtracts cached from
        plain input so the same byte isn't billed twice.
        """
        ii = max(0, int(input_tokens or 0))
        oo = max(0, int(output_tokens or 0))
        ci = max(0, int(cached_input_tokens or 0))
        if ii:
            self.costs["claude_input_tokens"] = int(
                self.costs.get("claude_input_tokens", 0) or 0
            ) + ii
        if oo:
            self.costs["claude_output_tokens"] = int(
                self.costs.get("claude_output_tokens", 0) or 0
            ) + oo
        if ci:
            self.costs["claude_cached_input_tokens"] = int(
                self.costs.get("claude_cached_input_tokens", 0) or 0
            ) + ci

    def elapsed_seconds(self) -> float:
        return time.time() - self.run_start

    # ── Per-step accounting ─────────────────────────────────────────────

    def record_step(
        self,
        step: "MicroIntent",
        step_result: "StepResult",
        *,
        time_meter: "TimeMeter | None" = None,
    ) -> None:
        """Apply a step's resource usage to the counters.

        Claude / proxy accounting unchanged. GPU accounting depends
        on ``time_meter``:

        * **With time_meter (production, since #351)** — bumps
          ``gpu_steps`` only; ``gpu_seconds`` is owned by
          :meth:`sync_gpu_seconds_from_time_meter` which sets it
          from the meter's ``think`` bucket (actual brain-inference
          wall time). Sync fires on EVERY ``record_step`` call,
          even when ``steps_used == 0``, so per-step deltas absorb
          think-bucket growth that happened during steps with no
          dispatch-attributed brain work (e.g. ``extract_data``,
          verify, and between-step critic / recovery handlers). Was
          previously gated on ``steps_used > 0``, which leaked
          inter-step think into the finalize sync where it landed
          on the run total but on no per-step row.
        * **Without time_meter (tests, legacy hosts)** — keeps the
          pre-#351 synthetic ``steps × per-step`` accumulation so
          existing callers see no behaviour change.
        """
        cfg = self.cost_config
        if step_result.steps_used > 0:
            self.costs["gpu_steps"] += step_result.steps_used
            if time_meter is None:
                # Legacy synthetic path — kept so tests / hosts that
                # don't wire a TimeMeter see the pre-#351 numbers.
                self.costs["gpu_seconds"] += (
                    step_result.steps_used * cfg.gpu_seconds_per_step
                )
        if time_meter is not None:
            # Production path — sync unconditionally so per-step
            # deltas pick up think growth from steps with
            # ``steps_used == 0`` and from inter-step handlers
            # (critic / recovery) that fired since the last sync.
            self.sync_gpu_seconds_from_time_meter(time_meter)
        if step.claude_only:
            self.costs["claude_extract"] += 1
        if step.grounding:
            self.costs["claude_grounding"] += step_result.steps_used  # ~1 grounding per click
        if step.type in ("click", "navigate", "paginate"):
            self.costs["proxy_mb"] += cfg.proxy_mb_per_nav
        elif step.type == "scroll":
            self.costs["proxy_mb"] += cfg.proxy_mb_per_scroll

    def totals_from(self, costs: dict[str, Any]) -> dict[str, float]:
        """Compute the USD-cost breakdown for an arbitrary counters dict
        (#350).

        Used for per-task cost attribution in
        :func:`mantis_agent.task_loop._run_loop` — callers snapshot
        ``cost_meter.costs`` before / after each task and pass the
        delta here to get a {gpu, claude, proxy, total} dict shaped
        like the run-level ``_final_costs``. Missing keys default to
        zero so partial dicts are safe.
        """
        cfg = self.cost_config
        gpu = cfg.gpu_cost(float(costs.get("gpu_seconds", 0.0) or 0.0))
        input_tokens = int(costs.get("claude_input_tokens", 0) or 0)
        output_tokens = int(costs.get("claude_output_tokens", 0) or 0)
        cached_tokens = int(costs.get("claude_cached_input_tokens", 0) or 0)
        if input_tokens or output_tokens:
            claude = cfg.claude_token_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_tokens,
            )
        else:
            claude = cfg.claude_cost(
                int(costs.get("claude_extract", 0) or 0)
                + int(costs.get("claude_grounding", 0) or 0)
            )
        proxy = cfg.proxy_cost(float(costs.get("proxy_mb", 0.0) or 0.0))
        return {
            "gpu": gpu,
            "claude": claude,
            "proxy": proxy,
            "total": gpu + claude + proxy,
            "gpu_steps": int(costs.get("gpu_steps", 0) or 0),
            "gpu_seconds": float(costs.get("gpu_seconds", 0.0) or 0.0),
            "claude_extract": int(costs.get("claude_extract", 0) or 0),
            "claude_grounding": int(costs.get("claude_grounding", 0) or 0),
            "claude_input_tokens": input_tokens,
            "claude_output_tokens": output_tokens,
            "claude_cached_input_tokens": cached_tokens,
            "proxy_mb": float(costs.get("proxy_mb", 0.0) or 0.0),
        }

    def sync_gpu_seconds_from_time_meter(self, time_meter: "TimeMeter") -> None:
        """Set ``costs["gpu_seconds"]`` from the meter's ``think``
        bucket (#351).

        ``think`` is the canonical brain-inference wall-time bucket
        (see :data:`~.time_meter.BUCKETS`) — credited by the brain
        ladder's ``publish_dispatch`` context whenever a Holo3 /
        Gemma4 / Claude inference call runs. Using it directly means
        per-run cost reflects what Modal actually billed for brain
        compute, instead of ``steps × 3s`` (the pre-#351 fake).

        Idempotent: sets (not increments) so calling repeatedly is
        safe. Best-effort: a missing ``think`` bucket leaves
        ``gpu_seconds`` alone — guards against monkey-patched
        TimeMeters in tests.
        """
        try:
            totals = time_meter.totals
        except AttributeError:
            return
        if "think" not in (totals or {}):
            # Missing bucket → no signal to sync from. Leave the
            # caller's pre-existing (possibly legacy-synthetic) value
            # intact rather than zeroing it.
            return
        try:
            think_seconds = float(totals.get("think", 0.0) or 0.0)
        except (TypeError, ValueError):
            return
        if think_seconds < 0:
            return
        self.costs["gpu_seconds"] = think_seconds

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


# ── Cost meter context publisher (#514) ────────────────────────────────


# Mirrors :data:`gym.time_meter._CURRENT_DISPATCH` — the runner
# publishes its CostMeter for the duration of an execution scope so
# deep helpers (the shared Anthropic client; future cost-emitting
# subsystems) can credit tokens without being passed the meter as an
# argument. PEP 567 contextvars keep this scope-correct for asyncio +
# threaded callers.
_CURRENT_COST_METER: contextvars.ContextVar["CostMeter | None"] = (
    contextvars.ContextVar("mantis_cost_meter_current", default=None)
)


def current_cost_meter() -> "CostMeter | None":
    """Return the runner's :class:`CostMeter` if one was published,
    else ``None``. Helpers that opportunistically credit costs bail
    out cleanly when this returns ``None`` — useful for tests +
    standalone-script runs of the Anthropic client.
    """
    return _CURRENT_COST_METER.get()


@contextlib.contextmanager
def publish_cost_meter(meter: "CostMeter | None") -> Iterator[None]:
    """Publish ``meter`` to the contextvar for the wrapped block.

    ``meter=None`` clears the context — used by tests that exercise
    Anthropic-client helpers in isolation without a runner present.
    Always resets on exit, including on exceptions, so a crashed
    scope can't leak the meter into the next one.
    """
    token = _CURRENT_COST_METER.set(meter)
    try:
        yield
    finally:
        _CURRENT_COST_METER.reset(token)


def record_claude_tokens_to_current(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
) -> None:
    """Credit per-call token usage to the currently-published meter (#514).

    Called by the Anthropic-client wrappers right after each
    ``messages.create`` response lands. No-op when no meter is
    published — protects test runs + standalone helpers that
    construct the client outside a runner context.
    """
    meter = _CURRENT_COST_METER.get()
    if meter is None:
        return
    meter.record_claude_tokens(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
    )


__all__ = [
    "CostMeter",
    "current_cost_meter",
    "make_initial_costs",
    "publish_cost_meter",
    "record_claude_tokens_to_current",
]
