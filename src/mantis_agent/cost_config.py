"""Cost configuration for CUA runs (#122).

Centralizes the per-resource rates so they can be overridden per-tenant or
per-deployment via environment variables, instead of being baked into the
runner as floating-point literals scattered across the file.

Defaults match the values previously hardcoded in
``MicroPlanRunner._cost_totals`` and ``_record_step_costs`` so behavior is
unchanged for runs that don't set any env vars.

Env vars (all USD where currency applies, all interpreted as floats):

* ``MANTIS_COST_GPU_HOURLY_USD``      — GPU compute, $/hour       (default 3.25)
* ``MANTIS_COST_CLAUDE_CALL_USD``     — per Claude API call       (default 0.003)
* ``MANTIS_COST_PROXY_PER_GB_USD``    — egress proxy bandwidth    (default 5.00)
* ``MANTIS_COST_GPU_SECONDS_PER_STEP``— ~GPU time per agent step  (default 3.0)
* ``MANTIS_COST_PROXY_MB_PER_NAV``    — proxy MB per page load    (default 5.0)
* ``MANTIS_COST_PROXY_MB_PER_SCROLL`` — proxy MB per scroll       (default 0.5)
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        # Bad config should not silently fall back — surface so operators see it.
        raise ValueError(
            f"{key} must be a number (got {raw!r}); unset or fix the env var"
        ) from None


@dataclass(frozen=True)
class CostConfig:
    """Per-resource pricing knobs for the CUA runtime.

    Claude cost has two parallel surfaces:

    * **Per-token** (preferred, since #514) — bills against actual
      ``input_tokens`` / ``output_tokens`` returned by the Anthropic
      API on every response. Defaults track Claude Sonnet 4 list
      prices ($3 / $15 / $0.30 per M tokens for input / output /
      cached-input). Override per deployment via env when the
      production model is different (Opus is ~5x; Haiku is ~5x lower).
    * **Per-call** (legacy) — flat ``claude_call_usd`` × call count.
      Kept for back-compat with callers + tests that still rely on
      the older :meth:`claude_cost` API. New code paths should use
      :meth:`claude_token_cost` which is the canonical surface.
    """

    gpu_hourly_usd: float = 3.25
    claude_call_usd: float = 0.003
    # Per-million-token rates (USD). Defaults: Claude Sonnet 4 list prices.
    claude_input_per_mtok: float = 3.0
    claude_output_per_mtok: float = 15.0
    # Cached-input rate when the prompt-cache breakpoint hits — Anthropic
    # charges ~10% of the standard input rate. Defaults match Sonnet 4.
    claude_cached_input_per_mtok: float = 0.3
    proxy_per_gb_usd: float = 5.0
    gpu_seconds_per_step: float = 3.0
    proxy_mb_per_nav: float = 5.0
    proxy_mb_per_scroll: float = 0.5

    @classmethod
    def from_env(cls) -> "CostConfig":
        """Construct a CostConfig with env-var overrides applied.

        Returns a fresh instance each call so tests can monkeypatch env between
        constructions. Operators set these once at deploy time.
        """
        return cls(
            gpu_hourly_usd=_env_float("MANTIS_COST_GPU_HOURLY_USD", 3.25),
            claude_call_usd=_env_float("MANTIS_COST_CLAUDE_CALL_USD", 0.003),
            claude_input_per_mtok=_env_float("MANTIS_COST_CLAUDE_INPUT_PER_MTOK", 3.0),
            claude_output_per_mtok=_env_float("MANTIS_COST_CLAUDE_OUTPUT_PER_MTOK", 15.0),
            claude_cached_input_per_mtok=_env_float(
                "MANTIS_COST_CLAUDE_CACHED_INPUT_PER_MTOK", 0.3,
            ),
            proxy_per_gb_usd=_env_float("MANTIS_COST_PROXY_PER_GB_USD", 5.0),
            gpu_seconds_per_step=_env_float("MANTIS_COST_GPU_SECONDS_PER_STEP", 3.0),
            proxy_mb_per_nav=_env_float("MANTIS_COST_PROXY_MB_PER_NAV", 5.0),
            proxy_mb_per_scroll=_env_float("MANTIS_COST_PROXY_MB_PER_SCROLL", 0.5),
        )

    # ── Cost computations ───────────────────────────────────────────────

    def gpu_cost(self, gpu_seconds: float) -> float:
        return (gpu_seconds / 3600.0) * self.gpu_hourly_usd

    def claude_cost(self, num_calls: int) -> float:
        """Legacy per-call billing (kept for back-compat).

        New code should use :meth:`claude_token_cost` — call-count
        billing under-counts real spend by roughly 40x at the
        $0.003-per-call default. See #514.
        """
        return num_calls * self.claude_call_usd

    def claude_token_cost(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
    ) -> float:
        """Token-billed Claude cost in USD (#514).

        ``cached_input_tokens`` counts INSIDE ``input_tokens`` on
        Anthropic responses (the API returns both fields; cached
        tokens are billed at the lower cached rate, the remainder at
        the standard input rate). We subtract the cached count from
        ``input_tokens`` here so callers can pass both fields exactly
        as Anthropic reports them.
        """
        ci = max(0, int(cached_input_tokens or 0))
        ii = max(0, int(input_tokens or 0) - ci)
        oo = max(0, int(output_tokens or 0))
        return (
            (ii / 1_000_000.0) * self.claude_input_per_mtok
            + (ci / 1_000_000.0) * self.claude_cached_input_per_mtok
            + (oo / 1_000_000.0) * self.claude_output_per_mtok
        )

    def proxy_cost(self, proxy_mb: float) -> float:
        return (proxy_mb / 1024.0) * self.proxy_per_gb_usd
