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
    """Per-resource pricing knobs for the CUA runtime."""

    gpu_hourly_usd: float = 3.25
    claude_call_usd: float = 0.003
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
            proxy_per_gb_usd=_env_float("MANTIS_COST_PROXY_PER_GB_USD", 5.0),
            gpu_seconds_per_step=_env_float("MANTIS_COST_GPU_SECONDS_PER_STEP", 3.0),
            proxy_mb_per_nav=_env_float("MANTIS_COST_PROXY_MB_PER_NAV", 5.0),
            proxy_mb_per_scroll=_env_float("MANTIS_COST_PROXY_MB_PER_SCROLL", 0.5),
        )

    # ── Cost computations ───────────────────────────────────────────────

    def gpu_cost(self, gpu_seconds: float) -> float:
        return (gpu_seconds / 3600.0) * self.gpu_hourly_usd

    def claude_cost(self, num_calls: int) -> float:
        return num_calls * self.claude_call_usd

    def proxy_cost(self, proxy_mb: float) -> float:
        return (proxy_mb / 1024.0) * self.proxy_per_gb_usd
