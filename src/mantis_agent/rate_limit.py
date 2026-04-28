"""In-process rate limiter for Tier 2 multi-tenant hardening.

Two dimensions:

* **Concurrency**: count of in-flight runs per tenant; rejects when at
  ``tenant.max_concurrent_runs``. Tracked as a simple dict + lock.
* **Rate** (token bucket): refilled at ``rate_limit_per_minute / 60`` tokens
  per second up to a burst of ``rate_limit_per_minute``. Each accepted
  request consumes one token.

Thread-safe (uses a single ``threading.Lock``). State is process-local —
behind a load balancer with N replicas, each replica tracks its own
counters so the effective per-tenant cap is ``N * configured_cap``. For
strict cluster-wide limits, swap this module for a Redis-backed limiter
(future Tier 2.5).
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Optional


@dataclasses.dataclass
class RateLimitDecision:
    """Outcome of a rate-limit check."""

    allowed: bool
    reason: str = ""
    retry_after_seconds: float = 0.0
    concurrent: int = 0
    rate_remaining: float = 0.0


@dataclasses.dataclass
class _ConcurrencyState:
    in_flight: int = 0


@dataclasses.dataclass
class _BucketState:
    tokens: float
    last_refill_ts: float


class TenantRateLimiter:
    """Per-tenant rate + concurrency limiter.

    Args:
        default_rpm: Default rate-per-minute for tenants whose config does
            not specify one (read from ``tenant.rate_limit_per_minute`` if
            present, else this default).
    """

    def __init__(self, default_rpm: int = 30) -> None:
        self._default_rpm = default_rpm
        self._lock = threading.Lock()
        self._concurrency: dict[str, _ConcurrencyState] = {}
        self._buckets: dict[str, _BucketState] = {}

    # ── Concurrency ─────────────────────────────────────────────────────
    def try_acquire_concurrency_slot(
        self, tenant_id: str, max_concurrent: int
    ) -> RateLimitDecision:
        """Reserve a concurrency slot for ``tenant_id`` if available.

        Caller MUST call :meth:`release_concurrency_slot` once the run
        finishes (success, failure, or cancellation).
        """
        with self._lock:
            state = self._concurrency.setdefault(tenant_id, _ConcurrencyState())
            if state.in_flight >= max_concurrent:
                return RateLimitDecision(
                    allowed=False,
                    reason=f"max_concurrent_runs ({max_concurrent}) exceeded",
                    retry_after_seconds=5.0,
                    concurrent=state.in_flight,
                )
            state.in_flight += 1
            return RateLimitDecision(allowed=True, concurrent=state.in_flight)

    def release_concurrency_slot(self, tenant_id: str) -> None:
        with self._lock:
            state = self._concurrency.get(tenant_id)
            if state and state.in_flight > 0:
                state.in_flight -= 1

    def get_concurrent(self, tenant_id: str) -> int:
        with self._lock:
            state = self._concurrency.get(tenant_id)
            return state.in_flight if state else 0

    # ── Rate (token bucket) ─────────────────────────────────────────────
    def try_consume_rate_token(
        self,
        tenant_id: str,
        rate_per_minute: Optional[int] = None,
    ) -> RateLimitDecision:
        """Try to consume one rate token; returns ``allowed=False`` with
        ``retry_after_seconds`` when the bucket is empty."""
        rpm = rate_per_minute if rate_per_minute is not None else self._default_rpm
        if rpm <= 0:
            # Disabled — no rate limit.
            return RateLimitDecision(allowed=True, rate_remaining=float("inf"))

        burst = float(rpm)
        refill_per_sec = rpm / 60.0
        now = time.monotonic()
        with self._lock:
            state = self._buckets.get(tenant_id)
            if state is None:
                state = _BucketState(tokens=burst, last_refill_ts=now)
                self._buckets[tenant_id] = state
            else:
                elapsed = now - state.last_refill_ts
                if elapsed > 0:
                    state.tokens = min(burst, state.tokens + elapsed * refill_per_sec)
                    state.last_refill_ts = now

            if state.tokens < 1.0:
                deficit = 1.0 - state.tokens
                retry_after = deficit / refill_per_sec
                return RateLimitDecision(
                    allowed=False,
                    reason=f"rate limit ({rpm}/min) exceeded",
                    retry_after_seconds=max(retry_after, 0.1),
                    rate_remaining=state.tokens,
                )

            state.tokens -= 1.0
            return RateLimitDecision(
                allowed=True, rate_remaining=state.tokens
            )

    # ── Test / admin helpers ────────────────────────────────────────────
    def reset(self) -> None:
        with self._lock:
            self._concurrency.clear()
            self._buckets.clear()


# Module-level singleton; the server reuses this across requests.
_LIMITER: TenantRateLimiter | None = None


def get_rate_limiter() -> TenantRateLimiter:
    global _LIMITER
    if _LIMITER is None:
        _LIMITER = TenantRateLimiter()
    return _LIMITER


def reset_rate_limiter() -> None:
    """Test helper: drop the singleton so a fresh state is constructed."""
    global _LIMITER
    _LIMITER = None
