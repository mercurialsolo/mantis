"""Tests for X-RateLimit-* response headers (#275).

Covered:

- ``RateLimitDecision`` carries ``limit`` and ``reset_after_seconds``.
- ``try_consume_rate_token`` populates both on allow and deny paths.
- ``_stash_rate_limit_headers`` writes the canonical triple to
  ``request.state``.
- ``_attach_rate_limit_headers`` middleware copies them onto the
  response — including on 4xx/5xx responses (so 429s carry the headers
  alongside ``Retry-After``).
- Routes that don't rate-limit (``/v1/health``, ``/v1/version``, etc.)
  return no ``X-RateLimit-*`` headers.

The middleware path is exercised against a minimal stub FastAPI app
that imports the real middleware + stash helper but doesn't drag the
production routes / runtime / model load.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from mantis_agent.baseten_server.routes import (
    _attach_rate_limit_headers,
    _stash_rate_limit_headers,
)
from mantis_agent.rate_limit import (
    RateLimitDecision,
    TenantRateLimiter,
)


# ── RateLimitDecision + limiter populate the new fields ────────────────────


def test_rate_limit_decision_has_limit_and_reset() -> None:
    d = RateLimitDecision(allowed=True)
    assert d.limit == 0
    assert d.reset_after_seconds == 0.0


def test_try_consume_token_populates_limit_and_reset_on_allow() -> None:
    limiter = TenantRateLimiter()
    decision = limiter.try_consume_rate_token("tenant-x", rate_per_minute=60)
    assert decision.allowed is True
    assert decision.limit == 60
    assert decision.rate_remaining == 59.0
    # One token spent → reset takes 1/refill = 1 sec to recover.
    assert 0.5 < decision.reset_after_seconds < 2.0


def test_try_consume_token_populates_fields_on_deny() -> None:
    """Even when the bucket is empty, the decision should carry the
    cap + remaining so the 429 response can surface them. Without
    this, callers can't tell whether to back off briefly or for a
    while just from the headers."""
    limiter = TenantRateLimiter()
    # Drain the bucket by consuming all tokens.
    for _ in range(10):
        limiter.try_consume_rate_token("tenant-y", rate_per_minute=10)
    decision = limiter.try_consume_rate_token("tenant-y", rate_per_minute=10)
    assert decision.allowed is False
    assert decision.limit == 10
    assert decision.rate_remaining < 1.0
    # The bucket needs (10 - remaining) tokens to refill at 10/min → ~60s.
    assert 50.0 < decision.reset_after_seconds < 60.5


def test_try_consume_token_disabled_returns_zero_limit() -> None:
    """rate_per_minute=0 disables rate-limiting. The decision carries
    limit=0 so the middleware can skip header emission entirely."""
    limiter = TenantRateLimiter()
    decision = limiter.try_consume_rate_token("tenant-z", rate_per_minute=0)
    assert decision.allowed is True
    assert decision.limit == 0


# ── _stash_rate_limit_headers ──────────────────────────────────────────────


def test_stash_sets_canonical_triple_on_request_state() -> None:
    request = SimpleNamespace(state=SimpleNamespace())
    decision = RateLimitDecision(
        allowed=True,
        rate_remaining=27.0,
        limit=30,
        reset_after_seconds=6.0,
    )
    now = time.time()
    _stash_rate_limit_headers(request, decision, 30)  # type: ignore[arg-type]
    headers = request.state.rate_limit_headers
    assert headers["X-RateLimit-Limit"] == 30
    assert headers["X-RateLimit-Remaining"] == 27
    # Reset is a Unix timestamp ~now + reset_after_seconds.
    assert now + 5 <= headers["X-RateLimit-Reset"] <= now + 8


def test_stash_no_op_when_rate_limit_disabled() -> None:
    """When the tenant has no rate limit (rate_per_minute=0) the
    decision carries limit=0 and there's no useful header to surface.
    Stash should skip the assignment so the middleware emits nothing."""
    request = SimpleNamespace(state=SimpleNamespace())
    decision = RateLimitDecision(
        allowed=True, rate_remaining=float("inf"), limit=0,
    )
    _stash_rate_limit_headers(request, decision, 0)  # type: ignore[arg-type]
    assert not hasattr(request.state, "rate_limit_headers")


def test_stash_clamps_negative_remaining_to_zero() -> None:
    """When the bucket is overdrawn (fractional negative tokens), the
    ``Remaining`` header should report 0, not a negative integer."""
    request = SimpleNamespace(state=SimpleNamespace())
    decision = RateLimitDecision(
        allowed=False,
        rate_remaining=-0.2,
        limit=10,
        reset_after_seconds=1.0,
    )
    _stash_rate_limit_headers(request, decision, 10)  # type: ignore[arg-type]
    assert request.state.rate_limit_headers["X-RateLimit-Remaining"] == 0


# ── Middleware integration on a stub FastAPI app ──────────────────────────


def _make_stub_app() -> FastAPI:
    """Stub app that wires the real middleware against synthetic routes.

    Avoids importing the Mantis runtime / model load — the only thing
    under test is the middleware contract, which is fully decoupled
    from the heavy plumbing.
    """
    app = FastAPI()
    app.middleware("http")(_attach_rate_limit_headers)

    @app.get("/limited")
    def limited(request: Request) -> dict:
        decision = RateLimitDecision(
            allowed=True,
            rate_remaining=29.0,
            limit=30,
            reset_after_seconds=2.0,
        )
        _stash_rate_limit_headers(request, decision, 30)
        return {"ok": True}

    @app.get("/limited-deny")
    def limited_deny(request: Request) -> dict:
        decision = RateLimitDecision(
            allowed=False,
            reason="rate limit (30/min) exceeded",
            retry_after_seconds=2.0,
            rate_remaining=0.0,
            limit=30,
            reset_after_seconds=60.0,
        )
        _stash_rate_limit_headers(request, decision, 30)
        raise HTTPException(
            status_code=429,
            detail=decision.reason,
            headers={"Retry-After": "3"},
        )

    @app.get("/unlimited")
    def unlimited() -> dict:
        # No stash → middleware should be a no-op.
        return {"ok": True}

    return app


def test_middleware_attaches_headers_on_success() -> None:
    client = TestClient(_make_stub_app())
    resp = client.get("/limited")
    assert resp.status_code == 200
    assert resp.headers["X-RateLimit-Limit"] == "30"
    assert resp.headers["X-RateLimit-Remaining"] == "29"
    assert "X-RateLimit-Reset" in resp.headers
    # Reset is a string-formatted unix timestamp; just spot-check it parses.
    int(resp.headers["X-RateLimit-Reset"])


def test_middleware_attaches_headers_on_429() -> None:
    """429s carry both ``Retry-After`` (legacy / required for backoff)
    and the ``X-RateLimit-*`` triple so callers can reconcile their
    local view of the bucket without having to wait for the next
    successful response."""
    client = TestClient(_make_stub_app())
    resp = client.get("/limited-deny")
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "3"
    assert resp.headers["X-RateLimit-Limit"] == "30"
    assert resp.headers["X-RateLimit-Remaining"] == "0"
    assert "X-RateLimit-Reset" in resp.headers


def test_middleware_noop_on_unlimited_routes() -> None:
    """Routes that don't stash anything (health probes, version,
    metrics) must not gain X-RateLimit-* headers — those are an
    explicit promise that the route is rate-limited."""
    client = TestClient(_make_stub_app())
    resp = client.get("/unlimited")
    assert resp.status_code == 200
    assert "X-RateLimit-Limit" not in resp.headers
    assert "X-RateLimit-Remaining" not in resp.headers
    assert "X-RateLimit-Reset" not in resp.headers
