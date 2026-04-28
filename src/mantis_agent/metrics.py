"""Tier 2 Prometheus metrics for the Mantis CUA service.

Lightweight wrappers around prometheus_client. Soft-imported so the
[orchestrator] extras (which omit prometheus_client) don't break library
imports — when the dep is absent, all of these become no-ops and
``/metrics`` returns 503.

Counters / histograms / gauges, all labeled by tenant when applicable:

* ``mantis_predict_requests_total{tenant_id, mode, outcome}``
* ``mantis_run_duration_seconds{tenant_id, model, status}``
* ``mantis_run_cost_usd{tenant_id, model, status}``
* ``mantis_concurrent_runs{tenant_id}``
* ``mantis_chat_completions_total{tenant_id, status}``
* ``mantis_rate_limit_rejections_total{tenant_id, kind}``
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("mantis_agent.metrics")

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on extras
    _PROM_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    def generate_latest(*_args: Any, **_kw: Any) -> bytes:  # type: ignore[no-redef]
        return b""


class _NullMetric:
    """No-op stand-in when prometheus_client isn't installed."""

    def labels(self, **_kw: Any) -> "_NullMetric":
        return self

    def inc(self, *_args: Any, **_kw: Any) -> None:
        pass

    def dec(self, *_args: Any, **_kw: Any) -> None:
        pass

    def observe(self, *_args: Any, **_kw: Any) -> None:
        pass

    def set(self, *_args: Any, **_kw: Any) -> None:
        pass


def _counter(name: str, doc: str, labels: tuple[str, ...]) -> Any:
    if _PROM_AVAILABLE:
        return Counter(name, doc, labels)
    return _NullMetric()


def _gauge(name: str, doc: str, labels: tuple[str, ...]) -> Any:
    if _PROM_AVAILABLE:
        return Gauge(name, doc, labels)
    return _NullMetric()


def _histogram(name: str, doc: str, labels: tuple[str, ...], buckets: tuple[float, ...]) -> Any:
    if _PROM_AVAILABLE:
        return Histogram(name, doc, labels, buckets=buckets)
    return _NullMetric()


# ── Metric handles ──────────────────────────────────────────────────────────
PREDICT_REQUESTS = _counter(
    "mantis_predict_requests_total",
    "Count of /predict + /v1/predict invocations.",
    ("tenant_id", "mode", "outcome"),
)

CHAT_COMPLETIONS = _counter(
    "mantis_chat_completions_total",
    "Count of /v1/chat/completions invocations forwarded to Holo3.",
    ("tenant_id", "outcome"),
)

RUN_DURATION_SECONDS = _histogram(
    "mantis_run_duration_seconds",
    "End-to-end duration of detached runs in seconds.",
    ("tenant_id", "model", "status"),
    buckets=(10, 30, 60, 120, 300, 600, 1200, 1800, 3600),
)

RUN_COST_USD = _histogram(
    "mantis_run_cost_usd",
    "Per-run total cost in USD (GPU + Claude + Proxy).",
    ("tenant_id", "model", "status"),
    buckets=(0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.5, 5.0, 10.0, 25.0),
)

CONCURRENT_RUNS = _gauge(
    "mantis_concurrent_runs",
    "Currently in-flight runs per tenant.",
    ("tenant_id",),
)

RATE_LIMIT_REJECTIONS = _counter(
    "mantis_rate_limit_rejections_total",
    "Requests rejected by the per-tenant rate limiter.",
    ("tenant_id", "kind"),
)


def is_available() -> bool:
    """True if prometheus_client is installed and metrics are live."""
    return _PROM_AVAILABLE


def render_text() -> bytes:
    """Render the current registry to Prometheus text-format bytes."""
    return generate_latest()
