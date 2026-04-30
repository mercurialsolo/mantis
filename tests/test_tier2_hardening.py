"""Tier 2 hardening tests: rate limits, idempotency, webhooks, allowlist, metrics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("pydantic")

from mantis_agent.api_schemas import (
    assert_hosts_allowed,
    extract_navigate_hosts,
)
from mantis_agent.idempotency import IdempotencyCache
from mantis_agent import metrics as mantis_metrics
from mantis_agent.rate_limit import TenantRateLimiter
from mantis_agent.tenant_auth import TenantConfig
from mantis_agent.webhooks import WebhookPayload, deliver_webhook_sync, sign_body


# ── Rate limiter ────────────────────────────────────────────────────────────
def test_rate_limiter_allows_within_burst():
    rl = TenantRateLimiter()
    decisions = [rl.try_consume_rate_token("t1", rate_per_minute=60) for _ in range(60)]
    assert all(d.allowed for d in decisions)


def test_rate_limiter_rejects_over_burst():
    rl = TenantRateLimiter()
    for _ in range(30):
        d = rl.try_consume_rate_token("t1", rate_per_minute=30)
        assert d.allowed
    d = rl.try_consume_rate_token("t1", rate_per_minute=30)
    assert not d.allowed
    assert d.retry_after_seconds > 0


def test_rate_limiter_zero_rpm_disables_limit():
    rl = TenantRateLimiter()
    for _ in range(1000):
        d = rl.try_consume_rate_token("t1", rate_per_minute=0)
        assert d.allowed


def test_rate_limiter_per_tenant_isolation():
    rl = TenantRateLimiter()
    for _ in range(5):
        assert rl.try_consume_rate_token("a", rate_per_minute=5).allowed
    assert not rl.try_consume_rate_token("a", rate_per_minute=5).allowed
    # Other tenant has its own bucket
    assert rl.try_consume_rate_token("b", rate_per_minute=5).allowed


def test_concurrency_acquire_and_release():
    rl = TenantRateLimiter()
    d1 = rl.try_acquire_concurrency_slot("t1", max_concurrent=2)
    d2 = rl.try_acquire_concurrency_slot("t1", max_concurrent=2)
    d3 = rl.try_acquire_concurrency_slot("t1", max_concurrent=2)
    assert d1.allowed and d2.allowed
    assert not d3.allowed
    assert d3.retry_after_seconds > 0
    rl.release_concurrency_slot("t1")
    d4 = rl.try_acquire_concurrency_slot("t1", max_concurrent=2)
    assert d4.allowed


def test_concurrency_release_floor_at_zero():
    rl = TenantRateLimiter()
    rl.release_concurrency_slot("t1")
    rl.release_concurrency_slot("t1")
    assert rl.get_concurrent("t1") == 0


# ── Idempotency cache ──────────────────────────────────────────────────────
def test_idempotency_get_returns_none_for_unknown_key(tmp_path: Path):
    cache = IdempotencyCache(root_dir=tmp_path)
    assert cache.get("t1", "unseen-key") is None


def test_idempotency_store_then_get(tmp_path: Path):
    cache = IdempotencyCache(root_dir=tmp_path)
    cache.store("t1", "key1", "20260428_xyz", {"status": "queued", "run_id": "20260428_xyz"})
    cached = cache.get("t1", "key1")
    assert cached is not None
    assert cached.run_id == "20260428_xyz"
    assert cached.response["status"] == "queued"


def test_idempotency_per_tenant_isolation(tmp_path: Path):
    cache = IdempotencyCache(root_dir=tmp_path)
    cache.store("t1", "shared-key", "run-A", {"run_id": "run-A"})
    cache.store("t2", "shared-key", "run-B", {"run_id": "run-B"})
    assert cache.get("t1", "shared-key").run_id == "run-A"
    assert cache.get("t2", "shared-key").run_id == "run-B"


def test_idempotency_survives_process_restart(tmp_path: Path):
    cache1 = IdempotencyCache(root_dir=tmp_path)
    cache1.store("t1", "persistent", "rid", {"run_id": "rid"})
    # Fresh cache reads the sidecar
    cache2 = IdempotencyCache(root_dir=tmp_path)
    cached = cache2.get("t1", "persistent")
    assert cached is not None and cached.run_id == "rid"


def test_idempotency_empty_key_is_noop(tmp_path: Path):
    cache = IdempotencyCache(root_dir=tmp_path)
    cache.store("t1", "", "rid", {"run_id": "rid"})
    cache.store("t1", "real-key", "", {})  # empty run_id also rejected
    assert cache.get("t1", "") is None


def test_idempotency_expired_entry_pruned(tmp_path: Path):
    cache = IdempotencyCache(root_dir=tmp_path)
    cache.store("t1", "old", "rid", {"run_id": "rid"})
    # Manipulate the stored timestamp to look ancient
    sidecar = next((tmp_path / "t1").glob("*.json"))
    raw = json.loads(sidecar.read_text())
    raw["stored_at"] = 0  # epoch
    sidecar.write_text(json.dumps(raw))
    cache._mem.clear()  # force re-read from disk
    assert cache.get("t1", "old") is None


# ── Webhook delivery + signing ──────────────────────────────────────────────
def test_webhook_signature_matches_hmac():
    body = b'{"foo":"bar"}'
    sig = sign_body(body, "shared-secret")
    # Reproduce manually
    import hashlib
    import hmac as _hmac
    expected = _hmac.new(b"shared-secret", body, hashlib.sha256).hexdigest()
    assert sig == expected


def test_webhook_signature_empty_secret_returns_empty():
    assert sign_body(b'{}', "") == ""


def _ok_response(status=200):
    r = MagicMock()
    r.status_code = status
    r.text = "ok"
    return r


def test_webhook_delivers_on_first_try():
    payload = WebhookPayload(run_id="r1", tenant_id="t1", status="succeeded", summary={})
    sess = MagicMock()
    sess.post.return_value = _ok_response(200)
    ok = deliver_webhook_sync(
        "https://example.com/hook", payload, secret="s",
        retry_delays=(0.0,), session=sess,
    )
    assert ok
    sess.post.assert_called_once()
    # Signature header is present
    call_kwargs = sess.post.call_args.kwargs
    assert "X-Mantis-Signature" in call_kwargs["headers"]


def test_webhook_retries_on_5xx_then_succeeds():
    payload = WebhookPayload(run_id="r1", tenant_id="t1", status="failed", summary={})
    sess = MagicMock()
    sess.post.side_effect = [_ok_response(503), _ok_response(503), _ok_response(200)]
    ok = deliver_webhook_sync(
        "https://example.com/hook", payload, secret="s",
        retry_delays=(0.0, 0.0), session=sess,
    )
    assert ok
    assert sess.post.call_count == 3


def test_webhook_gives_up_after_all_retries_fail():
    payload = WebhookPayload(run_id="r1", tenant_id="t1", status="failed", summary={})
    sess = MagicMock()
    sess.post.return_value = _ok_response(503)
    ok = deliver_webhook_sync(
        "https://example.com/hook", payload, secret="s",
        retry_delays=(0.0, 0.0), session=sess,
    )
    assert not ok
    assert sess.post.call_count == 3


def test_webhook_no_url_is_noop():
    payload = WebhookPayload(run_id="r1", tenant_id="t1", status="succeeded", summary={})
    assert not deliver_webhook_sync("", payload)


# ── URL allowlist enforcement ──────────────────────────────────────────────
def test_extract_hosts_from_micro_plan():
    plan = [
        {"intent": "Navigate to https://www.boattrader.com/boats/state-fl/",
         "type": "navigate"},
        {"intent": "Click listing", "type": "click"},
        {"intent": "Read URL", "type": "extract_url"},
    ]
    hosts = extract_navigate_hosts(plan)
    assert "www.boattrader.com" in hosts


def test_extract_hosts_from_task_suite():
    suite = {
        "base_url": "https://crm.example.com",
        "tasks": [
            {"task_id": "login", "intent": "Go to https://crm.example.com",
             "start_url": "https://crm.example.com"},
        ],
    }
    hosts = extract_navigate_hosts(suite)
    assert "crm.example.com" in hosts


def test_assert_hosts_allowed_passes_when_all_match():
    t = TenantConfig(tenant_id="t1", allowed_domains=("*.boattrader.com",))
    assert_hosts_allowed(["www.boattrader.com"], t.is_domain_allowed)


def test_assert_hosts_allowed_raises_on_off_list():
    t = TenantConfig(tenant_id="t1", allowed_domains=("*.boattrader.com",))
    with pytest.raises(PermissionError, match="evil.com"):
        assert_hosts_allowed(["www.boattrader.com", "evil.com"], t.is_domain_allowed)


def test_assert_hosts_allowed_empty_list_is_noop():
    t = TenantConfig(tenant_id="t1", allowed_domains=("only.example.com",))
    # No URLs in the plan = nothing to check
    assert_hosts_allowed([], t.is_domain_allowed)


# ── Metrics module ──────────────────────────────────────────────────────────
def test_metrics_handles_support_inc_set_observe():
    # Whether or not prometheus_client is installed, every metric handle must
    # support .labels(...).inc() / .set() / .observe() without raising. Don't
    # reload the module — that double-registers Counters in the global
    # CollectorRegistry.
    mantis_metrics.PREDICT_REQUESTS.labels(
        tenant_id="t-metrics-test", mode="run", outcome="ok"
    ).inc()
    mantis_metrics.RATE_LIMIT_REJECTIONS.labels(
        tenant_id="t-metrics-test", kind="rate"
    ).inc()
    mantis_metrics.CONCURRENT_RUNS.labels(tenant_id="t-metrics-test").set(2)
    mantis_metrics.RUN_DURATION_SECONDS.labels(
        tenant_id="t-metrics-test", model="holo3", status="succeeded"
    ).observe(123.4)


def test_metrics_render_text_returns_bytes():
    out = mantis_metrics.render_text()
    assert isinstance(out, bytes)
