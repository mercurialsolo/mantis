"""Multi-tenant auth + cap enforcement tests for Tier 1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mantis_agent.tenant_auth import (
    DEFAULT_TENANT,
    TenantConfig,
    TenantKeyStore,
    reset_key_store,
)


# ── TenantConfig domain matching ────────────────────────────────────────────
def test_empty_allowlist_allows_any_domain():
    t = TenantConfig(tenant_id="t1", allowed_domains=())
    assert t.is_domain_allowed("example.com")
    assert t.is_domain_allowed("internal.crm.example.com")


def test_exact_domain_match():
    t = TenantConfig(tenant_id="t1", allowed_domains=("crm.example.com",))
    assert t.is_domain_allowed("crm.example.com")
    assert not t.is_domain_allowed("evil.example.com")


def test_wildcard_subdomain_match():
    t = TenantConfig(tenant_id="t1", allowed_domains=("*.boattrader.com",))
    assert t.is_domain_allowed("www.boattrader.com")
    assert t.is_domain_allowed("api.boattrader.com")
    assert not t.is_domain_allowed("boattrader.com.evil.com")
    assert not t.is_domain_allowed("evil.com")


def test_scope_check():
    t = TenantConfig(tenant_id="t1", scopes=("status", "result"))
    assert t.has_scope("status")
    assert not t.has_scope("run")


# ── TenantKeyStore resolution ───────────────────────────────────────────────
def test_single_tenant_fallback_when_no_keys_file(monkeypatch):
    monkeypatch.delenv("MANTIS_TENANT_KEYS_PATH", raising=False)
    monkeypatch.setenv("MANTIS_API_TOKEN", "single-tenant-token")
    reset_key_store()
    store = TenantKeyStore()
    assert not store.is_multi_tenant
    assert store.resolve("single-tenant-token") is DEFAULT_TENANT
    assert store.resolve("wrong-token") is None


def test_no_keys_and_no_fallback_token_rejects(monkeypatch):
    monkeypatch.delenv("MANTIS_TENANT_KEYS_PATH", raising=False)
    monkeypatch.delenv("MANTIS_API_TOKEN", raising=False)
    reset_key_store()
    store = TenantKeyStore()
    assert store.resolve("anything") is None


def test_multi_tenant_keys_file(tmp_path: Path, monkeypatch):
    keys_file = tmp_path / "tenant_keys.json"
    keys_file.write_text(json.dumps({
        "tenant_keys": {
            "tok-vc-prod": {
                "tenant_id": "tenant_a",
                "scopes": ["run", "status", "result"],
                "max_concurrent_runs": 3,
                "max_cost_per_run": 5.0,
                "max_time_minutes_per_run": 30,
                "anthropic_secret_name": "anthropic_api_key_tenant_a",
                "allowed_domains": ["*.boattrader.com", "crm.example.com"],
            },
            "tok-readonly": {
                "tenant_id": "readonly_consumer",
                "scopes": ["status", "result"],
            },
        }
    }))
    monkeypatch.setenv("MANTIS_TENANT_KEYS_PATH", str(keys_file))
    monkeypatch.delenv("MANTIS_API_TOKEN", raising=False)
    reset_key_store()
    store = TenantKeyStore()
    assert store.is_multi_tenant

    cfg_run = store.resolve("tok-vc-prod")
    assert cfg_run is not None
    assert cfg_run.tenant_id == "tenant_a"
    assert cfg_run.has_scope("run")
    assert cfg_run.max_cost_per_run == 5.0
    assert cfg_run.is_domain_allowed("api.boattrader.com")

    cfg_ro = store.resolve("tok-readonly")
    assert cfg_ro is not None
    assert cfg_ro.tenant_id == "readonly_consumer"
    assert not cfg_ro.has_scope("run")
    assert cfg_ro.has_scope("status")

    assert store.resolve("unknown-token") is None


def test_keys_file_hot_reload(tmp_path: Path, monkeypatch):
    keys_file = tmp_path / "tenant_keys.json"
    keys_file.write_text(json.dumps({
        "tenant_keys": {"tok1": {"tenant_id": "t1"}}
    }))
    monkeypatch.setenv("MANTIS_TENANT_KEYS_PATH", str(keys_file))
    reset_key_store()
    store = TenantKeyStore()
    # Shorten TTL so test doesn't sleep
    store._CACHE_TTL_SECONDS = 0.0
    assert store.resolve("tok1") is not None
    assert store.resolve("tok2") is None

    keys_file.write_text(json.dumps({
        "tenant_keys": {"tok1": {"tenant_id": "t1"}, "tok2": {"tenant_id": "t2"}}
    }))
    cfg = store.resolve("tok2")
    assert cfg is not None and cfg.tenant_id == "t2"


def test_constant_time_compare_resists_empty_token(monkeypatch):
    monkeypatch.setenv("MANTIS_API_TOKEN", "real-token")
    reset_key_store()
    store = TenantKeyStore()
    assert store.resolve("") is None
    assert store.resolve(None) is None  # type: ignore[arg-type]


def test_invalid_json_keys_file_yields_no_tenants(tmp_path: Path, monkeypatch):
    keys_file = tmp_path / "tenant_keys.json"
    keys_file.write_text("{not json")
    monkeypatch.setenv("MANTIS_TENANT_KEYS_PATH", str(keys_file))
    reset_key_store()
    store = TenantKeyStore()
    assert store.resolve("tok1") is None


def test_missing_keys_file_yields_no_tenants(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MANTIS_TENANT_KEYS_PATH", str(tmp_path / "missing.json"))
    reset_key_store()
    store = TenantKeyStore()
    assert store.resolve("tok1") is None


# ── api_schemas: cap enforcement + plan validation ──────────────────────────
def test_predict_request_clamps_max_cost():
    pytest.importorskip("pydantic")
    from mantis_agent.api_schemas import MAX_COST_USD, PredictRequest

    req = PredictRequest(micro="plans/x.json", max_cost=MAX_COST_USD * 10)
    assert req.max_cost == MAX_COST_USD


def test_predict_request_clamps_max_time():
    pytest.importorskip("pydantic")
    from mantis_agent.api_schemas import MAX_RUNTIME_MINUTES, PredictRequest

    req = PredictRequest(micro="plans/x.json", max_time_minutes=MAX_RUNTIME_MINUTES * 10)
    assert req.max_time_minutes == MAX_RUNTIME_MINUTES


def test_predict_request_requires_a_plan_in_run_mode():
    pytest.importorskip("pydantic")
    from mantis_agent.api_schemas import PredictRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PredictRequest()  # no plan, no action


def test_predict_request_action_requires_run_id():
    pytest.importorskip("pydantic")
    from mantis_agent.api_schemas import PredictRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PredictRequest(action="status")


def test_predict_request_action_status_skips_plan_check():
    pytest.importorskip("pydantic")
    from mantis_agent.api_schemas import PredictRequest

    req = PredictRequest(action="status", run_id="20260428_abc")
    assert req.action == "status"
    assert req.run_id == "20260428_abc"


def test_validate_micro_steps_clamps_loop_count():
    pytest.importorskip("pydantic")
    from mantis_agent.api_schemas import MAX_LOOP_ITERATIONS, validate_micro_steps

    steps = [
        {"intent": "go", "type": "navigate"},
        {"intent": "loop", "type": "loop", "loop_count": 999_999, "loop_target": 0},
    ]
    out = validate_micro_steps(steps)
    assert out[1]["loop_count"] == MAX_LOOP_ITERATIONS


def test_validate_micro_steps_rejects_oversize_plan():
    pytest.importorskip("pydantic")
    from mantis_agent.api_schemas import MAX_STEPS_PER_PLAN, validate_micro_steps

    steps = [{"intent": "x", "type": "navigate"} for _ in range(MAX_STEPS_PER_PLAN + 1)]
    with pytest.raises(ValueError, match="server cap"):
        validate_micro_steps(steps)


def test_validate_micro_steps_rejects_missing_fields():
    pytest.importorskip("pydantic")
    from mantis_agent.api_schemas import validate_micro_steps

    with pytest.raises(ValueError, match="intent"):
        validate_micro_steps([{"type": "navigate"}])
    with pytest.raises(ValueError, match="type"):
        validate_micro_steps([{"intent": "go"}])
    with pytest.raises(ValueError, match="JSON object"):
        validate_micro_steps(["not a dict"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="JSON array"):
        validate_micro_steps({"not": "a list"})  # type: ignore[arg-type]
