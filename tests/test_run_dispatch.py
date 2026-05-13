"""Tests for the framework-agnostic dispatch helpers (#342).

The Modal HTTP endpoint and (eventually) the Baseten ASGI app both
prepare payloads through these helpers, so the dispatch logic stays
consistent across deployments. Tests use ``DEFAULT_TENANT`` to skip
the keyfile setup; allowlist and cap-clamp paths are exercised
explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mantis_agent.server.run_dispatch import (
    DispatchError,
    acquire_profile_lock,
    prepare_cua_payload,
    prepare_predict_payload,
    read_profile_lock,
    release_profile_lock,
)
from mantis_agent.tenant_auth import DEFAULT_TENANT, TenantConfig


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Each test gets its own MANTIS_DATA_DIR so file writes don't collide."""
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    yield


# ── Identity resolution (#341 wired into the dispatcher) ───────────


def test_prepare_predict_payload_routes_legacy_state_key_to_both() -> None:
    out = prepare_predict_payload(
        {"task_suite": {"tasks": []}, "state_key": "abc"},
        DEFAULT_TENANT,
    )
    assert out["profile_id"] == out["workflow_id"]
    assert out["profile_id"].endswith("__abc")
    # state_key tracks workflow_id for downstream consumers.
    assert out["state_key"] == out["workflow_id"]


def test_prepare_predict_payload_new_fields_decouple() -> None:
    out = prepare_predict_payload(
        {
            "task_suite": {"tasks": []},
            "profile_id": "alice",
            "workflow_id": "plan_v3",
        },
        DEFAULT_TENANT,
    )
    assert out["profile_id"].endswith("__alice")
    assert out["workflow_id"].endswith("__plan_v3")
    assert out["profile_id"] != out["workflow_id"]


def test_prepare_predict_payload_clamps_caps_to_tenant() -> None:
    tenant = TenantConfig(
        tenant_id="capped",
        max_cost_per_run=5.0,
        max_time_minutes_per_run=10,
    )
    out = prepare_predict_payload(
        {"task_suite": {"tasks": []}, "max_cost": 100.0, "max_time_minutes": 999},
        tenant,
    )
    assert out["max_cost"] == 5.0
    assert out["max_time_minutes"] == 10


def test_prepare_predict_payload_rejects_non_dict() -> None:
    with pytest.raises(DispatchError) as exc_info:
        prepare_predict_payload([], DEFAULT_TENANT)  # type: ignore[arg-type]
    assert exc_info.value.status_code == 400


def test_prepare_predict_payload_rejects_invalid_schema() -> None:
    # No plan-shape field provided — PredictRequest validator rejects.
    with pytest.raises(DispatchError) as exc_info:
        prepare_predict_payload({}, DEFAULT_TENANT)
    assert exc_info.value.status_code == 400


# ── Allowlist gate ─────────────────────────────────────────────────


def test_prepare_predict_payload_enforces_allowlist() -> None:
    tenant = TenantConfig(
        tenant_id="t1",
        allowed_domains=("example.com",),
    )
    bad = {
        "task_suite": {
            "base_url": "https://disallowed.io/x",
            "tasks": [{"intent": "x", "start_url": "https://disallowed.io/x"}],
        }
    }
    with pytest.raises(DispatchError) as exc_info:
        prepare_predict_payload(bad, tenant)
    assert exc_info.value.status_code == 403


def test_prepare_predict_payload_allows_listed_domain() -> None:
    tenant = TenantConfig(
        tenant_id="t1",
        allowed_domains=("example.com",),
    )
    ok = {
        "task_suite": {
            "base_url": "https://example.com/x",
            "tasks": [{"intent": "x", "start_url": "https://example.com/x"}],
        }
    }
    out = prepare_predict_payload(ok, tenant)
    assert out["task_suite"]["base_url"] == "https://example.com/x"


# ── Pure CUA flavor mirrors predict prep ───────────────────────────


def test_prepare_cua_payload_resolves_identity() -> None:
    out = prepare_cua_payload(
        {
            "instruction": "click foo",
            "start_url": "https://example.com",
            "profile_id": "alice",
            "workflow_id": "wf_v1",
        },
        DEFAULT_TENANT,
    )
    assert out["profile_id"].endswith("__alice")
    assert out["workflow_id"].endswith("__wf_v1")


def test_prepare_cua_payload_enforces_allowlist_on_start_url() -> None:
    tenant = TenantConfig(
        tenant_id="t1",
        allowed_domains=("example.com",),
    )
    with pytest.raises(DispatchError) as exc_info:
        prepare_cua_payload(
            {"instruction": "x", "start_url": "https://blocked.io"},
            tenant,
        )
    assert exc_info.value.status_code == 403


# ── Profile lock — concurrent-submission 409 (#342) ────────────────


def test_acquire_profile_lock_first_caller_wins() -> None:
    tenant = TenantConfig(tenant_id="t1")
    assert acquire_profile_lock(tenant, "alice", "run-1") is True
    assert read_profile_lock(tenant, "alice") == "run-1"


def test_acquire_profile_lock_second_caller_rejected() -> None:
    tenant = TenantConfig(tenant_id="t1")
    assert acquire_profile_lock(tenant, "alice", "run-1") is True
    assert acquire_profile_lock(tenant, "alice", "run-2") is False
    # The held run_id is exactly what the 409 surfaces.
    assert read_profile_lock(tenant, "alice") == "run-1"


def test_release_profile_lock_lets_next_run_proceed() -> None:
    tenant = TenantConfig(tenant_id="t1")
    assert acquire_profile_lock(tenant, "alice", "run-1") is True
    release_profile_lock(tenant, "alice")
    assert read_profile_lock(tenant, "alice") == ""
    assert acquire_profile_lock(tenant, "alice", "run-2") is True


def test_release_profile_lock_idempotent_on_missing_file() -> None:
    tenant = TenantConfig(tenant_id="t1")
    # No exception when the lock doesn't exist.
    release_profile_lock(tenant, "never-locked")


def test_acquire_profile_lock_stale_lock_is_overwritten(tmp_path, monkeypatch):
    """4h+ old lock files are treated as released — covers the crashed-executor case."""
    tenant = TenantConfig(tenant_id="t1")
    assert acquire_profile_lock(tenant, "alice", "run-1") is True

    # Backdate the lockfile to simulate a crash 5h ago.
    from mantis_agent.baseten_server.paths import tenant_lock_path
    lock_path = tenant_lock_path(tenant, "alice")
    old = lock_path.stat().st_mtime - 5 * 3600
    os.utime(lock_path, (old, old))

    # A fresh acquire on a stale lock succeeds.
    assert acquire_profile_lock(tenant, "alice", "run-2") is True
    assert read_profile_lock(tenant, "alice") == "run-2"


def test_profile_locks_are_independent_across_profiles() -> None:
    """Two distinct profile_ids are not serialized by each other's lock."""
    tenant = TenantConfig(tenant_id="t1")
    assert acquire_profile_lock(tenant, "alice", "run-1") is True
    # Different profile — should succeed.
    assert acquire_profile_lock(tenant, "bob", "run-2") is True
    assert read_profile_lock(tenant, "alice") == "run-1"
    assert read_profile_lock(tenant, "bob") == "run-2"


def test_profile_locks_are_independent_across_tenants() -> None:
    t1 = TenantConfig(tenant_id="t1")
    t2 = TenantConfig(tenant_id="t2")
    assert acquire_profile_lock(t1, "alice", "run-1") is True
    # Different tenant, same profile name — must not collide.
    assert acquire_profile_lock(t2, "alice", "run-2") is True
