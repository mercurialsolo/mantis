"""Tests for the fingerprint diagnostic endpoint (#827).

Submits a synthetic bot-detection diagnostic plan against a public
fingerprint test page. The route returns a detached-run envelope plus
a stealth-config snapshot so operators can correlate the scorecard
they see with the env flags that were active when the run was
submitted.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("modal")

from fastapi.testclient import TestClient  # noqa: E402 — guarded by importorskip

from mantis_agent import tenant_auth as ta_mod


_DEPLOY_MODAL = Path(__file__).resolve().parent.parent / "deploy" / "modal"
if str(_DEPLOY_MODAL) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_MODAL))


class _StubCall:
    object_id = "fc-stub-fp"

    def get(self, timeout: float = 0.1):
        return {}

    def cancel(self) -> None:
        return None


class _StubExecutor:
    def __init__(self) -> None:
        self.call = _StubCall()
        self.spawn_kwargs: dict = {}

    def spawn(self, *, task_file_contents: str, **kwargs):
        self.spawn_kwargs = {"task_file_contents": task_file_contents, **kwargs}
        return self.call


@pytest.fixture
def app_ctx(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_API_TOKEN", "test-token")
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "mantis-data"))
    monkeypatch.delenv("MANTIS_TENANT_KEYS_PATH", raising=False)
    ta_mod.reset_key_store()

    import modal_cua_server as mcs  # type: ignore[import-not-found]
    importlib.reload(mcs)

    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor.call,
    )
    return TestClient(app), executor


def _h() -> dict:
    return {"X-Mantis-Token": "test-token", "Content-Type": "application/json"}


# ── Happy path ─────────────────────────────────────────────────────


def test_diagnose_returns_run_envelope(app_ctx):
    client, _ex = app_ctx
    r = client.post("/v1/diagnose/fingerprint", headers=_h(), json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "queued"
    assert body["run_id"]
    assert body["target_url"] == "https://bot.sannysoft.com/"
    assert body["poll_via"] == f"/v1/runs/{body['run_id']}"
    assert body["rows_via"] == f"/v1/runs/{body['run_id']}/artifacts/extracted_rows.json"


def test_diagnose_accepts_custom_target_url(app_ctx):
    client, _ex = app_ctx
    r = client.post(
        "/v1/diagnose/fingerprint",
        headers=_h(),
        json={"target_url": "https://abrahamjuliot.github.io/creepjs/"},
    )
    assert r.status_code == 200
    assert r.json()["target_url"] == "https://abrahamjuliot.github.io/creepjs/"


def test_diagnose_rejects_non_http_url(app_ctx):
    client, _ex = app_ctx
    r = client.post(
        "/v1/diagnose/fingerprint",
        headers=_h(),
        json={"target_url": "file:///etc/passwd"},
    )
    assert r.status_code == 400


def test_diagnose_rejects_invalid_cua_model(app_ctx):
    client, _ex = app_ctx
    r = client.post(
        "/v1/diagnose/fingerprint",
        headers=_h(),
        json={"cua_model": "not-a-model"},
    )
    assert r.status_code == 400


# ── Stealth snapshot ───────────────────────────────────────────────


def test_diagnose_returns_stealth_snapshot(app_ctx):
    """The response carries which stealth flags were active when the
    run was submitted — operators correlate the scorecard with the
    flags so before/after comparison is unambiguous."""
    client, _ex = app_ctx
    r = client.post("/v1/diagnose/fingerprint", headers=_h(), json={})
    body = r.json()
    snap = body["stealth_snapshot"]
    assert isinstance(snap, dict)
    for key in (
        "honest_mode", "behavioral_jitter", "geo_consistency",
        "cdp_stealth", "proxy_provider",
    ):
        assert key in snap


def test_diagnose_snapshot_reflects_env_overrides(monkeypatch, app_ctx):
    client, _ex = app_ctx
    monkeypatch.setenv("MANTIS_BEHAVIORAL_JITTER", "0")
    monkeypatch.setenv("MANTIS_CDP_STEALTH", "0")
    r = client.post("/v1/diagnose/fingerprint", headers=_h(), json={})
    snap = r.json()["stealth_snapshot"]
    assert snap["behavioral_jitter"] is False
    assert snap["cdp_stealth"] is False


# ── Executor wiring ───────────────────────────────────────────────


def test_diagnose_spawns_executor_with_fingerprint_plan(app_ctx):
    """The synthesized plan should include the configured target URL
    and use ``extract_data`` with ``max_items > 1`` (multi-row branch
    from PR #820)."""
    import json

    client, executor = app_ctx
    r = client.post(
        "/v1/diagnose/fingerprint",
        headers=_h(),
        json={"target_url": "https://bot.sannysoft.com/"},
    )
    assert r.status_code == 200
    suite = json.loads(executor.spawn_kwargs["task_file_contents"])
    micro_plan = suite["_micro_plan"]
    # Two steps: navigate + extract_data.
    assert len(micro_plan) == 2
    assert micro_plan[0]["type"] == "navigate"
    assert micro_plan[0]["params"]["url"] == "https://bot.sannysoft.com/"
    assert micro_plan[1]["type"] == "extract_data"
    # Multi-row branch from #820.
    assert micro_plan[1]["extract"]["max_items"] >= 30


# ── Auth ───────────────────────────────────────────────────────────


def test_diagnose_requires_auth(app_ctx):
    client, _ex = app_ctx
    r = client.post("/v1/diagnose/fingerprint", json={})
    assert r.status_code in {401, 403}
