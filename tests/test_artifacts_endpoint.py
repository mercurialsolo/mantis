"""Integration tests for GET /v1/runs/{run_id}/artifacts/{name} (#508).

Mirrors ``test_video_endpoint.py`` — drop files into the per-tenant run
dir and assert the endpoint serves them (or 404s / 400s appropriately).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient

from mantis_agent import tenant_auth as ta_mod


@pytest.fixture
def client(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MANTIS_API_TOKEN", "test-token")
    monkeypatch.delenv("MANTIS_TENANT_KEYS_PATH", raising=False)
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "mantis-data"))
    ta_mod.reset_key_store()

    from mantis_agent import baseten_server as bs
    importlib.reload(bs)
    return TestClient(bs.app), tmp_path / "mantis-data"


def _run_dir(data_root: Path, tenant_id: str, run_id: str) -> Path:
    d = data_root / "tenants" / tenant_id / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_artifact_404_when_no_file(client):
    cli, _data = client
    r = cli.get(
        "/v1/runs/run_abc/artifacts/leads.csv",
        headers={"X-Mantis-Token": "test-token"},
    )
    assert r.status_code == 404
    assert "not available" in r.json()["detail"].lower()


def test_artifact_serves_extracted_rows_csv(client):
    cli, data = client
    rd = _run_dir(data, "default", "run_xyz")
    (rd / "extracted_rows.csv").write_text("title,url\nML,https://a\n")

    r = cli.get(
        "/v1/runs/run_xyz/artifacts/extracted_rows.csv",
        headers={"X-Mantis-Token": "test-token"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/csv; charset=utf-8"
    assert "title,url" in r.text
    assert "ML,https://a" in r.text


def test_artifact_serves_extracted_rows_json(client):
    cli, data = client
    rd = _run_dir(data, "default", "run_json")
    payload = [{"title": "ML", "url": "https://a"}]
    (rd / "extracted_rows.json").write_text(json.dumps(payload))

    r = cli.get(
        "/v1/runs/run_json/artifacts/extracted_rows.json",
        headers={"X-Mantis-Token": "test-token"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == payload


def test_artifact_serves_leads_csv(client):
    cli, data = client
    rd = _run_dir(data, "default", "run_leads")
    (rd / "leads.csv").write_text("status,url\nVIABLE,https://a\n")
    r = cli.get(
        "/v1/runs/run_leads/artifacts/leads.csv",
        headers={"X-Mantis-Token": "test-token"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/csv; charset=utf-8"


def test_artifact_404_for_unknown_name(client):
    """Allowlist guard: requesting a file that isn't in the allowlist
    returns 404 even if a same-named file happens to exist on disk."""
    cli, data = client
    rd = _run_dir(data, "default", "run_unk")
    (rd / "secret.env").write_text("API_KEY=hunter2")

    r = cli.get(
        "/v1/runs/run_unk/artifacts/secret.env",
        headers={"X-Mantis-Token": "test-token"},
    )
    assert r.status_code == 404
    assert "unknown artifact" in r.json()["detail"].lower()


def test_artifact_requires_auth(client):
    cli, _data = client
    r = cli.get("/v1/runs/anything/artifacts/leads.csv")
    assert r.status_code == 401
