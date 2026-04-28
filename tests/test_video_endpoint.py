"""Integration tests for GET /v1/runs/{run_id}/video.

We don't run a real recording; we drop a fake file on disk in the
expected per-tenant location and assert the endpoint serves it (and
returns 404 when missing or for the wrong tenant).
"""

from __future__ import annotations

import importlib
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


def _drop_recording(data_root: Path, tenant_id: str, run_id: str, fmt: str = "mp4") -> Path:
    runs_dir = data_root / "tenants" / tenant_id / "runs" / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)
    rec = runs_dir / f"recording.{fmt}"
    rec.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 256)  # plausible mp4 header
    return rec


def test_video_404_when_no_recording(client):
    cli, _data = client
    r = cli.get(
        "/v1/runs/20260428_abc/video",
        headers={"X-Mantis-Token": "test-token"},
    )
    assert r.status_code == 404
    assert "no recording" in r.json()["detail"].lower()


def test_video_serves_file_when_present(client):
    cli, data = client
    rec = _drop_recording(data, tenant_id="default", run_id="20260428_abc")
    r = cli.get(
        "/v1/runs/20260428_abc/video",
        headers={"X-Mantis-Token": "test-token"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert r.content == rec.read_bytes()


def test_video_picks_format_by_priority(client):
    cli, data = client
    # mp4 + webm both exist; mp4 wins (priority order)
    rec_mp4 = _drop_recording(data, "default", "20260428_xyz", fmt="mp4")
    rec_webm = _drop_recording(data, "default", "20260428_xyz", fmt="webm")
    r = cli.get(
        "/v1/runs/20260428_xyz/video",
        headers={"X-Mantis-Token": "test-token"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert r.content == rec_mp4.read_bytes()
    assert rec_webm.exists()  # untouched


def test_video_requires_auth(client):
    cli, _data = client
    r = cli.get("/v1/runs/anything/video")
    assert r.status_code == 401
