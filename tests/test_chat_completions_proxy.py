"""Tests for the /v1/chat/completions reverse proxy.

Exercises auth + body forwarding + error pass-through against a mocked
upstream llama.cpp. Uses fastapi.testclient.TestClient so we don't need
a live container.
"""

from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient

from mantis_agent import tenant_auth as ta_mod


@pytest.fixture
def client(monkeypatch, tmp_path):
    """TestClient with single-tenant auth wired up.

    Avoids loading the real llama.cpp server / GPU model — the runtime
    object stays loaded=False, which is fine because /v1/chat/completions
    proxies to upstream rather than going through `runtime.run()`.
    """
    monkeypatch.setenv("MANTIS_API_TOKEN", "test-token")
    monkeypatch.setenv("MANTIS_LLAMA_PORT", "18080")
    monkeypatch.delenv("MANTIS_TENANT_KEYS_PATH", raising=False)
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "mantis-data"))
    ta_mod.reset_key_store()

    # Importing baseten_server runs module-level setup (logging config + a
    # FastAPI() construction). It also imports a couple of GymRunner / xdotool
    # bits that are happy to live in import-time without a display.
    from mantis_agent import baseten_server as bs

    importlib.reload(bs)
    return TestClient(bs.app)


def _mock_upstream_post(payload: dict, status: int = 200):
    """Return a MagicMock that mimics requests.Response."""
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = payload
    mock.text = json.dumps(payload)
    return mock


def test_proxy_forwards_to_upstream(client, monkeypatch):
    expected = {"id": "chatcmpl-x", "choices": [{"message": {"content": "hi"}}]}
    captured = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _mock_upstream_post(expected)

    with patch("mantis_agent.baseten_server.requests.post", side_effect=fake_post):
        r = client.post(
            "/v1/chat/completions",
            json={"model": "holo3", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Mantis-Token": "test-token"},
        )
    assert r.status_code == 200
    assert r.json() == expected
    assert captured["url"] == "http://127.0.0.1:18080/v1/chat/completions"
    assert captured["headers"]["Content-Type"] == "application/json"
    # Mantis-side auth must NOT be forwarded to llama.cpp
    assert "x-mantis-token" not in {k.lower() for k in captured["headers"]}
    assert "authorization" not in {k.lower() for k in captured["headers"]}


def test_proxy_requires_mantis_token(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": []},
    )
    assert r.status_code == 401
    assert "missing" in r.json()["detail"].lower()


def test_proxy_rejects_wrong_token(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": []},
        headers={"X-Mantis-Token": "wrong"},
    )
    assert r.status_code == 401
    assert "invalid" in r.json()["detail"].lower()


def test_proxy_passes_through_upstream_4xx(client):
    err_body = {"error": {"message": "bad request from llama.cpp"}}
    with patch(
        "mantis_agent.baseten_server.requests.post",
        return_value=_mock_upstream_post(err_body, status=400),
    ):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": []},
            headers={"X-Mantis-Token": "test-token"},
        )
    assert r.status_code == 400
    assert r.json() == err_body


def test_proxy_returns_502_on_upstream_connection_error(client):
    import requests as real_requests

    def raise_connection_error(*args, **kwargs):
        raise real_requests.ConnectionError("upstream not reachable")

    with patch(
        "mantis_agent.baseten_server.requests.post",
        side_effect=raise_connection_error,
    ):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": []},
            headers={"X-Mantis-Token": "test-token"},
        )
    assert r.status_code == 502
    assert "upstream" in r.json()["detail"].lower()


def test_proxy_handles_non_json_upstream_response(client):
    bad = MagicMock()
    bad.status_code = 500
    bad.json.side_effect = ValueError("not json")
    bad.text = "<html>500 internal server error</html>"
    with patch("mantis_agent.baseten_server.requests.post", return_value=bad):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": []},
            headers={"X-Mantis-Token": "test-token"},
        )
    assert r.status_code == 500
    assert "upstream_error" in r.text or "upstream" in r.text.lower()


def test_v1_models_lists_holo3(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert any("holo3" in i.lower() or i == "" for i in ids)


def test_health_endpoints_are_open(client):
    r1 = client.get("/health")
    r2 = client.get("/v1/health")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json() == r2.json()
