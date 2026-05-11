"""Integration tests for the operator-facing surface added in #271 + #272.

Covered:

- ``GET /v1/version`` returns the runtime version + git/build env passthrough.
- ``GET /docs`` and ``GET /redoc`` are served by default, gated off by
  ``MANTIS_ENABLE_DOCS_UI=0`` for production tenant fleets that don't want
  the interactive UI exposed.
- ``GET /openapi.json`` is always served (FastAPI default) and reflects
  the title / version we configured on the FastAPI app.
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient

from mantis_agent import __version__ as MANTIS_VERSION
from mantis_agent import tenant_auth as ta_mod


def _build_client(monkeypatch, tmp_path):
    """Boot a fresh TestClient with single-tenant auth wired up.

    Reloading ``baseten_server`` (the package) alone is not enough — the
    package's ``__init__.py`` does ``from .routes import app``, which
    just re-fetches the existing binding from the already-cached
    ``routes`` submodule. ``_DOCS_UI_ENABLED`` is evaluated at
    ``routes`` import-time, so we must reload the ``routes`` submodule
    explicitly to make the test's ``MANTIS_ENABLE_DOCS_UI`` env override
    take effect. Without this, parallel CI workers that imported
    ``baseten_server.routes`` for a prior test see the FastAPI app
    constructed with the env unset (docs on by default) regardless of
    what the current test patches — passes locally on a clean worker
    but flakes in CI under pytest-xdist with hot module caches.
    """
    monkeypatch.setenv("MANTIS_API_TOKEN", "test-token")
    monkeypatch.delenv("MANTIS_TENANT_KEYS_PATH", raising=False)
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "mantis-data"))
    ta_mod.reset_key_store()

    from mantis_agent import baseten_server as bs
    from mantis_agent.baseten_server import routes as bs_routes

    importlib.reload(bs_routes)
    importlib.reload(bs)
    return TestClient(bs.app)


# ── /v1/version ──────────────────────────────────────────────────────────


def test_version_returns_runtime_snapshot(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path)
    resp = client.get("/v1/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == MANTIS_VERSION
    # ``model`` and ``ready`` come from the runtime singleton — booted in
    # TestClient land without a model load, so they're empty / False.
    assert "model" in body
    assert "ready" in body
    assert "git_sha" in body
    assert "build_time" in body


def test_version_includes_build_env_passthrough(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_GIT_SHA", "abc1234")
    monkeypatch.setenv("MANTIS_BUILD_TIME", "2026-05-11T20:00:00Z")
    client = _build_client(monkeypatch, tmp_path)
    body = client.get("/v1/version").json()
    assert body["git_sha"] == "abc1234"
    assert body["build_time"] == "2026-05-11T20:00:00Z"


def test_version_does_not_require_auth(monkeypatch, tmp_path):
    """``/v1/version`` is open like ``/health`` — no X-Mantis-Token needed."""
    client = _build_client(monkeypatch, tmp_path)
    resp = client.get("/v1/version")  # no headers
    assert resp.status_code == 200


# ── Swagger / Redoc / OpenAPI ────────────────────────────────────────────


def test_docs_ui_enabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("MANTIS_ENABLE_DOCS_UI", raising=False)
    client = _build_client(monkeypatch, tmp_path)
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200


@pytest.mark.parametrize("flag", ["0", "false", "no", "off"])
def test_docs_ui_disabled_by_env(monkeypatch, tmp_path, flag):
    monkeypatch.setenv("MANTIS_ENABLE_DOCS_UI", flag)
    client = _build_client(monkeypatch, tmp_path)
    # FastAPI returns 404 when the corresponding *_url is None.
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404


def test_openapi_json_always_served_and_reflects_metadata(
    monkeypatch, tmp_path,
):
    # Even with the UI disabled, the spec is still available — that's what
    # client SDKs and IDE plugins consume.
    monkeypatch.setenv("MANTIS_ENABLE_DOCS_UI", "0")
    client = _build_client(monkeypatch, tmp_path)
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    spec = resp.json()
    assert spec["info"]["title"] == "Mantis CUA"
    assert spec["info"]["version"] == MANTIS_VERSION
    # The new /v1/version endpoint should be visible in the spec.
    assert "/v1/version" in spec["paths"]
