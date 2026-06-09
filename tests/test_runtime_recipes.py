"""Tests for tenant-scoped runtime recipe persistence + HTTP CRUD (#809).

Exercises:

- ``mantis_agent.recipes.runtime_store`` — register / get / list /
  delete + name validation + payload validation.
- ``mantis_agent.recipes.load_schema`` precedence — tenant runtime
  wins over code-shipped when ``tenant_id`` supplied.
- ``POST /v1/recipes`` and friends mounted on modal_cua_server —
  round-trip + tenant isolation + auth + 404 + 400.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from mantis_agent import tenant_auth as ta_mod


# ── Pure store-level tests (no Modal dependency) ────────────────────


def _hn_payload() -> dict:
    return {
        "entity_name": "story",
        "fields": [
            {"name": "rank", "type": "str", "required": True},
            {"name": "title", "type": "str", "required": True},
            {"name": "url", "type": "str", "required": True},
        ],
        "required_fields": ["rank", "title", "url"],
    }


def test_register_round_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    from mantis_agent.recipes import runtime_store

    persisted = runtime_store.register("tenantA", "hn_top", _hn_payload())
    assert persisted["name"] == "hn_top"
    fetched = runtime_store.get("tenantA", "hn_top")
    assert fetched is not None
    assert fetched["schema"]["entity_name"] == "story"


def test_register_validates_schema_payload(monkeypatch, tmp_path):
    """Pydantic-style validation at write time, not at extract time."""
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    from mantis_agent.recipes import runtime_store

    with pytest.raises(runtime_store.RuntimeRecipeError):
        # Missing ``fields`` (required by ExtractionSchema.from_dict).
        runtime_store.register("tenantA", "broken", {"entity_name": "x"})


def test_register_rejects_invalid_names(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    from mantis_agent.recipes import runtime_store

    bad_names = ["", "../escape", "name/slash", "with spaces", "with.dot", "x" * 100]
    for name in bad_names:
        with pytest.raises(runtime_store.RuntimeRecipeError):
            runtime_store.register("tenantA", name, _hn_payload())


def test_register_accepts_slug_style_names(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    from mantis_agent.recipes import runtime_store

    for name in ["hn_top", "hn-top", "my_recipe_123", "ABC"]:
        runtime_store.register("tenantA", name, _hn_payload())


def test_list_recipes(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    from mantis_agent.recipes import runtime_store

    runtime_store.register("tenantA", "a", _hn_payload())
    runtime_store.register("tenantA", "b", _hn_payload())
    runtime_store.register("tenantA", "c", _hn_payload())
    listed = runtime_store.list_recipes("tenantA")
    assert sorted(r["name"] for r in listed) == ["a", "b", "c"]


def test_list_empty_for_fresh_tenant(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    from mantis_agent.recipes import runtime_store

    assert runtime_store.list_recipes("ghost") == []


def test_tenant_isolation(monkeypatch, tmp_path):
    """A recipe registered under tenant A must not be visible to tenant B."""
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    from mantis_agent.recipes import runtime_store

    runtime_store.register("tenantA", "shared_name", _hn_payload())
    assert runtime_store.get("tenantB", "shared_name") is None
    assert runtime_store.list_recipes("tenantB") == []


def test_delete_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    from mantis_agent.recipes import runtime_store

    runtime_store.register("tenantA", "x", _hn_payload())
    assert runtime_store.delete("tenantA", "x") is True
    assert runtime_store.delete("tenantA", "x") is False  # already gone — no error
    assert runtime_store.get("tenantA", "x") is None


def test_load_schema_returns_extraction_schema(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    from mantis_agent.extraction import ExtractionSchema
    from mantis_agent.recipes import runtime_store

    runtime_store.register("tenantA", "hn_top", _hn_payload())
    schema = runtime_store.load_schema("tenantA", "hn_top")
    assert isinstance(schema, ExtractionSchema)
    assert schema.entity_name == "story"
    assert "rank" in schema.field_names()


# ── load_schema precedence: runtime wins ────────────────────────────


def test_package_load_schema_falls_back_to_static_without_tenant(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    from mantis_agent import recipes

    schema = recipes.load_schema("marketplace_listings")
    assert schema.entity_name  # any populated value — proves code-shipped lookup wins


def test_package_load_schema_prefers_runtime_when_tenant_supplied(monkeypatch, tmp_path):
    """Runtime recipe with the same name as a code-shipped one MUST
    override per tenant — that's the whole point of runtime recipes."""
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    from mantis_agent import recipes
    from mantis_agent.recipes import runtime_store

    # Use the same name as a code-shipped recipe and check that we get
    # back the runtime version's entity_name (not the marketplace one).
    runtime_store.register(
        "tenantA",
        "marketplace_listings",
        {**_hn_payload(), "entity_name": "RUNTIME_OVERRIDE"},
    )
    schema = recipes.load_schema("marketplace_listings", tenant_id="tenantA")
    assert schema.entity_name == "RUNTIME_OVERRIDE"


def test_package_load_schema_falls_through_when_no_runtime(monkeypatch, tmp_path):
    """tenant_id supplied but no runtime recipe → fall through to code-shipped."""
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    from mantis_agent import recipes

    schema = recipes.load_schema("marketplace_listings", tenant_id="tenantA")
    # marketplace_listings ships a real schema — entity_name is populated.
    assert schema.entity_name


def test_package_load_schema_raises_when_neither_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    from mantis_agent import recipes

    with pytest.raises(ModuleNotFoundError):
        recipes.load_schema("does_not_exist", tenant_id="tenantA")


# ── HTTP CRUD via TestClient ────────────────────────────────────────

pytest.importorskip("modal")

from fastapi.testclient import TestClient  # noqa: E402 — guarded by importorskip above


_DEPLOY_MODAL = Path(__file__).resolve().parent.parent / "deploy" / "modal"
if str(_DEPLOY_MODAL) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_MODAL))


class _StubFunctionCall:
    def __init__(self) -> None:
        self.object_id = "fc-stub-recipes"

    def get(self, timeout: float = 0.1):
        return {}

    def cancel(self) -> None:
        return None


class _StubExecutor:
    def __init__(self, call: _StubFunctionCall) -> None:
        self.call = call
        self.spawn_kwargs: dict = {}

    def spawn(self, *, task_file_contents: str, **kwargs):
        self.spawn_kwargs = {"task_file_contents": task_file_contents, **kwargs}
        return self.call


@pytest.fixture
def http_ctx(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_API_TOKEN", "test-token")
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "mantis-data"))
    monkeypatch.delenv("MANTIS_TENANT_KEYS_PATH", raising=False)
    ta_mod.reset_key_store()

    import modal_cua_server as mcs  # type: ignore[import-not-found]
    importlib.reload(mcs)

    stub_call = _StubFunctionCall()
    stub_executor = _StubExecutor(stub_call)
    app = mcs.build_api_app(
        executor_resolver=lambda model: stub_executor,
        function_call_lookup=lambda call_id: stub_call,
    )
    return TestClient(app)


def _headers() -> dict:
    return {"X-Mantis-Token": "test-token", "Content-Type": "application/json"}


def test_http_register_round_trips(http_ctx):
    r = http_ctx.post(
        "/v1/recipes",
        headers=_headers(),
        json={"name": "hn_top", "schema": _hn_payload()},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "hn_top"
    assert body["schema"]["entity_name"] == "story"


def test_http_register_rejects_malformed_schema(http_ctx):
    r = http_ctx.post(
        "/v1/recipes",
        headers=_headers(),
        json={"name": "bad", "schema": {"entity_name": "x"}},  # missing fields
    )
    assert r.status_code == 400


def test_http_register_rejects_invalid_name(http_ctx):
    r = http_ctx.post(
        "/v1/recipes",
        headers=_headers(),
        json={"name": "../escape", "schema": _hn_payload()},
    )
    assert r.status_code == 400


def test_http_list_recipes(http_ctx):
    http_ctx.post(
        "/v1/recipes",
        headers=_headers(),
        json={"name": "a", "schema": _hn_payload()},
    )
    http_ctx.post(
        "/v1/recipes",
        headers=_headers(),
        json={"name": "b", "schema": _hn_payload()},
    )
    r = http_ctx.get("/v1/recipes", headers=_headers())
    assert r.status_code == 200
    body = r.json()
    assert sorted(rec["name"] for rec in body["recipes"]) == ["a", "b"]


def test_http_get_recipe(http_ctx):
    http_ctx.post(
        "/v1/recipes",
        headers=_headers(),
        json={"name": "hn_top", "schema": _hn_payload()},
    )
    r = http_ctx.get("/v1/recipes/hn_top", headers=_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "hn_top"
    assert "rank" in [f["name"] for f in body["schema"]["fields"]]


def test_http_get_unknown_recipe_404(http_ctx):
    r = http_ctx.get("/v1/recipes/missing", headers=_headers())
    assert r.status_code == 404


def test_http_delete_recipe(http_ctx):
    http_ctx.post(
        "/v1/recipes",
        headers=_headers(),
        json={"name": "trash", "schema": _hn_payload()},
    )
    r = http_ctx.delete("/v1/recipes/trash", headers=_headers())
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    # And follow-up get → 404
    r2 = http_ctx.get("/v1/recipes/trash", headers=_headers())
    assert r2.status_code == 404


def test_http_delete_idempotent(http_ctx):
    """Deleting a missing recipe returns 200 with deleted=false — not 404 —
    so client-side cleanup loops don't have to special-case the not-found
    response."""
    r = http_ctx.delete("/v1/recipes/never_existed", headers=_headers())
    assert r.status_code == 200
    assert r.json()["deleted"] is False


def test_http_routes_require_auth(http_ctx):
    for method, path in [
        ("POST", "/v1/recipes"),
        ("GET", "/v1/recipes"),
        ("GET", "/v1/recipes/x"),
        ("DELETE", "/v1/recipes/x"),
    ]:
        r = http_ctx.request(method, path, json={"name": "x", "schema": _hn_payload()})
        assert r.status_code in {401, 403}, f"{method} {path} returned {r.status_code}"
