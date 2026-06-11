"""Tests for the Augur runtime surface — gap-closing routes (#840+).

Two ergonomic gaps in Augur observability surfaced during the audit:

1. The Mantis ``run_id`` → Augur ``run_id`` mapping isn't visible
   anywhere user-facing. Operators with a Mantis run_id can't easily
   find the corresponding Augur session in the workspace.

2. The Augur bundle lives at ``/data/augur/<augur_run_id>/`` but no
   HTTP route on the Mantis API serves it. To inspect a bundle you
   had to either SSH into the Modal volume or open the Augur SDK
   workspace.

This PR adds:

- A side-channel ``augur.json`` written by the executor, mapping
  ``api_run_id`` → ``augur_run_id``.
- ``augur_run_id`` + ``augur_bundle_url`` surfaced on the cheap-poll
  lifecycle response (``GET /v1/runs/{id}``).
- ``GET /v1/runs/{id}/augur`` — envelope: augur_run_id, bundle_dir,
  list of fetchable files, workspace URL.
- ``GET /v1/runs/{id}/augur/files/{path}`` — streams a specific
  bundle file. Allowlist on extension (``.json``, ``.jsonl``,
  ``.png``); path-traversal guarded.
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


@pytest.fixture
def mcs(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_API_TOKEN", "test-token")
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "mantis-data"))
    monkeypatch.delenv("MANTIS_TENANT_KEYS_PATH", raising=False)
    monkeypatch.setenv("AUGUR_DSN_WORKSPACE_URL", "https://augur.example.com/runs")
    ta_mod.reset_key_store()

    import modal_cua_server as mod  # type: ignore[import-not-found]
    importlib.reload(mod)
    from mantis_agent.run_state_store import RunStateStore
    mod._RUN_STATE_STORE = RunStateStore(backing={})
    mod._RUN_STATE_STORE_INIT = True
    return mod


def _h() -> dict:
    return {"X-Mantis-Token": "test-token", "Content-Type": "application/json"}


class _StubCall:
    object_id = "fc-stub"

    def get(self, timeout: float = 0.1):
        return {}

    def cancel(self) -> None:
        return None


class _StubExecutor:
    def spawn(self, *, task_file_contents, **kwargs):
        return _StubCall()


# ── side-channel helpers ──────────────────────────────────────────


def test_write_then_read_augur_metadata(mcs):
    mcs._write_augur_metadata("default", "r1", "augur-r1-abc123")
    got = mcs._read_augur_metadata("default", "r1")
    assert got is not None
    assert got["augur_run_id"] == "augur-r1-abc123"
    # Bundle dir derives under MANTIS_DATA_DIR/augur/<augur_run_id>.
    assert "augur" in got["bundle_dir"]
    assert "augur-r1-abc123" in got["bundle_dir"]
    # Workspace url surfaced from env.
    assert got["dsn_workspace"] == "https://augur.example.com/runs"


def test_write_skips_empty_augur_run_id(mcs):
    """Empty id should be a no-op — never write garbage metadata."""
    mcs._write_augur_metadata("default", "r1", "")
    assert mcs._read_augur_metadata("default", "r1") is None


def test_read_metadata_returns_none_for_unknown_run(mcs):
    assert mcs._read_augur_metadata("default", "never-submitted") is None


def test_bundle_dir_honors_data_dir_env(mcs, monkeypatch, tmp_path):
    """The bundle path must use ``MANTIS_DATA_DIR/augur/<id>`` —
    parity with ``observability.augur.default_out_dir`` so the API
    and the runner agree on where to look."""
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    importlib.reload(mcs)
    p = mcs._augur_bundle_dir("augur-xyz")
    assert str(p) == str(tmp_path / "augur" / "augur-xyz")


def test_bundle_dir_honors_explicit_override(mcs, monkeypatch, tmp_path):
    """``MANTIS_AUGUR_DIR`` wins over ``MANTIS_DATA_DIR`` for the
    Augur-specific path, matching the runner-side helper."""
    monkeypatch.setenv("MANTIS_AUGUR_DIR", str(tmp_path / "custom"))
    importlib.reload(mcs)
    p = mcs._augur_bundle_dir("augur-xyz")
    assert str(p) == str(tmp_path / "custom" / "augur-xyz")


# ── Lifecycle response surfaces augur metadata (gap #1) ───────────


def test_lifecycle_response_includes_augur_run_id(mcs):
    """``GET /v1/runs/{id}`` returns ``augur_run_id`` + bundle URL when
    the executor has stamped the metadata."""
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)

    # Seed status + augur metadata for the run.
    mcs._write_status("default", "r1", {
        "run_id": "r1", "status": "running",
        "updated_at": "2026-06-09T20:00:00+00:00",
    })
    mcs._write_augur_metadata("default", "r1", "augur-r1-abc123")

    r = client.get("/v1/runs/r1", headers=_h())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["augur_run_id"] == "augur-r1-abc123"
    assert body["augur_bundle_url"] == "/v1/runs/r1/augur"


def test_lifecycle_response_omits_augur_when_missing(mcs):
    """No augur metadata stamped → lifecycle response doesn't carry
    the keys (rather than emitting empty strings)."""
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)
    mcs._write_status("default", "r-noaugur", {
        "run_id": "r-noaugur", "status": "running",
        "updated_at": "2026-06-09T20:00:00+00:00",
    })
    r = client.get("/v1/runs/r-noaugur", headers=_h())
    assert r.status_code == 200
    body = r.json()
    assert "augur_run_id" not in body
    assert "augur_bundle_url" not in body


# ── Bundle envelope route (gap #2) ────────────────────────────────


def test_augur_envelope_returns_metadata_and_files(mcs, tmp_path):
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)

    mcs._write_augur_metadata("default", "r-bundle", "augur-bundle-xyz")
    # Materialise a small fake bundle.
    bundle = mcs._augur_bundle_dir("augur-bundle-xyz")
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "manifest.json").write_text('{"schema_version": "1"}')
    (bundle / "trace.json").write_text('{"steps": []}')
    (bundle / "events").mkdir(exist_ok=True)
    (bundle / "events" / "0001.jsonl").write_text('{"ts": "x", "kind": "step"}\n')

    r = client.get("/v1/runs/r-bundle/augur", headers=_h())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["augur_run_id"] == "augur-bundle-xyz"
    assert body["bundle_present"] is True
    names = sorted(f["name"] for f in body["files"])
    assert "manifest.json" in names
    assert "trace.json" in names
    assert any(n.endswith("0001.jsonl") for n in names)


def test_augur_envelope_404_when_no_metadata(mcs):
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)
    r = client.get("/v1/runs/never/augur", headers=_h())
    assert r.status_code == 404


def test_augur_envelope_bundle_absent_but_metadata_present(mcs):
    """Metadata stamped but bundle dir doesn't exist yet (race or
    DebugSession opt-out) → envelope still 200s with bundle_present=False."""
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)
    mcs._write_augur_metadata("default", "r-no-bundle", "augur-no-bundle")

    r = client.get("/v1/runs/r-no-bundle/augur", headers=_h())
    assert r.status_code == 200
    body = r.json()
    assert body["augur_run_id"] == "augur-no-bundle"
    assert body["bundle_present"] is False
    assert body["files"] == []


# ── File-fetch route ──────────────────────────────────────────────


def test_augur_file_fetch_serves_manifest(mcs):
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)

    mcs._write_augur_metadata("default", "r-fetch", "augur-fetch-id")
    bundle = mcs._augur_bundle_dir("augur-fetch-id")
    bundle.mkdir(parents=True, exist_ok=True)
    payload = '{"schema_version": "1", "run_id": "fetch"}'
    (bundle / "manifest.json").write_text(payload)

    r = client.get("/v1/runs/r-fetch/augur/files/manifest.json", headers=_h())
    assert r.status_code == 200
    assert r.text == payload
    assert r.headers["content-type"].startswith("application/json")


def test_augur_file_fetch_serves_nested_path(mcs):
    """Nested files like ``events/0001.jsonl`` round-trip through the
    path matcher correctly."""
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)
    mcs._write_augur_metadata("default", "r-nest", "augur-nest")
    bundle = mcs._augur_bundle_dir("augur-nest")
    (bundle / "events").mkdir(parents=True, exist_ok=True)
    (bundle / "events" / "0001.jsonl").write_text(
        '{"ts": "1", "kind": "step"}\n'
    )

    r = client.get(
        "/v1/runs/r-nest/augur/files/events/0001.jsonl", headers=_h(),
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/jsonl")
    assert "kind" in r.text


def test_augur_file_fetch_blocks_path_traversal(mcs, tmp_path):
    """``..`` segments must NOT escape the bundle dir."""
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)
    mcs._write_augur_metadata("default", "r-trav", "augur-trav")
    bundle = mcs._augur_bundle_dir("augur-trav")
    bundle.mkdir(parents=True, exist_ok=True)
    # Plant a sibling file outside the bundle.
    parent = bundle.parent
    secret = parent / "secret.txt"
    secret.write_text("you should not see this")

    r = client.get(
        "/v1/runs/r-trav/augur/files/../secret.txt", headers=_h(),
    )
    # Either 400 (caught by our guard) or 404 (FastAPI route resolution
    # normalised the path before the handler ran). Both are acceptable;
    # what's NOT acceptable is 200 + the secret content.
    assert r.status_code in {400, 404}
    if r.status_code == 200:
        assert "you should not see this" not in r.text


def test_augur_file_fetch_rejects_disallowed_extension(mcs):
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)
    mcs._write_augur_metadata("default", "r-ext", "augur-ext")
    bundle = mcs._augur_bundle_dir("augur-ext")
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "weird.exe").write_bytes(b"\x00\x00")

    r = client.get(
        "/v1/runs/r-ext/augur/files/weird.exe", headers=_h(),
    )
    assert r.status_code == 400


def test_augur_file_fetch_404_for_missing_file(mcs):
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)
    mcs._write_augur_metadata("default", "r-miss", "augur-miss")
    bundle = mcs._augur_bundle_dir("augur-miss")
    bundle.mkdir(parents=True, exist_ok=True)
    r = client.get(
        "/v1/runs/r-miss/augur/files/manifest.json", headers=_h(),
    )
    assert r.status_code == 404


# ── Auth ───────────────────────────────────────────────────────────


def test_augur_envelope_requires_auth(mcs):
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)
    r = client.get("/v1/runs/r1/augur")
    assert r.status_code in {401, 403}


def test_augur_file_fetch_requires_auth(mcs):
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)
    r = client.get("/v1/runs/r1/augur/files/manifest.json")
    assert r.status_code in {401, 403}
