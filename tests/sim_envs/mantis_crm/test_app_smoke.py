"""End-to-end smoke against the FastAPI app via httpx TestClient.

Covers:

* Boot — startup hook seeds the DB; counts match the spec.
* Harness gating — ``/__env__/*`` 401s without ``X-Env-Admin``.
* Health is open.
* The five agent-facing surfaces all render 200 on a happy path.
* T01 oracle round-trip: bulk-tag the right contacts → oracle passes.

We do not boot uvicorn — TestClient drives the ASGI app in-process, which
is what FastAPI's tests do and what CI can run without Docker.
"""

from __future__ import annotations

import pytest


pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("jinja2")
pytest.importorskip("multipart")  # python-multipart is needed for Form parsing


@pytest.fixture
def client():
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import app, _BOOT_TIME  # noqa: PLC0415,F401

    with TestClient(app) as c:
        yield c


def _admin_headers():
    return {"X-Env-Admin": "test-admin-token"}


# ── harness gating ─────────────────────────────────────────────────────


@pytest.mark.parametrize("path", [
    "/__env__/state",
    "/__env__/events",
    "/__env__/oracle?task_id=T01_tag_reengage",
])
def test_admin_routes_401_without_token(client, path):
    r = client.get(path)
    assert r.status_code == 401


def test_health_is_open(client):
    r = client.get("/__env__/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["seed"] == 42


def test_admin_oracle_passes_with_token(client):
    r = client.get("/__env__/oracle?task_id=T01_tag_reengage",
                   headers=_admin_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["task_id"] == "T01_tag_reengage"
    assert "passed" in body and "score" in body


# ── agent-facing surfaces ──────────────────────────────────────────────


def test_root_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "mantis-crm" in r.text


def test_contacts_list_renders(client):
    r = client.get("/contacts")
    assert r.status_code == 200
    assert "contact_00001" in r.text


def test_contact_detail_renders(client):
    r = client.get("/contacts/contact_00001")
    assert r.status_code == 200
    assert "alice.lead@acme.com" in r.text


def test_companies_list_renders(client):
    r = client.get("/companies")
    assert r.status_code == 200
    assert "company_00001" in r.text


def test_deals_pipeline_renders(client):
    r = client.get("/deals")
    assert r.status_code == 200
    # All seven stages present as kanban columns
    for stage in ("Prospect", "Qualified", "Proposal", "Negotiation",
                  "At Risk", "Closed Won", "Closed Lost"):
        assert stage in r.text


def test_search_renders(client):
    r = client.get("/search?q=Acme")
    assert r.status_code == 200
    assert "Acme" in r.text


def test_reports_render(client):
    r = client.get("/reports")
    assert r.status_code == 200
    assert "Deals by stage" in r.text


# ── T01 round-trip: bulk-tag → oracle passes ──────────────────────────


def test_t01_oracle_passes_after_bulk_tag(client):
    """Synthesise the action: tag every qualifying contact with 'reengage'."""
    from app import db as env_db  # noqa: PLC0415
    from app.oracles.t01_tag_reengage import _target_contact_ids

    conn = env_db.connect()
    targets = sorted(_target_contact_ids(conn, now="2026-01-15T09:00:00Z"))
    assert targets, "seed should produce some qualifying contacts"

    # Bulk-tag via the agent-facing form.
    r = client.post(
        "/contacts/bulk/tag",
        data={"tag": "reengage", "ids": ",".join(targets)},
        follow_redirects=False,
    )
    assert r.status_code in (200, 303)

    # Oracle now passes.
    r = client.get("/__env__/oracle?task_id=T01_tag_reengage",
                   headers=_admin_headers())
    body = r.json()
    assert body["passed"], body["reasons"]
    assert body["score"] == 1.0


# ── reset round-trip ───────────────────────────────────────────────────


def test_reset_clears_mutations(client):
    # Touch a contact, then reset, then assert the mutation is gone.
    client.post(
        "/contacts/contact_00001/tag",
        data={"tag": "vip"},
        follow_redirects=False,
    )
    state = client.get("/__env__/state", headers=_admin_headers()).json()
    assert state["counts"]["mutations"] >= 1

    r = client.post("/__env__/reset", headers=_admin_headers())
    assert r.status_code == 200

    state = client.get("/__env__/state", headers=_admin_headers()).json()
    assert state["counts"]["mutations"] == 0
