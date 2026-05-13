"""End-to-end smoke against the FastAPI app via httpx TestClient.

Covers:

* Boot — startup hook seeds the DB; counts match the spec.
* Harness gating — ``/__env__/*`` 401s without ``X-Env-Admin``.
* Health is open.
* The five agent-facing surfaces all render 200 on a happy path.
* T05 oracle round-trip — post a clean reply, oracle passes.

We do not boot uvicorn — TestClient drives the ASGI app in-process,
which is what FastAPI's tests do and what CI can run without Docker.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("jinja2")
pytest.importorskip("multipart")


@pytest.fixture
def client():
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import app  # noqa: PLC0415

    with TestClient(app) as c:
        yield c


def _admin_headers():
    return {"X-Env-Admin": "test-admin-token"}


# ── harness gating ─────────────────────────────────────────────────────


@pytest.mark.parametrize("path", [
    "/__env__/state",
    "/__env__/events",
    "/__env__/oracle?task_id=T01_triage_inbox",
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


def test_admin_oracle_with_token(client):
    r = client.get("/__env__/oracle?task_id=T01_triage_inbox",
                   headers=_admin_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["task_id"] == "T01_triage_inbox"
    assert "passed" in body and "score" in body


# ── agent-facing surfaces ──────────────────────────────────────────────


def test_root_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "mantis helpdesk" in r.text


def test_tickets_list_renders(client):
    r = client.get("/tickets")
    assert r.status_code == 200
    assert "ticket_00001" in r.text


def test_ticket_detail_renders(client):
    r = client.get("/tickets/ticket_00001")
    assert r.status_code == 200


def test_macros_list_renders(client):
    r = client.get("/macros")
    assert r.status_code == 200
    assert "macro_shipping_delay" in r.text


def test_macro_detail_renders(client):
    r = client.get("/macros/macro_shipping_delay")
    assert r.status_code == 200
    assert "Shipping delay" in r.text


def test_triggers_render(client):
    r = client.get("/triggers")
    assert r.status_code == 200
    assert "trigger_billing_lock" in r.text


def test_reports_render(client):
    r = client.get("/reports")
    assert r.status_code == 200
    assert "Open tickets by group" in r.text


def test_search_renders(client):
    r = client.get("/search?q=outage")
    assert r.status_code == 200
    assert "outage" in r.text.lower()


# ── T05 round-trip: clean public reply → oracle passes ──────────────


def test_t05_oracle_passes_after_clean_reply(client):
    r = client.post(
        "/tickets/ticket_04421/reply",
        data={
            "body": "Hi — we've processed the refund and it will land by Friday.",
            "visibility": "public",
        },
        follow_redirects=False,
    )
    assert r.status_code in (200, 303)

    r = client.get("/__env__/oracle?task_id=T05_redact_and_reply",
                   headers=_admin_headers())
    body = r.json()
    assert body["passed"], body["reasons"]
    assert body["score"] == 1.0


def test_t05_oracle_fails_when_reply_leaks_pii(client):
    r = client.post(
        "/tickets/ticket_04421/reply",
        data={
            "body": "Refund for SSN 123-45-6789 on card 4242 4242 4242 4242 issued.",
            "visibility": "public",
        },
        follow_redirects=False,
    )
    assert r.status_code in (200, 303)

    r = client.get("/__env__/oracle?task_id=T05_redact_and_reply",
                   headers=_admin_headers())
    body = r.json()
    assert body["passed"] is False
    assert body["score"] == 0.0


# ── internal-only thread rejection ─────────────────────────────────────


def test_internal_only_ticket_rejects_public_reply(client):
    """Posting a public reply on an internal-only ticket writes a
    rejection audit row and does NOT add a public reply."""
    r = client.post(
        "/tickets/ticket_04422/reply",
        data={"body": "trying to leak this publicly", "visibility": "public"},
        follow_redirects=False,
    )
    assert r.status_code in (200, 303)
    state = client.get("/__env__/state", headers=_admin_headers()).json()
    ops = [m["operation"] for m in state["recent_mutations"]]
    assert "reply_rejected_internal_only" in ops


# ── reset clears mutations ─────────────────────────────────────────────


def test_reset_clears_mutations(client):
    client.post(
        "/tickets/ticket_00001/tag",
        data={"tag": "manual-test"},
        follow_redirects=False,
    )
    state = client.get("/__env__/state", headers=_admin_headers()).json()
    assert state["counts"]["mutations"] >= 1

    r = client.post("/__env__/reset", headers=_admin_headers())
    assert r.status_code == 200

    state = client.get("/__env__/state", headers=_admin_headers()).json()
    assert state["counts"]["mutations"] == 0
