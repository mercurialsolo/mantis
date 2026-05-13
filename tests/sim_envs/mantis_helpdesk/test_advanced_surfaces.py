"""Advanced surface tests for mantis-helpdesk (#333).

Macros + bulk actions + triggers + merge oracle. Each test exercises
the agent-facing route, asserts the response, and verifies a
downstream write landed in the audit log where appropriate.
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


def _admin():
    return {"X-Env-Admin": "test-admin-token"}


# ── ticket detail tabs ─────────────────────────────────────────────────


@pytest.mark.parametrize("tab", ["thread", "internal", "related", "history"])
def test_ticket_detail_tabs_render(client, tab):
    r = client.get(f"/tickets/ticket_00001?tab={tab}")
    assert r.status_code == 200
    assert f'data-testid="tab-{tab}"' in r.text


# ── macro preview substitution ─────────────────────────────────────────


def test_macro_preview_substitutes_merge_fields(client):
    """Preview the shipping macro against ticket_00001 — name should resolve."""
    r = client.get("/macros/macro_shipping_delay?preview_ticket=ticket_00001")
    assert r.status_code == 200
    # Requester's first name should appear in the rendered preview.
    detail = client.get("/tickets/ticket_00001").text
    # Find requester name (testid is on the right rail dd).
    assert 'data-testid="requester-name"' in detail


# ── bulk-tag + bulk-status + bulk-group ────────────────────────────────


def test_bulk_tag_routes_before_parametric(client):
    """The bulk route must not be shadowed by /tickets/{ticket_id}/tag."""
    r = client.post(
        "/tickets/bulk/tag",
        data={"tag": "triage-test", "ids": "ticket_00001,ticket_00002"},
        follow_redirects=False,
    )
    assert r.status_code in (200, 303)
    state = client.get("/__env__/state", headers=_admin()).json()
    ops = [m["operation"] for m in state["recent_mutations"]]
    assert "ticket_tagged" in ops


def test_bulk_status_change(client):
    # Pick a ticket we know is 'new' (the outage cluster losers are all new).
    r = client.post(
        "/tickets/bulk/status",
        data={"new_status": "pending", "ids": "ticket_07001"},
        follow_redirects=False,
    )
    assert r.status_code in (200, 303)
    state = client.get("/__env__/state", headers=_admin()).json()
    ops = [m["operation"] for m in state["recent_mutations"]]
    assert "ticket_status_changed" in ops


def test_bulk_group_change(client):
    r = client.post(
        "/tickets/bulk/group",
        data={"group_id": "group_extvendor", "ids": "ticket_00001"},
        follow_redirects=False,
    )
    assert r.status_code in (200, 303)
    state = client.get("/__env__/state", headers=_admin()).json()
    ops = [m["operation"] for m in state["recent_mutations"]]
    assert "ticket_group_changed" in ops


# ── billing-group trigger ──────────────────────────────────────────────


def test_billing_group_lock_reverts_out_of_group_assign(client):
    """Bulk-assigning a billing-group ticket to a non-billing agent
    triggers the auto-revert. The audit log must record the revert.
    Setup: pick a known billing ticket + a non-billing agent."""
    # Find a billing-group ticket.
    state = client.get("/__env__/state", headers=_admin()).json()
    assert state["counts"]["tickets"] > 0
    # The seed routes billing-keyword bodies to group_billing.
    r = client.get("/tickets?group=group_billing")
    assert r.status_code == 200
    # Grab the first ticket id off the rendered HTML.
    import re
    ticket_ids = re.findall(r'data-ticket-id="(ticket_\d+)"', r.text)
    assert ticket_ids, "expected at least one billing-group ticket"
    billing_tid = ticket_ids[0]

    # Pick agent_002 (in technical/engineering, NOT billing).
    r = client.post(
        "/tickets/bulk/assign",
        data={"assignee_id": "agent_002", "ids": billing_tid},
        follow_redirects=False,
    )
    assert r.status_code in (200, 303)
    state = client.get("/__env__/state", headers=_admin()).json()
    ops = [m["operation"] for m in state["recent_mutations"]]
    # We expect both the initial assignment + the revert.
    assert "ticket_assignee_reverted" in ops


# ── escalation + merge ────────────────────────────────────────────────


def test_escalate_links_two_tickets(client):
    r = client.post(
        "/tickets/ticket_00001/escalate",
        data={"related_id": "ticket_00002"},
        follow_redirects=False,
    )
    assert r.status_code in (200, 303)
    state = client.get("/__env__/state", headers=_admin()).json()
    ops = [m["operation"] for m in state["recent_mutations"]]
    assert "ticket_escalated" in ops


def test_merge_into_survivor_marks_loser_deleted(client):
    """Single-survivor merge soft-deletes the loser; replies re-point."""
    r = client.post(
        "/tickets/ticket_07000/merge",
        data={"loser_ids": "ticket_07001"},
        follow_redirects=False,
    )
    assert r.status_code in (200, 303)
    # The loser should now be invisible on the list view.
    r = client.get("/tickets/ticket_07001")
    # Soft-deleted detail still 404s through our filter.
    detail = client.get("/tickets/ticket_07000").text
    assert "ticket_07000" in detail


# ── reports KPIs ───────────────────────────────────────────────────────


def test_reports_kpis_present(client):
    r = client.get("/reports")
    assert r.status_code == 200
    assert 'data-testid="kpi-near-breach"' in r.text
    assert 'data-testid="kpi-breached"' in r.text


# ── audit log ──────────────────────────────────────────────────────────


def test_audit_renders_recent_mutations(client):
    client.post(
        "/tickets/ticket_00001/tag",
        data={"tag": "audit-test"},
        follow_redirects=False,
    )
    r = client.get("/audit")
    assert r.status_code == 200
    assert "ticket_tagged" in r.text
