"""Smoke + behavioural tests for the depth surfaces (#332).

Tasks, notes, email templates, audit history, CSV export, forecast.
Each test exercises the agent-facing route, asserts the response, and
where appropriate verifies a downstream write landed in the audit log.
"""

from __future__ import annotations

import csv
import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("jinja2")


@pytest.fixture
def client():
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import app  # noqa: PLC0415

    with TestClient(app) as c:
        yield c


def _admin():
    return {"X-Env-Admin": "test-admin-token"}


# ── tasks ──────────────────────────────────────────────────────────────


def test_tasks_list_renders(client):
    r = client.get("/tasks")
    assert r.status_code == 200
    assert "Tasks" in r.text
    # Seed plants ~6k tasks; at least one row should be visible.
    assert "task_" in r.text


def test_tasks_overdue_filter(client):
    r = client.get("/tasks?status=open&overdue=1&assignee=user_00005")
    assert r.status_code == 200
    # The 20 pinned overdue tasks for user_00005 are guaranteed by the seed.
    assert "user_00005" in r.text


def test_complete_task_writes_audit(client):
    """Completing a task creates a task_completed mutation."""
    r = client.post("/tasks/task_00001/complete", follow_redirects=False)
    assert r.status_code in (200, 303)
    state = client.get("/__env__/state", headers=_admin()).json()
    assert any(m["operation"] == "task_completed"
               for m in state["recent_mutations"])


def test_create_task_with_target_lands_on_contact(client):
    r = client.post(
        "/tasks/create",
        data={
            "title": "Follow up — Q3 deck",
            "target_type": "contact",
            "target_id": "contact_00042",
            "due_date": "2026-02-01",
            "priority": "high",
            "assignee_id": "user_00005",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    # Redirect should land us back on the contact detail.
    assert r.headers["location"] == "/contacts/contact_00042"


# ── notes ──────────────────────────────────────────────────────────────


def test_create_note_pins_and_audits(client):
    r = client.post(
        "/notes/create",
        data={
            "target_type": "contact",
            "target_id": "contact_00010",
            "body_md": "Reminder: VP Eng prefers Slack DMs.",
            "pinned": "1",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    state = client.get("/__env__/state", headers=_admin()).json()
    note_ops = [m for m in state["recent_mutations"]
                if m["operation"] == "note_added"]
    assert note_ops, "note_added mutation missing from audit log"

    # Notes show up on the contact detail under the notes tab.
    r = client.get("/contacts/contact_00010?tab=notes")
    assert "Reminder: VP Eng" in r.text


# ── templates ──────────────────────────────────────────────────────────


def test_templates_list_renders(client):
    r = client.get("/templates")
    assert r.status_code == 200
    assert "Email templates" in r.text
    assert "template_001" in r.text


def test_template_detail_preview_substitutes_merge_fields(client):
    """Preview with a contact resolves {{contact.first_name}} etc."""
    r = client.get(
        "/templates/template_001?preview_contact=contact_00042"
    )
    assert r.status_code == 200
    # Canonical Sarah Chen is at contact_00042; her first name should
    # appear in the rendered subject preview.
    assert "Sarah" in r.text


# ── audit + CSV export + forecast ──────────────────────────────────────


def test_audit_log_renders_recent_mutations(client):
    # Trigger a mutation we can match on.
    client.post(
        "/contacts/contact_00100/tag",
        data={"tag": "vip-test"},
        follow_redirects=False,
    )
    r = client.get("/audit")
    assert r.status_code == 200
    assert "tag_added" in r.text


def test_audit_for_specific_target(client):
    client.post(
        "/contacts/contact_00100/tag",
        data={"tag": "vip-test"},
        follow_redirects=False,
    )
    r = client.get("/audit/contact/contact_00100")
    assert r.status_code == 200
    assert "tag_added" in r.text


def test_export_contacts_csv_streams_rows(client):
    r = client.get("/export/contacts.csv?owner=user_00001")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    reader = csv.reader(io.StringIO(r.text))
    headers = next(reader)
    assert "id" in headers and "email" in headers
    body_rows = list(reader)
    assert len(body_rows) > 0


def test_forecast_renders_weighted_total(client):
    r = client.get("/forecast")
    assert r.status_code == 200
    assert "Weighted pipeline" in r.text
    # Each stage gets a row + a win-prob column.
    assert "Negotiation" in r.text
    assert "%" in r.text


# ── contact-detail tabs ────────────────────────────────────────────────


@pytest.mark.parametrize("tab", ["activity", "tasks", "notes", "deals", "history"])
def test_contact_detail_tabs_render(client, tab):
    r = client.get(f"/contacts/contact_00042?tab={tab}")
    assert r.status_code == 200
    assert "Sarah Chen" in r.text
    # The tab marker should be in the active tab class.
    assert f'href="?tab={tab}"' in r.text


def test_contact_detail_lead_score_present(client):
    r = client.get("/contacts/contact_00001")
    assert r.status_code == 200
    assert 'data-testid="lead-score"' in r.text
