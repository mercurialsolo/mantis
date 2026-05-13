"""Insights — forecast, audit history, CSV export.

Three views the agent has to reach for during realistic CRM workflows:

* ``/forecast`` — weighted pipeline by stage (sum amount × win probability).
* ``/audit`` and ``/audit/contact/<id>`` — recent mutation history.
* ``/export/contacts.csv?…`` — bulk export honouring the contact filters.
"""

from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from .. import db, main as app_main

router = APIRouter()


# Win probabilities per stage. Matches the convention used by HubSpot /
# Salesforce out-of-the-box pipelines: early stages low, late stages high,
# closed-won = 1.0, lost = 0.
STAGE_WIN_PROB = {
    "Prospect": 0.10,
    "Qualified": 0.25,
    "Proposal": 0.50,
    "Negotiation": 0.75,
    "At Risk": 0.20,
    "Closed Won": 1.00,
    "Closed Lost": 0.00,
}


@router.get("/forecast", response_class=HTMLResponse)
async def forecast(request: Request) -> HTMLResponse:
    conn = db.connect()
    rows = conn.execute(
        "SELECT stage, COUNT(*) AS deal_count, SUM(amount) AS amount_sum "
        "FROM deals GROUP BY stage ORDER BY stage"
    ).fetchall()
    breakdown = []
    weighted_total = 0.0
    raw_total = 0.0
    for r in rows:
        amount = float(r["amount_sum"] or 0.0)
        prob = STAGE_WIN_PROB.get(r["stage"], 0.0)
        weighted = amount * prob
        weighted_total += weighted
        raw_total += amount
        breakdown.append({
            "stage": r["stage"],
            "deal_count": r["deal_count"],
            "amount_sum": amount,
            "win_prob": prob,
            "weighted": weighted,
        })
    return app_main.app.state.templates.TemplateResponse(
        "forecast.html",
        {
            "request": request,
            "breakdown": breakdown,
            "raw_total": raw_total,
            "weighted_total": weighted_total,
        },
    )


@router.get("/audit", response_class=HTMLResponse)
async def audit_recent(request: Request) -> HTMLResponse:
    conn = db.connect()
    limit = int(request.query_params.get("limit") or 100)
    rows = conn.execute(
        "SELECT * FROM mutations ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    items = []
    for r in rows:
        try:
            payload = json.loads(r["payload_json"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        items.append({
            "id": r["id"],
            "occurred_at": r["occurred_at"],
            "operation": r["operation"],
            "target_type": r["target_type"],
            "target_id": r["target_id"],
            "payload": payload,
        })
    return app_main.app.state.templates.TemplateResponse(
        "audit.html",
        {"request": request, "items": items, "limit": limit},
    )


@router.get("/audit/{target_type}/{target_id}", response_class=HTMLResponse)
async def audit_for_target(
    request: Request, target_type: str, target_id: str
) -> HTMLResponse:
    if target_type not in {"contact", "deal", "company", "task", "list"}:
        return HTMLResponse("Bad target_type", status_code=400)
    conn = db.connect()
    rows = conn.execute(
        "SELECT * FROM mutations WHERE target_type = ? AND target_id = ? "
        "ORDER BY id DESC LIMIT 200",
        (target_type, target_id),
    ).fetchall()
    items = []
    for r in rows:
        try:
            payload = json.loads(r["payload_json"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        items.append({
            "id": r["id"], "occurred_at": r["occurred_at"],
            "operation": r["operation"], "payload": payload,
        })
    return app_main.app.state.templates.TemplateResponse(
        "audit.html",
        {"request": request, "items": items, "limit": 200,
         "target_type": target_type, "target_id": target_id},
    )


@router.get("/export/contacts.csv")
async def export_contacts_csv(request: Request) -> Response:
    """CSV export honouring the contact-list filters. Capped at 5000 rows."""
    q = (request.query_params.get("q") or "").strip()
    stage = (request.query_params.get("stage") or "").strip()
    owner = (request.query_params.get("owner") or "").strip()
    tag = (request.query_params.get("tag") or "").strip()

    where = ["c.deleted_at IS NULL"]
    args: list = []
    if q:
        where.append("(c.name LIKE ? OR c.email LIKE ? OR co.name LIKE ?)")
        like = f"%{q}%"
        args += [like, like, like]
    if stage:
        where.append("c.lifecycle_stage = ?")
        args.append(stage)
    if owner:
        where.append("c.owner_id = ?")
        args.append(owner)
    if tag:
        where.append("c.tags LIKE ?")
        args.append(f'%"{tag}"%')

    where_clause = " AND ".join(where) or "1=1"
    conn = db.connect()
    rows = conn.execute(
        f"SELECT c.id, c.name, c.email, c.phone, c.lifecycle_stage, "
        f"       c.owner_id, c.tags, c.last_activity_at, co.name AS company_name "
        f"FROM contacts c LEFT JOIN companies co ON c.company_id = co.id "
        f"WHERE {where_clause} ORDER BY c.id LIMIT 5000",
        args,
    ).fetchall()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "name", "email", "phone", "lifecycle_stage",
        "owner_id", "tags", "last_activity_at", "company_name",
    ])
    for r in rows:
        try:
            tags = ",".join(json.loads(r["tags"] or "[]"))
        except json.JSONDecodeError:
            tags = ""
        writer.writerow([
            r["id"], r["name"], r["email"] or "", r["phone"] or "",
            r["lifecycle_stage"] or "", r["owner_id"] or "",
            tags, r["last_activity_at"] or "", r["company_name"] or "",
        ])
    return PlainTextResponse(
        buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="contacts.csv"'},
    )
