"""Reports — Open by group, SLA breach trend, audit log."""

from __future__ import annotations

import json
from datetime import timedelta

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .. import db, main as app_main
from ..seed import _parse_iso

router = APIRouter()


@router.get("/reports", response_class=HTMLResponse)
async def reports(request: Request) -> HTMLResponse:
    conn = db.connect()
    now = app_main.now_value()
    now_dt = _parse_iso(now)

    by_group_rows = conn.execute(
        "SELECT COALESCE(g.name, '(unassigned group)') AS group_name, "
        "       COUNT(*) AS open_count "
        "FROM tickets t LEFT JOIN groups g ON t.group_id = g.id "
        "WHERE t.status IN ('new','open','pending') AND t.deleted_at IS NULL "
        "GROUP BY g.name ORDER BY open_count DESC"
    ).fetchall()
    by_group = [dict(r) for r in by_group_rows]

    soon_cutoff = (now_dt + timedelta(hours=2)).isoformat()
    near_breach = conn.execute(
        "SELECT COUNT(*) FROM tickets "
        "WHERE status IN ('new','open','pending') AND deleted_at IS NULL "
        "AND sla_breach_at > ? AND sla_breach_at <= ?",
        (now, soon_cutoff),
    ).fetchone()[0]
    already_breached = conn.execute(
        "SELECT COUNT(*) FROM tickets "
        "WHERE status IN ('new','open','pending') AND deleted_at IS NULL "
        "AND sla_breach_at <= ?",
        (now,),
    ).fetchone()[0]

    by_priority_rows = conn.execute(
        "SELECT priority, COUNT(*) AS open_count "
        "FROM tickets WHERE status IN ('new','open','pending') AND deleted_at IS NULL "
        "GROUP BY priority"
    ).fetchall()
    by_priority = {r["priority"]: r["open_count"] for r in by_priority_rows}

    return app_main.app.state.templates.TemplateResponse(
        "reports.html",
        {
            "request": request,
            "by_group": by_group,
            "near_breach": near_breach,
            "already_breached": already_breached,
            "by_priority": by_priority,
        },
    )


@router.get("/audit", response_class=HTMLResponse)
async def audit(request: Request) -> HTMLResponse:
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
