"""Triggers — read-only browse surface (see ``triggers.py`` for runtime).

The 12 seeded triggers are displayed for the agent so a careful agent
can avoid violating them; the only one enforced in v1 is the
billing-group assignee lock, applied by ``triggers.apply_bulk_assign_revert``.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .. import db, main as app_main

router = APIRouter()


@router.get("/triggers", response_class=HTMLResponse)
async def list_triggers(request: Request) -> HTMLResponse:
    conn = db.connect()
    rows = conn.execute(
        "SELECT * FROM triggers ORDER BY is_active DESC, id"
    ).fetchall()
    items = []
    for r in rows:
        try:
            condition = json.loads(r["condition_json"] or "{}")
            action = json.loads(r["action_json"] or "{}")
        except json.JSONDecodeError:
            condition = {}
            action = {}
        items.append({
            "id": r["id"],
            "name": r["name"],
            "description": r["description"],
            "is_active": bool(r["is_active"]),
            "condition": condition,
            "action": action,
        })
    return app_main.app.state.templates.TemplateResponse(
        "triggers_list.html",
        {"request": request, "items": items},
    )
