"""Three prebuilt reports + a /lists surface used by T05 (export to list)."""

from __future__ import annotations

import json

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()


@router.get("/reports", response_class=HTMLResponse)
async def reports_home(request: Request) -> HTMLResponse:
    conn = db.connect()
    deals_by_stage = [
        {"stage": r["stage"], "count": r["c"]} for r in conn.execute(
            "SELECT stage, COUNT(*) AS c FROM deals GROUP BY stage ORDER BY stage"
        ).fetchall()
    ]
    contacts_by_source = [
        {"source": r["source"], "count": r["c"]} for r in conn.execute(
            "SELECT source, COUNT(*) AS c FROM contacts "
            "WHERE deleted_at IS NULL GROUP BY source ORDER BY source"
        ).fetchall()
    ]
    activity_by_user = [
        {"user_id": r["actor_id"], "count": r["c"]} for r in conn.execute(
            "SELECT actor_id, COUNT(*) AS c FROM activities "
            "GROUP BY actor_id ORDER BY actor_id"
        ).fetchall()
    ]
    return app_main.app.state.templates.TemplateResponse(
        "reports.html",
        {
            "request": request,
            "deals_by_stage": deals_by_stage,
            "contacts_by_source": contacts_by_source,
            "activity_by_user": activity_by_user,
        },
    )


@router.get("/lists/{list_id}", response_class=HTMLResponse)
async def list_detail(request: Request, list_id: str) -> HTMLResponse:
    conn = db.connect()
    row = conn.execute("SELECT * FROM lists WHERE id = ?",
                       (list_id,)).fetchone()
    if row is None:
        return HTMLResponse("Not found", status_code=404)
    members = [dict(m) for m in conn.execute(
        "SELECT * FROM list_members WHERE list_id = ? ORDER BY member_id "
        "LIMIT 500",
        (list_id,),
    ).fetchall()]
    return app_main.app.state.templates.TemplateResponse(
        "list_detail.html",
        {
            "request": request,
            "list": dict(row),
            "members": members,
            "filter_def": json.loads(row["filter_json"] or "{}"),
        },
    )


@router.post("/lists/{list_id}/add")
async def add_to_list(
    list_id: str,
    member_type: str = Form(...),
    member_ids: str = Form(...),
) -> RedirectResponse:
    if member_type not in {"contact", "deal"}:
        return RedirectResponse(f"/lists/{list_id}", status_code=303)
    ids = [s.strip() for s in member_ids.split(",") if s.strip()]
    if not ids:
        return RedirectResponse(f"/lists/{list_id}", status_code=303)
    with db.transaction() as txn:
        for mid in ids:
            txn.execute(
                "INSERT OR IGNORE INTO list_members (list_id, member_type, member_id) "
                "VALUES (?, ?, ?)",
                (list_id, member_type, mid),
            )
            db.log_mutation(
                txn,
                occurred_at=app_main.now_value(),
                operation="list_member_added",
                target_type="list",
                target_id=list_id,
                payload={"member_type": member_type, "member_id": mid},
            )
        app_main.emit("mutation.list_member_added",
                      {"list_id": list_id, "count": len(ids)})
    return RedirectResponse(f"/lists/{list_id}", status_code=303)
