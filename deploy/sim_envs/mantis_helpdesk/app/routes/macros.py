"""Macros library — mirror of mantis-crm's ``/templates`` page.

Read-only listing + detail with a preview against a requester. The
substitution logic lives next to the composer in ``tickets.py`` so the
preview here uses the same renderer. Bulk actions live under
``/macros/bulk/...`` declared BEFORE the parametric ``/macros/{id}``
to keep the route-ordering invariant.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .. import db, main as app_main
from .tickets import _render_merge_body

router = APIRouter()


@router.get("/macros", response_class=HTMLResponse)
async def list_macros(request: Request) -> HTMLResponse:
    folder = (request.query_params.get("folder") or "").strip()
    conn = db.connect()
    where = []
    args: list = []
    if folder:
        where.append("folder = ?")
        args.append(folder)
    where_clause = " AND ".join(where) or "1=1"
    rows = conn.execute(
        f"SELECT id, name, folder FROM macros WHERE {where_clause} ORDER BY id",
        args,
    ).fetchall()
    folders = [r["folder"] for r in conn.execute(
        "SELECT DISTINCT folder FROM macros ORDER BY folder"
    ).fetchall()]
    return app_main.app.state.templates.TemplateResponse(
        "macros_list.html",
        {
            "request": request,
            "macros": [dict(r) for r in rows],
            "folders": folders,
            "active_folder": folder,
        },
    )


# Declared BEFORE the parametric ``/macros/{macro_id}`` so a future
# ``/macros/bulk/...`` route doesn't get shadowed. The list endpoint is
# read-only in v1 — leaving the slot open for v2.


@router.get("/macros/{macro_id}", response_class=HTMLResponse)
async def macro_detail(request: Request, macro_id: str) -> HTMLResponse:
    conn = db.connect()
    row = conn.execute(
        "SELECT * FROM macros WHERE id = ?", (macro_id,),
    ).fetchone()
    if row is None:
        return HTMLResponse("Not found", status_code=404)

    preview_body = row["body"]
    preview_ticket_id = (request.query_params.get("preview_ticket") or "").strip()
    if preview_ticket_id:
        t = conn.execute(
            "SELECT t.*, u.name AS requester_name, u.email AS requester_email "
            "FROM tickets t LEFT JOIN users u ON t.requester_id = u.id WHERE t.id = ?",
            (preview_ticket_id,),
        ).fetchone()
        if t is not None:
            requester = {"name": t["requester_name"] or "", "email": t["requester_email"] or ""}
            agent_obj = None
            if t["assignee_id"]:
                a = conn.execute(
                    "SELECT name, email FROM users WHERE id = ?", (t["assignee_id"],),
                ).fetchone()
                if a is not None:
                    agent_obj = {"name": a["name"], "email": a["email"]}
            preview_body = _render_merge_body(
                row["body"], requester=requester, agent=agent_obj,
            )

    return app_main.app.state.templates.TemplateResponse(
        "macro_detail.html",
        {
            "request": request,
            "macro": dict(row),
            "preview_ticket_id": preview_ticket_id,
            "preview_body": preview_body,
        },
    )
