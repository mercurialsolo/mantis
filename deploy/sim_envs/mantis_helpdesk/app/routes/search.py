"""Global search — substring match across tickets + requesters."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .. import db, main as app_main

router = APIRouter()


@router.get("/search", response_class=HTMLResponse)
async def search(request: Request) -> HTMLResponse:
    q = (request.query_params.get("q") or "").strip()
    tickets: list[dict] = []
    requesters: list[dict] = []
    if q:
        like = f"%{q}%"
        conn = db.connect()
        tickets = [
            dict(r) for r in conn.execute(
                "SELECT id, subject, status, priority FROM tickets "
                "WHERE deleted_at IS NULL AND (subject LIKE ? OR body LIKE ?) "
                "ORDER BY id LIMIT 50",
                (like, like),
            ).fetchall()
        ]
        requesters = [
            dict(r) for r in conn.execute(
                "SELECT id, name, email FROM users "
                "WHERE role='requester' AND (name LIKE ? OR email LIKE ?) "
                "ORDER BY id LIMIT 30",
                (like, like),
            ).fetchall()
        ]
    return app_main.app.state.templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "q": q,
            "tickets": tickets,
            "requesters": requesters,
        },
    )
