"""Global fuzzy search — name / email / company / deal name."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .. import db, main as app_main

router = APIRouter()


@router.get("/search", response_class=HTMLResponse)
async def search(request: Request) -> HTMLResponse:
    q = (request.query_params.get("q") or "").strip()
    contacts: list = []
    companies: list = []
    deals: list = []
    if q:
        like = f"%{q}%"
        conn = db.connect()
        contacts = [dict(r) for r in conn.execute(
            "SELECT id, name, email FROM contacts "
            "WHERE deleted_at IS NULL AND (name LIKE ? OR email LIKE ?) "
            "ORDER BY id LIMIT 20",
            (like, like),
        ).fetchall()]
        companies = [dict(r) for r in conn.execute(
            "SELECT id, name, domain FROM companies "
            "WHERE name LIKE ? OR domain LIKE ? ORDER BY id LIMIT 20",
            (like, like),
        ).fetchall()]
        deals = [dict(r) for r in conn.execute(
            "SELECT id, name, stage, amount FROM deals WHERE name LIKE ? "
            "ORDER BY id LIMIT 20",
            (like,),
        ).fetchall()]
    return app_main.app.state.templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "q": q,
            "contacts": contacts,
            "companies": companies,
            "deals": deals,
        },
    )
