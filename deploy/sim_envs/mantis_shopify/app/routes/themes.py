"""Themes — `/themes`, `/themes/new`."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()


@router.get("/themes", response_class=HTMLResponse)
async def themes(request: Request):
    templates = request.app.state.templates
    conn = db.connect()
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM themes ORDER BY created_at DESC"
    ).fetchall()]
    return templates.TemplateResponse(
        "themes.html",
        {
            "request": request,
            "active_section": "themes",
            "rows": rows,
        },
    )


@router.get("/themes/new", response_class=HTMLResponse)
async def theme_new_form(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "theme_new.html",
        {"request": request, "active_section": "themes"},
    )


@router.post("/themes/new")
async def theme_new_submit(
    request: Request,
    name: str = Form(...),
):
    conn = db.connect()
    tid = f"theme_user_{int(conn.execute('SELECT COUNT(*) FROM themes').fetchone()[0]) + 1:05d}"
    conn.execute(
        "INSERT INTO themes (id, name, status, created_at) VALUES (?,?,?,?)",
        (tid, name.strip(), "in_review", app_main.now_value()),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="theme_submitted",
        target_type="theme",
        target_id=tid,
        payload={"name": name.strip()},
    )
    conn.commit()
    app_main.emit("theme_submitted", {"theme_id": tid})
    return RedirectResponse("/themes", status_code=303)
