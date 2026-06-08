"""Marketing surface — `/` home + `/experts` landing."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .. import db, main as app_main

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    templates = request.app.state.templates
    conn = db.connect()
    latest = [
        dict(r) for r in conn.execute(
            "SELECT j.*, c.name AS company_name "
            "FROM jobs j JOIN companies c ON c.id = j.company_id "
            "WHERE j.status = 'open' "
            "ORDER BY j.id LIMIT 9"
        ).fetchall()
    ]
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "latest_jobs": latest,
            "stats": {
                "avg_pay": 95,
                "roles_created_k": 12,
                "daily_payouts_usd": 240_000,
            },
            "now": app_main.now_value(),
        },
    )


@router.get("/experts", response_class=HTMLResponse)
async def experts(request: Request):
    templates = request.app.state.templates
    conn = db.connect()
    latest = [
        dict(r) for r in conn.execute(
            "SELECT j.*, c.name AS company_name "
            "FROM jobs j JOIN companies c ON c.id = j.company_id "
            "WHERE j.status = 'open' "
            "ORDER BY j.id LIMIT 12"
        ).fetchall()
    ]
    return templates.TemplateResponse(
        "experts.html",
        {
            "request": request,
            "latest_jobs": latest,
        },
    )
