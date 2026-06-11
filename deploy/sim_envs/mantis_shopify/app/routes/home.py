"""Home dashboard — `/`."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .. import db

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    templates = request.app.state.templates
    conn = db.connect()

    stores = [dict(r) for r in conn.execute(
        "SELECT * FROM stores ORDER BY last_login_at DESC LIMIT 5"
    ).fetchall()]
    total_stores = int(conn.execute(
        "SELECT COUNT(*) FROM stores"
    ).fetchone()[0])

    pending = conn.execute(
        "SELECT * FROM payouts WHERE status='pending' "
        "ORDER BY period_end DESC LIMIT 1"
    ).fetchone()
    active_referrals = int(conn.execute(
        "SELECT COUNT(*) FROM referrals WHERE status='active'"
    ).fetchone()[0])
    lifetime_leads = int(conn.execute(
        "SELECT COUNT(*) FROM leads"
    ).fetchone()[0])

    changelog = [dict(r) for r in conn.execute(
        "SELECT * FROM changelog ORDER BY published_at DESC LIMIT 6"
    ).fetchall()]

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "active_section": "home",
            "stores": stores,
            "total_stores": total_stores,
            "pending_payout": dict(pending) if pending else None,
            "active_referrals": active_referrals,
            "lifetime_leads": lifetime_leads,
            "changelog": changelog,
        },
    )


@router.get("/notifications", response_class=HTMLResponse)
async def notifications(request: Request):
    templates = request.app.state.templates
    conn = db.connect()
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM notifications ORDER BY occurred_at DESC"
    ).fetchall()]
    return templates.TemplateResponse(
        "notifications.html",
        {"request": request, "active_section": "", "notifications": rows},
    )
