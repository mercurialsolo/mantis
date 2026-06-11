"""Payouts — `/payouts`, `/payouts/<id>`, `/payouts/export`."""

from __future__ import annotations

import io

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from .. import db, main as app_main

router = APIRouter()


@router.get("/payouts", response_class=HTMLResponse)
async def payouts(request: Request):
    templates = request.app.state.templates
    conn = db.connect()
    pending = conn.execute(
        "SELECT * FROM payouts WHERE status='pending' "
        "ORDER BY period_end DESC LIMIT 1"
    ).fetchone()
    sent = [dict(r) for r in conn.execute(
        "SELECT * FROM payouts WHERE status='paid' "
        "ORDER BY period_end DESC LIMIT 24"
    ).fetchall()]
    return templates.TemplateResponse(
        "payouts.html",
        {
            "request": request,
            "active_section": "payouts",
            "pending": dict(pending) if pending else None,
            "sent": sent,
        },
    )


@router.get("/payouts/export")
async def payouts_export(request: Request):
    conn = db.connect()
    rows = conn.execute(
        "SELECT id, period_start, period_end, sent_at, amount_cents, currency, "
        "method, status FROM payouts ORDER BY period_end DESC"
    ).fetchall()
    buf = io.StringIO()
    buf.write("id,period_start,period_end,sent_at,amount,currency,method,status\n")
    for r in rows:
        amount = (r["amount_cents"] / 100.0)
        buf.write(
            f"{r['id']},{r['period_start']},{r['period_end']},{r['sent_at']},"
            f"{amount:.2f},{r['currency']},\"{r['method']}\",{r['status']}\n"
        )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="payouts_export_requested",
        target_type="payouts",
        target_id="export",
        payload={"rows": len(rows)},
    )
    conn.commit()
    app_main.emit("payouts_export_requested", {"rows": len(rows)})
    buf.seek(0)
    return StreamingResponse(
        iter([buf.read()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=payouts.csv"},
    )


@router.get("/payouts/{payout_id}", response_class=HTMLResponse)
async def payout_detail(payout_id: str, request: Request):
    templates = request.app.state.templates
    conn = db.connect()
    payout = conn.execute(
        "SELECT * FROM payouts WHERE id=?", (payout_id,),
    ).fetchone()
    if payout is None:
        return HTMLResponse("Payout not found", status_code=404)
    items = [dict(r) for r in conn.execute(
        "SELECT * FROM payout_line_items WHERE payout_id=? "
        "ORDER BY occurred_at ASC",
        (payout_id,),
    ).fetchall()]
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="payout_viewed",
        target_type="payout",
        target_id=payout_id,
        payload={"line_items": len(items)},
    )
    conn.commit()
    app_main.emit("payout_viewed", {"payout_id": payout_id})
    return templates.TemplateResponse(
        "payout_detail.html",
        {
            "request": request,
            "active_section": "payouts",
            "payout": dict(payout),
            "items": items,
        },
    )
