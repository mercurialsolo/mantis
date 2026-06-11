"""Sales (Partner leads) — `/sales`, `/sales/referrals`, `/sales/leads/new`."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()

VALID_PRODUCTS = {"plus", "pos", "plus_b2b"}


@router.get("/sales", response_class=HTMLResponse)
async def sales_leads(request: Request):
    templates = request.app.state.templates
    conn = db.connect()
    submitted = [dict(r) for r in conn.execute(
        "SELECT * FROM leads ORDER BY submitted_at DESC LIMIT 20"
    ).fetchall()]
    return templates.TemplateResponse(
        "sales_leads.html",
        {
            "request": request,
            "active_section": "sales",
            "sub_section": "leads",
            "leads": submitted,
        },
    )


@router.get("/sales/referrals", response_class=HTMLResponse)
async def sales_referrals(request: Request):
    templates = request.app.state.templates
    conn = db.connect()
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM referrals ORDER BY created_at DESC"
    ).fetchall()]
    return templates.TemplateResponse(
        "sales_referrals.html",
        {
            "request": request,
            "active_section": "sales",
            "sub_section": "referrals",
            "rows": rows,
        },
    )


@router.get("/sales/leads/new", response_class=HTMLResponse)
async def lead_new_form(request: Request, product: str = "plus"):
    templates = request.app.state.templates
    if product not in VALID_PRODUCTS:
        product = "plus"
    titles = {
        "plus": "Submit a Plus lead",
        "pos": "Submit a POS lead",
        "plus_b2b": "Submit a Plus B2B lead",
    }
    return templates.TemplateResponse(
        "lead_new.html",
        {
            "request": request,
            "active_section": "sales",
            "sub_section": "leads",
            "product": product,
            "title": titles[product],
        },
    )


@router.post("/sales/leads/new")
async def lead_new_submit(
    request: Request,
    product: str = Form("plus"),
    merchant_name: str = Form(...),
    contact_email: str = Form(...),
    contact_name: str = Form(""),
    notes: str = Form(""),
):
    if product not in VALID_PRODUCTS:
        product = "plus"
    conn = db.connect()
    nid = f"lead_user_{int(conn.execute('SELECT COUNT(*) FROM leads').fetchone()[0]) + 1:05d}"
    conn.execute(
        "INSERT INTO leads (id, product, merchant_name, contact_email, "
        "contact_name, status, earnings_cents, submitted_at, notes) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (
            nid, product, merchant_name.strip(),
            contact_email.strip(), contact_name.strip(),
            "submitted", 0, app_main.now_value(), notes.strip(),
        ),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="lead_submitted",
        target_type="lead",
        target_id=nid,
        payload={
            "product": product,
            "merchant_name": merchant_name.strip(),
            "contact_email": contact_email.strip(),
        },
    )
    conn.commit()
    app_main.emit("lead_submitted", {"lead_id": nid, "product": product})
    return RedirectResponse("/sales", status_code=303)
