"""Support — `/support`, `/support/contact`, `/support/tickets`."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()

CATEGORIES = [
    "Account access",
    "Payouts",
    "App distribution",
    "Themes",
    "Partner Directory",
    "Other",
]


@router.get("/support", response_class=HTMLResponse)
async def support(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "support.html",
        {"request": request, "active_section": "support"},
    )


@router.get("/support/contact", response_class=HTMLResponse)
async def support_contact_form(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "support_contact.html",
        {
            "request": request,
            "active_section": "support",
            "categories": CATEGORIES,
        },
    )


@router.post("/support/contact")
async def support_contact_submit(
    request: Request,
    subject: str = Form(...),
    category: str = Form(...),
    description: str = Form(...),
):
    conn = db.connect()
    tid = f"ticket_{int(conn.execute('SELECT COUNT(*) FROM tickets').fetchone()[0]) + 1:05d}"
    conn.execute(
        "INSERT INTO tickets (id, subject, category, description, status, "
        "created_at) VALUES (?,?,?,?,?,?)",
        (tid, subject.strip(), category.strip(), description.strip(),
         "open", app_main.now_value()),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="support_ticket_created",
        target_type="ticket",
        target_id=tid,
        payload={
            "subject": subject.strip(),
            "category": category.strip(),
            "description_len": len(description.strip()),
        },
    )
    conn.commit()
    app_main.emit("support_ticket_created", {"ticket_id": tid})

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "support_confirm.html",
        {
            "request": request,
            "active_section": "support",
            "ticket_id": tid,
        },
        status_code=201,
    )
