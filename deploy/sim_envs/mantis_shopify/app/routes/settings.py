"""Settings — `/settings` + business/emergency/profile forms."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()


def _partner_row(conn):
    return conn.execute("SELECT * FROM partners LIMIT 1").fetchone()


@router.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    templates = request.app.state.templates
    conn = db.connect()
    partner = dict(_partner_row(conn))
    banner_dismissed = bool(conn.execute(
        "SELECT 1 FROM audit_log WHERE operation='banner_dismissed' "
        "AND target_id='emergency_contact' LIMIT 1"
    ).fetchone())
    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "active_section": "settings",
            "partner": partner,
            "owner": request.state.effective_user,
            "banner_dismissed": banner_dismissed,
            "now": app_main.now_value(),
        },
    )


@router.post("/settings/business")
async def settings_business(
    request: Request,
    business_name: str = Form(""),
    website: str = Form(""),
    business_email: str = Form(""),
    support_email: str = Form(""),
    phone: str = Form(""),
    address1: str = Form(""),
    address2: str = Form(""),
    city: str = Form(""),
    zip: str = Form(""),
    state: str = Form(""),
    country: str = Form(""),
):
    conn = db.connect()
    changed_fields: dict[str, str] = {}
    raw_kwargs = {
        "business_name": business_name, "website": website,
        "business_email": business_email, "support_email": support_email,
        "phone": phone, "address1": address1, "address2": address2,
        "city": city, "zip": zip, "state": state, "country": country,
    }
    existing = dict(_partner_row(conn))
    for k, v in raw_kwargs.items():
        if v != existing.get(k, ""):
            changed_fields[k] = v.strip()

    cols = ", ".join(f"{k}=?" for k in raw_kwargs)
    conn.execute(
        f"UPDATE partners SET {cols}, updated_at=? WHERE id=?",
        (*[v.strip() for v in raw_kwargs.values()],
         app_main.now_value(), existing["id"]),
    )
    for k, v in changed_fields.items():
        db.log_audit(
            conn,
            occurred_at=app_main.now_value(),
            operation="settings_updated",
            target_type="partner",
            target_id=existing["id"],
            payload={"field": k, "value": v},
        )
    conn.commit()
    app_main.emit("settings_updated", {"fields": list(changed_fields.keys())})
    return RedirectResponse("/settings", status_code=303)


@router.post("/settings/emergency")
async def settings_emergency(
    request: Request,
    emergency_name: str = Form(""),
    emergency_email: str = Form(""),
    emergency_phone: str = Form(""),
):
    conn = db.connect()
    existing = dict(_partner_row(conn))
    conn.execute(
        "UPDATE partners SET emergency_name=?, emergency_email=?, "
        "emergency_phone=?, updated_at=? WHERE id=?",
        (emergency_name.strip(), emergency_email.strip(),
         emergency_phone.strip(), app_main.now_value(), existing["id"]),
    )
    for k, v in [("emergency_name", emergency_name),
                 ("emergency_email", emergency_email),
                 ("emergency_phone", emergency_phone)]:
        db.log_audit(
            conn,
            occurred_at=app_main.now_value(),
            operation="emergency_contact_updated",
            target_type="partner",
            target_id=existing["id"],
            payload={"field": k, "value": v.strip()},
        )
    conn.commit()
    app_main.emit("emergency_contact_updated", {})
    return RedirectResponse("/settings", status_code=303)
