"""Team — `/team`, `/team/invite`, `/team/banner/dismiss`."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()

VALID_STAFF_ROLES = {
    "owner", "staff_business", "staff_dev", "staff_marketing",
    "staff_support", "staff_finance",
}


@router.get("/team", response_class=HTMLResponse)
async def team(request: Request, tab: str = "active"):
    templates = request.app.state.templates
    conn = db.connect()
    owners = [dict(r) for r in conn.execute(
        "SELECT * FROM users WHERE role='owner' ORDER BY name"
    ).fetchall()]
    staff_active = [dict(r) for r in conn.execute(
        "SELECT * FROM users WHERE role != 'owner' AND status='active' "
        "ORDER BY name"
    ).fetchall()]
    staff_invited = [dict(r) for r in conn.execute(
        "SELECT * FROM users WHERE role != 'owner' AND status='invited' "
        "ORDER BY name"
    ).fetchall()]
    # Was the banner dismissed?
    banner_dismissed = bool(conn.execute(
        "SELECT 1 FROM audit_log WHERE operation='banner_dismissed' "
        "AND target_id='emergency_contact' LIMIT 1"
    ).fetchone())
    return templates.TemplateResponse(
        "team.html",
        {
            "request": request,
            "active_section": "team",
            "owners": owners,
            "staff_active": staff_active,
            "staff_invited": staff_invited,
            "tab": tab,
            "banner_dismissed": banner_dismissed,
            "now": app_main.now_value(),
        },
    )


@router.post("/team/banner/dismiss")
async def banner_dismiss(request: Request):
    conn = db.connect()
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="banner_dismissed",
        target_type="banner",
        target_id="emergency_contact",
        payload={"page": "team"},
    )
    conn.commit()
    app_main.emit("banner_dismissed", {"banner": "emergency_contact"})
    return RedirectResponse(request.headers.get("referer") or "/team",
                            status_code=303)


@router.get("/team/invite", response_class=HTMLResponse)
async def team_invite_form(request: Request, role: str = "staff_business"):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "team_invite.html",
        {
            "request": request,
            "active_section": "team",
            "role_default": role,
        },
    )


@router.post("/team/invite")
async def team_invite_submit(
    request: Request,
    email: str = Form(...),
    name: str = Form(""),
    role: str = Form("staff_business"),
):
    if role not in VALID_STAFF_ROLES:
        role = "staff_business"
    conn = db.connect()
    uid = f"invite_{int(conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]) + 1:05d}"
    conn.execute(
        "INSERT INTO users (id, email, password_hash, name, role, status, "
        "last_login_at, avatar_color, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            uid, email.strip().lower(), "", name.strip() or email.split("@")[0],
            "owner" if role == "owner" else role,
            "invited", "", "#9c6ade", app_main.now_value(),
        ),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="staff_invited",
        target_type="user",
        target_id=uid,
        payload={"email": email.strip(), "role": role},
    )
    conn.commit()
    app_main.emit("staff_invited", {"user_id": uid, "role": role})
    return RedirectResponse("/team?tab=invited", status_code=303)
