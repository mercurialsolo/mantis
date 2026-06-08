"""Login / signup / logout routes."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import auth as auth_mod, db, main as app_main

router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, next: str = "/dashboard",
                     error: str | None = None):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "next": next, "error": error},
    )


@router.post("/login")
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form("/dashboard"),
):
    user = auth_mod.lookup_user_by_email(email)
    if not user or not auth_mod.verify_password(password, user["password_hash"]):
        return RedirectResponse(
            f"/login?next={next}&error=Invalid+credentials", status_code=303,
        )
    response = RedirectResponse(next, status_code=303)
    auth_mod.set_session_cookie(response, user["id"])

    with db.transaction() as conn:
        db.log_audit(
            conn,
            occurred_at=app_main.now_value(),
            operation="login_succeeded",
            target_type="user",
            target_id=user["id"],
            payload={"email": user["email"], "role": user["role"]},
        )
    app_main.emit("login_succeeded", {"user_id": user["id"]})
    return response


@router.get("/signup", response_class=HTMLResponse)
async def signup_form(request: Request, error: str | None = None):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "signup.html",
        {"request": request, "error": error},
    )


@router.post("/signup")
async def signup_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    name: str = Form(""),
    role: str = Form("candidate"),
):
    if role not in {"candidate", "client"}:
        role = "candidate"
    email_lc = email.strip().lower()
    existing = auth_mod.lookup_user_by_email(email_lc)
    if existing:
        return RedirectResponse(
            "/signup?error=Email+already+in+use", status_code=303,
        )

    now = app_main.now_value()
    user_id = f"{role}_signup_{abs(hash(email_lc)) % 1_000_000:06d}"
    with db.transaction() as conn:
        company_id = None
        if role == "client":
            company_id = f"company_signup_{user_id[-6:]}"
            conn.execute(
                "INSERT INTO companies (id, name, domain, owner_user_id) "
                "VALUES (?, ?, ?, ?)",
                (company_id, name or "Self-serve client", "", user_id),
            )
        conn.execute(
            "INSERT INTO users (id, email, password_hash, role, name, company_id, "
            "created_at) VALUES (?,?,?,?,?,?,?)",
            (user_id, email_lc, auth_mod.hash_password(password), role,
             name or "", company_id, now),
        )
        if role == "candidate":
            conn.execute(
                "INSERT INTO candidate_profiles (user_id, headline, skills_json, "
                "hourly_rate, availability, resume_text, updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (user_id, "", "[]", 0.0, "Part-time", "", now),
            )
        db.log_audit(
            conn,
            occurred_at=now,
            operation="user_signed_up",
            target_type="user",
            target_id=user_id,
            payload={"email": email_lc, "role": role},
        )

    response = RedirectResponse("/dashboard", status_code=303)
    auth_mod.set_session_cookie(response, user_id)
    app_main.emit("user_signed_up", {"user_id": user_id})
    return response


@router.post("/logout")
async def logout(request: Request):
    response = RedirectResponse("/", status_code=303)
    auth_mod.clear_session_cookie(response)
    return response
