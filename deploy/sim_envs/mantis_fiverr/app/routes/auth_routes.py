"""/login + /signup + /logout — email/password only."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import auth, db, main as app_main

router = APIRouter()


def _templates(request: Request):
    return request.app.state.templates


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return _templates(request).TemplateResponse(
        "login.html",
        {
            "request": request,
            "next": request.query_params.get("next", "/"),
            "error": request.query_params.get("error", ""),
        },
    )


@router.post("/login")
async def login_submit(
    request: Request,
    login: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
):
    user = auth.lookup_user_by_login(login)
    if user is None or not auth.verify_password(password, user["password_hash"]):
        return RedirectResponse(
            url=f"/login?error=invalid+credentials&next={next}",
            status_code=303,
        )
    if not next.startswith("/"):
        next = "/"
    resp = RedirectResponse(url=next, status_code=303)
    auth.set_session_cookie(resp, user["id"])
    with db.transaction() as tx:
        db.log_audit(
            tx,
            occurred_at=app_main.now_value(),
            operation="login_succeeded",
            target_type="user",
            target_id=user["id"],
            payload={"role": user["role"]},
        )
    return resp


@router.get("/signup", response_class=HTMLResponse)
async def signup_form(request: Request):
    return _templates(request).TemplateResponse(
        "signup.html",
        {"request": request, "error": request.query_params.get("error", "")},
    )


@router.post("/signup")
async def signup_submit(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    display_name: str = Form(""),
):
    conn = db.connect()
    username = username.strip().lower()
    email = email.strip().lower()
    if not username or not email or not password:
        return RedirectResponse(
            url="/signup?error=all+fields+required", status_code=303,
        )
    if conn.execute("SELECT 1 FROM users WHERE username=? OR email=?",
                    (username, email)).fetchone():
        return RedirectResponse(
            url="/signup?error=username+or+email+taken", status_code=303,
        )
    count = int(conn.execute("SELECT COUNT(*) FROM users WHERE role='buyer'").fetchone()[0])
    uid = f"buyer_n{count+1:05d}"
    with db.transaction() as tx:
        tx.execute(
            "INSERT INTO users (id, username, email, password_hash, role, "
            "display_name, created_at) VALUES (?, ?, ?, ?, 'buyer', ?, ?)",
            (uid, username, email, auth.hash_password(password),
             display_name or username, app_main.now_value()),
        )
        db.log_audit(
            tx,
            occurred_at=app_main.now_value(),
            operation="signup_succeeded",
            target_type="user",
            target_id=uid,
            payload={"username": username},
        )
    resp = RedirectResponse(url="/", status_code=303)
    auth.set_session_cookie(resp, uid)
    return resp


@router.get("/logout")
async def logout(request: Request):
    resp = RedirectResponse(url="/", status_code=303)
    auth.clear_session_cookie(resp)
    return resp
