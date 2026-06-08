"""Login / signup / logout."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from .. import auth, db, main as app_main

router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request) -> Response:
    return request.app.state.templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "error": request.query_params.get("error"),
            "next": request.query_params.get("next") or "/",
        },
    )


@router.post("/login")
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
) -> Response:
    user = auth.lookup_user_by_email(email)
    if user is None or not auth.verify_password(password, user["password_hash"]):
        return request.app.state.templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Email or password incorrect.",
                "next": next,
            },
            status_code=400,
        )
    conn = db.connect()
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="login_succeeded",
        target_type="user",
        target_id=user["id"],
        payload={"email": user["email"]},
    )
    conn.commit()
    response = RedirectResponse(next or "/", status_code=303)
    auth.set_session_cookie(response, user["id"])
    return response


@router.post("/logout")
async def logout(request: Request) -> Response:
    response = RedirectResponse("/", status_code=303)
    auth.clear_session_cookie(response)
    return response


@router.get("/signup", response_class=HTMLResponse)
async def signup_form(request: Request) -> Response:
    return request.app.state.templates.TemplateResponse(
        "signup.html",
        {"request": request, "error": None},
    )


@router.post("/signup")
async def signup_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    confirm: str = Form(...),
    role: str = Form("seeker"),
) -> Response:
    if password != confirm:
        return request.app.state.templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Passwords do not match."},
            status_code=400,
        )
    existing = auth.lookup_user_by_email(email)
    if existing:
        return request.app.state.templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Email already registered."},
            status_code=400,
        )
    conn = db.connect()
    # Generate a fresh user id by counting.
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    uid = f"user_signup_{count + 1:05d}"
    if role == "employer":
        uid = f"user_emp_signup_{count + 1:05d}"
    conn.execute(
        "INSERT INTO users (id, email, password_hash, role, name, created_at) "
        "VALUES (?, ?, ?, ?, '', ?)",
        (uid, email.strip().lower(), auth.hash_password(password),
         role if role in {"seeker", "employer"} else "seeker",
         app_main.now_value()),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="user_created",
        target_type="user",
        target_id=uid,
        payload={"email": email, "role": role},
    )
    conn.commit()
    response = RedirectResponse("/", status_code=303)
    auth.set_session_cookie(response, uid)
    return response
