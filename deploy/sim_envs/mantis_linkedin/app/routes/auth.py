"""Login / logout / signup routes."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import auth, db, main as app_main

router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, next: str = "/feed/",
                     error: str | None = None):
    tpls = request.app.state.templates
    return tpls.TemplateResponse("login.html", {
        "request": request, "next_path": next, "error": error,
    })


@router.post("/login")
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form("/feed/"),
):
    user = auth.lookup_user_by_email(email)
    if not user or not auth.verify_password(password, user["password_hash"]):
        tpls = request.app.state.templates
        return tpls.TemplateResponse("login.html", {
            "request": request, "next_path": next,
            "error": "Wrong email or password.",
        }, status_code=400)

    resp = RedirectResponse(next or "/feed/", status_code=303)
    auth.set_session_cookie(resp, user["id"])

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
    app_main.emit("login_succeeded", {"user_id": user["id"]})
    app_main.emit_mutation(
        op="login_succeeded", target_type="user",
        target_id=user["id"], payload={"email": user["email"]},
    )
    return resp


@router.post("/logout")
async def logout():
    resp = RedirectResponse("/login", status_code=303)
    auth.clear_session_cookie(resp)
    return resp


@router.get("/signup", response_class=HTMLResponse)
async def signup_form(request: Request):
    tpls = request.app.state.templates
    return tpls.TemplateResponse("signup.html", {"request": request})
