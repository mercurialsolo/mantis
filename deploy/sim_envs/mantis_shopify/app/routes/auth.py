"""Login / logout for mantis-shopify (optional; default is no-auth)."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import auth as auth_lib

router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, next: str = "/"):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "next_path": next,
            "error": None,
        },
    )


@router.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
):
    templates = request.app.state.templates
    row = auth_lib.lookup_user_by_email(email)
    if row is None or not auth_lib.verify_password(password, row.get("password_hash", "")):
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "next_path": next,
                "error": "Incorrect email or password.",
            },
            status_code=400,
        )
    response = RedirectResponse(next or "/", status_code=303)
    auth_lib.set_session_cookie(response, row["id"])
    return response


@router.post("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=303)
    auth_lib.clear_session_cookie(response)
    return response
