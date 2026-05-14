"""Mock auth routes — ``/login``, ``/logout``, OAuth-mock pair (#387).

Mirrors ``mantis_shop/app/routes/auth.py``. Audit goes into the CRM's
``mutations`` table via ``db.log_mutation`` (shop uses ``audit_log`` +
``log_audit`` — same shape, different table name per env design).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from .. import auth, db, main as app_main

router = APIRouter()

OAUTH_CLIENT_ID = "mantis-crm"
OAUTH_DEFAULT_REDIRECT = "/oauth/callback"


def _audit(operation: str, target_id: str, payload: dict[str, Any]) -> None:
    conn = db.connect()
    db.log_mutation(
        conn,
        occurred_at=app_main.now_value(),
        operation=operation,
        target_type="user",
        target_id=target_id,
        payload=payload,
    )
    conn.commit()
    app_main.emit(f"mutation.{operation}", {"target_id": target_id, **payload})


@router.get("/login", response_class=HTMLResponse)
async def login_get(request: Request) -> HTMLResponse:
    return app_main.app.state.templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "next": request.query_params.get("next") or "/",
            "error": request.query_params.get("error") or "",
            "current_user": auth.current_user(request),
        },
    )


@router.post("/login")
async def login_post(
    request: Request,
    email: str = Form(""),
    password: str = Form(""),
    next: str = Form("/"),
) -> Response:
    user = auth.lookup_user_by_email(email)
    if user is None or not auth.verify_password(password, user["password_hash"]):
        _audit(
            "login_failed",
            target_id="anon",
            payload={"email": email.strip().lower(), "via": "password"},
        )
        return RedirectResponse(
            f"/login?error=Invalid+email+or+password&next={next}",
            status_code=303,
        )

    resp = RedirectResponse(next or "/", status_code=303)
    auth.set_session_cookie(resp, user["id"])
    _audit(
        "login_succeeded",
        target_id=user["id"],
        payload={"email": user["email"], "role": user["role"], "via": "password"},
    )
    return resp


@router.post("/logout")
async def logout_post(request: Request) -> Response:
    user = auth.current_user(request)
    resp = RedirectResponse("/login", status_code=303)
    auth.clear_session_cookie(resp)
    if user is not None:
        _audit("logout", target_id=user["id"], payload={"email": user["email"]})
    return resp


@router.get("/oauth/authorize", response_class=HTMLResponse)
async def oauth_authorize_get(request: Request) -> HTMLResponse:
    client_id = request.query_params.get("client_id", "")
    redirect_uri = request.query_params.get("redirect_uri", OAUTH_DEFAULT_REDIRECT)
    state = request.query_params.get("state", "")
    error = ""
    if client_id and client_id != OAUTH_CLIENT_ID:
        error = f"unknown client_id: {client_id}"
    return app_main.app.state.templates.TemplateResponse(
        "oauth_authorize.html",
        {
            "request": request,
            "client_id": client_id or OAUTH_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "state": state,
            "error": error,
            "accounts": [
                {"sub": "google-oauth2|crm-demo-001",
                 "label": "demo@mantis.example"},
                {"sub": "google-oauth2|crm-admin-001",
                 "label": "admin@mantis.example"},
            ],
        },
    )


@router.post("/oauth/consent", response_class=HTMLResponse)
async def oauth_consent_get(
    request: Request,
    oauth_subject: str = Form(""),
    redirect_uri: str = Form(OAUTH_DEFAULT_REDIRECT),
    state: str = Form(""),
) -> Response:
    """Step 2 of the OAuth mock — consent screen for the picked account."""
    conn = db.connect()
    row = conn.execute(
        "SELECT id, email, oauth_subject FROM users WHERE oauth_subject = ?",
        (oauth_subject,),
    ).fetchone()
    if row is None:
        return RedirectResponse(
            f"/oauth/authorize?error=unknown+account&state={state}",
            status_code=303,
        )
    return app_main.app.state.templates.TemplateResponse(
        "oauth_consent.html",
        {
            "request": request,
            "user": dict(row),
            "client_id": OAUTH_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "state": state,
        },
    )


@router.post("/oauth/consent/grant", response_class=HTMLResponse)
async def oauth_consent_grant(
    request: Request,
    oauth_subject: str = Form(""),
    redirect_uri: str = Form(OAUTH_DEFAULT_REDIRECT),
    state: str = Form(""),
) -> Response:
    """Step 3 — mint a single-use code and render the redirect interstitial."""
    conn = db.connect()
    row = conn.execute(
        "SELECT id FROM users WHERE oauth_subject = ?", (oauth_subject,),
    ).fetchone()
    if row is None:
        return RedirectResponse(
            f"/oauth/authorize?error=unknown+account&state={state}",
            status_code=303,
        )
    code = auth.issue_oauth_code(
        user_id=row["id"], redirect_uri=redirect_uri, state=state,
    )
    _audit(
        "oauth_consent",
        target_id=row["id"],
        payload={"oauth_subject": oauth_subject, "client_id": OAUTH_CLIENT_ID},
    )
    sep = "&" if "?" in redirect_uri else "?"
    target = f"{redirect_uri}{sep}code={code}&state={state}"
    return app_main.app.state.templates.TemplateResponse(
        "oauth_redirect.html",
        {
            "request": request,
            "target": target,
            "client_id": OAUTH_CLIENT_ID,
        },
    )


@router.get("/oauth/callback")
async def oauth_callback(request: Request) -> Response:
    code = request.query_params.get("code", "")
    if not code:
        return RedirectResponse(
            "/login?error=missing+oauth+code", status_code=303,
        )
    entry = auth.consume_oauth_code(code)
    if entry is None:
        return RedirectResponse(
            "/login?error=oauth+code+expired+or+used", status_code=303,
        )
    user = auth.lookup_user_by_id(entry["user_id"])
    if user is None:
        return RedirectResponse(
            "/login?error=oauth+user+not+found", status_code=303,
        )
    resp = RedirectResponse("/", status_code=303)
    auth.set_session_cookie(resp, user["id"])
    _audit(
        "login_succeeded",
        target_id=user["id"],
        payload={"email": user["email"], "role": user["role"], "via": "oauth"},
    )
    return resp
