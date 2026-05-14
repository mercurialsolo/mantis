"""Mock auth routes — ``/login``, ``/logout``, OAuth-mock pair (#387).

The five surfaces (all server-rendered HTML, no JS-only validation, to
match the rest of the env):

* ``GET  /login``                — render the login form. Honours
  ``?next=`` to bounce the user back where they were headed.
* ``POST /login``                — verify creds, set session cookie,
  audit ``login_succeeded`` (or ``login_failed``), redirect.
* ``POST /logout``               — clear cookie, audit ``logout``,
  redirect to ``/login``.
* ``GET  /oauth/authorize``      — Google-style consent screen. POSTs
  to itself to grant the code.
* ``GET  /oauth/callback``       — exchanges the code for a session.

The OAuth flow is single-issuer: the env is both the IdP and the
relying party. ``client_id=mantis-shop`` is the only registered client.
That's enough to give an agent the *shape* of "click sign-in →
consent → land back logged in" without dragging in a real OAuth lib.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from .. import auth, db, main as app_main

router = APIRouter()

OAUTH_CLIENT_ID = "mantis-shop"
OAUTH_DEFAULT_REDIRECT = "/oauth/callback"


# ── audit helper ──────────────────────────────────────────────────────


def _audit(operation: str, target_id: str, payload: dict[str, Any]) -> None:
    conn = db.connect()
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation=operation,
        target_type="user",
        target_id=target_id,
        payload=payload,
    )
    conn.commit()
    app_main.emit(f"mutation.{operation}", {"target_id": target_id, **payload})


# ── /login ────────────────────────────────────────────────────────────


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


# ── /logout ───────────────────────────────────────────────────────────


@router.post("/logout")
async def logout_post(request: Request) -> Response:
    user = auth.current_user(request)
    resp = RedirectResponse("/login", status_code=303)
    auth.clear_session_cookie(resp)
    if user is not None:
        _audit("logout", target_id=user["id"], payload={"email": user["email"]})
    return resp


# ── /oauth/authorize ─────────────────────────────────────────────────


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
            # Two known IdP subjects so the consent screen offers a
            # realistic "choose-an-account" UI.
            "accounts": [
                {"sub": "google-oauth2|demo-001",
                 "label": "demo@mantis.example"},
                {"sub": "google-oauth2|admin-001",
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
    """Render the consent screen for the picked account.

    Real Google OAuth is two screens — account picker, then a separate
    consent screen listing scopes and showing the chosen account. This
    matches that shape so an agent has to actually navigate both pages
    before the code is issued.
    """
    user = auth.lookup_user_by_oauth_subject(oauth_subject) \
        if hasattr(auth, "lookup_user_by_oauth_subject") else None
    if user is None:
        # Fallback to a direct DB lookup so we don't have to add an extra
        # helper to ``auth.py``.
        conn = db.connect()
        row = conn.execute(
            "SELECT id, email, oauth_subject FROM users WHERE oauth_subject = ?",
            (oauth_subject,),
        ).fetchone()
        user = dict(row) if row else None
    if user is None:
        return RedirectResponse(
            f"/oauth/authorize?error=unknown+account&state={state}",
            status_code=303,
        )
    return app_main.app.state.templates.TemplateResponse(
        "oauth_consent.html",
        {
            "request": request,
            "user": user,
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
    """User clicked 'Allow' — mint a single-use code + render the
    'Redirecting back to {client_id}…' interstitial.

    The interstitial auto-redirects via ``<meta http-equiv=refresh>``
    so the agent sees the realistic two-hop shape (consent → IdP
    redirect page → app callback) rather than an instant 303.
    """
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


# ── /oauth/callback ──────────────────────────────────────────────────


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
