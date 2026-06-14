"""Product surface behind the auth wall — the thing you sign in to reach.

Intentionally small: a dashboard greeting and a "security" account page
that lists the signed-in user's enrolled auth methods. Reaching either
page at all is the agent's reward for completing a login, so we emit a
``console_viewed`` event the oracle can cross-check against the login.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from .. import db, main as app_main

router = APIRouter()


def _enrolled_methods(user_id: str) -> dict[str, object]:
    conn = db.connect()
    has_pw = conn.execute(
        "SELECT password_hash FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    providers = [r["provider"] for r in conn.execute(
        "SELECT provider FROM oauth_identities WHERE user_id = ? ORDER BY provider",
        (user_id,)).fetchall()]
    passkeys = [r["label"] for r in conn.execute(
        "SELECT label FROM passkey_credentials WHERE user_id = ? ORDER BY label",
        (user_id,)).fetchall()]
    return {
        "password": bool(has_pw and has_pw["password_hash"]),
        "oauth_providers": providers,
        "passkeys": passkeys,
    }


@router.get("/console", response_class=HTMLResponse)
async def console(request: Request) -> Response:
    user = getattr(request.state, "current_user", None)
    if user is None:
        return RedirectResponse("/login?next=/console", status_code=303)
    app_main.emit("console_viewed", {"target_id": user["id"]})
    return app_main.app.state.templates.TemplateResponse(
        "console.html",
        {"request": request, "config": app_main.app.state.auth_config,
         "current_user": user, "enrolled": _enrolled_methods(user["id"])},
    )


@router.get("/account", response_class=HTMLResponse)
async def account(request: Request) -> Response:
    user = getattr(request.state, "current_user", None)
    if user is None:
        return RedirectResponse("/login?next=/account", status_code=303)
    return app_main.app.state.templates.TemplateResponse(
        "account.html",
        {"request": request, "config": app_main.app.state.auth_config,
         "current_user": user, "enrolled": _enrolled_methods(user["id"])},
    )
