"""Mock auth surface for mantis-fiverr.

Same shape as mantis_shop: signed-cookie sessions via stdlib hmac,
plain SHA-256 password hash (sim env). ``ENV_REQUIRE_AUTH=1`` flips
the gate on protected routes; default 0 keeps the oracle contracts
working against the canonical `buyer_00001` user.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any

from fastapi import Request, Response
from fastapi.responses import RedirectResponse

from . import db

SESSION_COOKIE = "mantis_session"
SESSION_TTL_DEFAULT_S = 60 * 60  # 1h
LOGIN_PATH = "/login"


def session_secret() -> str:
    return os.environ.get("ENV_SESSION_SECRET", "dev-only-do-not-use-in-prod")


def session_ttl_s() -> int:
    raw = os.environ.get("ENV_SESSION_TTL_S")
    try:
        return int(raw) if raw else SESSION_TTL_DEFAULT_S
    except ValueError:
        return SESSION_TTL_DEFAULT_S


def auth_required() -> bool:
    return os.environ.get("ENV_REQUIRE_AUTH", "0").lower() in {"1", "true", "yes"}


def hash_password(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def verify_password(plain: str, stored: str) -> bool:
    return hmac.compare_digest(hash_password(plain), stored)


def mint_session(user_id: str, *, now: float | None = None) -> str:
    issued = int(now if now is not None else time.time())
    payload = f"{user_id}|{issued}"
    sig = hmac.new(
        session_secret().encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload}|{sig}"


def verify_session(cookie_val: str, *, now: float | None = None) -> str | None:
    if not cookie_val:
        return None
    parts = cookie_val.split("|")
    if len(parts) != 3:
        return None
    user_id, issued_at_s, sig = parts
    try:
        issued_at = int(issued_at_s)
    except ValueError:
        return None
    payload = f"{user_id}|{issued_at}"
    expected = hmac.new(
        session_secret().encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    now_ts = now if now is not None else time.time()
    if now_ts - issued_at > session_ttl_s():
        return None
    return user_id


def set_session_cookie(response: Response, user_id: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        mint_session(user_id),
        httponly=True,
        samesite="lax",
        max_age=session_ttl_s(),
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


def lookup_user_by_id(user_id: str) -> dict[str, Any] | None:
    conn = db.connect()
    row = conn.execute(
        "SELECT id, username, email, role, display_name FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    return dict(row) if row else None


def lookup_user_by_login(login: str) -> dict[str, Any] | None:
    conn = db.connect()
    login = login.strip().lower()
    row = conn.execute(
        "SELECT id, username, email, password_hash, role, display_name "
        "FROM users WHERE email = ? OR username = ?",
        (login, login),
    ).fetchone()
    return dict(row) if row else None


def current_user(request: Request) -> dict[str, Any] | None:
    raw = request.cookies.get(SESSION_COOKIE)
    user_id = verify_session(raw or "")
    if user_id is None:
        return None
    return lookup_user_by_id(user_id)


def require_user(
    request: Request, *, roles: list[str] | None = None
) -> Response | None:
    if not auth_required():
        return None
    user = current_user(request)
    next_path = request.url.path
    if user is None:
        return RedirectResponse(
            f"{LOGIN_PATH}?next={next_path}", status_code=303
        )
    if roles and user.get("role") not in roles:
        return RedirectResponse(LOGIN_PATH, status_code=303)
    return None


def effective_buyer_id(request: Request) -> str:
    """Resolve the acting buyer for checkout flows.

    Mirrors mantis_shop pattern: when auth is required, pull from the
    session; otherwise fall back to the canonical buyer_00001 so the
    oracle contracts keep grading.
    """
    user = current_user(request)
    if user and user.get("role") == "buyer":
        return str(user["id"])
    return "buyer_00001"
