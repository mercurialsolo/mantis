"""Mock auth surface — email-password + Google-style OAuth (#387).

Mirrors ``mantis_shop/app/auth.py``. CRM users in this env are all
*agents* (no "buyer" role), so no ``effective_customer_id`` indirection
— the session user IS the actor on every mutation.

Two extra columns are added to the existing ``users`` table:
``password_hash`` + ``oauth_subject``. Existing CRM seed users get a
default password so any of them can be impersonated for tests.

``ENV_REQUIRE_AUTH=1`` flips the gate on every non-``/`` route except
``/login``, ``/logout``, ``/oauth/*``, ``/__env__/*``, ``/static/*``.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from typing import Any

from fastapi import Request, Response

from . import db

SESSION_COOKIE = "mantis_session"
SESSION_TTL_DEFAULT_S = 60 * 60
LOGIN_PATH = "/login"
OAUTH_AUTHORIZE_PATH = "/oauth/authorize"
OAUTH_CALLBACK_PATH = "/oauth/callback"

_OAUTH_CODE_TTL_S = 60
_OAUTH_CODES: dict[str, dict[str, Any]] = {}


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
    if not stored:
        return False
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
        "SELECT id, email, name, role, oauth_subject FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    return dict(row) if row else None


def lookup_user_by_email(email: str) -> dict[str, Any] | None:
    conn = db.connect()
    row = conn.execute(
        "SELECT id, email, name, password_hash, role, oauth_subject "
        "FROM users WHERE LOWER(email) = ?",
        (email.strip().lower(),),
    ).fetchone()
    return dict(row) if row else None


def current_user(request: Request) -> dict[str, Any] | None:
    raw = request.cookies.get(SESSION_COOKIE)
    user_id = verify_session(raw or "")
    if user_id is None:
        return None
    return lookup_user_by_id(user_id)


def issue_oauth_code(*, user_id: str, redirect_uri: str, state: str) -> str:
    code = secrets.token_urlsafe(16)
    _OAUTH_CODES[code] = {
        "user_id": user_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "expires_at": time.time() + _OAUTH_CODE_TTL_S,
    }
    return code


def consume_oauth_code(code: str) -> dict[str, Any] | None:
    entry = _OAUTH_CODES.pop(code, None)
    if entry is None:
        return None
    if time.time() > entry["expires_at"]:
        return None
    return entry


def clear_oauth_codes() -> None:
    _OAUTH_CODES.clear()
