"""Stateless signed session cookies — shared by every auth method.

Format: ``user_id|issued_at|hmac_sha256``. The secret comes from
``AUTH_SESSION_SECRET`` and the TTL from ``AUTH_SESSION_TTL_S`` so an env
embedding this flow only has to set two env vars.

This module is deliberately storage-free: a session is verifiable from
the cookie alone, so the same helper works whether the embedding env
keeps users in SQLite, Postgres, or a dict.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time

from fastapi import Request, Response

SESSION_COOKIE = "mantis_auth_session"
SESSION_TTL_DEFAULT_S = 60 * 60


def session_secret() -> str:
    return os.environ.get("AUTH_SESSION_SECRET", "dev-only-do-not-use-in-prod")


def session_ttl_s() -> int:
    raw = os.environ.get("AUTH_SESSION_TTL_S")
    try:
        return int(raw) if raw else SESSION_TTL_DEFAULT_S
    except ValueError:
        return SESSION_TTL_DEFAULT_S


def _sign(payload: str) -> str:
    return hmac.new(
        session_secret().encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def mint_session(user_id: str, *, now: float | None = None) -> str:
    issued = int(now if now is not None else time.time())
    payload = f"{user_id}|{issued}"
    return f"{payload}|{_sign(payload)}"


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
    if not hmac.compare_digest(sig, _sign(f"{user_id}|{issued_at}")):
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


def session_user_id(request: Request) -> str | None:
    return verify_session(request.cookies.get(SESSION_COOKIE) or "")
