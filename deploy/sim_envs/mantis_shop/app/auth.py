"""Mock auth surface — email-password + Google-style OAuth.

Per issue #387: real B2B SaaS starts with a login screen and a chunk of
production agent failures happen *at the login step*. This module gives
the sim env that surface without dragging in a real IdP.

Shape
-----

* ``users`` table holds canonical creds. Password is plain sha256 hex
  (no bcrypt — these are sim envs, not prod).
* Session = signed cookie ``user_id|issued_at|hmac_sha256``. HMAC key
  is ``ENV_SESSION_SECRET`` (defaulted for dev; harness should override
  per run so signatures don't leak across resets).
* ``ENV_REQUIRE_AUTH=1`` flips the gate on ``/admin/*`` and
  ``/checkout/*``. Default ``0`` keeps the existing T01–T05 oracle
  contract — they grade DB state and don't care about the audit row.
* The new ``T06_login_then_buy_jacket`` oracle reads
  ``audit_log.operation='login_succeeded'`` and only passes if the
  agent went through the real flow (cookie-mint via
  ``/__env__/login_as`` does NOT write that row).

OAuth mock
----------

Same env is both IdP and relying party. ``/oauth/authorize`` renders a
Google-style consent page ("Continue as demo@mantis.example"); POST
mints a one-time code, ``/oauth/callback`` exchanges it for a session.
No real crypto — shape only.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from typing import Any

from fastapi import Request, Response
from fastapi.responses import RedirectResponse

from . import db

SESSION_COOKIE = "mantis_session"
SESSION_TTL_DEFAULT_S = 60 * 60  # 1h
LOGIN_PATH = "/login"
OAUTH_AUTHORIZE_PATH = "/oauth/authorize"
OAUTH_CALLBACK_PATH = "/oauth/callback"

# OAuth code TTL — short, like Google's.
_OAUTH_CODE_TTL_S = 60
_OAUTH_CODES: dict[str, dict[str, Any]] = {}


# ── env knobs ─────────────────────────────────────────────────────────


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


# ── password ──────────────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def verify_password(plain: str, stored: str) -> bool:
    return hmac.compare_digest(hash_password(plain), stored)


# ── session cookie ────────────────────────────────────────────────────


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


# ── user lookup ───────────────────────────────────────────────────────


def lookup_user_by_id(user_id: str) -> dict[str, Any] | None:
    conn = db.connect()
    row = conn.execute(
        "SELECT id, email, role, customer_id, oauth_subject FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    return dict(row) if row else None


def lookup_user_by_email(email: str) -> dict[str, Any] | None:
    conn = db.connect()
    row = conn.execute(
        "SELECT id, email, password_hash, role, customer_id, oauth_subject "
        "FROM users WHERE email = ?",
        (email.strip().lower(),),
    ).fetchone()
    return dict(row) if row else None


def current_user(request: Request) -> dict[str, Any] | None:
    raw = request.cookies.get(SESSION_COOKIE)
    user_id = verify_session(raw or "")
    if user_id is None:
        return None
    return lookup_user_by_id(user_id)


# ── gating ────────────────────────────────────────────────────────────


def require_user(
    request: Request, *, roles: list[str] | None = None
) -> Response | None:
    """Return a redirect-to-login if the request is gated but unauthed.

    Returns ``None`` if auth is disabled, or if the user is present
    *and* matches one of ``roles`` (when given). Callers do::

        if redirect := require_user(request, roles=["admin"]):
            return redirect
    """
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


def effective_customer_id(request: Request) -> str:
    """Resolve the acting customer for checkout flows.

    When auth is required: pull from the session. When disabled
    (legacy behaviour): fall back to the seeded canonical customer so
    T01–T05 oracles keep grading against the same shape.
    """
    user = current_user(request)
    if user and user.get("customer_id"):
        return str(user["customer_id"])
    return "customer_00001"


# ── OAuth mock store ──────────────────────────────────────────────────


def issue_oauth_code(*, user_id: str, redirect_uri: str, state: str) -> str:
    """Mint a one-time code bound to (user, redirect_uri). Expires in 60s."""
    code = secrets.token_urlsafe(16)
    _OAUTH_CODES[code] = {
        "user_id": user_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "expires_at": time.time() + _OAUTH_CODE_TTL_S,
    }
    return code


def consume_oauth_code(code: str) -> dict[str, Any] | None:
    """Atomic consume — code is single-use."""
    entry = _OAUTH_CODES.pop(code, None)
    if entry is None:
        return None
    if time.time() > entry["expires_at"]:
        return None
    return entry


def clear_oauth_codes() -> None:
    _OAUTH_CODES.clear()
