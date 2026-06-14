"""``build_auth_router(config)`` — the drop-in auth surface.

Mount it on any FastAPI app::

    from app.authflow import build_auth_router, AuthConfig
    app.include_router(build_auth_router(AuthConfig(templates=..., backend=...)))

It contributes every login route (password, OAuth × N providers, email
magic-link, email OTP, passkey) plus ``/logout``. Each successful method
funnels through :func:`_complete_login`, which sets the signed session
cookie and writes exactly one ``login_succeeded`` audit row (tagged with
``via`` and, for OAuth, ``provider``) — the single fact every oracle
grades on.
"""

from __future__ import annotations

import secrets
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from . import providers as providers_mod
from . import sessions, tokens
from .config import AuthConfig
from .passwords import verify_password


def _otp_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def build_auth_router(config: AuthConfig) -> APIRouter:  # noqa: C901
    router = APIRouter()
    be = config.backend
    tpl = config.templates

    def render(name: str, request: Request, **ctx: Any) -> HTMLResponse:
        return tpl.TemplateResponse(
            name,
            {"request": request, "config": config,
             "current_user": _current_user(request), **ctx},
        )

    def _current_user(request: Request) -> dict[str, Any] | None:
        uid = sessions.session_user_id(request)
        return be.lookup_user_by_id(uid) if uid else None

    def _complete_login(
        target: str, user: dict[str, Any], *, via: str,
        provider: str | None = None,
    ) -> Response:
        resp = RedirectResponse(target or config.post_login_redirect,
                                status_code=303)
        sessions.set_session_cookie(resp, user["id"])
        be.record_login(user_id=user["id"], via=via,
                        email=user["email"], provider=provider)
        return resp

    # ── password ────────────────────────────────────────────────────────

    @router.get("/login", response_class=HTMLResponse)
    async def login_get(request: Request) -> HTMLResponse:
        layout = request.query_params.get("layout") or config.default_layout
        return render(
            "login.html", request,
            layout=layout,
            next=request.query_params.get("next") or config.post_login_redirect,
            error=request.query_params.get("error") or "",
            providers=[providers_mod.PROVIDERS[p]
                       for p in config.enabled_providers
                       if p in providers_mod.PROVIDERS],
        )

    @router.post("/login")
    async def login_post(
        request: Request,
        email: str = Form(""),
        password: str = Form(""),
        next: str = Form(""),
    ) -> Response:
        user = be.lookup_user_by_email(email)
        if user is None or not verify_password(password, user.get("password_hash")):
            be.record_event(
                operation="login_failed", target_id="anon",
                payload={"email": email.strip().lower(), "via": "password"},
            )
            nxt = next or config.post_login_redirect
            return RedirectResponse(
                f"/login?error=Invalid+email+or+password&next={nxt}",
                status_code=303,
            )
        return _complete_login(next, user, via="password")

    @router.post("/logout")
    async def logout_post(request: Request) -> Response:
        user = _current_user(request)
        resp = RedirectResponse("/login", status_code=303)
        sessions.clear_session_cookie(resp)
        if user is not None:
            be.record_event(operation="logout", target_id=user["id"],
                            payload={"email": user["email"]})
        return resp

    # ── OAuth (multi-provider) ──────────────────────────────────────────

    def _provider_or_404(slug: str) -> providers_mod.Provider | None:
        if slug not in config.enabled_providers:
            return None
        return providers_mod.get_provider(slug)

    @router.get("/auth/oauth/{slug}/authorize", response_class=HTMLResponse)
    async def oauth_authorize(request: Request, slug: str) -> Response:
        provider = _provider_or_404(slug)
        if provider is None:
            return RedirectResponse(
                "/login?error=unknown+oauth+provider", status_code=303)
        redirect_uri = request.query_params.get(
            "redirect_uri", f"/auth/oauth/{slug}/callback")
        state = request.query_params.get("state", "")
        return render(
            provider.template, request,
            provider=provider, stage="picker",
            accounts=be.lookup_oauth_accounts(slug),
            redirect_uri=redirect_uri, state=state, error="",
        )

    @router.post("/auth/oauth/{slug}/consent", response_class=HTMLResponse)
    async def oauth_consent(
        request: Request, slug: str,
        subject: str = Form(""),
        redirect_uri: str = Form(""),
        state: str = Form(""),
    ) -> Response:
        provider = _provider_or_404(slug)
        if provider is None:
            return RedirectResponse(
                "/login?error=unknown+oauth+provider", status_code=303)
        user = be.lookup_user_by_oauth(slug, subject)
        if user is None:
            return RedirectResponse(
                f"/auth/oauth/{slug}/authorize?error=unknown+account&state={state}",
                status_code=303)
        return render(
            provider.template, request,
            provider=provider, stage="consent",
            account=user, subject=subject,
            redirect_uri=redirect_uri or f"/auth/oauth/{slug}/callback",
            state=state,
        )

    @router.post("/auth/oauth/{slug}/grant", response_class=HTMLResponse)
    async def oauth_grant(
        request: Request, slug: str,
        subject: str = Form(""),
        redirect_uri: str = Form(""),
        state: str = Form(""),
    ) -> Response:
        provider = _provider_or_404(slug)
        if provider is None:
            return RedirectResponse(
                "/login?error=unknown+oauth+provider", status_code=303)
        user = be.lookup_user_by_oauth(slug, subject)
        if user is None:
            return RedirectResponse(
                f"/auth/oauth/{slug}/authorize?error=unknown+account&state={state}",
                status_code=303)
        redirect_uri = redirect_uri or f"/auth/oauth/{slug}/callback"
        code = tokens.issue_oauth_code(
            user_id=user["id"], provider=slug,
            redirect_uri=redirect_uri, state=state)
        be.record_event(
            operation="oauth_consent", target_id=user["id"],
            payload={"provider": slug, "subject": subject})
        sep = "&" if "?" in redirect_uri else "?"
        target = f"{redirect_uri}{sep}code={code}&state={state}"
        return render("oauth_redirect.html", request,
                      provider=provider, target=target)

    @router.get("/auth/oauth/{slug}/callback")
    async def oauth_callback(request: Request, slug: str) -> Response:
        code = request.query_params.get("code", "")
        if not code:
            return RedirectResponse(
                "/login?error=missing+oauth+code", status_code=303)
        entry = tokens.consume_oauth_code(code)
        if entry is None or entry["provider"] != slug:
            return RedirectResponse(
                "/login?error=oauth+code+expired+or+used", status_code=303)
        user = be.lookup_user_by_id(entry["user_id"])
        if user is None:
            return RedirectResponse(
                "/login?error=oauth+user+not+found", status_code=303)
        return _complete_login(
            config.post_login_redirect, user, via="oauth", provider=slug)

    # ── email magic link ────────────────────────────────────────────────

    @router.get("/auth/magic", response_class=HTMLResponse)
    async def magic_get(request: Request) -> HTMLResponse:
        return render("magic_request.html", request,
                      error=request.query_params.get("error") or "")

    @router.post("/auth/magic", response_class=HTMLResponse)
    async def magic_post(request: Request, email: str = Form("")) -> Response:
        user = be.lookup_user_by_email(email)
        token = secrets.token_urlsafe(20)
        # Always render the same "sent" page (don't leak which emails exist).
        if user is not None:
            link = f"/auth/magic/verify?token={token}"
            be.deliver_email(
                to_email=user["email"], kind="magic_link",
                subject=f"Your {config.app_name} sign-in link",
                body=("Click to finish signing in:\n\n"
                      f"{link}\n\nThis link expires shortly and works once."),
                token=token,
            )
            be.record_event(operation="magic_link_requested",
                            target_id=user["id"],
                            payload={"email": user["email"]})
        return render("magic_sent.html", request, email=email.strip())

    @router.get("/auth/magic/verify")
    async def magic_verify(request: Request) -> Response:
        token = request.query_params.get("token", "")
        user = be.consume_magic_token(token) if token else None
        if user is None:
            return RedirectResponse(
                "/auth/magic?error=link+expired+or+already+used",
                status_code=303)
        be.record_event(operation="magic_link_consumed",
                        target_id=user["id"], payload={"email": user["email"]})
        return _complete_login(config.post_login_redirect, user,
                               via="magic_link")

    # ── email OTP ───────────────────────────────────────────────────────

    @router.get("/auth/otp", response_class=HTMLResponse)
    async def otp_get(request: Request) -> HTMLResponse:
        return render("otp_request.html", request,
                      error=request.query_params.get("error") or "")

    @router.post("/auth/otp", response_class=HTMLResponse)
    async def otp_post(request: Request, email: str = Form("")) -> Response:
        user = be.lookup_user_by_email(email)
        code = _otp_code()
        if user is not None:
            be.deliver_email(
                to_email=user["email"], kind="otp",
                subject=f"{config.app_name} verification code: {code}",
                body=(f"Your verification code is {code}\n\n"
                      "Enter it on the sign-in screen. It expires shortly."),
                code=code,
            )
            be.record_event(operation="otp_requested", target_id=user["id"],
                            payload={"email": user["email"]})
        return render("otp_verify.html", request, email=email.strip(), error="")

    @router.post("/auth/otp/verify")
    async def otp_verify(
        request: Request, email: str = Form(""), code: str = Form(""),
    ) -> Response:
        user = be.consume_otp(email.strip(), code.strip())
        if user is None:
            return render("otp_verify.html", request, email=email.strip(),
                          error="Invalid or expired code.")
        be.record_event(operation="otp_verified", target_id=user["id"],
                        payload={"email": user["email"]})
        return _complete_login(config.post_login_redirect, user,
                               via="email_otp")

    # ── passkey (simulated WebAuthn) ────────────────────────────────────

    @router.get("/auth/passkey", response_class=HTMLResponse)
    async def passkey_get(request: Request) -> HTMLResponse:
        challenge = tokens.issue_passkey_challenge()
        return render("passkey.html", request,
                      challenge=challenge,
                      passkeys=be.list_passkeys(),
                      error=request.query_params.get("error") or "")

    @router.post("/auth/passkey/assert")
    async def passkey_assert(
        request: Request, cred_id: str = Form(""), challenge: str = Form(""),
    ) -> Response:
        # A real authenticator signs the challenge; we accept any
        # registered credential whose challenge is live + unused.
        if not tokens.consume_passkey_challenge(challenge):
            return RedirectResponse(
                "/auth/passkey?error=challenge+expired", status_code=303)
        cred = be.get_passkey(cred_id)
        if cred is None:
            return RedirectResponse(
                "/auth/passkey?error=unknown+passkey", status_code=303)
        user = be.lookup_user_by_id(cred["user_id"])
        if user is None:
            return RedirectResponse(
                "/auth/passkey?error=passkey+user+not+found", status_code=303)
        be.bump_passkey_sign_count(cred_id)
        be.record_event(operation="passkey_asserted", target_id=user["id"],
                        payload={"cred_id": cred_id, "label": cred.get("label")})
        return _complete_login(config.post_login_redirect, user, via="passkey")

    return router
