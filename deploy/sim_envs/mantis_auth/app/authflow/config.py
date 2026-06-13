"""Wiring contract between the embeddable auth flow and its host env.

``build_auth_router`` (see ``router.py``) takes an :class:`AuthConfig`.
The config carries the host's Jinja templates plus an :class:`AuthBackend`
— a small protocol the host implements over its own user store. Nothing
in ``authflow`` imports the env's ``db`` module directly, so the flow
drops into any FastAPI env that can satisfy the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from fastapi.templating import Jinja2Templates

# All auth methods this flow knows how to render/handle.
ALL_METHODS = ("password", "oauth", "magic_link", "email_otp", "passkey")
# Visual variants for the password sign-in page.
ALL_LAYOUTS = ("centered", "split", "minimal")


class AuthBackend(Protocol):
    """Everything the auth flow needs from its host's data layer.

    Returned user dicts must carry at least ``id``, ``email``, ``name``,
    ``role`` (password lookups additionally carry ``password_hash``).
    """

    # clock ----------------------------------------------------------------
    def now(self) -> str: ...

    # users ----------------------------------------------------------------
    def lookup_user_by_id(self, user_id: str) -> dict[str, Any] | None: ...
    def lookup_user_by_email(self, email: str) -> dict[str, Any] | None: ...

    # oauth ----------------------------------------------------------------
    def lookup_oauth_accounts(self, provider: str) -> list[dict[str, Any]]: ...
    def lookup_user_by_oauth(
        self, provider: str, subject: str
    ) -> dict[str, Any] | None: ...

    # passkeys -------------------------------------------------------------
    def list_passkeys(self) -> list[dict[str, Any]]: ...
    def get_passkey(self, cred_id: str) -> dict[str, Any] | None: ...
    def bump_passkey_sign_count(self, cred_id: str) -> None: ...

    # email (magic link + OTP) --------------------------------------------
    def deliver_email(
        self, *, to_email: str, kind: str, subject: str, body: str,
        token: str | None = None, code: str | None = None,
    ) -> None: ...
    def consume_magic_token(self, token: str) -> dict[str, Any] | None: ...
    def consume_otp(self, email: str, code: str) -> dict[str, Any] | None: ...

    # audit ----------------------------------------------------------------
    def record_login(
        self, *, user_id: str, via: str, email: str, provider: str | None = None,
    ) -> None: ...
    def record_event(
        self, *, operation: str, target_id: str, payload: dict[str, Any],
    ) -> None: ...


@dataclass
class AuthConfig:
    templates: Jinja2Templates
    backend: AuthBackend
    app_name: str = "Mantis Console"
    # Where to land the session after any successful sign-in.
    post_login_redirect: str = "/console"
    # Subset of ALL_METHODS to expose. Order drives the login page.
    enabled_methods: tuple[str, ...] = ALL_METHODS
    # Subset of provider slugs to expose for OAuth.
    enabled_providers: tuple[str, ...] = ("google", "github", "microsoft", "okta")
    default_layout: str = "centered"
    extra: dict[str, Any] = field(default_factory=dict)

    def method_enabled(self, method: str) -> bool:
        return method in self.enabled_methods
