"""``authflow`` — an embeddable, multi-method auth surface for sim envs.

Public surface::

    from app.authflow import build_auth_router, AuthConfig, AuthBackend
    from app.authflow import sessions  # cookie verify for your middleware

Supported methods: password, OAuth (Google / GitHub / Microsoft / Okta),
email magic-link, email OTP, and passkey. The whole flow is storage-free
except through the :class:`AuthBackend` protocol the host supplies, so it
drops into any FastAPI env. See ``router.build_auth_router`` and the
env's ``backend.py`` for a worked implementation.
"""

from __future__ import annotations

from . import providers, sessions, tokens
from .config import ALL_LAYOUTS, ALL_METHODS, AuthBackend, AuthConfig
from .passwords import hash_password, verify_password
from .providers import PROVIDERS, Provider, get_provider
from .router import build_auth_router

__all__ = [
    "ALL_LAYOUTS",
    "ALL_METHODS",
    "AuthBackend",
    "AuthConfig",
    "PROVIDERS",
    "Provider",
    "build_auth_router",
    "get_provider",
    "hash_password",
    "providers",
    "sessions",
    "tokens",
    "verify_password",
]
