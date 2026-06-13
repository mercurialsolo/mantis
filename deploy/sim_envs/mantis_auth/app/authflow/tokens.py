"""Ephemeral, single-use challenge stores (in-memory, short TTL).

Two kinds of short-lived secret never need to outlive a single auth
ceremony, so they live in memory rather than the DB:

* **OAuth authorization codes** — minted at consent, exchanged once at
  the callback.
* **Passkey challenges** — minted when the passkey page loads, echoed
  back by the (simulated) authenticator assertion.

Magic-link tokens and email OTP codes are *not* here — those are
persisted to the env's ``emails`` table so the agent can read them in
the mock inbox, and so the oracle can prove they were consumed.
"""

from __future__ import annotations

import secrets
import time
from typing import Any

OAUTH_CODE_TTL_S = 60
PASSKEY_CHALLENGE_TTL_S = 120

_OAUTH_CODES: dict[str, dict[str, Any]] = {}
_PASSKEY_CHALLENGES: dict[str, float] = {}


# ── OAuth authorization codes ───────────────────────────────────────────


def issue_oauth_code(*, user_id: str, provider: str,
                     redirect_uri: str, state: str) -> str:
    code = secrets.token_urlsafe(16)
    _OAUTH_CODES[code] = {
        "user_id": user_id,
        "provider": provider,
        "redirect_uri": redirect_uri,
        "state": state,
        "expires_at": time.time() + OAUTH_CODE_TTL_S,
    }
    return code


def consume_oauth_code(code: str) -> dict[str, Any] | None:
    entry = _OAUTH_CODES.pop(code, None)
    if entry is None or time.time() > entry["expires_at"]:
        return None
    return entry


# ── passkey challenges ──────────────────────────────────────────────────


def issue_passkey_challenge() -> str:
    challenge = secrets.token_urlsafe(24)
    _PASSKEY_CHALLENGES[challenge] = time.time() + PASSKEY_CHALLENGE_TTL_S
    return challenge


def consume_passkey_challenge(challenge: str) -> bool:
    expires = _PASSKEY_CHALLENGES.pop(challenge, None)
    return expires is not None and time.time() <= expires


def clear_all() -> None:
    """Wipe ephemeral stores — used by ``/__env__/reset``."""
    _OAUTH_CODES.clear()
    _PASSKEY_CHALLENGES.clear()
