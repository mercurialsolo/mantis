"""Password hashing for the embeddable auth flow.

Plain sha256 hex — this is a *simulated* environment for agent
evaluation, never a production credential store. Kept in its own module
so an embedding env could swap in bcrypt/argon2 without touching the
routes.
"""

from __future__ import annotations

import hashlib
import hmac


def hash_password(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def verify_password(plain: str, stored: str | None) -> bool:
    if not stored:
        return False
    return hmac.compare_digest(hash_password(plain), stored)
