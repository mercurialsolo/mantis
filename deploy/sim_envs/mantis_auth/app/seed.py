"""Deterministic seed for mantis-auth.

The data model is small on purpose — the env's value is the *auth
surface*, not the records behind it. We seed a handful of accounts, each
enrolled in a different mix of methods so every scenario has a concrete
target:

================  =====================  ===================================
account           email                  enrolled methods
================  =====================  ===================================
user_00001 (Ada)  ada@mantis.example     password, google, github,
                                         microsoft, okta, passkey  ← canonical
user_00002 (Gra)  grace@mantis.example   password (admin), google,
                                         microsoft, passkey
user_00003 (Ala)  alan@mantis.example    okta only (SSO-only, no password)
user_00004..6     <name>@mantis.example  password only (filler)
================  =====================  ===================================

Calling :func:`seed` again hard-resets every table, so it doubles as the
``/__env__/reset`` implementation.
"""

from __future__ import annotations

import sqlite3
from typing import Any

FAKE_NOW_DEFAULT = "2026-01-15T09:00:00Z"

N_USERS = 6
PROVIDERS = ("google", "github", "microsoft", "okta")

_FIRST = ["Ada", "Grace", "Alan", "Katherine", "Edsger", "Barbara"]
_LAST = ["Lovelace", "Hopper", "Turing", "Johnson", "Dijkstra", "Liskov"]

# Fixed credentials for the three named accounts the scenarios target.
PASSWORDS = {
    "user_00001": "hunter2",
    "user_00002": "compiler1",
    # user_00003 is SSO-only — no password.
}

# Which providers each user holds an identity with.
OAUTH_ENROLLMENT = {
    "user_00001": ("google", "github", "microsoft", "okta"),
    "user_00002": ("google", "microsoft"),
    "user_00003": ("okta",),
    "user_00004": ("github",),
}

PASSKEY_ENROLLMENT = {
    "user_00001": "MacBook Touch ID",
    "user_00002": "YubiKey 5C",
}


def _id(prefix: str, i: int) -> str:
    return f"{prefix}_{i:05d}"


def _email(i: int, first: str, last: str) -> str:
    named = {1: "ada", 2: "grace", 3: "alan"}
    if i in named:
        return f"{named[i]}@mantis.example"
    return f"{first.lower()}.{last.lower()}@mantis.example"


def seed(conn: sqlite3.Connection, *, seed_val: int,
         fake_now: str = FAKE_NOW_DEFAULT) -> None:
    """Populate ``conn`` deterministically from ``seed_val``."""
    from .authflow import hash_password  # local import avoids cycle at load

    cur = conn.cursor()

    for table in ("mutations", "emails", "passkey_credentials",
                  "oauth_identities", "users"):
        cur.execute(f"DELETE FROM {table}")

    users: list[tuple[Any, ...]] = []
    oauth: list[tuple[Any, ...]] = []
    passkeys: list[tuple[Any, ...]] = []

    for i in range(1, N_USERS + 1):
        uid = _id("user", i)
        first = _FIRST[(i - 1) % len(_FIRST)]
        last = _LAST[(i - 1) % len(_LAST)]
        name = f"{first} {last}"
        email = _email(i, first, last)
        role = "admin" if i == 2 else "member"
        pw = PASSWORDS.get(uid)
        # user_00003 is deliberately password-less (SSO-only); fillers
        # (4..6) get a deterministic password so they can sign in too.
        if pw is None and i >= 4:
            pw = f"pw-{uid}"
        password_hash = hash_password(pw) if pw else None
        users.append((uid, name, email, 1, password_hash, role, fake_now))

        for provider in OAUTH_ENROLLMENT.get(uid, ()):  # noqa: PLC0206
            oauth.append((
                f"oid_{provider[:2]}_{i:05d}", uid, provider,
                f"{provider}-oauth2|{uid}", email,
            ))

        label = PASSKEY_ENROLLMENT.get(uid)
        if label:
            passkeys.append((_id("cred", i), uid, label, 0, fake_now))

    cur.executemany(
        "INSERT INTO users (id, name, email, is_active, password_hash, "
        "role, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)", users)
    cur.executemany(
        "INSERT INTO oauth_identities (id, user_id, provider, subject, email) "
        "VALUES (?, ?, ?, ?, ?)", oauth)
    cur.executemany(
        "INSERT INTO passkey_credentials (id, user_id, label, sign_count, "
        "created_at) VALUES (?, ?, ?, ?, ?)", passkeys)

    conn.commit()
