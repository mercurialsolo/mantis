"""Each oracle returns the same verdict twice on the same DB state, and
fails out of the box (no one has signed in yet).
"""

from __future__ import annotations

import sqlite3

import pytest

TASK_IDS = [
    "T01_password_login",
    "T02_oauth_google",
    "T03_oauth_github",
    "T04_oauth_microsoft",
    "T05_oauth_okta",
    "T06_magic_link_email",
    "T07_email_otp",
    "T08_passkey",
]


@pytest.fixture
def seeded_db():
    from app import db, seed as seed_mod  # noqa: PLC0415

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(db.SCHEMA)
    seed_mod.seed(conn, seed_val=42, fake_now="2026-01-15T09:00:00Z")
    return conn


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_oracle_is_deterministic(seeded_db, task_id):
    from app.oracles import grade  # noqa: PLC0415

    r1 = grade(task_id, seeded_db, now="2026-01-15T09:00:00Z", seed_val=42)
    r2 = grade(task_id, seeded_db, now="2026-01-15T09:00:00Z", seed_val=42)
    assert r1 == r2, f"{task_id}: oracle is non-deterministic"


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_oracle_fails_on_fresh_seed(seeded_db, task_id):
    """No login has happened yet, so every scenario must fail."""
    from app.oracles import grade  # noqa: PLC0415

    r = grade(task_id, seeded_db, now="2026-01-15T09:00:00Z", seed_val=42)
    assert r["passed"] is False
    assert r["score"] == 0.0


def test_oracle_unregistered_task_fails_gracefully(seeded_db):
    from app.oracles import grade  # noqa: PLC0415

    r = grade("DOES_NOT_EXIST", seeded_db,
              now="2026-01-15T09:00:00Z", seed_val=42)
    assert r["passed"] is False
    assert "no oracle" in " ".join(r["reasons"]).lower()
    assert r["task_id"] == "DOES_NOT_EXIST"


def test_seed_enrolls_demo_user_across_methods(seeded_db):
    """user_00001 (Ada) is the canonical all-methods account."""
    uid = "user_00001"
    pw = seeded_db.execute(
        "SELECT password_hash FROM users WHERE id = ?", (uid,)).fetchone()
    assert pw["password_hash"], "demo user should have a password"
    providers = {r["provider"] for r in seeded_db.execute(
        "SELECT provider FROM oauth_identities WHERE user_id = ?", (uid,))}
    assert providers == {"google", "github", "microsoft", "okta"}
    passkeys = seeded_db.execute(
        "SELECT COUNT(*) FROM passkey_credentials WHERE user_id = ?",
        (uid,)).fetchone()[0]
    assert passkeys >= 1
