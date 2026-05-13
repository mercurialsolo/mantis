"""Each oracle returns the same verdict twice on the same DB state.

This is the easy half of "5 reruns → 5 identical scores". The harder
half (running a real agent five times) lives in CI integration, not
unit tests, but oracle determinism guarantees the unit-level invariant.
"""

from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture
def seeded_db():
    from app import db, seed as seed_mod  # noqa: PLC0415

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(db.SCHEMA)
    seed_mod.seed(conn, seed_val=42, fake_now="2026-01-15T09:00:00Z")
    return conn


@pytest.mark.parametrize("task_id", [
    "T01_triage_inbox",
    "T02_shipping_macro",
    "T03_merge_outage_dupes",
    "T04_sla_rescue",
    "T05_redact_and_reply",
])
def test_oracle_is_deterministic(seeded_db, task_id):
    from app.oracles import grade  # noqa: PLC0415

    r1 = grade(task_id, seeded_db, now="2026-01-15T09:00:00Z", seed_val=42)
    r2 = grade(task_id, seeded_db, now="2026-01-15T09:00:00Z", seed_val=42)
    assert r1 == r2, f"{task_id}: oracle is non-deterministic"


def test_oracle_unregistered_task_fails_gracefully(seeded_db):
    from app.oracles import grade  # noqa: PLC0415

    r = grade("DOES_NOT_EXIST", seeded_db,
              now="2026-01-15T09:00:00Z", seed_val=42)
    assert r["passed"] is False
    assert "no oracle" in " ".join(r["reasons"]).lower()


def test_oracle_t03_fails_without_merge(seeded_db):
    """Out of the box, the 30 outage dupes are live → oracle fails."""
    from app.oracles import grade  # noqa: PLC0415

    r = grade("T03_merge_outage_dupes", seeded_db,
              now="2026-01-15T09:00:00Z", seed_val=42)
    assert r["passed"] is False


def test_oracle_t05_fails_when_no_reply(seeded_db):
    """No public reply yet → T05 must fail."""
    from app.oracles import grade  # noqa: PLC0415

    r = grade("T05_redact_and_reply", seeded_db,
              now="2026-01-15T09:00:00Z", seed_val=42)
    assert r["passed"] is False
    assert any("no public reply" in s.lower() for s in r["reasons"])


def test_oracle_t05_passes_with_clean_reply(seeded_db):
    """Synthesise a clean reply + audit row and confirm T05 passes."""
    from app import db as env_db  # noqa: PLC0415
    from app.oracles import grade  # noqa: PLC0415

    now = "2026-01-15T09:00:00Z"
    new_reply_id = "reply_test_001"
    seeded_db.execute(
        "INSERT INTO replies (id, ticket_id, author_id, body, visibility, cc, bcc, created_at) "
        "VALUES (?, 'ticket_04421', 'agent_001', "
        "'Hi, we have processed your refund and it will land by Friday.', "
        "'public', '[]', '[]', ?)",
        (new_reply_id, now),
    )
    env_db.log_mutation(
        seeded_db, occurred_at=now, operation="reply_posted",
        target_type="ticket", target_id="ticket_04421",
        payload={"reply_id": new_reply_id, "visibility": "public",
                 "body_preview": "Hi, we have processed your refund"},
    )
    seeded_db.commit()

    r = grade("T05_redact_and_reply", seeded_db, now=now, seed_val=42)
    assert r["passed"], r["reasons"]
    assert r["score"] == 1.0


def test_oracle_t05_fails_when_reply_leaks_pii(seeded_db):
    """A reply containing PII flips the score to 0 even if everything else is fine."""
    from app import db as env_db  # noqa: PLC0415
    from app.oracles import grade  # noqa: PLC0415

    now = "2026-01-15T09:00:00Z"
    new_reply_id = "reply_test_002"
    seeded_db.execute(
        "INSERT INTO replies (id, ticket_id, author_id, body, visibility, cc, bcc, created_at) "
        "VALUES (?, 'ticket_04421', 'agent_001', "
        "'Refund issued for SSN 123-45-6789 on card 4242 4242 4242 4242', "
        "'public', '[]', '[]', ?)",
        (new_reply_id, now),
    )
    env_db.log_mutation(
        seeded_db, occurred_at=now, operation="reply_posted",
        target_type="ticket", target_id="ticket_04421",
        payload={"reply_id": new_reply_id, "visibility": "public",
                 "body_preview": "Refund issued..."},
    )
    seeded_db.commit()

    r = grade("T05_redact_and_reply", seeded_db, now=now, seed_val=42)
    assert r["passed"] is False
    assert r["score"] == 0.0
    assert any("pii" in s.lower() for s in r["reasons"])


def test_oracle_t02_fails_on_seed(seeded_db):
    """No replies posted yet → T02 must fail (recall=0)."""
    from app.oracles import grade  # noqa: PLC0415

    r = grade("T02_shipping_macro", seeded_db,
              now="2026-01-15T09:00:00Z", seed_val=42)
    assert r["passed"] is False
    assert r["diff"]["target_count"] >= 1
