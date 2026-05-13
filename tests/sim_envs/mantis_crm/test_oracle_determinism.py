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
    "T01_tag_reengage",
    "T02_merge_acme_dupes",
    "T03_at_risk_deals",
    "T04_add_meeting_note",
    "T05_pipeline_review",
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


def test_oracle_t01_fails_on_seed(seeded_db):
    """Out of the box, no contact has the 'reengage' tag → oracle must fail."""
    from app.oracles import grade  # noqa: PLC0415

    r = grade("T01_tag_reengage", seeded_db,
              now="2026-01-15T09:00:00Z", seed_val=42)
    # The target set is non-empty (companies at 1M+ ARR + stale) and
    # nothing is tagged yet.
    assert r["passed"] is False
    assert r["diff"]["target_count"] > 0
    assert r["diff"]["tagged_count"] == 0


def test_oracle_t02_fails_without_merge(seeded_db):
    """Out of the box, all 4 acme dupes are live → oracle must fail."""
    from app.oracles import grade  # noqa: PLC0415

    r = grade("T02_merge_acme_dupes", seeded_db,
              now="2026-01-15T09:00:00Z", seed_val=42)
    assert r["passed"] is False
    # 4 live survivors with same email == failure.
    assert len(r["diff"]["survivor_ids"]) == 4


def test_oracle_t05_passes_when_correct_deals_added(seeded_db):
    """Synthesise the target action and confirm the oracle approves."""
    from app.oracles import grade  # noqa: PLC0415
    from app.oracles.t05_pipeline_review import (
        LIST_ID,
        AMOUNT_FLOOR,
        OWNER_ID,
        _end_of_quarter,
        _parse_iso,
    )

    now = "2026-01-15T09:00:00Z"
    now_dt = _parse_iso(now)
    eoq = _end_of_quarter(now_dt).isoformat()
    target_rows = seeded_db.execute(
        "SELECT id FROM deals WHERE owner_id = ? AND amount > ? "
        "AND expected_close >= ? AND expected_close <= ?",
        (OWNER_ID, AMOUNT_FLOOR, now_dt.isoformat(), eoq),
    ).fetchall()
    assert len(target_rows) >= 1, "seed must contain at least one T05 target"

    for r in target_rows:
        seeded_db.execute(
            "INSERT INTO list_members (list_id, member_type, member_id) "
            "VALUES (?, 'deal', ?)",
            (LIST_ID, r["id"]),
        )
    seeded_db.commit()

    verdict = grade("T05_pipeline_review", seeded_db,
                    now=now, seed_val=42)
    assert verdict["passed"], verdict["reasons"]
    assert verdict["score"] == 1.0
