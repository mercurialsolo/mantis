"""Seed determinism — same SEED → identical row IDs + counts + key joins.

Acceptance criterion (#332): "Same SEED → identical row IDs across boots".
We boot two in-memory DBs with the same seed and assert their snapshots
match column-for-column on a representative sample.
"""

from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture
def fresh_db():
    """Build a fresh in-memory DB with a clean schema and return it."""
    def _build(seed_val: int = 42) -> sqlite3.Connection:
        from app import db, seed as seed_mod  # noqa: PLC0415

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(db.SCHEMA)
        seed_mod.seed(conn, seed_val=seed_val,
                      fake_now="2026-01-15T09:00:00Z")
        return conn

    return _build


def _snapshot(conn: sqlite3.Connection, table: str, *, limit: int = 50) -> list[tuple]:
    return [tuple(r) for r in conn.execute(
        f"SELECT * FROM {table} ORDER BY id LIMIT ?", (limit,)
    ).fetchall()]


def _count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_row_counts_match_spec(fresh_db):
    conn = fresh_db(42)
    assert _count(conn, "users") == 12
    assert _count(conn, "companies") == 8_000
    assert _count(conn, "contacts") == 50_000
    assert _count(conn, "deals") == 12_000
    assert _count(conn, "activities") == 200_000
    # 50 random + 1 seeded pipeline review list
    assert _count(conn, "lists") == 51


def test_same_seed_same_snapshot(fresh_db):
    conn_a = fresh_db(42)
    conn_b = fresh_db(42)

    for table in ("users", "companies", "contacts", "deals", "activities"):
        assert _snapshot(conn_a, table) == _snapshot(conn_b, table), \
            f"snapshot mismatch in {table}"


def test_different_seeds_yield_different_data(fresh_db):
    conn_a = fresh_db(42)
    conn_b = fresh_db(43)
    a = _snapshot(conn_a, "contacts", limit=20)
    b = _snapshot(conn_b, "contacts", limit=20)
    # Counts identical, content different.
    assert len(a) == len(b)
    assert a != b


def test_ids_are_zero_padded_strings(fresh_db):
    conn = fresh_db(42)
    sample = conn.execute(
        "SELECT id FROM contacts WHERE id LIKE 'contact_%' "
        "ORDER BY id LIMIT 3"
    ).fetchall()
    ids = [r["id"] for r in sample]
    assert ids == ["contact_00001", "contact_00002", "contact_00003"]


def test_acme_dupes_are_seeded(fresh_db):
    """T02_merge_acme_dupes relies on a known set of 4 dupes."""
    conn = fresh_db(42)
    rows = conn.execute(
        "SELECT id FROM contacts WHERE email = 'alice.lead@acme.com' "
        "AND deleted_at IS NULL ORDER BY id"
    ).fetchall()
    assert [r["id"] for r in rows] == [
        "contact_00001", "contact_00002", "contact_00003", "contact_00004",
    ]


def test_sarah_chen_is_unique(fresh_db):
    """T04 expects exactly one Sarah Chen."""
    conn = fresh_db(42)
    rows = conn.execute(
        "SELECT id FROM contacts WHERE name = 'Sarah Chen' "
        "AND deleted_at IS NULL"
    ).fetchall()
    assert len(rows) == 1


def test_pipeline_review_list_exists(fresh_db):
    """T05 expects a target list to already exist."""
    conn = fresh_db(42)
    row = conn.execute(
        "SELECT id FROM lists WHERE id = 'list_pipeline_review'"
    ).fetchone()
    assert row is not None


def test_acme_activity_counts_match_oracle_expectation(fresh_db):
    """The merge winner is contact_00001 (12 activities). Oracle relies on this."""
    conn = fresh_db(42)
    counts = {
        cid: conn.execute(
            "SELECT COUNT(*) FROM activities "
            "WHERE target_type='contact' AND target_id=?",
            (cid,),
        ).fetchone()[0]
        for cid in ("contact_00001", "contact_00002", "contact_00003", "contact_00004")
    }
    assert counts == {
        "contact_00001": 12,
        "contact_00002": 4,
        "contact_00003": 2,
        "contact_00004": 1,
    }
