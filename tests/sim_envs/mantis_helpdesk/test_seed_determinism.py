"""Seed determinism — same SEED → identical row IDs + counts + key joins.

Acceptance criterion (#333): "Same SEED → identical row IDs across boots".
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
        seed_mod.seed(conn, seed_val=seed_val, fake_now="2026-01-15T09:00:00Z")
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
    from app.seed import N_TICKETS, N_REPLIES, N_MACROS, N_TRIGGERS, N_GROUPS  # noqa: PLC0415

    assert _count(conn, "groups") == N_GROUPS
    assert _count(conn, "tickets") == N_TICKETS
    assert _count(conn, "replies") == N_REPLIES
    assert _count(conn, "macros") == N_MACROS
    assert _count(conn, "triggers") == N_TRIGGERS


def test_same_seed_same_snapshot(fresh_db):
    conn_a = fresh_db(42)
    conn_b = fresh_db(42)
    for table in ("groups", "users", "macros", "triggers", "tickets", "replies"):
        assert _snapshot(conn_a, table) == _snapshot(conn_b, table), \
            f"snapshot mismatch in {table}"


def test_different_seeds_yield_different_data(fresh_db):
    conn_a = fresh_db(42)
    conn_b = fresh_db(43)
    a = _snapshot(conn_a, "tickets", limit=20)
    b = _snapshot(conn_b, "tickets", limit=20)
    assert len(a) == len(b)
    assert a != b


def test_ids_are_zero_padded_strings(fresh_db):
    conn = fresh_db(42)
    sample = conn.execute(
        "SELECT id FROM tickets WHERE id LIKE 'ticket_%' ORDER BY id LIMIT 3"
    ).fetchall()
    assert [r["id"] for r in sample] == ["ticket_00001", "ticket_00002", "ticket_00003"]


def test_outage_cluster_is_seeded(fresh_db):
    """T03 relies on 30 dupes ticket_07001..07030 + survivor ticket_07000."""
    conn = fresh_db(42)
    rows = conn.execute(
        "SELECT id, status, tags FROM tickets "
        "WHERE id BETWEEN 'ticket_07000' AND 'ticket_07030' ORDER BY id"
    ).fetchall()
    assert len(rows) == 31
    # All carry the 'outage' tag (planted by the seed).
    for r in rows:
        assert '"outage"' in (r["tags"] or "")


def test_pii_fixture_ticket_is_seeded(fresh_db):
    """T05 relies on ticket_04421 carrying PII in the body."""
    conn = fresh_db(42)
    from app import pii  # noqa: PLC0415

    row = conn.execute(
        "SELECT body FROM tickets WHERE id='ticket_04421'"
    ).fetchone()
    assert row is not None
    assert pii.contains_ssn(row["body"])
    assert pii.contains_credit_card(row["body"])


def test_internal_only_ticket_is_seeded(fresh_db):
    """T05 plants ticket_04422 with visibility=internal so the
    composer's internal-only rejection path is reachable."""
    conn = fresh_db(42)
    row = conn.execute(
        "SELECT visibility FROM tickets WHERE id='ticket_04422'"
    ).fetchone()
    assert row is not None
    assert row["visibility"] == "internal"


def test_shipping_macro_seeded(fresh_db):
    """T02 relies on macro_shipping_delay."""
    conn = fresh_db(42)
    row = conn.execute(
        "SELECT id FROM macros WHERE id='macro_shipping_delay'"
    ).fetchone()
    assert row is not None


def test_agents_split_across_groups(fresh_db):
    """T04 needs agents in every group — assert each of the 6 groups
    has at least one active agent."""
    conn = fresh_db(42)
    rows = conn.execute(
        "SELECT group_id, COUNT(*) AS n FROM users "
        "WHERE role='agent' AND is_active=1 GROUP BY group_id"
    ).fetchall()
    by_group = {r["group_id"]: r["n"] for r in rows}
    assert len(by_group) == 6
    for n in by_group.values():
        assert n >= 1


def test_six_digit_order_targets_exist(fresh_db):
    """T02 needs the 12xxxx body bodies — there should be several."""
    import re

    conn = fresh_db(42)
    rows = conn.execute("SELECT body FROM tickets").fetchall()
    pattern = re.compile(r"\b12\d{4}\b")
    hits = sum(1 for r in rows if pattern.search(r["body"] or ""))
    assert hits >= 5, f"expected several shipping order tickets, got {hits}"
