"""Each oracle returns the same verdict twice on the same store state.

This is the easy half of "5 reruns → 5 identical scores". The harder
half (running a real agent five times) lives in CI integration, not
unit tests, but oracle determinism guarantees the unit-level invariant.

The foundation PR (issue #588) ships with an empty grader table — this
file scaffolds the determinism contract so it's enforced from day one;
concrete graders register themselves in :mod:`app.oracles` and acquire
a parametrize entry as they land (#589 onwards).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def seeded_store():
    from app import db  # noqa: PLC0415

    # Force a fresh in-memory store seeded deterministically.
    db._store = None  # type: ignore[attr-defined]
    s = db.store()
    # Sanity — seed produced something to grade against.
    assert s.boats, "expected the seed to produce at least one boat"
    return s


@pytest.mark.parametrize("task_id", [
    "BT01_lead_capture_filtered_search",
    "BT02_spec_lookup_engine",
    "BT03_byowner_phone_reveal",
])
def test_oracle_is_deterministic(seeded_store, task_id):
    from app.oracles import grade  # noqa: PLC0415

    r1 = grade(task_id, seeded_store, now="2026-01-15T09:00:00Z", seed_val=42)
    r2 = grade(task_id, seeded_store, now="2026-01-15T09:00:00Z", seed_val=42)
    assert r1 == r2, f"{task_id}: oracle is non-deterministic"


def test_oracle_unregistered_task_fails_gracefully(seeded_store):
    from app.oracles import grade  # noqa: PLC0415

    r = grade(
        "DOES_NOT_EXIST",
        seeded_store,
        now="2026-01-15T09:00:00Z",
        seed_val=42,
    )
    assert r["passed"] is False
    assert r["score"] == 0.0
    assert r["task_id"] == "DOES_NOT_EXIST"
    assert "no oracle" in " ".join(r["reasons"]).lower()
    assert r["diff"] == {}


def test_oracle_dispatcher_round_trips_task_id(seeded_store):
    """``grade`` sets ``task_id`` even when a grader forgot to."""
    from app import oracles  # noqa: PLC0415

    def fake_grader(_store, *, now, seed_val):
        return {"passed": True, "score": 1.0, "reasons": [], "diff": {}}

    oracles.GRADERS["FAKE_TASK"] = fake_grader
    try:
        r = oracles.grade(
            "FAKE_TASK",
            seeded_store,
            now="2026-01-15T09:00:00Z",
            seed_val=42,
        )
        assert r["task_id"] == "FAKE_TASK"
        assert r["passed"] is True
    finally:
        oracles.GRADERS.pop("FAKE_TASK", None)


# ── BT01_lead_capture_filtered_search ──────────────────────────────────


def test_oracle_bt01_fails_on_seed(seeded_store):
    """No leads submitted yet → BT01 must fail."""
    from app.oracles import grade  # noqa: PLC0415

    r = grade(
        "BT01_lead_capture_filtered_search",
        seeded_store,
        now="2026-01-15T09:00:00Z",
        seed_val=42,
    )
    assert r["passed"] is False
    assert r["score"] == 0.0
    assert r["diff"]["leads_total"] == 0
    assert r["diff"]["qualifying_boat_count"] >= 1, (
        "expected the seed catalog to include at least one boat matching "
        "BT01's filter constraints"
    )


def test_oracle_bt01_passes_when_qualifying_lead_submitted(seeded_store):
    """Synthesise the target action via the public ``record_lead`` API
    and confirm the oracle approves."""
    from app import db  # noqa: PLC0415
    from app.oracles import grade  # noqa: PLC0415
    from app.oracles.bt01_lead_capture_filtered_search import _boat_matches_filters  # noqa: PLC0415

    # Pick a qualifying boat from the seeded store.
    qualifying = next(
        b for b in seeded_store.boats if _boat_matches_filters(b)
    )
    db.record_lead({
        "boat_id": qualifying.id,
        "boat_title": qualifying.title,
        "dealer_id": qualifying.dealer_id,
        "name": "Test Buyer",
        "email": "buyer@example.test",
        "phone": "",
        "message": "Interested in this boat.",
    })

    r = grade(
        "BT01_lead_capture_filtered_search",
        seeded_store,
        now="2026-01-15T09:00:00Z",
        seed_val=42,
    )
    assert r["passed"] is True, r
    assert r["score"] == 1.0
    assert r["diff"]["hits"] == 1
    assert r["diff"]["misses"] == 0
    assert r["diff"]["malformed"] == 0


def test_oracle_bt01_fails_on_non_qualifying_lead(seeded_store):
    """A lead on a boat that doesn't match the filter constraints is
    a fail — collateral damage counts even with no qualifying hit."""
    from app import db  # noqa: PLC0415
    from app.oracles import grade  # noqa: PLC0415
    from app.oracles.bt01_lead_capture_filtered_search import _boat_matches_filters  # noqa: PLC0415

    non_qualifying = next(
        b for b in seeded_store.boats if not _boat_matches_filters(b)
    )
    db.record_lead({
        "boat_id": non_qualifying.id,
        "boat_title": non_qualifying.title,
        "dealer_id": non_qualifying.dealer_id,
        "name": "Test Buyer",
        "email": "buyer@example.test",
        "phone": "",
        "message": "Interested in this boat.",
    })

    r = grade(
        "BT01_lead_capture_filtered_search",
        seeded_store,
        now="2026-01-15T09:00:00Z",
        seed_val=42,
    )
    assert r["passed"] is False
    assert r["diff"]["hits"] == 0
    assert r["diff"]["misses"] == 1


def test_oracle_bt01_rejects_malformed_payload(seeded_store):
    """A lead with empty name or missing-@ email counts as malformed,
    not as a hit — the agent must submit a real-shaped payload."""
    from app import db  # noqa: PLC0415
    from app.oracles import grade  # noqa: PLC0415
    from app.oracles.bt01_lead_capture_filtered_search import _boat_matches_filters  # noqa: PLC0415

    qualifying = next(
        b for b in seeded_store.boats if _boat_matches_filters(b)
    )
    db.record_lead({
        "boat_id": qualifying.id,
        "boat_title": qualifying.title,
        "dealer_id": qualifying.dealer_id,
        "name": "",  # empty name
        "email": "buyer@example.test",
        "phone": "",
        "message": "Interested.",
    })

    r = grade(
        "BT01_lead_capture_filtered_search",
        seeded_store,
        now="2026-01-15T09:00:00Z",
        seed_val=42,
    )
    assert r["passed"] is False
    assert r["diff"]["hits"] == 0
    assert r["diff"]["malformed"] == 1


# ── BT02_spec_lookup_engine ────────────────────────────────────────────


def test_oracle_bt02_fails_on_seed(seeded_store):
    """No leads submitted yet → BT02 must fail, but a qualifying boat exists."""
    from app.oracles import grade  # noqa: PLC0415

    r = grade(
        "BT02_spec_lookup_engine",
        seeded_store,
        now="2026-01-15T09:00:00Z",
        seed_val=42,
    )
    assert r["passed"] is False
    assert r["score"] == 0.0
    assert r["diff"]["leads_total"] == 0
    assert r["diff"]["qualifying_boat_count"] >= 1, (
        "expected the seed catalog to include at least one Caterpillar-"
        "powered boat"
    )


def test_oracle_bt02_passes_when_caterpillar_lead_submitted(seeded_store):
    from app import db  # noqa: PLC0415
    from app.oracles import grade  # noqa: PLC0415
    from app.oracles.bt02_spec_lookup_engine import _boat_matches_spec  # noqa: PLC0415

    qualifying = next(
        b for b in seeded_store.boats if _boat_matches_spec(b)
    )
    db.record_lead({
        "boat_id": qualifying.id,
        "boat_title": qualifying.title,
        "dealer_id": qualifying.dealer_id,
        "name": "Test Buyer",
        "email": "buyer@example.test",
        "phone": "",
        "message": "Interested in this boat.",
    })

    r = grade(
        "BT02_spec_lookup_engine",
        seeded_store,
        now="2026-01-15T09:00:00Z",
        seed_val=42,
    )
    assert r["passed"] is True, r
    assert r["score"] == 1.0
    assert r["diff"]["hits"] == 1
    assert r["diff"]["misses"] == 0
    assert r["diff"]["malformed"] == 0


def test_oracle_bt02_fails_on_non_caterpillar_lead(seeded_store):
    from app import db  # noqa: PLC0415
    from app.oracles import grade  # noqa: PLC0415
    from app.oracles.bt02_spec_lookup_engine import _boat_matches_spec  # noqa: PLC0415

    non_qualifying = next(
        b for b in seeded_store.boats if not _boat_matches_spec(b)
    )
    db.record_lead({
        "boat_id": non_qualifying.id,
        "boat_title": non_qualifying.title,
        "dealer_id": non_qualifying.dealer_id,
        "name": "Test Buyer",
        "email": "buyer@example.test",
        "phone": "",
        "message": "Interested in this boat.",
    })

    r = grade(
        "BT02_spec_lookup_engine",
        seeded_store,
        now="2026-01-15T09:00:00Z",
        seed_val=42,
    )
    assert r["passed"] is False
    assert r["diff"]["hits"] == 0
    assert r["diff"]["misses"] == 1


def test_oracle_bt02_rejects_malformed_payload(seeded_store):
    from app import db  # noqa: PLC0415
    from app.oracles import grade  # noqa: PLC0415
    from app.oracles.bt02_spec_lookup_engine import _boat_matches_spec  # noqa: PLC0415

    qualifying = next(
        b for b in seeded_store.boats if _boat_matches_spec(b)
    )
    db.record_lead({
        "boat_id": qualifying.id,
        "boat_title": qualifying.title,
        "dealer_id": qualifying.dealer_id,
        "name": "",  # empty name
        "email": "buyer@example.test",
        "phone": "",
        "message": "Interested.",
    })

    r = grade(
        "BT02_spec_lookup_engine",
        seeded_store,
        now="2026-01-15T09:00:00Z",
        seed_val=42,
    )
    assert r["passed"] is False
    assert r["diff"]["hits"] == 0
    assert r["diff"]["malformed"] == 1


# ── BT03_byowner_phone_reveal ──────────────────────────────────────────


def test_oracle_bt03_fails_on_seed(seeded_store):
    """No phone revealed yet → BT03 must fail, but a by-owner boat exists."""
    from app.oracles import grade  # noqa: PLC0415

    r = grade(
        "BT03_byowner_phone_reveal",
        seeded_store,
        now="2026-01-15T09:00:00Z",
        seed_val=42,
    )
    assert r["passed"] is False
    assert r["score"] == 0.0
    assert r["diff"]["reveals_total"] == 0
    assert r["diff"]["qualifying_boat_count"] >= 1, (
        "expected the seed catalog to include at least one by-owner listing"
    )


def test_oracle_bt03_passes_when_owner_phone_revealed(seeded_store):
    """Synthesise the reveal via the same ``emit_mutation`` channel the
    ``/show-phone`` route uses, and confirm the oracle approves."""
    from app import db  # noqa: PLC0415
    from app.oracles import grade  # noqa: PLC0415
    from app.oracles.bt03_byowner_phone_reveal import _is_owner_listed  # noqa: PLC0415

    owner_boat = next(
        b for b in seeded_store.boats if _is_owner_listed(b)
    )
    db.emit_mutation(
        operation="phone_revealed",
        target_type="boat",
        target_id=owner_boat.id,
        payload={"slug": owner_boat.slug},
    )

    r = grade(
        "BT03_byowner_phone_reveal",
        seeded_store,
        now="2026-01-15T09:00:00Z",
        seed_val=42,
    )
    assert r["passed"] is True, r
    assert r["score"] == 1.0
    assert r["diff"]["hits"] == 1
    assert r["diff"]["misses"] == 0


def test_oracle_bt03_fails_on_nonowner_reveal(seeded_store):
    """Revealing a dealer/sponsored listing's phone is the policy mistake."""
    from app import db  # noqa: PLC0415
    from app.oracles import grade  # noqa: PLC0415
    from app.oracles.bt03_byowner_phone_reveal import _is_owner_listed  # noqa: PLC0415

    dealer_boat = next(
        b for b in seeded_store.boats if not _is_owner_listed(b)
    )
    db.emit_mutation(
        operation="phone_revealed",
        target_type="boat",
        target_id=dealer_boat.id,
        payload={"slug": dealer_boat.slug},
    )

    r = grade(
        "BT03_byowner_phone_reveal",
        seeded_store,
        now="2026-01-15T09:00:00Z",
        seed_val=42,
    )
    assert r["passed"] is False
    assert r["diff"]["hits"] == 0
    assert r["diff"]["misses"] == 1
