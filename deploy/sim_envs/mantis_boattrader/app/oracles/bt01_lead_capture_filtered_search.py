"""BT01_lead_capture_filtered_search — submit a lead on a filtered listing.

Task contract
-------------

The agent navigates to ``/boats/``, applies the filters
``condition=used`` + ``make=Sea Ray`` + ``price_max=200000``, clicks any
matching listing card, and submits the dealer contact form with a
non-empty name + email.

The catalog at seed=42 has 9 boats matching these constraints.

Grader
------

Reads ``store.leads`` and ``store.boats_by_id``:

* **Hit set**: leads whose boat satisfies all three filter constraints
  (condition, make, price_max). The lead must also carry non-empty
  ``name`` and a vaguely email-shaped ``email`` (contains ``@``) —
  loose payload shape, since the agent has license on the canonical
  values but not on submitting blank fields.

* **Miss set**: leads on boats that don't satisfy at least one of the
  constraints. Each miss penalises the score; the pass gate requires
  zero misses.

* **Pass condition**: ``len(hits) >= 1`` AND ``len(misses) == 0``.

* **Score**: F1 over (precision, recall) where precision = hits /
  (hits + misses), recall = 1 if hits >= 1 else 0. Rounded to 4 dp.

The grader is deterministic — same store snapshot → same verdict.
"""

from __future__ import annotations

from typing import Any


# Canonical task constraints. Lower-cased for ``condition`` because
# ``seed.build`` ships ``condition`` values as lower-case strings and
# ``db.query_boats`` does an exact match against the param.
TASK_FILTERS: dict[str, Any] = {
    "condition": "used",
    "make": "Sea Ray",
    "price_max": 200000,
}


def _lead_payload_ok(lead: dict) -> bool:
    """Lead has a non-empty name + email-shaped email."""
    name = str(lead.get("name") or "").strip()
    email = str(lead.get("email") or "").strip()
    if not name:
        return False
    if "@" not in email or "." not in email.split("@", 1)[-1]:
        return False
    return True


def _boat_matches_filters(boat: Any) -> bool:
    if str(getattr(boat, "condition", "")) != TASK_FILTERS["condition"]:
        return False
    if str(getattr(boat, "make", "")) != TASK_FILTERS["make"]:
        return False
    price = getattr(boat, "price", None)
    if price is None:
        return False
    if int(price) > int(TASK_FILTERS["price_max"]):
        return False
    return True


def grade(store: Any, *, now: str, seed_val: int) -> dict[str, Any]:
    leads = list(getattr(store, "leads", []) or [])
    boats_by_id = dict(getattr(store, "boats_by_id", {}) or {})

    # Universe of boats that satisfy the constraints — the "right set"
    # the agent could have picked from. We don't require coverage of
    # every matching boat; one matching lead is sufficient to pass.
    qualifying_ids = {
        boat_id for boat_id, b in boats_by_id.items()
        if _boat_matches_filters(b)
    }

    hits: list[dict] = []
    misses: list[dict] = []
    malformed: list[dict] = []
    for lead in leads:
        boat_id = str(lead.get("boat_id") or "")
        boat = boats_by_id.get(boat_id)
        if boat is None:
            # Lead points at a boat we don't recognise — treat as miss
            # so the grader is robust to seed drift.
            misses.append(lead)
            continue
        if not _lead_payload_ok(lead):
            malformed.append(lead)
            continue
        if boat_id in qualifying_ids:
            hits.append(lead)
        else:
            misses.append(lead)

    n_hits = len(hits)
    n_misses = len(misses) + len(malformed)
    precision = n_hits / (n_hits + n_misses) if (n_hits + n_misses) else 0.0
    recall = 1.0 if n_hits >= 1 else 0.0
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    passed = n_hits >= 1 and n_misses == 0

    reasons: list[str] = []
    if passed:
        reasons.append(
            f"{n_hits} qualifying lead(s) submitted with no collateral"
        )
    else:
        if n_hits == 0:
            reasons.append("no leads submitted on a qualifying boat")
        if misses:
            reasons.append(
                f"{len(misses)} lead(s) submitted on non-qualifying boats"
            )
        if malformed:
            reasons.append(
                f"{len(malformed)} lead(s) had a malformed name/email payload"
            )

    def _lead_summary(leads_subset: list[dict]) -> list[dict]:
        return [
            {
                "id": ld.get("id"),
                "boat_id": ld.get("boat_id"),
                "boat_title": ld.get("boat_title"),
                "email": ld.get("email"),
            }
            for ld in leads_subset[:5]
        ]

    return {
        "passed": passed,
        "score": round(f1, 4),
        "reasons": reasons,
        "diff": {
            "qualifying_boat_count": len(qualifying_ids),
            "leads_total": len(leads),
            "hits": n_hits,
            "misses": len(misses),
            "malformed": len(malformed),
            "hit_examples": _lead_summary(hits),
            "miss_examples": _lead_summary(misses),
            "malformed_examples": _lead_summary(malformed),
        },
    }
