"""BT02_spec_lookup_engine — submit a lead on a Caterpillar-powered boat.

Task contract
-------------

The agent must find a boat whose engine is made by **Caterpillar** and
submit the dealer contact form on it with a non-empty name + email.

This is a *knowledge* task, not a *capability* one: ``engine_make`` is a
spec-sheet field, **not** a listing-page facet/filter. There is no
``?engine_make=Caterpillar`` query the agent can lean on — it has to open
listings and read the spec block to learn which boats qualify. That is
exactly the regime where an S0 retrieval rung wins: once the agent (or a
prior run) has learned "boat X is Caterpillar-powered / the engine spec
lives here", caching that anchor short-circuits the lookup. A fresh agent
with no memory has to re-discover it every time.

The catalog has 82 Caterpillar-powered boats at seed=42 (76 at seed=7),
so at least one qualifying target always exists; the grader never depends
on a hand-counted per-seed id.

Grader
------

Reads ``store.leads`` and ``store.boats_by_id`` (mirrors BT01):

* **Hit set**: leads whose boat is Caterpillar-powered, carrying a
  non-empty ``name`` and an email-shaped ``email``.
* **Miss set**: leads on boats with any other engine make (collateral).
* **Malformed set**: leads on a qualifying boat but with a blank
  name / non-email email — the agent reached the right boat but didn't
  submit a real-shaped payload.
* **Pass condition**: ``len(hits) >= 1`` AND ``len(misses) == 0`` AND
  ``len(malformed) == 0``.
* **Score**: F1 over (precision, recall), precision = hits /
  (hits + misses + malformed), recall = 1 if hits >= 1 else 0. 4 dp.

Deterministic — same store snapshot → same verdict.
"""

from __future__ import annotations

from typing import Any

# The spec value that defines the task. Engine make is a detail-page spec
# field, not a listing facet — that's what makes this a lookup task.
TARGET_ENGINE_MAKE = "Caterpillar"


def _lead_payload_ok(lead: dict) -> bool:
    """Lead has a non-empty name + email-shaped email."""
    name = str(lead.get("name") or "").strip()
    email = str(lead.get("email") or "").strip()
    if not name:
        return False
    if "@" not in email or "." not in email.split("@", 1)[-1]:
        return False
    return True


def _boat_matches_spec(boat: Any) -> bool:
    return str(getattr(boat, "engine_make", "") or "") == TARGET_ENGINE_MAKE


def grade(store: Any, *, now: str, seed_val: int) -> dict[str, Any]:
    leads = list(getattr(store, "leads", []) or [])
    boats_by_id = dict(getattr(store, "boats_by_id", {}) or {})

    # The "right set": every boat whose spec block names the target engine.
    qualifying_ids = {
        boat_id for boat_id, b in boats_by_id.items()
        if _boat_matches_spec(b)
    }

    hits: list[dict] = []
    misses: list[dict] = []
    malformed: list[dict] = []
    for lead in leads:
        boat_id = str(lead.get("boat_id") or "")
        boat = boats_by_id.get(boat_id)
        if boat is None:
            # Lead points at an unknown boat — miss, robust to seed drift.
            misses.append(lead)
            continue
        if boat_id not in qualifying_ids:
            misses.append(lead)
            continue
        if not _lead_payload_ok(lead):
            malformed.append(lead)
            continue
        hits.append(lead)

    n_hits = len(hits)
    n_bad = len(misses) + len(malformed)
    precision = n_hits / (n_hits + n_bad) if (n_hits + n_bad) else 0.0
    recall = 1.0 if n_hits >= 1 else 0.0
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    passed = n_hits >= 1 and n_bad == 0

    reasons: list[str] = []
    if passed:
        reasons.append(
            f"{n_hits} lead(s) submitted on a {TARGET_ENGINE_MAKE}-powered "
            "boat with no collateral"
        )
    else:
        if n_hits == 0:
            reasons.append(
                f"no leads submitted on a {TARGET_ENGINE_MAKE}-powered boat"
            )
        if misses:
            reasons.append(
                f"{len(misses)} lead(s) submitted on a differently-powered boat"
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
