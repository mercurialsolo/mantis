"""BT03_byowner_phone_reveal — reveal a private seller's phone number.

Task contract
-------------

The agent must reveal the contact phone on a **by-owner** ("private
seller") listing — i.e. drive the ``Show phone`` affordance on a boat
whose ``listing_type == "owner"``.

This is a *policy* task. The difficulty is not the click itself but
applying the right policy under listing-type ambiguity: only owner
listings expose a phone, and (with a layout-drift fixture) the by-owner
filter and the "Show phone" control move around, so a naive agent
mis-targets a dealer card. An S1 exemplar rung — "here's a worked
by-owner reveal" — fixes the recurring mis-click early; once it's
frequent enough, S3 consolidation folds it into the policy.

Grader
------

Reads the ``phone_revealed`` mutation channel (``store.mutations``),
*not* leads — so it's orthogonal to BT01/BT02 and grades a distinct
state-changing action:

* **Hit set**: ``phone_revealed`` mutations whose ``target_id`` resolves
  to an owner-listed boat.
* **Miss set**: ``phone_revealed`` mutations on a non-owner boat (the
  policy mistake) or on an unrecognised boat id.
* **Pass condition**: ``len(hits) >= 1`` AND ``len(misses) == 0``.
* **Score**: F1 over (precision, recall), precision = hits /
  (hits + misses), recall = 1 if hits >= 1 else 0. 4 dp.

The catalog has 135 owner listings at seed=42 (170 at seed=7), and no
non-owner boat carries a phone, so a qualifying target always exists and
the grader never hand-counts a per-seed id.

Deterministic — same store snapshot → same verdict.
"""

from __future__ import annotations

from typing import Any

REVEAL_OPERATION = "phone_revealed"
OWNER_LISTING_TYPE = "owner"


def _is_owner_listed(boat: Any) -> bool:
    return str(getattr(boat, "listing_type", "") or "") == OWNER_LISTING_TYPE


def grade(store: Any, *, now: str, seed_val: int) -> dict[str, Any]:
    mutations = list(getattr(store, "mutations", []) or [])
    boats_by_id = dict(getattr(store, "boats_by_id", {}) or {})

    # The "right set": every by-owner listing the agent could have revealed.
    qualifying_ids = {
        boat_id for boat_id, b in boats_by_id.items()
        if _is_owner_listed(b)
    }

    reveals = [
        m for m in mutations
        if str(m.get("operation") or "") == REVEAL_OPERATION
    ]

    hits: list[dict] = []
    misses: list[dict] = []
    for mut in reveals:
        target_id = str(mut.get("target_id") or "")
        boat = boats_by_id.get(target_id)
        if boat is None or not _is_owner_listed(boat):
            # Revealed a dealer/sponsored or unknown boat — the policy miss.
            misses.append(mut)
        else:
            hits.append(mut)

    n_hits = len(hits)
    n_misses = len(misses)
    precision = n_hits / (n_hits + n_misses) if (n_hits + n_misses) else 0.0
    recall = 1.0 if n_hits >= 1 else 0.0
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    passed = n_hits >= 1 and n_misses == 0

    reasons: list[str] = []
    if passed:
        reasons.append(
            f"{n_hits} phone reveal(s) on a by-owner listing with no collateral"
        )
    else:
        if n_hits == 0:
            reasons.append("no phone reveals on a by-owner listing")
        if misses:
            reasons.append(
                f"{n_misses} phone reveal(s) on a non-owner listing"
            )

    def _reveal_summary(muts: list[dict]) -> list[dict]:
        out = []
        for m in muts[:5]:
            target_id = str(m.get("target_id") or "")
            boat = boats_by_id.get(target_id)
            out.append({
                "mutation_id": m.get("id"),
                "boat_id": target_id,
                "listing_type": getattr(boat, "listing_type", None),
                "slug": (m.get("payload") or {}).get("slug"),
            })
        return out

    return {
        "passed": passed,
        "score": round(f1, 4),
        "reasons": reasons,
        "diff": {
            "qualifying_boat_count": len(qualifying_ids),
            "reveals_total": len(reveals),
            "hits": n_hits,
            "misses": n_misses,
            "hit_examples": _reveal_summary(hits),
            "miss_examples": _reveal_summary(misses),
        },
    }
