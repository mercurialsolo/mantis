"""Trigger rules — read-only routing rules applied as post-mutation hooks.

The seed plants 12 triggers in the ``triggers`` table for the agent to
see in the UI. Only one is *enforced* at server runtime in v1: the
billing-group assignee lock, applied by the bulk-assign path. We
explicitly localise the enforcement here so the rule is documented in
one place + easy to find when oracles assert behaviour.

Why not a generic engine? Each helpdesk product has 50+ triggers
configurable per-account; modelling that engine v1 is overkill. The
shape we mirror — a list of declarative rules the agent can read +
respect — is the part that matters for benchmarking.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

BILLING_GROUP_ID = "group_billing"


def apply_bulk_assign_revert(
    conn: sqlite3.Connection,
    *,
    ticket_ids: Iterable[str],
    new_assignee_id: str,
) -> list[str]:
    """Revert assignee on billing-group tickets if the new assignee is
    outside the billing group. Returns the list of reverted ticket ids.

    Called by the bulk-assign route after the assignee write. The route
    logs a ``ticket_assignee_reverted`` mutation for each reverted row
    so the oracle (and the audit page) can see the revert happened.
    """
    if not new_assignee_id:
        return []
    row = conn.execute(
        "SELECT group_id FROM users WHERE id = ?", (new_assignee_id,),
    ).fetchone()
    if row is None:
        return []
    if row["group_id"] == BILLING_GROUP_ID:
        return []                                  # new assignee is in billing — allowed

    reverted: list[str] = []
    for tid in ticket_ids:
        t = conn.execute(
            "SELECT group_id FROM tickets WHERE id = ? AND deleted_at IS NULL",
            (tid,),
        ).fetchone()
        if t is None:
            continue
        if t["group_id"] != BILLING_GROUP_ID:
            continue
        conn.execute(
            "UPDATE tickets SET assignee_id = NULL WHERE id = ?", (tid,),
        )
        reverted.append(tid)
    return reverted
