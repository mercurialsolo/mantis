"""Contact list (paginated) + detail + inline edit + bulk-tag."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()

PAGE_SIZE = 50


def _now_iso() -> str:
    return app_main.now_value()


def _log_mutation(conn, *, operation: str, target_type: str, target_id: str,
                  payload: dict[str, Any] | None = None) -> None:
    db.log_mutation(
        conn,
        occurred_at=_now_iso(),
        operation=operation,
        target_type=target_type,
        target_id=target_id,
        payload=payload or {},
    )
    app_main.emit(f"mutation.{operation}", {
        "target_type": target_type, "target_id": target_id,
        **(payload or {}),
    })


@router.get("/contacts", response_class=HTMLResponse)
async def list_contacts(request: Request) -> HTMLResponse:
    """Paginated, filterable list. Query params:

    * ``page`` — 1-based
    * ``q`` — substring match on name OR email OR company.name
    * ``stage`` — lifecycle_stage filter
    * ``owner`` — owner_id filter
    * ``tag`` — tag membership filter
    """
    page = max(1, int(request.query_params.get("page") or 1))
    q = (request.query_params.get("q") or "").strip()
    stage = (request.query_params.get("stage") or "").strip()
    owner = (request.query_params.get("owner") or "").strip()
    tag = (request.query_params.get("tag") or "").strip()

    conn = db.connect()
    where = ["c.deleted_at IS NULL"]
    args: list[Any] = []
    if q:
        where.append("(c.name LIKE ? OR c.email LIKE ? OR co.name LIKE ?)")
        like = f"%{q}%"
        args += [like, like, like]
    if stage:
        where.append("c.lifecycle_stage = ?")
        args.append(stage)
    if owner:
        where.append("c.owner_id = ?")
        args.append(owner)
    if tag:
        where.append("c.tags LIKE ?")
        args.append(f'%"{tag}"%')

    where_clause = " AND ".join(where) or "1=1"
    total = conn.execute(
        f"SELECT COUNT(*) FROM contacts c LEFT JOIN companies co ON c.company_id = co.id "
        f"WHERE {where_clause}",
        args,
    ).fetchone()[0]

    offset = (page - 1) * PAGE_SIZE
    rows = conn.execute(
        f"SELECT c.id, c.name, c.email, c.phone, c.lifecycle_stage, c.owner_id, "
        f"       c.tags, c.last_activity_at, co.name AS company_name, co.id AS company_id "
        f"FROM contacts c LEFT JOIN companies co ON c.company_id = co.id "
        f"WHERE {where_clause} ORDER BY c.id LIMIT ? OFFSET ?",
        args + [PAGE_SIZE, offset],
    ).fetchall()

    contacts = [
        dict(r) | {"tags": db.unpack_tags(r["tags"])}
        for r in rows
    ]

    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    return app_main.app.state.templates.TemplateResponse(
        "contacts_list.html",
        {
            "request": request,
            "contacts": contacts,
            "total": total,
            "page": page,
            "pages": pages,
            "page_size": PAGE_SIZE,
            "filters": {"q": q, "stage": stage, "owner": owner, "tag": tag},
        },
    )


def _lead_score(contact: dict, activity_count: int, deal_count: int) -> int:
    """Cheap deterministic lead score in 0..100.

    Mirrors what HubSpot/Salesforce call a 'fit + engagement' score —
    aggregating a few signals into a single number so the rep can sort.
    """
    score = 30  # baseline
    stage_bonus = {
        "lead": 0, "mql": 10, "sql": 20,
        "customer": 30, "evangelist": 35, "churned": -20,
    }
    score += stage_bonus.get(contact.get("lifecycle_stage") or "", 0)
    score += min(activity_count, 30)  # cap engagement contribution
    score += min(deal_count * 5, 25)
    if contact.get("email") and "@" in (contact.get("email") or ""):
        score += 5
    if contact.get("phone"):
        score += 3
    if contact.get("tags") and "do-not-contact" in contact["tags"]:
        score -= 40
    return max(0, min(100, score))


@router.get("/contacts/{contact_id}", response_class=HTMLResponse)
async def contact_detail(request: Request, contact_id: str) -> HTMLResponse:
    conn = db.connect()
    row = conn.execute(
        "SELECT c.*, co.name AS company_name FROM contacts c "
        "LEFT JOIN companies co ON c.company_id = co.id WHERE c.id = ?",
        (contact_id,),
    ).fetchone()
    if row is None:
        return HTMLResponse("Not found", status_code=404)

    contact = dict(row) | {
        "tags": db.unpack_tags(row["tags"]),
        "custom_fields": db.unpack_custom_fields(row["custom_fields"]),
    }
    activities = [
        dict(a) for a in conn.execute(
            "SELECT * FROM activities WHERE target_type='contact' AND target_id=? "
            "ORDER BY occurred_at DESC LIMIT 100",
            (contact_id,),
        ).fetchall()
    ]
    related_deals = [
        dict(d) for d in conn.execute(
            "SELECT * FROM deals WHERE contact_id=? ORDER BY id", (contact_id,),
        ).fetchall()
    ]
    tasks = [
        dict(t) for t in conn.execute(
            "SELECT * FROM tasks WHERE target_type='contact' AND target_id=? "
            "ORDER BY CASE WHEN completed_at IS NULL THEN 0 ELSE 1 END, "
            "due_date NULLS LAST LIMIT 50",
            (contact_id,),
        ).fetchall()
    ]
    notes = [
        dict(n) for n in conn.execute(
            "SELECT * FROM notes WHERE target_type='contact' AND target_id=? "
            "ORDER BY pinned DESC, created_at DESC LIMIT 50",
            (contact_id,),
        ).fetchall()
    ]
    transitions = [
        dict(t) for t in conn.execute(
            "SELECT * FROM lifecycle_transitions WHERE contact_id=? "
            "ORDER BY occurred_at DESC LIMIT 50",
            (contact_id,),
        ).fetchall()
    ]

    tab = (request.query_params.get("tab") or "activity").strip().lower()
    if tab not in {"activity", "tasks", "notes", "deals", "history"}:
        tab = "activity"

    lead_score = _lead_score(
        contact, activity_count=len(activities), deal_count=len(related_deals)
    )

    return app_main.app.state.templates.TemplateResponse(
        "contact_detail.html",
        {
            "request": request,
            "contact": contact,
            "activities": activities,
            "related_deals": related_deals,
            "tasks": tasks,
            "notes": notes,
            "transitions": transitions,
            "tab": tab,
            "lead_score": lead_score,
        },
    )


# ── bulk operations ───────────────────────────────────────────────────
# Declared BEFORE the per-contact routes so ``/contacts/bulk/tag`` doesn't
# get shadowed by ``/contacts/{contact_id}/tag`` (Starlette routes in
# declaration order).


@router.post("/contacts/bulk/tag")
async def bulk_tag(
    request: Request,
    tag: str = Form(...),
    ids: str = Form(""),
) -> RedirectResponse:
    """Bulk-tag the supplied comma-separated contact ids.

    Used by both the bulk-edit modal (form post) and by API-style
    automation (just call with ids=…&tag=…). Idempotent per contact.
    """
    if not tag.strip():
        return RedirectResponse("/contacts", status_code=303)
    target_ids = [s.strip() for s in ids.split(",") if s.strip()]
    if not target_ids:
        return RedirectResponse("/contacts", status_code=303)

    with db.transaction() as txn:
        for cid in target_ids:
            row = txn.execute("SELECT tags FROM contacts WHERE id = ?",
                              (cid,)).fetchone()
            if row is None:
                continue
            current = db.unpack_tags(row["tags"])
            if tag in current:
                continue
            current.append(tag)
            txn.execute("UPDATE contacts SET tags = ? WHERE id = ?",
                        (db.pack_tags(current), cid))
            _log_mutation(txn, operation="tag_added", target_type="contact",
                          target_id=cid, payload={"tag": tag, "via": "bulk"})
    return RedirectResponse("/contacts", status_code=303)


@router.post("/contacts/{contact_id}/tag")
async def add_tag(contact_id: str, tag: str = Form(...)) -> RedirectResponse:
    """Add a tag to a contact. Idempotent — duplicate tags collapse."""
    with db.transaction() as txn:
        row = txn.execute("SELECT tags FROM contacts WHERE id = ?",
                          (contact_id,)).fetchone()
        if row is None:
            return RedirectResponse("/contacts", status_code=303)
        tags = db.unpack_tags(row["tags"])
        if tag and tag not in tags:
            tags.append(tag)
            txn.execute("UPDATE contacts SET tags = ? WHERE id = ?",
                        (db.pack_tags(tags), contact_id))
            _log_mutation(txn, operation="tag_added", target_type="contact",
                          target_id=contact_id, payload={"tag": tag})
    return RedirectResponse(f"/contacts/{contact_id}", status_code=303)


@router.post("/contacts/{contact_id}/activity")
async def add_activity(
    contact_id: str,
    activity_type: str = Form(...),
    body: str = Form(""),
    occurred_at: str = Form(""),
) -> RedirectResponse:
    """Log an activity (call, email, note, meeting) against a contact."""
    when = occurred_at or _now_iso()
    aid_n = int(time.time() * 1000)
    new_id = f"activity_{aid_n:09d}"
    with db.transaction() as txn:
        txn.execute(
            "INSERT INTO activities (id, target_type, target_id, activity_type, "
            "body, actor_id, occurred_at) VALUES (?, 'contact', ?, ?, ?, NULL, ?)",
            (new_id, contact_id, activity_type, body, when),
        )
        # Bump last_activity_at if the new activity is more recent.
        txn.execute(
            "UPDATE contacts SET last_activity_at = ? "
            "WHERE id = ? AND (last_activity_at IS NULL OR last_activity_at < ?)",
            (when, contact_id, when),
        )
        _log_mutation(txn, operation="activity_added", target_type="contact",
                      target_id=contact_id,
                      payload={"activity_type": activity_type, "body": body,
                               "occurred_at": when, "activity_id": new_id})
    return RedirectResponse(f"/contacts/{contact_id}", status_code=303)


@router.post("/contacts/{contact_id}/merge")
async def merge_into(
    contact_id: str,
    loser_ids: str = Form(""),
) -> RedirectResponse:
    """Merge ``loser_ids`` into ``contact_id``.

    Survivor keeps its identity; activities + deals re-point to the
    survivor; losers are soft-deleted. Tags from the losers union into
    the survivor's tag set so no metadata is lost.
    """
    losers = [s.strip() for s in loser_ids.split(",") if s.strip()]
    losers = [cid for cid in losers if cid != contact_id]
    if not losers:
        return RedirectResponse(f"/contacts/{contact_id}", status_code=303)

    with db.transaction() as txn:
        survivor = txn.execute("SELECT tags FROM contacts WHERE id = ?",
                               (contact_id,)).fetchone()
        if survivor is None:
            return RedirectResponse("/contacts", status_code=303)
        survivor_tags = set(db.unpack_tags(survivor["tags"]))

        for loser in losers:
            row = txn.execute("SELECT tags FROM contacts WHERE id = ?",
                              (loser,)).fetchone()
            if row is None:
                continue
            survivor_tags.update(db.unpack_tags(row["tags"]))
            # Re-point activities and deals to the survivor.
            txn.execute(
                "UPDATE activities SET target_id = ? "
                "WHERE target_type='contact' AND target_id = ?",
                (contact_id, loser),
            )
            txn.execute(
                "UPDATE deals SET contact_id = ? WHERE contact_id = ?",
                (contact_id, loser),
            )
            txn.execute(
                "UPDATE contacts SET deleted_at = ? WHERE id = ?",
                (_now_iso(), loser),
            )
            _log_mutation(txn, operation="contact_merged",
                          target_type="contact", target_id=loser,
                          payload={"into": contact_id})

        txn.execute(
            "UPDATE contacts SET tags = ? WHERE id = ?",
            (db.pack_tags(sorted(survivor_tags)), contact_id),
        )
        _log_mutation(txn, operation="contact_merge_completed",
                      target_type="contact", target_id=contact_id,
                      payload={"absorbed": losers})

    return RedirectResponse(f"/contacts/{contact_id}", status_code=303)
