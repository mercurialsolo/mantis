"""Ticket inbox + detail + composer + bulk actions.

Mirrors ``mantis_crm/routes/contacts.py`` shape:

* List view (paginated) with filters: status, assignee, group, priority,
  tag, SLA-soon.
* Detail view with tabs (Thread / Internal notes / Related / History).
* Composer: reply (public/internal), apply macro, CC/BCC (stored).
* Mutations: status, priority, assignee, group, tags, apply macro,
  merge, escalate.
* Bulk actions: bulk-status / bulk-tag / bulk-assign / bulk-close /
  bulk-merge.

Route ordering: every ``/tickets/bulk/<op>`` route is declared BEFORE
the parametric ``/tickets/{ticket_id}/<op>`` so the bulk paths don't
get shadowed by the parametric matcher. This bit the mantis-crm PR;
we apply the same fix preemptively here.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main
from ..triggers import apply_bulk_assign_revert

router = APIRouter()

PAGE_SIZE = 50
VALID_STATUSES = {"new", "open", "pending", "solved", "closed"}
VALID_PRIORITIES = {"low", "normal", "high", "urgent"}


def _now_iso() -> str:
    return app_main.now_value()


def _log(conn, *, operation: str, target_id: str,
         payload: dict[str, Any] | None = None,
         target_type: str = "ticket") -> None:
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


def _render_merge_body(body: str, *, requester: dict, agent: dict | None,
                       order_number: str = "") -> str:
    """Substitute ``{{requester.first_name}}`` style tokens used by macros.

    Mirrors the merge logic in mantis-crm's template renderer but the
    vocabulary is helpdesk-shaped (requester / agent / order).
    """
    first_name = (requester.get("name") or "").split(" ")[0]
    ctx = {
        "{{requester.first_name}}": first_name,
        "{{requester.name}}": requester.get("name") or "",
        "{{requester.email}}": requester.get("email") or "",
        "{{agent.name}}": (agent or {}).get("name", "") if agent else "",
        "{{agent.email}}": (agent or {}).get("email", "") if agent else "",
        "{{order.number}}": order_number or "",
    }
    out = body
    for token, value in ctx.items():
        out = out.replace(token, value)
    return out


# ── inbox list ─────────────────────────────────────────────────────────


@router.get("/tickets", response_class=HTMLResponse)
async def list_tickets(request: Request) -> HTMLResponse:
    """Filterable inbox. Query params:

    * ``status`` — comma-separated subset of new/open/pending/solved/closed
    * ``assignee`` — agent id
    * ``group`` — group id
    * ``priority`` — low / normal / high / urgent
    * ``tag`` — tag membership filter
    * ``sla_soon`` — '1' = within 2h of breach
    * ``q`` — substring on subject / body
    * ``page`` — 1-based pagination
    """
    page = max(1, int(request.query_params.get("page") or 1))
    status_filter = (request.query_params.get("status") or "").strip()
    assignee = (request.query_params.get("assignee") or "").strip()
    group = (request.query_params.get("group") or "").strip()
    priority = (request.query_params.get("priority") or "").strip()
    tag = (request.query_params.get("tag") or "").strip()
    sla_soon = request.query_params.get("sla_soon") == "1"
    q = (request.query_params.get("q") or "").strip()

    where: list[str] = ["t.deleted_at IS NULL"]
    args: list[Any] = []
    if status_filter:
        parts = [s.strip() for s in status_filter.split(",") if s.strip()]
        if parts:
            placeholders = ",".join("?" for _ in parts)
            where.append(f"t.status IN ({placeholders})")
            args.extend(parts)
    if assignee:
        where.append("t.assignee_id = ?")
        args.append(assignee)
    if group:
        where.append("t.group_id = ?")
        args.append(group)
    if priority:
        where.append("t.priority = ?")
        args.append(priority)
    if tag:
        where.append("t.tags LIKE ?")
        args.append(f'%"{tag}"%')
    if sla_soon:
        # Within 2 hours of now or already breached, and still open.
        from datetime import timedelta
        from ..seed import _parse_iso
        cutoff = (_parse_iso(_now_iso()) + timedelta(hours=2)).isoformat()
        where.append("t.sla_breach_at <= ?")
        args.append(cutoff)
        where.append("t.status IN ('new','open','pending')")
    if q:
        where.append("(t.subject LIKE ? OR t.body LIKE ?)")
        like = f"%{q}%"
        args.extend([like, like])

    where_clause = " AND ".join(where) or "1=1"
    conn = db.connect()
    total = conn.execute(
        f"SELECT COUNT(*) FROM tickets t WHERE {where_clause}", args,
    ).fetchone()[0]

    offset = (page - 1) * PAGE_SIZE
    rows = conn.execute(
        f"SELECT t.id, t.subject, t.status, t.priority, t.assignee_id, "
        f"       t.group_id, t.tags, t.sla_breach_at, t.created_at, "
        f"       t.requester_id, t.locale, t.visibility, "
        f"       u.name AS requester_name "
        f"FROM tickets t LEFT JOIN users u ON t.requester_id = u.id "
        f"WHERE {where_clause} ORDER BY t.id LIMIT ? OFFSET ?",
        args + [PAGE_SIZE, offset],
    ).fetchall()

    tickets = [
        dict(r) | {"tags": db.unpack_tags(r["tags"])}
        for r in rows
    ]

    groups = [dict(g) for g in conn.execute(
        "SELECT id, name, slug FROM groups ORDER BY name"
    ).fetchall()]

    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    return app_main.app.state.templates.TemplateResponse(
        "tickets_list.html",
        {
            "request": request,
            "tickets": tickets,
            "total": total,
            "page": page,
            "pages": pages,
            "page_size": PAGE_SIZE,
            "groups": groups,
            "filters": {
                "status": status_filter, "assignee": assignee, "group": group,
                "priority": priority, "tag": tag, "sla_soon": sla_soon, "q": q,
            },
            "now": _now_iso(),
        },
    )


# ── bulk actions ──────────────────────────────────────────────────────
# Declared BEFORE per-ticket routes so ``/tickets/bulk/<op>`` doesn't
# get shadowed by ``/tickets/{ticket_id}/<op>`` matching ``{id}="bulk"``.
# This bit #332's bulk-tag path; preemptive fix here.


def _parse_ids(raw: str) -> list[str]:
    return [s.strip() for s in (raw or "").split(",") if s.strip()]


@router.post("/tickets/bulk/status")
async def bulk_status(
    request: Request,
    new_status: str = Form(...),
    ids: str = Form(""),
) -> RedirectResponse:
    new_status = new_status.strip()
    if new_status not in VALID_STATUSES:
        return RedirectResponse("/tickets", status_code=303)
    target_ids = _parse_ids(ids)
    if not target_ids:
        return RedirectResponse("/tickets", status_code=303)
    with db.transaction() as txn:
        for tid in target_ids:
            row = txn.execute("SELECT status FROM tickets WHERE id = ? AND deleted_at IS NULL",
                              (tid,)).fetchone()
            if row is None:
                continue
            if row["status"] == new_status:
                continue
            txn.execute(
                "UPDATE tickets SET status = ?, updated_at = ? WHERE id = ?",
                (new_status, _now_iso(), tid),
            )
            _log(txn, operation="ticket_status_changed", target_id=tid,
                 payload={"from": row["status"], "to": new_status, "via": "bulk"})
    return RedirectResponse("/tickets", status_code=303)


@router.post("/tickets/bulk/tag")
async def bulk_tag(
    request: Request,
    tag: str = Form(...),
    ids: str = Form(""),
) -> RedirectResponse:
    tag = (tag or "").strip()
    target_ids = _parse_ids(ids)
    if not tag or not target_ids:
        return RedirectResponse("/tickets", status_code=303)
    with db.transaction() as txn:
        for tid in target_ids:
            row = txn.execute("SELECT tags FROM tickets WHERE id = ? AND deleted_at IS NULL",
                              (tid,)).fetchone()
            if row is None:
                continue
            current = db.unpack_tags(row["tags"])
            if tag in current:
                continue
            current.append(tag)
            txn.execute("UPDATE tickets SET tags = ?, updated_at = ? WHERE id = ?",
                        (db.pack_tags(current), _now_iso(), tid))
            _log(txn, operation="ticket_tagged", target_id=tid,
                 payload={"tag": tag, "via": "bulk"})
    return RedirectResponse("/tickets", status_code=303)


@router.post("/tickets/bulk/assign")
async def bulk_assign(
    request: Request,
    assignee_id: str = Form(""),
    ids: str = Form(""),
) -> RedirectResponse:
    """Bulk-assign tickets. Applies the billing-group revert trigger."""
    target_ids = _parse_ids(ids)
    if not target_ids:
        return RedirectResponse("/tickets", status_code=303)
    new_assignee = (assignee_id or "").strip() or None
    with db.transaction() as txn:
        for tid in target_ids:
            row = txn.execute("SELECT assignee_id FROM tickets "
                              "WHERE id = ? AND deleted_at IS NULL",
                              (tid,)).fetchone()
            if row is None:
                continue
            if row["assignee_id"] == new_assignee:
                continue
            txn.execute(
                "UPDATE tickets SET assignee_id = ?, updated_at = ? WHERE id = ?",
                (new_assignee, _now_iso(), tid),
            )
            _log(txn, operation="ticket_assigned", target_id=tid,
                 payload={"from": row["assignee_id"], "to": new_assignee, "via": "bulk"})
        # Apply the billing-group revert trigger.
        if new_assignee:
            reverted = apply_bulk_assign_revert(
                txn, ticket_ids=target_ids, new_assignee_id=new_assignee,
            )
            for tid in reverted:
                _log(txn, operation="ticket_assignee_reverted", target_id=tid,
                     payload={"reason": "billing_group_lock",
                              "rejected_assignee": new_assignee})
    return RedirectResponse("/tickets", status_code=303)


@router.post("/tickets/bulk/group")
async def bulk_group(
    request: Request,
    group_id: str = Form(""),
    ids: str = Form(""),
) -> RedirectResponse:
    target_ids = _parse_ids(ids)
    new_group = (group_id or "").strip() or None
    if not target_ids:
        return RedirectResponse("/tickets", status_code=303)
    if new_group:
        row = db.connect().execute(
            "SELECT id FROM groups WHERE id = ?", (new_group,),
        ).fetchone()
        if row is None:
            return RedirectResponse("/tickets", status_code=303)
    with db.transaction() as txn:
        for tid in target_ids:
            row = txn.execute("SELECT group_id FROM tickets "
                              "WHERE id = ? AND deleted_at IS NULL",
                              (tid,)).fetchone()
            if row is None or row["group_id"] == new_group:
                continue
            txn.execute(
                "UPDATE tickets SET group_id = ?, updated_at = ? WHERE id = ?",
                (new_group, _now_iso(), tid),
            )
            _log(txn, operation="ticket_group_changed", target_id=tid,
                 payload={"from": row["group_id"], "to": new_group, "via": "bulk"})
    return RedirectResponse("/tickets", status_code=303)


@router.post("/tickets/bulk/close")
async def bulk_close(
    request: Request,
    ids: str = Form(""),
) -> RedirectResponse:
    target_ids = _parse_ids(ids)
    if not target_ids:
        return RedirectResponse("/tickets", status_code=303)
    with db.transaction() as txn:
        for tid in target_ids:
            row = txn.execute("SELECT status FROM tickets "
                              "WHERE id = ? AND deleted_at IS NULL",
                              (tid,)).fetchone()
            if row is None or row["status"] == "closed":
                continue
            txn.execute(
                "UPDATE tickets SET status='closed', updated_at = ? WHERE id = ?",
                (_now_iso(), tid),
            )
            _log(txn, operation="ticket_status_changed", target_id=tid,
                 payload={"from": row["status"], "to": "closed", "via": "bulk-close"})
    return RedirectResponse("/tickets", status_code=303)


@router.post("/tickets/bulk/merge")
async def bulk_merge(
    request: Request,
    survivor_id: str = Form(...),
    loser_ids: str = Form(""),
) -> RedirectResponse:
    """Bulk-merge: same survivor + many losers. Useful for the outage cluster."""
    survivor = survivor_id.strip()
    losers = [tid for tid in _parse_ids(loser_ids) if tid != survivor]
    if not survivor or not losers:
        return RedirectResponse(f"/tickets/{survivor}", status_code=303)
    _merge_into_survivor(survivor, losers)
    return RedirectResponse(f"/tickets/{survivor}", status_code=303)


# ── detail view ────────────────────────────────────────────────────────


@router.get("/tickets/{ticket_id}", response_class=HTMLResponse)
async def ticket_detail(request: Request, ticket_id: str) -> HTMLResponse:
    conn = db.connect()
    row = conn.execute(
        "SELECT t.*, u.name AS requester_name, u.email AS requester_email, "
        "       u.locale AS requester_locale "
        "FROM tickets t LEFT JOIN users u ON t.requester_id = u.id WHERE t.id = ?",
        (ticket_id,),
    ).fetchone()
    if row is None:
        return HTMLResponse("Not found", status_code=404)

    ticket = dict(row) | {"tags": db.unpack_tags(row["tags"])}

    replies = [
        dict(r) | {
            "cc": db.unpack_emails(r["cc"]),
            "bcc": db.unpack_emails(r["bcc"]),
        }
        for r in conn.execute(
            "SELECT * FROM replies WHERE ticket_id = ? ORDER BY created_at, id",
            (ticket_id,),
        ).fetchall()
    ]

    public_thread = [r for r in replies if r["visibility"] == "public"]
    internal_thread = [r for r in replies if r["visibility"] == "internal"]

    # Side panel: requester profile + their other tickets + related tickets.
    other_tickets = [
        dict(r) for r in conn.execute(
            "SELECT id, subject, status FROM tickets "
            "WHERE requester_id = ? AND id != ? AND deleted_at IS NULL "
            "ORDER BY created_at DESC LIMIT 10",
            (ticket["requester_id"], ticket_id),
        ).fetchall()
    ]

    related = [
        dict(r) for r in conn.execute(
            "SELECT el.related_ticket_id AS id, el.relation, t.subject, t.status "
            "FROM escalation_links el "
            "JOIN tickets t ON el.related_ticket_id = t.id "
            "WHERE el.ticket_id = ? AND t.deleted_at IS NULL LIMIT 20",
            (ticket_id,),
        ).fetchall()
    ]

    history = [
        dict(r) for r in conn.execute(
            "SELECT * FROM mutations WHERE target_type='ticket' AND target_id=? "
            "ORDER BY id DESC LIMIT 50",
            (ticket_id,),
        ).fetchall()
    ]

    tab = (request.query_params.get("tab") or "thread").strip().lower()
    if tab not in {"thread", "internal", "related", "history"}:
        tab = "thread"

    macros = [
        dict(m) for m in conn.execute(
            "SELECT id, name, folder FROM macros ORDER BY id LIMIT 200"
        ).fetchall()
    ]
    agents = [
        dict(a) for a in conn.execute(
            "SELECT id, name, group_id FROM users WHERE role='agent' ORDER BY id"
        ).fetchall()
    ]
    groups = [dict(g) for g in conn.execute(
        "SELECT id, name, slug FROM groups ORDER BY name"
    ).fetchall()]

    return app_main.app.state.templates.TemplateResponse(
        "ticket_detail.html",
        {
            "request": request,
            "ticket": ticket,
            "public_thread": public_thread,
            "internal_thread": internal_thread,
            "other_tickets": other_tickets,
            "related": related,
            "history": history,
            "tab": tab,
            "macros": macros,
            "agents": agents,
            "groups": groups,
            "now": _now_iso(),
        },
    )


# ── per-ticket mutations ──────────────────────────────────────────────


@router.post("/tickets/{ticket_id}/reply")
async def post_reply(
    ticket_id: str,
    body: str = Form(""),
    visibility: str = Form("public"),
    cc: str = Form(""),
    bcc: str = Form(""),
    macro_id: str = Form(""),
) -> RedirectResponse:
    """Post a reply. ``visibility`` = public | internal.

    If ``macro_id`` is supplied, the macro body is merged-rendered with
    the requester's profile + the assigned agent then appended to the
    composer body. The renderer is intentionally simple — we only
    substitute the documented tokens.

    A public reply on an internal-only thread (``ticket.visibility='internal'``)
    is REJECTED. We return a 303 to the ticket but record a rejection
    mutation so the oracle can detect the attempt — and so the agent's
    transcript carries the audit.
    """
    body = body or ""
    visibility = (visibility or "public").strip().lower()
    if visibility not in {"public", "internal"}:
        visibility = "public"

    conn = db.connect()
    ticket = conn.execute(
        "SELECT t.*, u.name AS requester_name, u.email AS requester_email "
        "FROM tickets t LEFT JOIN users u ON t.requester_id = u.id "
        "WHERE t.id = ? AND t.deleted_at IS NULL",
        (ticket_id,),
    ).fetchone()
    if ticket is None:
        return RedirectResponse("/tickets", status_code=303)

    # If macro requested, merge into the body.
    if macro_id:
        macro_row = conn.execute("SELECT body FROM macros WHERE id = ?",
                                 (macro_id,)).fetchone()
        if macro_row is not None:
            requester = {
                "name": ticket["requester_name"] or "",
                "email": ticket["requester_email"] or "",
            }
            agent_obj = None
            if ticket["assignee_id"]:
                agent_row = conn.execute(
                    "SELECT name, email FROM users WHERE id = ?",
                    (ticket["assignee_id"],),
                ).fetchone()
                if agent_row is not None:
                    agent_obj = {"name": agent_row["name"], "email": agent_row["email"]}
            merged = _render_merge_body(
                macro_row["body"], requester=requester, agent=agent_obj,
            )
            body = (body + ("\n\n" if body and merged else "") + merged).strip()

    # Reject public reply on internal-only thread.
    if visibility == "public" and ticket["visibility"] == "internal":
        with db.transaction() as txn:
            _log(txn, operation="reply_rejected_internal_only",
                 target_id=ticket_id,
                 payload={"reason": "thread is internal-only",
                          "body_preview": body[:120]})
        return RedirectResponse(f"/tickets/{ticket_id}?tab=thread", status_code=303)

    cc_list = [s.strip() for s in (cc or "").split(",") if s.strip()]
    bcc_list = [s.strip() for s in (bcc or "").split(",") if s.strip()]

    new_id = f"reply_user_{int(time.time() * 1000):013d}"
    # author: assigned agent if any, else first agent (simple default).
    author = ticket["assignee_id"] or "agent_001"

    with db.transaction() as txn:
        txn.execute(
            "INSERT INTO replies (id, ticket_id, author_id, body, visibility, cc, bcc, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new_id, ticket_id, author, body, visibility,
                db.pack_emails(cc_list), db.pack_emails(bcc_list), _now_iso(),
            ),
        )
        # Touch the ticket — public reply on a 'new' ticket promotes to 'open'.
        if visibility == "public" and ticket["status"] == "new":
            txn.execute("UPDATE tickets SET status='open', updated_at=? WHERE id=?",
                        (_now_iso(), ticket_id))
            _log(txn, operation="ticket_status_changed", target_id=ticket_id,
                 payload={"from": "new", "to": "open", "via": "first-reply"})
        _log(txn, operation="reply_posted", target_id=ticket_id,
             payload={"reply_id": new_id, "visibility": visibility,
                      "macro_id": macro_id or None, "body_preview": body[:160]})

    return RedirectResponse(f"/tickets/{ticket_id}?tab=thread", status_code=303)


@router.post("/tickets/{ticket_id}/priority")
async def set_priority(
    ticket_id: str, priority: str = Form(...),
) -> RedirectResponse:
    priority = priority.strip()
    if priority not in VALID_PRIORITIES:
        return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)
    with db.transaction() as txn:
        row = txn.execute("SELECT priority FROM tickets WHERE id=? AND deleted_at IS NULL",
                          (ticket_id,)).fetchone()
        if row is None or row["priority"] == priority:
            return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)
        txn.execute(
            "UPDATE tickets SET priority=?, updated_at=? WHERE id=?",
            (priority, _now_iso(), ticket_id),
        )
        _log(txn, operation="ticket_priority_changed", target_id=ticket_id,
             payload={"from": row["priority"], "to": priority})
    return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)


@router.post("/tickets/{ticket_id}/status")
async def set_status(
    ticket_id: str, status: str = Form(...),
) -> RedirectResponse:
    status = status.strip()
    if status not in VALID_STATUSES:
        return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)
    with db.transaction() as txn:
        row = txn.execute("SELECT status FROM tickets WHERE id=? AND deleted_at IS NULL",
                          (ticket_id,)).fetchone()
        if row is None or row["status"] == status:
            return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)
        txn.execute(
            "UPDATE tickets SET status=?, updated_at=? WHERE id=?",
            (status, _now_iso(), ticket_id),
        )
        _log(txn, operation="ticket_status_changed", target_id=ticket_id,
             payload={"from": row["status"], "to": status})
    return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)


@router.post("/tickets/{ticket_id}/assign")
async def assign(
    ticket_id: str, assignee_id: str = Form(""),
) -> RedirectResponse:
    new = (assignee_id or "").strip() or None
    with db.transaction() as txn:
        row = txn.execute("SELECT assignee_id FROM tickets "
                          "WHERE id=? AND deleted_at IS NULL", (ticket_id,)).fetchone()
        if row is None or row["assignee_id"] == new:
            return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)
        txn.execute(
            "UPDATE tickets SET assignee_id=?, updated_at=? WHERE id=?",
            (new, _now_iso(), ticket_id),
        )
        _log(txn, operation="ticket_assigned", target_id=ticket_id,
             payload={"from": row["assignee_id"], "to": new})
    return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)


@router.post("/tickets/{ticket_id}/group")
async def set_group(
    ticket_id: str, group_id: str = Form(""),
) -> RedirectResponse:
    new = (group_id or "").strip() or None
    if new is not None:
        ok = db.connect().execute(
            "SELECT id FROM groups WHERE id=?", (new,)
        ).fetchone()
        if ok is None:
            return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)
    with db.transaction() as txn:
        row = txn.execute("SELECT group_id FROM tickets "
                          "WHERE id=? AND deleted_at IS NULL", (ticket_id,)).fetchone()
        if row is None or row["group_id"] == new:
            return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)
        txn.execute(
            "UPDATE tickets SET group_id=?, updated_at=? WHERE id=?",
            (new, _now_iso(), ticket_id),
        )
        _log(txn, operation="ticket_group_changed", target_id=ticket_id,
             payload={"from": row["group_id"], "to": new})
    return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)


@router.post("/tickets/{ticket_id}/tag")
async def add_tag(
    ticket_id: str, tag: str = Form(...),
) -> RedirectResponse:
    tag = (tag or "").strip()
    if not tag:
        return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)
    with db.transaction() as txn:
        row = txn.execute("SELECT tags FROM tickets "
                          "WHERE id=? AND deleted_at IS NULL", (ticket_id,)).fetchone()
        if row is None:
            return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)
        tags = db.unpack_tags(row["tags"])
        if tag in tags:
            return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)
        tags.append(tag)
        txn.execute(
            "UPDATE tickets SET tags=?, updated_at=? WHERE id=?",
            (db.pack_tags(tags), _now_iso(), ticket_id),
        )
        _log(txn, operation="ticket_tagged", target_id=ticket_id,
             payload={"tag": tag})
    return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)


@router.post("/tickets/{ticket_id}/apply_macro")
async def apply_macro(
    ticket_id: str, macro_id: str = Form(...),
    visibility: str = Form("public"),
) -> RedirectResponse:
    """Apply a macro: posts the merged macro body as a reply.

    This is the canonical "fast-path" for T02. The composer route also
    supports macros embedded into a manual reply (see ``post_reply``).
    """
    return await post_reply(
        ticket_id=ticket_id, body="", visibility=visibility,
        cc="", bcc="", macro_id=macro_id,
    )


@router.post("/tickets/{ticket_id}/escalate")
async def escalate(
    ticket_id: str, related_id: str = Form(...),
) -> RedirectResponse:
    """Create a symmetric escalation link between two tickets."""
    related_id = related_id.strip()
    if not related_id or related_id == ticket_id:
        return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)
    with db.transaction() as txn:
        for a, b in ((ticket_id, related_id), (related_id, ticket_id)):
            txn.execute(
                "INSERT INTO escalation_links (ticket_id, related_ticket_id, relation, created_at) "
                "VALUES (?, ?, 'escalates_to', ?)",
                (a, b, _now_iso()),
            )
        _log(txn, operation="ticket_escalated", target_id=ticket_id,
             payload={"related_ticket_id": related_id})
    return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)


@router.post("/tickets/{ticket_id}/merge")
async def merge(
    ticket_id: str, loser_ids: str = Form(""),
) -> RedirectResponse:
    """Single-survivor merge. Survivor = ``ticket_id``; losers re-point."""
    losers = [tid for tid in _parse_ids(loser_ids) if tid != ticket_id]
    if not losers:
        return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)
    _merge_into_survivor(ticket_id, losers)
    return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)


# ── shared merge helper ───────────────────────────────────────────────


def _merge_into_survivor(survivor_id: str, loser_ids: list[str]) -> None:
    """Move all replies + escalation links onto the survivor; mark losers
    deleted; carry tags forward. Oracle asserts no orphan replies."""
    if not loser_ids:
        return
    with db.transaction() as txn:
        survivor = txn.execute(
            "SELECT tags FROM tickets WHERE id = ? AND deleted_at IS NULL",
            (survivor_id,),
        ).fetchone()
        if survivor is None:
            return
        survivor_tags = set(db.unpack_tags(survivor["tags"]))
        for loser in loser_ids:
            row = txn.execute(
                "SELECT tags FROM tickets WHERE id = ? AND deleted_at IS NULL",
                (loser,),
            ).fetchone()
            if row is None:
                continue
            survivor_tags.update(db.unpack_tags(row["tags"]))
            # Re-point replies to the survivor.
            txn.execute(
                "UPDATE replies SET ticket_id = ? WHERE ticket_id = ?",
                (survivor_id, loser),
            )
            # Re-point escalation links to the survivor.
            txn.execute(
                "UPDATE escalation_links SET ticket_id = ? WHERE ticket_id = ?",
                (survivor_id, loser),
            )
            txn.execute(
                "UPDATE escalation_links SET related_ticket_id = ? "
                "WHERE related_ticket_id = ?",
                (survivor_id, loser),
            )
            # Soft-delete the loser.
            txn.execute(
                "UPDATE tickets SET deleted_at = ?, updated_at = ? WHERE id = ?",
                (_now_iso(), _now_iso(), loser),
            )
            _log(txn, operation="ticket_merged", target_id=loser,
                 payload={"into": survivor_id})
        txn.execute(
            "UPDATE tickets SET tags = ?, updated_at = ? WHERE id = ?",
            (db.pack_tags(sorted(survivor_tags)), _now_iso(), survivor_id),
        )
        _log(txn, operation="ticket_merge_completed", target_id=survivor_id,
             payload={"absorbed": loser_ids})
