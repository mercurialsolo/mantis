"""Inbox + thread surfaces. Buyer-facing only for now."""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import auth, db, main as app_main

router = APIRouter()


def _templates(request: Request):
    return request.app.state.templates


@router.get("/inbox", response_class=HTMLResponse)
async def inbox_list(request: Request):
    conn = db.connect()
    me = auth.effective_buyer_id(request)
    rows = conn.execute(
        """SELECT c.id, c.seller_id, c.last_msg_at,
                  u.display_name AS seller_display,
                  u.username AS seller_username,
                  s.avatar_palette AS seller_avatar_palette,
                  (SELECT body FROM messages m WHERE m.conversation_id = c.id
                   ORDER BY m.created_at DESC LIMIT 1) AS last_body
           FROM conversations c
           JOIN users u ON c.seller_id = u.id
           JOIN sellers s ON c.seller_id = s.user_id
           WHERE c.buyer_id = ?
           ORDER BY c.last_msg_at DESC""",
        (me,),
    ).fetchall()
    return _templates(request).TemplateResponse(
        "inbox.html",
        {
            "request": request,
            "threads": [dict(r) for r in rows],
            "active_thread": None,
            "messages": [],
            "me": me,
        },
    )


@router.get("/inbox/{thread_id}", response_class=HTMLResponse)
async def inbox_thread(request: Request, thread_id: str):
    conn = db.connect()
    me = auth.effective_buyer_id(request)
    thread = conn.execute(
        """SELECT c.id, c.seller_id, c.last_msg_at,
                  u.display_name AS seller_display,
                  u.username AS seller_username,
                  s.country AS seller_country,
                  s.avatar_palette AS seller_avatar_palette
           FROM conversations c
           JOIN users u ON c.seller_id = u.id
           JOIN sellers s ON c.seller_id = s.user_id
           WHERE c.id = ? AND c.buyer_id = ?""",
        (thread_id, me),
    ).fetchone()
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")
    rows = conn.execute(
        """SELECT m.*, u.display_name AS sender_display, u.username AS sender_username
           FROM messages m JOIN users u ON m.sender_id = u.id
           WHERE m.conversation_id = ? ORDER BY m.created_at""",
        (thread_id,),
    ).fetchall()
    # Other threads for sidebar.
    threads = conn.execute(
        """SELECT c.id, c.seller_id, c.last_msg_at,
                  u.display_name AS seller_display,
                  u.username AS seller_username,
                  s.avatar_palette AS seller_avatar_palette,
                  (SELECT body FROM messages m WHERE m.conversation_id = c.id
                   ORDER BY m.created_at DESC LIMIT 1) AS last_body
           FROM conversations c
           JOIN users u ON c.seller_id = u.id
           JOIN sellers s ON c.seller_id = s.user_id
           WHERE c.buyer_id = ?
           ORDER BY c.last_msg_at DESC""",
        (me,),
    ).fetchall()
    return _templates(request).TemplateResponse(
        "inbox.html",
        {
            "request": request,
            "threads": [dict(r) for r in threads],
            "active_thread": dict(thread),
            "messages": [dict(r) for r in rows],
            "me": me,
        },
    )


@router.post("/inbox/{thread_id}/send")
async def send_message(
    request: Request,
    thread_id: str,
    body: str = Form(...),
):
    body = body.strip()
    if not body:
        return RedirectResponse(url=f"/inbox/{thread_id}", status_code=303)
    conn = db.connect()
    me = auth.effective_buyer_id(request)
    thread = conn.execute(
        "SELECT id, buyer_id, seller_id FROM conversations WHERE id = ? AND buyer_id = ?",
        (thread_id, me),
    ).fetchone()
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")
    msg_count = int(conn.execute(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
        (thread_id,)
    ).fetchone()[0])
    msg_id = f"msg_{thread_id}_{msg_count+1}"
    now = app_main.now_value()
    with db.transaction() as tx:
        tx.execute(
            "INSERT INTO messages (id, conversation_id, sender_id, body, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (msg_id, thread_id, me, body, now),
        )
        tx.execute(
            "UPDATE conversations SET last_msg_at = ? WHERE id = ?",
            (now, thread_id),
        )
        db.log_audit(
            tx,
            occurred_at=now,
            operation="message_sent",
            target_type="conversation",
            target_id=thread_id,
            payload={
                "sender_id": me,
                "recipient_id": thread["seller_id"],
                "message_id": msg_id,
                "body": body,
            },
        )
    return RedirectResponse(url=f"/inbox/{thread_id}", status_code=303)


@router.get("/inbox/new/{seller_id}")
async def open_thread_with_seller(request: Request, seller_id: str):
    """Open or create a thread with a seller; redirect to it."""
    conn = db.connect()
    me = auth.effective_buyer_id(request)
    row = conn.execute(
        "SELECT id FROM conversations WHERE buyer_id = ? AND seller_id = ?",
        (me, seller_id),
    ).fetchone()
    if row is not None:
        return RedirectResponse(url=f"/inbox/{row['id']}", status_code=303)
    count = int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0])
    cid = f"conv_{count+1:05d}"
    now = app_main.now_value()
    with db.transaction() as tx:
        tx.execute(
            "INSERT INTO conversations (id, buyer_id, seller_id, last_msg_at) "
            "VALUES (?, ?, ?, ?)",
            (cid, me, seller_id, now),
        )
        db.log_audit(
            tx,
            occurred_at=now,
            operation="conversation_opened",
            target_type="conversation",
            target_id=cid,
            payload={"buyer_id": me, "seller_id": seller_id},
        )
    return RedirectResponse(url=f"/inbox/{cid}", status_code=303)
