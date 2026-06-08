"""Messaging surface: /messaging/, thread view, send."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()


def _me_id(request: Request) -> str:
    me = request.state.current_user or {"id": "user_00001"}
    return me["id"]


def _user_chip(conn, uid: str):
    r = conn.execute(
        "SELECT id, handle, name, headline, avatar_color FROM users WHERE id=?",
        (uid,),
    ).fetchone()
    return dict(r) if r else None


def _thread_summary(conn, thread_row, me_id: str):
    parts = db.unpack_json(thread_row["participants"])
    peer_id = next((p for p in parts if p != me_id), None)
    peer = _user_chip(conn, peer_id) if peer_id else None
    last = conn.execute(
        "SELECT body, created_at FROM messages WHERE thread_id = ? "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (thread_row["id"],),
    ).fetchone()
    snippet = last["body"] if last else ""
    if len(snippet) > 60:
        snippet = snippet[:57].rstrip() + "…"
    return {
        "id": thread_row["id"],
        "peer": peer,
        "snippet": snippet,
        "last_at": thread_row["last_message_at"],
    }


@router.get("/messaging/", response_class=HTMLResponse)
async def messaging(request: Request, thread: str | None = None):
    conn = db.connect()
    me_id = _me_id(request)

    thread_rows = conn.execute(
        "SELECT id, participants, last_message_at FROM threads "
        "WHERE participants LIKE ? "
        "ORDER BY last_message_at DESC, id DESC",
        (f'%"{me_id}"%',),
    ).fetchall()
    threads = [_thread_summary(conn, r, me_id) for r in thread_rows]

    active_id = thread or (threads[0]["id"] if threads else None)
    active = None
    messages = []
    if active_id:
        row = conn.execute(
            "SELECT id, participants FROM threads WHERE id = ?",
            (active_id,),
        ).fetchone()
        if row:
            parts = db.unpack_json(row["participants"])
            peer_id = next((p for p in parts if p != me_id), None)
            peer = _user_chip(conn, peer_id) if peer_id else None
            active = {"id": active_id, "peer": peer}
            messages = [
                dict(m) for m in conn.execute(
                    "SELECT id, sender_id, body, created_at "
                    "FROM messages WHERE thread_id = ? "
                    "ORDER BY created_at, id",
                    (active_id,),
                ).fetchall()
            ]

    return request.app.state.templates.TemplateResponse(
        "messaging.html",
        {
            "request": request,
            "me_id": me_id,
            "threads": threads,
            "active": active,
            "messages": messages,
        },
    )


@router.post("/messaging/{thread_id}/send")
async def send_message(request: Request, thread_id: str,
                       body: str = Form(...)):
    conn = db.connect()
    me_id = _me_id(request)
    text = (body or "").strip()
    if not text:
        return RedirectResponse(f"/messaging/?thread={thread_id}",
                                status_code=303)
    now = app_main.now_value()

    next_idx = conn.execute(
        "SELECT COALESCE(MAX(CAST(SUBSTR(id, 5) AS INTEGER)), 0) + 1 AS n "
        "FROM messages WHERE id LIKE 'msg_n%'"
    ).fetchone()["n"]
    mid = f"msg_n{next_idx:06d}"

    conn.execute(
        "INSERT INTO messages (id, thread_id, sender_id, body, created_at) "
        "VALUES (?,?,?,?,?)",
        (mid, thread_id, me_id, text, now),
    )
    conn.execute(
        "UPDATE threads SET last_message_at = ? WHERE id = ?",
        (now, thread_id),
    )
    db.log_audit(
        conn,
        occurred_at=now,
        operation="message_sent",
        target_type="message",
        target_id=mid,
        payload={"thread_id": thread_id, "sender_id": me_id, "body": text},
    )
    conn.commit()
    app_main.emit("message_sent",
                  {"message_id": mid, "thread_id": thread_id})
    app_main.emit_mutation(
        op="message_sent", target_type="message", target_id=mid,
        payload={"thread_id": thread_id, "sender_id": me_id, "body": text},
    )
    return RedirectResponse(f"/messaging/?thread={thread_id}",
                            status_code=303)
