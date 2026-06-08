"""Profile surface: /in/<handle>/ and the Connect action."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()


def _user_by_handle(conn, handle: str):
    r = conn.execute(
        "SELECT id, handle, name, headline, about, location, avatar_color, "
        "email FROM users WHERE handle = ?",
        (handle,),
    ).fetchone()
    return dict(r) if r else None


def _user_by_id(conn, uid: str):
    r = conn.execute(
        "SELECT id, handle, name, headline, about, location, avatar_color "
        "FROM users WHERE id = ?",
        (uid,),
    ).fetchone()
    return dict(r) if r else None


@router.get("/in/{handle}/", response_class=HTMLResponse)
@router.get("/in/{handle}", response_class=HTMLResponse)
async def profile(request: Request, handle: str):
    conn = db.connect()
    profile = _user_by_handle(conn, handle)
    if not profile:
        return HTMLResponse("Profile not found", status_code=404)

    me = request.state.current_user or _user_by_id(conn, "user_00001")
    me_id = me["id"] if me else "user_00001"
    is_self = me_id == profile["id"]

    experience = [dict(r) for r in conn.execute(
        "SELECT * FROM experience WHERE user_id = ? ORDER BY sort_idx",
        (profile["id"],),
    ).fetchall()]
    education = [dict(r) for r in conn.execute(
        "SELECT * FROM education WHERE user_id = ? ORDER BY sort_idx",
        (profile["id"],),
    ).fetchall()]
    skills = [dict(r) for r in conn.execute(
        "SELECT * FROM skills WHERE user_id = ? ORDER BY sort_idx",
        (profile["id"],),
    ).fetchall()]

    conn_count = conn.execute(
        "SELECT COUNT(*) AS n FROM connections "
        "WHERE (from_user_id = ? OR to_user_id = ?) AND status = 'accepted'",
        (profile["id"], profile["id"]),
    ).fetchone()["n"]

    existing_conn = conn.execute(
        "SELECT status FROM connections WHERE "
        "(from_user_id = ? AND to_user_id = ?) "
        "OR (from_user_id = ? AND to_user_id = ?)",
        (me_id, profile["id"], profile["id"], me_id),
    ).fetchone()
    conn_state = existing_conn["status"] if existing_conn else None

    # Activity: this user's posts (most recent 5)
    activity = [
        dict(r) for r in conn.execute(
            "SELECT id, body, created_at FROM posts "
            "WHERE author_id = ? ORDER BY created_at DESC, id DESC LIMIT 5",
            (profile["id"],),
        ).fetchall()
    ]

    return request.app.state.templates.TemplateResponse(
        "profile.html",
        {
            "request": request,
            "profile": profile,
            "me": me,
            "is_self": is_self,
            "experience": experience,
            "education": education,
            "skills": skills,
            "connection_count": conn_count,
            "conn_state": conn_state,
            "activity": activity,
        },
    )


@router.post("/in/{handle}/connect")
async def connect(
    request: Request, handle: str,
    note: str = Form(""),
):
    conn = db.connect()
    target = _user_by_handle(conn, handle)
    if not target:
        return HTMLResponse("Profile not found", status_code=404)

    me = request.state.current_user or _user_by_id(conn, "user_00001")
    me_id = me["id"] if me else "user_00001"

    if me_id == target["id"]:
        return RedirectResponse(f"/in/{handle}/", status_code=303)

    # Reject duplicate edges.
    existing = conn.execute(
        "SELECT id FROM connections WHERE "
        "(from_user_id = ? AND to_user_id = ?) "
        "OR (from_user_id = ? AND to_user_id = ?)",
        (me_id, target["id"], target["id"], me_id),
    ).fetchone()
    if existing:
        return RedirectResponse(f"/in/{handle}/", status_code=303)

    next_idx = conn.execute(
        "SELECT COALESCE(MAX(CAST(SUBSTR(id, 6) AS INTEGER)), 0) + 1 AS n "
        "FROM connections"
    ).fetchone()["n"]
    cid = f"conn_{next_idx:06d}"
    now = app_main.now_value()

    conn.execute(
        "INSERT INTO connections (id, from_user_id, to_user_id, status, "
        "note, created_at, accepted_at) VALUES (?,?,?,?,?,?,?)",
        (cid, me_id, target["id"], "pending", note.strip(), now, None),
    )
    db.log_audit(
        conn,
        occurred_at=now,
        operation="connection_requested",
        target_type="connection",
        target_id=cid,
        payload={
            "from_user_id": me_id,
            "to_user_id": target["id"],
            "note": note.strip(),
        },
    )
    conn.commit()
    app_main.emit("connection_requested",
                  {"connection_id": cid, "from": me_id, "to": target["id"]})
    app_main.emit_mutation(
        op="connection_requested", target_type="connection", target_id=cid,
        payload={
            "from_user_id": me_id, "to_user_id": target["id"],
            "note": note.strip(),
        },
    )
    return RedirectResponse(f"/in/{handle}/", status_code=303)
