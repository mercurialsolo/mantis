"""Connections surface: /mynetwork/ and /mynetwork/invitation-manager/."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()


def _me(request: Request) -> dict:
    me = request.state.current_user or {}
    if not me:
        return {"id": "user_00001"}
    return me


def _user_chip(conn, uid: str):
    r = conn.execute(
        "SELECT id, handle, name, headline, avatar_color FROM users WHERE id=?",
        (uid,),
    ).fetchone()
    return dict(r) if r else None


@router.get("/mynetwork/", response_class=HTMLResponse)
async def mynetwork(request: Request):
    conn = db.connect()
    me = _me(request)
    me_id = me["id"]

    counts = {
        "connections": conn.execute(
            "SELECT COUNT(*) AS n FROM connections "
            "WHERE (from_user_id = ? OR to_user_id = ?) AND status = 'accepted'",
            (me_id, me_id),
        ).fetchone()["n"],
        "received_pending": conn.execute(
            "SELECT COUNT(*) AS n FROM connections "
            "WHERE to_user_id = ? AND status = 'pending'",
            (me_id,),
        ).fetchone()["n"],
        "sent_pending": conn.execute(
            "SELECT COUNT(*) AS n FROM connections "
            "WHERE from_user_id = ? AND status = 'pending'",
            (me_id,),
        ).fetchone()["n"],
    }

    received = [
        {
            "id": r["id"],
            "note": r["note"],
            "user": _user_chip(conn, r["from_user_id"]),
            "created_at": r["created_at"],
        }
        for r in conn.execute(
            "SELECT * FROM connections WHERE to_user_id = ? "
            "AND status = 'pending' ORDER BY id DESC LIMIT 3",
            (me_id,),
        ).fetchall()
    ]

    # People you may know — same shape as feed suggestion.
    suggested_rows = conn.execute(
        "SELECT id, name, headline, handle, avatar_color FROM users "
        "WHERE id != ? "
        "AND id NOT IN (SELECT to_user_id FROM connections WHERE from_user_id = ?) "
        "AND id NOT IN (SELECT from_user_id FROM connections WHERE to_user_id = ?) "
        "ORDER BY id LIMIT 12",
        (me_id, me_id, me_id),
    ).fetchall()
    suggested = [dict(r) for r in suggested_rows]

    return request.app.state.templates.TemplateResponse(
        "mynetwork.html",
        {
            "request": request,
            "me": me,
            "counts": counts,
            "received": received,
            "suggested": suggested,
        },
    )


@router.get("/mynetwork/invitation-manager/", response_class=HTMLResponse)
async def invitation_manager(request: Request, tab: str = "received"):
    conn = db.connect()
    me = _me(request)
    me_id = me["id"]

    if tab == "sent":
        rows = conn.execute(
            "SELECT * FROM connections WHERE from_user_id = ? "
            "AND status = 'pending' ORDER BY id DESC",
            (me_id,),
        ).fetchall()
        items = [
            {
                "id": r["id"],
                "note": r["note"],
                "user": _user_chip(conn, r["to_user_id"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    else:
        rows = conn.execute(
            "SELECT * FROM connections WHERE to_user_id = ? "
            "AND status = 'pending' ORDER BY id DESC",
            (me_id,),
        ).fetchall()
        items = [
            {
                "id": r["id"],
                "note": r["note"],
                "user": _user_chip(conn, r["from_user_id"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    return request.app.state.templates.TemplateResponse(
        "invitation_manager.html",
        {
            "request": request,
            "me": me,
            "tab": tab,
            "items": items,
        },
    )


@router.post("/mynetwork/invitations/{conn_id}/accept")
async def accept_invitation(request: Request, conn_id: str):
    conn = db.connect()
    me = _me(request)
    me_id = me["id"]
    now = app_main.now_value()

    row = conn.execute(
        "SELECT * FROM connections WHERE id = ? AND to_user_id = ? "
        "AND status = 'pending'",
        (conn_id, me_id),
    ).fetchone()
    if not row:
        return RedirectResponse("/mynetwork/invitation-manager/",
                                status_code=303)

    conn.execute(
        "UPDATE connections SET status = 'accepted', accepted_at = ? "
        "WHERE id = ?",
        (now, conn_id),
    )
    db.log_audit(
        conn,
        occurred_at=now,
        operation="connection_accepted",
        target_type="connection",
        target_id=conn_id,
        payload={"from_user_id": row["from_user_id"], "to_user_id": me_id},
    )
    conn.commit()
    app_main.emit("connection_accepted", {"connection_id": conn_id})
    app_main.emit_mutation(
        op="connection_accepted", target_type="connection",
        target_id=conn_id,
        payload={"from_user_id": row["from_user_id"], "to_user_id": me_id},
    )
    return RedirectResponse("/mynetwork/invitation-manager/", status_code=303)


@router.post("/mynetwork/invitations/{conn_id}/ignore")
async def ignore_invitation(request: Request, conn_id: str):
    conn = db.connect()
    me = _me(request)
    me_id = me["id"]
    now = app_main.now_value()
    conn.execute(
        "DELETE FROM connections WHERE id = ? AND to_user_id = ?",
        (conn_id, me_id),
    )
    db.log_audit(
        conn,
        occurred_at=now,
        operation="connection_ignored",
        target_type="connection",
        target_id=conn_id,
    )
    conn.commit()
    app_main.emit("connection_ignored", {"connection_id": conn_id})
    app_main.emit_mutation(
        op="connection_ignored", target_type="connection",
        target_id=conn_id,
    )
    return RedirectResponse("/mynetwork/invitation-manager/", status_code=303)


@router.post("/mynetwork/invitations/{conn_id}/withdraw")
async def withdraw_invitation(request: Request, conn_id: str):
    conn = db.connect()
    me = _me(request)
    me_id = me["id"]
    now = app_main.now_value()
    conn.execute(
        "DELETE FROM connections WHERE id = ? AND from_user_id = ?",
        (conn_id, me_id),
    )
    db.log_audit(
        conn,
        occurred_at=now,
        operation="connection_withdrawn",
        target_type="connection",
        target_id=conn_id,
    )
    conn.commit()
    app_main.emit("connection_withdrawn", {"connection_id": conn_id})
    app_main.emit_mutation(
        op="connection_withdrawn", target_type="connection",
        target_id=conn_id,
    )
    return RedirectResponse("/mynetwork/invitation-manager/?tab=sent",
                            status_code=303)
