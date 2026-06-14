"""Mock mailbox — where magic-link + OTP emails land.

Open without a session by design: the email flows require the agent to
read the message *before* it has authenticated. A single shared dev
mailbox lists every delivered message newest-first; the message view
surfaces the magic-link as a real anchor and the OTP code in large type
so a vision agent can read it.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from .. import db, main as app_main

router = APIRouter()

_LINK_RE = re.compile(r"(/auth/magic/verify\?token=[A-Za-z0-9_\-]+)")


def _extract_link(body: str) -> str | None:
    m = _LINK_RE.search(body or "")
    return m.group(1) if m else None


@router.get("/inbox", response_class=HTMLResponse)
async def inbox(request: Request) -> Response:
    conn = db.connect()
    rows = conn.execute(
        "SELECT id, to_email, subject, kind, created_at, consumed_at "
        "FROM emails ORDER BY id DESC").fetchall()
    return app_main.app.state.templates.TemplateResponse(
        "inbox.html",
        {"request": request, "config": app_main.app.state.auth_config,
         "current_user": getattr(request.state, "current_user", None),
         "messages": [dict(r) for r in rows]},
    )


@router.get("/inbox/{email_id}", response_class=HTMLResponse)
async def inbox_message(request: Request, email_id: str) -> Response:
    conn = db.connect()
    row = conn.execute(
        "SELECT * FROM emails WHERE id = ?", (email_id,)).fetchone()
    if row is None:
        return app_main.app.state.templates.TemplateResponse(
            "inbox.html",
            {"request": request, "config": app_main.app.state.auth_config,
             "current_user": getattr(request.state, "current_user", None),
             "messages": [], "error": f"no message {email_id}"},
            status_code=404,
        )
    msg = dict(row)
    return app_main.app.state.templates.TemplateResponse(
        "inbox_message.html",
        {"request": request, "config": app_main.app.state.auth_config,
         "current_user": getattr(request.state, "current_user", None),
         "msg": msg, "magic_link": _extract_link(msg.get("body", ""))},
    )
