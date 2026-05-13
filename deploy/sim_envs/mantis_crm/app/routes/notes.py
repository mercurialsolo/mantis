"""Notes — free-form markdown anchored to a record. Pinned notes float."""

from __future__ import annotations

import time

from fastapi import APIRouter, Form
from fastapi.responses import RedirectResponse

from .. import db, main as app_main

router = APIRouter()


@router.post("/notes/create")
async def create_note(
    target_type: str = Form(...),
    target_id: str = Form(...),
    body_md: str = Form(...),
    pinned: str = Form("0"),
) -> RedirectResponse:
    if target_type not in {"contact", "deal", "company"}:
        return RedirectResponse("/contacts", status_code=303)
    new_id = f"note_user_{int(time.time() * 1000):013d}"
    is_pinned = 1 if pinned in ("1", "true", "on") else 0
    with db.transaction() as txn:
        txn.execute(
            "INSERT INTO notes (id, target_type, target_id, body_md, "
            "pinned, author_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?)",
            (new_id, target_type, target_id, body_md, is_pinned,
             app_main.now_value()),
        )
        db.log_mutation(
            txn, occurred_at=app_main.now_value(),
            operation="note_added", target_type=target_type, target_id=target_id,
            payload={"note_id": new_id, "pinned": bool(is_pinned)},
        )
    app_main.emit("mutation.note_added",
                  {"target_type": target_type, "target_id": target_id})
    redirect = f"/{target_type}s/{target_id}"
    return RedirectResponse(redirect, status_code=303)


@router.post("/notes/{note_id}/pin")
async def toggle_pin(note_id: str) -> RedirectResponse:
    with db.transaction() as txn:
        row = txn.execute(
            "SELECT target_type, target_id, pinned FROM notes WHERE id = ?",
            (note_id,),
        ).fetchone()
        if row is None:
            return RedirectResponse("/contacts", status_code=303)
        new_val = 0 if row["pinned"] else 1
        txn.execute("UPDATE notes SET pinned = ? WHERE id = ?",
                    (new_val, note_id))
    return RedirectResponse(
        f"/{row['target_type']}s/{row['target_id']}", status_code=303
    )
