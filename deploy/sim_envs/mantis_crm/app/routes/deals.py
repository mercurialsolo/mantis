"""Deal pipeline (kanban) + bulk stage change."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()


VALID_STAGES = [
    "Prospect", "Qualified", "Proposal", "Negotiation",
    "At Risk", "Closed Won", "Closed Lost",
]


def _log_mutation(conn, *, operation: str, target_type: str, target_id: str,
                  payload: dict[str, Any] | None = None) -> None:
    db.log_mutation(
        conn,
        occurred_at=app_main.now_value(),
        operation=operation,
        target_type=target_type,
        target_id=target_id,
        payload=payload or {},
    )
    app_main.emit(f"mutation.{operation}", {
        "target_type": target_type, "target_id": target_id,
        **(payload or {}),
    })


@router.get("/deals", response_class=HTMLResponse)
async def list_deals(request: Request) -> HTMLResponse:
    """Pipeline kanban — one column per stage. Within a column we sort
    by amount descending so big deals are easy to spot."""
    conn = db.connect()
    owner_filter = (request.query_params.get("owner") or "").strip()
    stage_filter = (request.query_params.get("stage") or "").strip()

    where = ["1=1"]
    args: list = []
    if owner_filter:
        where.append("owner_id = ?")
        args.append(owner_filter)
    if stage_filter:
        where.append("stage = ?")
        args.append(stage_filter)
    where_clause = " AND ".join(where)

    # Each column shows up to 200 rows — the agent shouldn't be paging
    # through the entire kanban; if they want everything they should
    # filter by owner / stage.
    columns: dict[str, list[dict[str, Any]]] = {s: [] for s in VALID_STAGES}
    rows = conn.execute(
        f"SELECT id, name, stage, amount, expected_close, owner_id, contact_id "
        f"FROM deals WHERE {where_clause} ORDER BY amount DESC LIMIT 5000",
        args,
    ).fetchall()
    for r in rows:
        bucket = r["stage"] if r["stage"] in columns else "Prospect"
        if len(columns[bucket]) < 200:
            columns[bucket].append(dict(r))

    return app_main.app.state.templates.TemplateResponse(
        "deals_list.html",
        {
            "request": request,
            "columns": columns,
            "stages": VALID_STAGES,
            "filters": {"owner": owner_filter, "stage": stage_filter},
        },
    )


@router.get("/deals/{deal_id}", response_class=HTMLResponse)
async def deal_detail(request: Request, deal_id: str) -> HTMLResponse:
    conn = db.connect()
    row = conn.execute(
        "SELECT d.*, c.name AS contact_name, co.name AS company_name "
        "FROM deals d "
        "LEFT JOIN contacts c ON d.contact_id = c.id "
        "LEFT JOIN companies co ON d.company_id = co.id "
        "WHERE d.id = ?",
        (deal_id,),
    ).fetchone()
    if row is None:
        return HTMLResponse("Not found", status_code=404)
    activities = [
        dict(a) for a in conn.execute(
            "SELECT * FROM activities WHERE target_type='deal' AND target_id=? "
            "ORDER BY occurred_at DESC LIMIT 100",
            (deal_id,),
        ).fetchall()
    ]
    return app_main.app.state.templates.TemplateResponse(
        "deal_detail.html",
        {"request": request, "deal": dict(row), "activities": activities,
         "stages": VALID_STAGES},
    )


# Bulk route declared BEFORE the per-deal route so /deals/bulk/stage
# isn't shadowed by /deals/{deal_id}/stage (Starlette matches in order).


@router.post("/deals/bulk/stage")
async def bulk_change_stage(
    request: Request,
    ids: str = Form(...),
    stage: str = Form(...),
) -> RedirectResponse:
    if stage not in VALID_STAGES:
        return RedirectResponse("/deals", status_code=303)
    target_ids = [s.strip() for s in ids.split(",") if s.strip()]
    if not target_ids:
        return RedirectResponse("/deals", status_code=303)

    with db.transaction() as txn:
        for did in target_ids:
            cur = txn.execute("SELECT stage FROM deals WHERE id = ?", (did,))
            row = cur.fetchone()
            if row is None or row["stage"] == stage:
                continue
            txn.execute("UPDATE deals SET stage = ? WHERE id = ?", (stage, did))
            _log_mutation(txn, operation="deal_stage_changed", target_type="deal",
                          target_id=did,
                          payload={"from": row["stage"], "to": stage, "via": "bulk"})
    return RedirectResponse("/deals", status_code=303)


@router.post("/deals/{deal_id}/stage")
async def change_stage(deal_id: str, stage: str = Form(...)) -> RedirectResponse:
    if stage not in VALID_STAGES:
        return RedirectResponse(f"/deals/{deal_id}", status_code=303)
    with db.transaction() as txn:
        cur = txn.execute("SELECT stage FROM deals WHERE id = ?", (deal_id,))
        row = cur.fetchone()
        if row is None:
            return RedirectResponse("/deals", status_code=303)
        if row["stage"] == stage:
            return RedirectResponse(f"/deals/{deal_id}", status_code=303)
        txn.execute("UPDATE deals SET stage = ? WHERE id = ?", (stage, deal_id))
        _log_mutation(txn, operation="deal_stage_changed", target_type="deal",
                      target_id=deal_id, payload={"from": row["stage"], "to": stage})
    return RedirectResponse(f"/deals/{deal_id}", status_code=303)
