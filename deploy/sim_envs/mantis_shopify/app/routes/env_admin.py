"""Harness contract — ``/__env__/{health,reset,seed,clock,oracle,state,events,mutations}``."""

from __future__ import annotations

import json
import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .. import db, main as app_main, seed

router = APIRouter(prefix="/__env__")


@router.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "ok": True,
        "seed": app_main.seed_value(),
        "now": app_main.now_value(),
        "boot_time": app_main.boot_time(),
    })


@router.post("/reset")
async def reset(request: Request) -> JSONResponse:
    if not app_main.admin_token_ok(request):
        return app_main.admin_required_response()
    conn = db.connect()
    seed.seed(conn, seed_val=app_main.seed_value(), fake_now=app_main.now_value())
    app_main.clear_events()
    app_main.emit("reset")
    return JSONResponse({"ok": True})


@router.post("/seed")
async def reseed(request: Request) -> JSONResponse:
    if not app_main.admin_token_ok(request):
        return app_main.admin_required_response()
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        payload = {}
    new_seed = int(payload.get("seed", app_main.seed_value()))
    conn = db.connect()
    seed.seed(conn, seed_val=new_seed, fake_now=app_main.now_value())
    app_main.emit("reseeded", {"seed": new_seed})
    return JSONResponse({"ok": True, "seed": new_seed})


@router.post("/clock")
async def clock(request: Request) -> JSONResponse:
    if not app_main.admin_token_ok(request):
        return app_main.admin_required_response()
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        payload = {}
    new_now = str(payload.get("now") or app_main.now_value())
    os.environ["FAKE_NOW"] = new_now
    app_main.emit("clock_set", {"now": new_now})
    return JSONResponse({"ok": True, "now": new_now})


@router.get("/oracle")
async def oracle(request: Request) -> JSONResponse:
    if not app_main.admin_token_ok(request):
        return app_main.admin_required_response()
    task_id = (request.query_params.get("task_id") or "").strip()
    if not task_id:
        return JSONResponse({"passed": False, "score": 0.0,
                             "reasons": ["task_id is required"], "diff": {}})

    from ..oracles import grade
    result = grade(task_id, db.connect(), now=app_main.now_value(),
                   seed_val=app_main.seed_value())
    app_main.emit("oracle_query", {"task_id": task_id, "passed": result["passed"]})
    return JSONResponse(result)


@router.get("/state")
async def state(request: Request) -> JSONResponse:
    if not app_main.admin_token_ok(request):
        return app_main.admin_required_response()
    conn = db.connect()

    def count(table: str) -> int:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    recent_audit = [
        {
            "id": r["id"],
            "occurred_at": r["occurred_at"],
            "operation": r["operation"],
            "target_type": r["target_type"],
            "target_id": r["target_id"],
        }
        for r in conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT 50"
        ).fetchall()
    ]

    return JSONResponse({
        "seed": app_main.seed_value(),
        "now": app_main.now_value(),
        "counts": {
            "partners": count("partners"),
            "users": count("users"),
            "stores": count("stores"),
            "payouts": count("payouts"),
            "payout_line_items": count("payout_line_items"),
            "leads": count("leads"),
            "referrals": count("referrals"),
            "directory_listings": count("directory_listings"),
            "tickets": count("tickets"),
            "themes": count("themes"),
            "catalogs": count("catalogs"),
            "changelog": count("changelog"),
            "notifications": count("notifications"),
            "audit_log": count("audit_log"),
        },
        "recent_audit": recent_audit,
    })


@router.get("/events")
async def events(request: Request) -> JSONResponse:
    if not app_main.admin_token_ok(request):
        return app_main.admin_required_response()
    since = float(request.query_params.get("since") or 0)
    return JSONResponse({"events": app_main.events_since(since)})


@router.get("/mutations")
async def mutations(request: Request) -> JSONResponse:
    """Return ordered mutation records from audit_log since ``since``."""
    if not app_main.admin_token_ok(request):
        return app_main.admin_required_response()
    try:
        since_id = int(request.query_params.get("since") or 0)
    except ValueError:
        since_id = 0
    conn = db.connect()
    rows = conn.execute(
        "SELECT id, occurred_at, operation, target_type, target_id, payload_json "
        "FROM audit_log WHERE id > ? ORDER BY id ASC",
        (since_id,),
    ).fetchall()
    return JSONResponse({
        "mutations": [
            {
                "id": r["id"],
                "ts": r["occurred_at"],
                "op": r["operation"],
                "target_type": r["target_type"],
                "target_id": r["target_id"],
                "payload": json.loads(r["payload_json"] or "{}"),
            }
            for r in rows
        ]
    })
