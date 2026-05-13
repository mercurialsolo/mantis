"""Harness endpoints — ``/__env__/{health,reset,seed,clock,oracle,state,events}``.

Every route except ``health`` is gated on the ``X-Env-Admin`` header.
The check lives in the route bodies (rather than middleware) so a typo
in one route can't accidentally bypass auth on another.
"""

from __future__ import annotations


from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .. import db, main as app_main, seed

router = APIRouter(prefix="/__env__")


@router.get("/health")
async def health() -> JSONResponse:
    """Open by design — health checks can't authenticate."""
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
    payload = {}
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 — empty body is fine
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
    # We don't actually re-seed; we just acknowledge the clock change.
    # The seed snapshot is the canonical "now" — re-seed via ``/__env__/seed``
    # if the test wants both.
    import os
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
    """Debug-only dump of high-level table counts + recent mutations."""
    if not app_main.admin_token_ok(request):
        return app_main.admin_required_response()
    conn = db.connect()

    def count(table: str) -> int:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    mutations = [
        {
            "id": r["id"],
            "occurred_at": r["occurred_at"],
            "operation": r["operation"],
            "target_type": r["target_type"],
            "target_id": r["target_id"],
        }
        for r in conn.execute(
            "SELECT * FROM mutations ORDER BY id DESC LIMIT 50"
        ).fetchall()
    ]

    return JSONResponse({
        "seed": app_main.seed_value(),
        "now": app_main.now_value(),
        "counts": {
            "users": count("users"),
            "companies": count("companies"),
            "contacts": count("contacts"),
            "deals": count("deals"),
            "activities": count("activities"),
            "lists": count("lists"),
            "list_members": count("list_members"),
            "mutations": count("mutations"),
        },
        "recent_mutations": mutations,
    })


@router.get("/events")
async def events(request: Request) -> JSONResponse:
    if not app_main.admin_token_ok(request):
        return app_main.admin_required_response()
    since = float(request.query_params.get("since") or 0)
    return JSONResponse({"events": app_main.events_since(since)})
