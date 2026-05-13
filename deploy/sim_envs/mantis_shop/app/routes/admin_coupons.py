"""Admin coupons surface — list, create, edit, disable."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()


def _log_audit(conn, *, operation: str, target_type: str, target_id: str,
               payload: dict[str, Any] | None = None) -> None:
    db.log_audit(
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


@router.get("/admin/coupons", response_class=HTMLResponse)
async def list_coupons(request: Request) -> HTMLResponse:
    conn = db.connect()
    rows = conn.execute(
        "SELECT * FROM coupons ORDER BY code"
    ).fetchall()
    coupons = [dict(r) for r in rows]
    categories = [
        r["slug"] for r in conn.execute(
            "SELECT slug FROM collections WHERE kind = 'rule' "
            "AND slug NOT IN ('on-sale', 'under-50') ORDER BY slug"
        ).fetchall()
    ]
    error = request.query_params.get("error") or ""
    return app_main.app.state.templates.TemplateResponse(
        "admin_coupons.html",
        {"request": request, "coupons": coupons, "categories": categories,
         "error": error},
    )


@router.post("/admin/coupons/create")
async def create_coupon(
    request: Request,
    code: str = Form(...),
    kind: str = Form(...),
    value: float = Form(...),
    scope_category: str = Form(""),
    stackable: str = Form(""),
    expires_at: str = Form(""),
    max_uses: str = Form(""),
) -> RedirectResponse:
    """Create a new coupon. Code is uppercased server-side."""
    code = code.strip().upper()
    if not code:
        return RedirectResponse(
            "/admin/coupons?error=Code%20required", status_code=303,
        )
    if kind not in {"pct", "amount", "bogo"}:
        return RedirectResponse(
            "/admin/coupons?error=Invalid%20kind", status_code=303,
        )
    with db.transaction() as txn:
        existing = txn.execute(
            "SELECT code FROM coupons WHERE code = ?", (code,),
        ).fetchone()
        if existing:
            return RedirectResponse(
                f"/admin/coupons?error=Coupon%20{code}%20already%20exists",
                status_code=303,
            )
        scope = scope_category.strip() or None
        stackable_flag = 1 if stackable.strip() in {"1", "on", "true", "yes"} else 0
        expiry = expires_at.strip() or None
        try:
            max_uses_val: int | None = int(max_uses.strip()) if max_uses.strip() else None
        except ValueError:
            max_uses_val = None
        txn.execute(
            "INSERT INTO coupons (code, kind, value, scope_category, stackable, "
            "stacking_exclusions, expires_at, max_uses, uses_count, disabled_at) "
            "VALUES (?, ?, ?, ?, ?, '[]', ?, ?, 0, NULL)",
            (code, kind, value, scope, stackable_flag, expiry, max_uses_val),
        )
        _log_audit(txn, operation="coupon_created", target_type="coupon",
                   target_id=code,
                   payload={"kind": kind, "value": value,
                            "scope_category": scope,
                            "stackable": bool(stackable_flag),
                            "expires_at": expiry,
                            "max_uses": max_uses_val})
    return RedirectResponse("/admin/coupons", status_code=303)


@router.post("/admin/coupons/{code}/disable")
async def disable_coupon(code: str) -> RedirectResponse:
    with db.transaction() as txn:
        existing = txn.execute(
            "SELECT code FROM coupons WHERE code = ?", (code,),
        ).fetchone()
        if not existing:
            return RedirectResponse("/admin/coupons", status_code=303)
        txn.execute(
            "UPDATE coupons SET disabled_at = ? WHERE code = ?",
            (app_main.now_value(), code),
        )
        _log_audit(txn, operation="coupon_disabled", target_type="coupon",
                   target_id=code, payload={})
    return RedirectResponse("/admin/coupons", status_code=303)
