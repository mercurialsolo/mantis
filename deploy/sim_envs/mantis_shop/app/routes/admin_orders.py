"""Admin orders surface — list, detail, refund/fulfill/cancel, saved views.

Refund, fulfill, and cancel all go through the order detail page.
``/saved_views/<id>/add`` lets the admin slot a target order into a
saved view (T04 export path).
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()

PAGE_SIZE = 25


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


@router.get("/admin/orders", response_class=HTMLResponse)
async def list_orders(request: Request) -> HTMLResponse:
    """Filter by status, coupon code, since-days, or customer_id."""
    status = (request.query_params.get("status") or "").strip()
    coupon = (request.query_params.get("coupon") or "").strip().upper()
    customer = (request.query_params.get("customer") or "").strip()
    since_days = (request.query_params.get("since_days") or "").strip()
    page = max(1, int(request.query_params.get("page") or 1))

    conn = db.connect()
    where = ["1=1"]
    args: list[Any] = []
    if status:
        where.append("status = ?")
        args.append(status)
    if coupon:
        where.append("coupon_codes LIKE ?")
        args.append(f'%"{coupon}"%')
    if customer:
        where.append("customer_id = ?")
        args.append(customer)
    if since_days:
        try:
            from datetime import timedelta
            from ..seed import _parse_iso
            cutoff = (_parse_iso(app_main.now_value())
                      - timedelta(days=int(since_days))).isoformat()
            where.append("placed_at >= ?")
            args.append(cutoff)
        except (ValueError, ImportError):
            pass

    where_sql = " AND ".join(where)
    total = conn.execute(
        f"SELECT COUNT(*) FROM orders WHERE {where_sql}", args,
    ).fetchone()[0]
    offset = (page - 1) * PAGE_SIZE
    rows = conn.execute(
        f"SELECT * FROM orders WHERE {where_sql} "
        f"ORDER BY placed_at DESC LIMIT ? OFFSET ?",
        args + [PAGE_SIZE, offset],
    ).fetchall()
    orders = [dict(r) for r in rows]
    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

    saved_views = [
        dict(r) for r in conn.execute(
            "SELECT * FROM saved_views ORDER BY id"
        ).fetchall()
    ]

    return app_main.app.state.templates.TemplateResponse(
        "admin_orders.html",
        {
            "request": request,
            "orders": orders,
            "page": page,
            "pages": pages,
            "total": total,
            "filters": {"status": status, "coupon": coupon,
                        "customer": customer, "since_days": since_days},
            "saved_views": saved_views,
        },
    )


# Declared BEFORE /admin/orders/{order_id}/... so /admin/orders/bulk
# isn't shadowed (Starlette declaration-order rule, #332 gotcha).


@router.post("/admin/orders/bulk/add_to_view")
async def bulk_add_to_view(
    request: Request,
    view_id: str = Form(...),
    ids: str = Form(""),
) -> RedirectResponse:
    """Slot many orders into a saved view in one call."""
    target_ids = [s.strip() for s in ids.split(",") if s.strip()]
    if not target_ids:
        return RedirectResponse("/admin/orders", status_code=303)
    with db.transaction() as txn:
        # Confirm the view exists.
        view = txn.execute(
            "SELECT id FROM saved_views WHERE id = ?", (view_id,),
        ).fetchone()
        if view is None:
            return RedirectResponse("/admin/orders", status_code=303)
        for oid in target_ids:
            exists = txn.execute(
                "SELECT 1 FROM orders WHERE id = ?", (oid,),
            ).fetchone()
            if not exists:
                continue
            txn.execute(
                "INSERT OR IGNORE INTO saved_view_members (view_id, order_id) "
                "VALUES (?, ?)",
                (view_id, oid),
            )
            _log_audit(txn, operation="saved_view_member_added",
                       target_type="saved_view", target_id=view_id,
                       payload={"order_id": oid, "via": "bulk"})
    return RedirectResponse(f"/admin/saved_views/{view_id}", status_code=303)


@router.get("/admin/orders/{order_id}", response_class=HTMLResponse)
async def order_detail(request: Request, order_id: str) -> HTMLResponse:
    conn = db.connect()
    row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    if row is None:
        return HTMLResponse("Not found", status_code=404)
    order = dict(row)
    items = [
        dict(r) for r in conn.execute(
            "SELECT * FROM order_items WHERE order_id = ? ORDER BY line_no",
            (order_id,),
        ).fetchall()
    ]
    refunds = [
        dict(r) for r in conn.execute(
            "SELECT * FROM order_refunds WHERE order_id = ? ORDER BY id",
            (order_id,),
        ).fetchall()
    ]
    audit_rows = [
        dict(r) for r in conn.execute(
            "SELECT * FROM audit_log WHERE target_type = 'order' AND target_id = ? "
            "ORDER BY id DESC LIMIT 50",
            (order_id,),
        ).fetchall()
    ]
    addr = conn.execute(
        "SELECT * FROM customer_addresses WHERE id = ?",
        (order["shipping_address_id"],),
    ).fetchone()

    tab = (request.query_params.get("tab") or "items").strip().lower()
    if tab not in {"items", "refunds", "audit"}:
        tab = "items"

    return app_main.app.state.templates.TemplateResponse(
        "admin_order_detail.html",
        {
            "request": request,
            "order": order,
            "items": items,
            "refunds": refunds,
            "audit_rows": audit_rows,
            "address": dict(addr) if addr else {},
            "tab": tab,
        },
    )


@router.post("/admin/orders/{order_id}/refund")
async def refund_order(
    order_id: str,
    line_no: str = Form(""),
    amount: str = Form(""),
    reason: str = Form(""),
    notify_customer: str = Form(""),
) -> RedirectResponse:
    """Refund a line item (partial) or the entire order (line_no empty).

    The amount can be specified; if blank we default to the line item's
    full value (or order's outstanding balance for full-order refunds).
    """
    with db.transaction() as txn:
        order_row = txn.execute(
            "SELECT * FROM orders WHERE id = ?", (order_id,),
        ).fetchone()
        if order_row is None:
            return RedirectResponse("/admin/orders", status_code=303)

        target_line: int | None = None
        if line_no.strip():
            try:
                target_line = int(line_no.strip())
            except ValueError:
                return RedirectResponse(
                    f"/admin/orders/{order_id}?tab=refunds", status_code=303,
                )

        if target_line is not None:
            line = txn.execute(
                "SELECT * FROM order_items WHERE order_id = ? AND line_no = ?",
                (order_id, target_line),
            ).fetchone()
            if line is None:
                return RedirectResponse(
                    f"/admin/orders/{order_id}?tab=refunds", status_code=303,
                )
            default_amt = float(line["unit_price"]) * int(line["quantity"])
        else:
            default_amt = float(order_row["total"])

        try:
            amt = float(amount.strip()) if amount.strip() else default_amt
        except ValueError:
            amt = default_amt

        notify_flag = 1 if notify_customer in {"1", "on", "true", "yes"} else 0
        refund_id = f"refund_{int(time.time() * 1000)}"
        txn.execute(
            "INSERT INTO order_refunds "
            "(id, order_id, line_no, amount, reason, notify_customer, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (refund_id, order_id, target_line, round(amt, 2), reason.strip(),
             notify_flag, app_main.now_value()),
        )

        # Update notify_customer + status reflection on the order.
        txn.execute(
            "UPDATE orders SET notify_customer = ? WHERE id = ?",
            (notify_flag, order_id),
        )
        # If a full-order refund (no line_no) — mark status refunded.
        if target_line is None:
            txn.execute(
                "UPDATE orders SET status = 'refunded' WHERE id = ?",
                (order_id,),
            )
        _log_audit(txn, operation="order_refunded", target_type="order",
                   target_id=order_id,
                   payload={"refund_id": refund_id, "line_no": target_line,
                            "amount": round(amt, 2), "reason": reason.strip(),
                            "notify_customer": bool(notify_flag)})

    return RedirectResponse(
        f"/admin/orders/{order_id}?tab=refunds", status_code=303,
    )


@router.post("/admin/orders/{order_id}/fulfill")
async def fulfill_order(order_id: str) -> RedirectResponse:
    with db.transaction() as txn:
        row = txn.execute("SELECT status FROM orders WHERE id = ?",
                          (order_id,)).fetchone()
        if row is None or row["status"] in {"refunded", "cancelled"}:
            return RedirectResponse(f"/admin/orders/{order_id}", status_code=303)
        txn.execute(
            "UPDATE orders SET status = 'fulfilled', fulfilled_at = ? WHERE id = ?",
            (app_main.now_value(), order_id),
        )
        _log_audit(txn, operation="order_fulfilled", target_type="order",
                   target_id=order_id, payload={})
    return RedirectResponse(f"/admin/orders/{order_id}", status_code=303)


@router.post("/admin/orders/{order_id}/cancel")
async def cancel_order(order_id: str) -> RedirectResponse:
    with db.transaction() as txn:
        row = txn.execute("SELECT status FROM orders WHERE id = ?",
                          (order_id,)).fetchone()
        if row is None or row["status"] == "cancelled":
            return RedirectResponse(f"/admin/orders/{order_id}", status_code=303)
        txn.execute(
            "UPDATE orders SET status = 'cancelled', cancelled_at = ? WHERE id = ?",
            (app_main.now_value(), order_id),
        )
        _log_audit(txn, operation="order_cancelled", target_type="order",
                   target_id=order_id, payload={})
    return RedirectResponse(f"/admin/orders/{order_id}", status_code=303)


# ── saved views ──────────────────────────────────────────────────────


@router.get("/admin/saved_views/{view_id}", response_class=HTMLResponse)
async def saved_view_detail(request: Request, view_id: str) -> HTMLResponse:
    conn = db.connect()
    view_row = conn.execute(
        "SELECT * FROM saved_views WHERE id = ?", (view_id,),
    ).fetchone()
    if view_row is None:
        return HTMLResponse("Not found", status_code=404)
    members = [
        dict(r) for r in conn.execute(
            "SELECT o.* FROM saved_view_members svm "
            "JOIN orders o ON svm.order_id = o.id "
            "WHERE svm.view_id = ? ORDER BY o.placed_at DESC",
            (view_id,),
        ).fetchall()
    ]
    return app_main.app.state.templates.TemplateResponse(
        "admin_saved_view.html",
        {"request": request, "view": dict(view_row), "members": members},
    )


@router.post("/admin/saved_views/{view_id}/add")
async def add_to_view(view_id: str, order_id: str = Form(...)) -> RedirectResponse:
    with db.transaction() as txn:
        view = txn.execute(
            "SELECT id FROM saved_views WHERE id = ?", (view_id,),
        ).fetchone()
        if view is None:
            return RedirectResponse("/admin/orders", status_code=303)
        order = txn.execute(
            "SELECT id FROM orders WHERE id = ?", (order_id,),
        ).fetchone()
        if order is None:
            return RedirectResponse(f"/admin/saved_views/{view_id}", status_code=303)
        txn.execute(
            "INSERT OR IGNORE INTO saved_view_members (view_id, order_id) "
            "VALUES (?, ?)",
            (view_id, order_id),
        )
        _log_audit(txn, operation="saved_view_member_added",
                   target_type="saved_view", target_id=view_id,
                   payload={"order_id": order_id})
    return RedirectResponse(f"/admin/saved_views/{view_id}", status_code=303)
