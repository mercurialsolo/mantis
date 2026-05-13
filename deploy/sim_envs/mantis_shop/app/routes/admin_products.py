"""Admin product editor — variant matrix + per-variant inventory adjustment."""

from __future__ import annotations

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


@router.get("/admin/products", response_class=HTMLResponse)
async def list_products(request: Request) -> HTMLResponse:
    q = (request.query_params.get("q") or "").strip()
    category = (request.query_params.get("category") or "").strip()
    page = max(1, int(request.query_params.get("page") or 1))

    conn = db.connect()
    where = ["1=1"]
    args: list[Any] = []
    if q:
        where.append("(title LIKE ? OR sku LIKE ?)")
        args += [f"%{q}%", f"%{q}%"]
    if category:
        where.append("category = ?")
        args.append(category)

    where_sql = " AND ".join(where)
    total = conn.execute(
        f"SELECT COUNT(*) FROM products WHERE {where_sql}", args,
    ).fetchone()[0]
    offset = (page - 1) * PAGE_SIZE
    rows = conn.execute(
        f"SELECT * FROM products WHERE {where_sql} "
        f"ORDER BY id LIMIT ? OFFSET ?",
        args + [PAGE_SIZE, offset],
    ).fetchall()
    products = [dict(r) for r in rows]
    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

    return app_main.app.state.templates.TemplateResponse(
        "admin_products.html",
        {
            "request": request,
            "products": products,
            "page": page,
            "pages": pages,
            "total": total,
            "filters": {"q": q, "category": category},
        },
    )


# Declared BEFORE /admin/products/{product_id}... to avoid shadowing
# (Starlette declaration-order rule, #332 gotcha).


@router.post("/admin/products/inventory/adjust_by_sku")
async def adjust_inventory_by_sku(
    sku: str = Form(...),
    delta: int = Form(...),
    reason: str = Form(""),
) -> RedirectResponse:
    """Adjust a variant's inventory by SKU. The T05 entry point.

    The form takes the SKU (which is more discoverable than the
    internal variant_id), looks the variant up, applies ``delta``
    (positive or negative), and records the reason in the audit log.
    """
    sku = sku.strip()
    if not sku:
        return RedirectResponse(
            "/admin/products?error=SKU%20required", status_code=303,
        )
    with db.transaction() as txn:
        variant = txn.execute(
            "SELECT id, product_id FROM variants WHERE sku = ?", (sku,),
        ).fetchone()
        if variant is None:
            return RedirectResponse(
                f"/admin/products?error=Unknown%20SKU%20{sku}", status_code=303,
            )
        cur = txn.execute(
            "SELECT quantity FROM inventory WHERE variant_id = ?",
            (variant["id"],),
        ).fetchone()
        existing = cur["quantity"] if cur else 0
        new_qty = max(0, existing + delta)
        if cur:
            txn.execute(
                "UPDATE inventory SET quantity = ? WHERE variant_id = ?",
                (new_qty, variant["id"]),
            )
        else:
            txn.execute(
                "INSERT INTO inventory (variant_id, quantity) VALUES (?, ?)",
                (variant["id"], new_qty),
            )
        _log_audit(txn, operation="inventory_adjusted", target_type="variant",
                   target_id=variant["id"],
                   payload={"sku": sku, "delta": delta,
                            "old_quantity": existing,
                            "new_quantity": new_qty,
                            "reason": reason.strip()})
    return RedirectResponse(
        f"/admin/products/{variant['product_id']}", status_code=303,
    )


@router.get("/admin/products/{product_id}", response_class=HTMLResponse)
async def product_detail(request: Request, product_id: str) -> HTMLResponse:
    conn = db.connect()
    row = conn.execute(
        "SELECT * FROM products WHERE id = ?", (product_id,),
    ).fetchone()
    if row is None:
        return HTMLResponse("Not found", status_code=404)
    variants = [
        dict(r) for r in conn.execute(
            "SELECT v.*, COALESCE(i.quantity, 0) AS stock "
            "FROM variants v LEFT JOIN inventory i ON v.id = i.variant_id "
            "WHERE v.product_id = ? ORDER BY v.size, v.color",
            (product_id,),
        ).fetchall()
    ]
    return app_main.app.state.templates.TemplateResponse(
        "admin_product_detail.html",
        {"request": request, "product": dict(row), "variants": variants},
    )


@router.post("/admin/products/{product_id}/edit")
async def edit_product(
    product_id: str,
    title: str = Form(""),
    description_md: str = Form(""),
    base_price: str = Form(""),
    sale_price: str = Form(""),
) -> RedirectResponse:
    with db.transaction() as txn:
        existing = txn.execute(
            "SELECT * FROM products WHERE id = ?", (product_id,),
        ).fetchone()
        if existing is None:
            return RedirectResponse("/admin/products", status_code=303)
        changes: dict[str, Any] = {}
        if title.strip() and title.strip() != existing["title"]:
            changes["title"] = title.strip()
        if description_md.strip() and description_md.strip() != existing["description_md"]:
            changes["description_md"] = description_md.strip()
        if base_price.strip():
            try:
                bp = float(base_price.strip())
                if bp != existing["base_price"]:
                    changes["base_price"] = bp
            except ValueError:
                pass
        if sale_price.strip():
            try:
                sp = float(sale_price.strip())
                if sp != existing["sale_price"]:
                    changes["sale_price"] = sp
            except ValueError:
                pass
        for col, val in changes.items():
            txn.execute(
                f"UPDATE products SET {col} = ? WHERE id = ?",
                (val, product_id),
            )
        if changes:
            _log_audit(txn, operation="product_edited",
                       target_type="product", target_id=product_id,
                       payload={"changes": changes})
    return RedirectResponse(f"/admin/products/{product_id}", status_code=303)


@router.post("/admin/products/{product_id}/inventory/{variant_id}")
async def adjust_inventory(
    product_id: str, variant_id: str,
    delta: int = Form(...),
    reason: str = Form(""),
) -> RedirectResponse:
    """Adjust a single variant's inventory by id (the per-row in-page form)."""
    with db.transaction() as txn:
        cur = txn.execute(
            "SELECT quantity FROM inventory WHERE variant_id = ?",
            (variant_id,),
        ).fetchone()
        existing = cur["quantity"] if cur else 0
        new_qty = max(0, existing + delta)
        if cur:
            txn.execute(
                "UPDATE inventory SET quantity = ? WHERE variant_id = ?",
                (new_qty, variant_id),
            )
        else:
            txn.execute(
                "INSERT INTO inventory (variant_id, quantity) VALUES (?, ?)",
                (variant_id, new_qty),
            )
        # Look up the sku for the audit payload.
        v = txn.execute(
            "SELECT sku FROM variants WHERE id = ?", (variant_id,),
        ).fetchone()
        sku = v["sku"] if v else ""
        _log_audit(txn, operation="inventory_adjusted",
                   target_type="variant", target_id=variant_id,
                   payload={"sku": sku, "delta": delta,
                            "old_quantity": existing,
                            "new_quantity": new_qty,
                            "reason": reason.strip()})
    return RedirectResponse(f"/admin/products/{product_id}", status_code=303)
