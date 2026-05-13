"""mantis-shop FastAPI app — entrypoint mounted by both Docker + Modal.

Two route surfaces (mirrors mantis-crm):

* ``/`` — agent-facing storefront + admin SPA (server-rendered HTML).
* ``/__env__/*`` — harness-only, gated on ``X-Env-Admin`` header.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import db, seed

APP_DIR = Path(__file__).parent
ADMIN_TOKEN_ENV = "ENV_ADMIN_TOKEN"
ADMIN_HEADER = "X-Env-Admin"


def _admin_token() -> str:
    token = os.environ.get(ADMIN_TOKEN_ENV, "").strip()
    if not token:
        raise RuntimeError(
            f"{ADMIN_TOKEN_ENV} is required — harness generates it per run and "
            "passes it to the container as an env var."
        )
    return token


def _now() -> str:
    return os.environ.get("FAKE_NOW") or seed.FAKE_NOW_DEFAULT


def _seed_val() -> int:
    return int(os.environ.get("SEED") or 42)


# ── event log ─────────────────────────────────────────────────────────


_EVENTS: list[dict[str, Any]] = []
_BOOT_TIME = time.time()


def _emit_event(event: str, data: dict[str, Any] | None = None) -> None:
    _EVENTS.append({
        "ts": time.time(),
        "event": event,
        "data": data or {},
    })


# ── app factory ───────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(title="mantis-shop", docs_url=None, redoc_url=None)

    templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
    app.state.templates = templates
    app.state.admin_token = _admin_token()

    # Bootstrap the DB. Mirrors mantis-crm: re-seed only when empty so
    # dev iteration under uvicorn --reload preserves state.
    @app.on_event("startup")
    def _bootstrap() -> None:
        conn = db.connect()
        cur = conn.execute("SELECT COUNT(*) FROM products")
        if cur.fetchone()[0] == 0:
            seed.seed(conn, seed_val=_seed_val(), fake_now=_now())
            _emit_event("seeded", {"seed": _seed_val(), "now": _now()})

    app.mount(
        "/static",
        StaticFiles(directory=str(APP_DIR / "static")),
        name="static",
    )

    # Routers — split by UI surface for readability.
    from .routes import admin_coupons as admin_coupons_router
    from .routes import admin_orders as admin_orders_router
    from .routes import admin_products as admin_products_router
    from .routes import cart as cart_router
    from .routes import checkout as checkout_router
    from .routes import env_admin as env_admin_router
    from .routes import storefront as storefront_router

    app.include_router(env_admin_router.router)
    app.include_router(storefront_router.router)
    app.include_router(cart_router.router)
    app.include_router(checkout_router.router)
    app.include_router(admin_orders_router.router)
    app.include_router(admin_products_router.router)
    app.include_router(admin_coupons_router.router)

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request) -> Response:
        conn = db.connect()
        product_count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        order_count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        customer_count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        # Feature a few sale products on the home page.
        featured = [
            dict(r) for r in conn.execute(
                "SELECT id, title, category, base_price, sale_price, image_url "
                "FROM products WHERE sale_price IS NOT NULL "
                "ORDER BY id LIMIT 8"
            ).fetchall()
        ]
        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                "product_count": product_count,
                "order_count": order_count,
                "customer_count": customer_count,
                "featured": featured,
                "now": _now(),
            },
        )

    return app


app = create_app()


# Helpers exposed to the routes package -----------------------------


def admin_token_ok(request: Request) -> bool:
    return request.headers.get(ADMIN_HEADER) == request.app.state.admin_token


def admin_required_response() -> JSONResponse:
    return JSONResponse({"error": "admin token required"}, status_code=401)


def now_value() -> str:
    return _now()


def seed_value() -> int:
    return _seed_val()


def boot_time() -> float:
    return _BOOT_TIME


def emit(event: str, data: dict[str, Any] | None = None) -> None:
    _emit_event(event, data)


def events_since(ts: float) -> list[dict[str, Any]]:
    return [e for e in _EVENTS if e["ts"] >= ts]


def clear_events() -> None:
    _EVENTS.clear()
