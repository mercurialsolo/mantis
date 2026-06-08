"""mantis-fiverr FastAPI app — entrypoint mounted by both Docker + Modal.

Two route surfaces (mirrors mantis-shop / mantis-boattrader):

* ``/`` — buyer-facing marketplace pages (home, search, gig detail,
  checkout, inbox, orders, auth).
* ``/__env__/*`` — harness-only, gated on ``X-Env-Admin`` header.
"""

from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import auth, db, seed

APP_DIR = Path(__file__).parent
ADMIN_TOKEN_ENV = "ENV_ADMIN_TOKEN"
ADMIN_HEADER = "X-Env-Admin"


def _admin_token() -> str:
    token = os.environ.get(ADMIN_TOKEN_ENV, "").strip()
    if not token:
        # Generate one if missing — local/dev ergonomics.
        token = secrets.token_urlsafe(32)
        os.environ[ADMIN_TOKEN_ENV] = token
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


def _ensure_seeded() -> None:
    """Seed the DB if empty. Safe to call from startup hook or directly."""
    conn = db.connect()
    cur = conn.execute("SELECT COUNT(*) FROM gigs")
    if cur.fetchone()[0] == 0:
        seed.seed(conn, seed_val=_seed_val(), fake_now=_now())
        _emit_event("seeded", {"seed": _seed_val(), "now": _now()})


# ── app factory ───────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(title="mantis-fiverr", docs_url=None, redoc_url=None)

    templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
    # Globals exposed to every template.
    templates.env.globals["site_name"] = "Fiverr"
    templates.env.globals["env_admin_required"] = False
    app.state.templates = templates
    app.state.admin_token = _admin_token()

    # Auth gate + per-request user resolution. Mirrors mantis_shop.
    @app.middleware("http")
    async def _auth_gate(request: Request, call_next):  # noqa: ANN202
        try:
            request.state.current_user = auth.current_user(request)
        except Exception:  # noqa: BLE001
            request.state.current_user = None
        if auth.auth_required():
            path = request.url.path
            open_prefixes = (
                "/login", "/signup", "/logout", "/__env__/", "/static/",
                "/assets/", "/categories/", "/search/",
            )
            if path == "/" or path.startswith(open_prefixes):
                pass
            elif path.startswith(("/inbox", "/orders", "/checkout")):
                user = request.state.current_user
                if user is None:
                    return RedirectResponse(
                        f"/login?next={path}", status_code=303,
                    )
        return await call_next(request)

    @app.on_event("startup")
    def _bootstrap() -> None:
        _ensure_seeded()

    # Eager-seed at app construction too. TestClient + Modal both rely
    # on lifespan; with synchronous tests / inspections that bypass
    # the lifespan (``from app.main import app`` direct), we still
    # need a populated DB or the first request 404s.
    _ensure_seeded()

    # Static
    static_dir = APP_DIR / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Routers
    from .routes import auth_routes as auth_router
    from .routes import env_admin as env_admin_router
    from .routes import storefront as storefront_router
    from .routes import gig as gig_router
    from .routes import checkout as checkout_router
    from .routes import inbox as inbox_router
    from .routes import orders as orders_router
    from .routes import assets as assets_router

    app.include_router(env_admin_router.router)
    app.include_router(auth_router.router)
    app.include_router(assets_router.router)
    app.include_router(storefront_router.router)
    app.include_router(checkout_router.router)
    app.include_router(inbox_router.router)
    app.include_router(orders_router.router)
    # gig detail uses the catch-all ``/{username}/{slug}`` pattern; must
    # be the LAST router so /checkout, /inbox, /orders etc. all match
    # their specific routes first.
    app.include_router(gig_router.router)

    return app


app = create_app()


# Helpers exposed to other modules ----------------------------------


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
