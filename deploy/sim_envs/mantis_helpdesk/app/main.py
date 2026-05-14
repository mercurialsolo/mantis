"""mantis-helpdesk FastAPI app — entrypoint mounted by Docker + Modal.

Two route surfaces, mirroring mantis-crm (#332):

* ``/``  (everything else) — agent-facing helpdesk UI (server-rendered HTML)
* ``/__env__/*`` — harness-only, gated on ``X-Env-Admin`` header

The admin token is read from ``ENV_ADMIN_TOKEN`` at app construction
and never embedded in any agent-visible response. ``/__env__/health``
is intentionally NOT gated — health-check probes can't authenticate.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import auth, db, seed

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


# ── event log ──────────────────────────────────────────────────────────


_EVENTS: list[dict[str, Any]] = []
_BOOT_TIME = time.time()


def _emit_event(event: str, data: dict[str, Any] | None = None) -> None:
    _EVENTS.append({
        "ts": time.time(),
        "event": event,
        "data": data or {},
    })


# ── app factory ────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(title="mantis-helpdesk", docs_url=None, redoc_url=None)

    templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
    app.state.templates = templates
    app.state.admin_token = _admin_token()

    # See #387. Mirrors mantis_crm's gate.
    @app.middleware("http")
    async def _auth_gate(request: Request, call_next):  # noqa: ANN202
        try:
            request.state.current_user = auth.current_user(request)
        except Exception:  # noqa: BLE001
            request.state.current_user = None

        if auth.auth_required():
            path = request.url.path
            open_prefixes = (
                "/login", "/logout", "/oauth/", "/__env__/", "/static/",
            )
            if path != "/" and not path.startswith(open_prefixes):
                if request.state.current_user is None:
                    return RedirectResponse(
                        f"/login?next={path}", status_code=303,
                    )
        return await call_next(request)

    @app.on_event("startup")
    def _bootstrap() -> None:
        conn = db.connect()
        cur = conn.execute("SELECT COUNT(*) FROM tickets")
        if cur.fetchone()[0] == 0:
            seed.seed(conn, seed_val=_seed_val(), fake_now=_now())
            _emit_event("seeded", {"seed": _seed_val(), "now": _now()})

    app.mount(
        "/static",
        StaticFiles(directory=str(APP_DIR / "static")),
        name="static",
    )

    from .routes import auth as auth_router
    from .routes import env_admin as env_admin_router
    from .routes import macros as macros_router
    from .routes import reports as reports_router
    from .routes import search as search_router
    from .routes import tickets as tickets_router
    from .routes import triggers_ui as triggers_router

    app.include_router(env_admin_router.router)
    app.include_router(auth_router.router)
    app.include_router(tickets_router.router)
    app.include_router(macros_router.router)
    app.include_router(triggers_router.router)
    app.include_router(reports_router.router)
    app.include_router(search_router.router)

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request) -> Response:
        conn = db.connect()
        open_count = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE status IN ('new', 'open', 'pending') "
            "AND deleted_at IS NULL"
        ).fetchone()[0]
        breach_count = conn.execute(
            "SELECT COUNT(*) FROM tickets "
            "WHERE status IN ('new','open','pending') AND deleted_at IS NULL "
            "AND sla_breach_at <= ?",
            (_now(),),
        ).fetchone()[0]
        total = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE deleted_at IS NULL"
        ).fetchone()[0]
        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                "open_count": open_count,
                "breach_count": breach_count,
                "total": total,
                "now": _now(),
            },
        )

    return app


# Convenience: importable ASGI app for uvicorn / Modal.
app = create_app()


# Helpers exposed to the routes package -------------------------------


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
