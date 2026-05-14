"""mantis-crm FastAPI app — entrypoint mounted by both Docker + Modal.

Two route surfaces:

* ``/``  (everything else) — agent-facing CRM SPA (server-rendered HTML)
* ``/__env__/*`` — harness-only, gated on ``X-Env-Admin`` header

The admin middleware lives here so every harness route is gated
uniformly. ``health`` is intentionally NOT gated — health-check probes
can't authenticate.

The agent never sees the admin token in its DOM. The same image
rendering for the agent omits it; the harness reads it from
``ENV_ADMIN_TOKEN`` at app construction.
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
    """Frozen wall-clock for the env. Real envs would also surrogate the
    in-app clock; we just return the configured value."""
    return os.environ.get("FAKE_NOW") or seed.FAKE_NOW_DEFAULT


def _seed_val() -> int:
    return int(os.environ.get("SEED") or 42)


# ── event log ──────────────────────────────────────────────────────────


# Held in memory; the harness pulls it via ``/__env__/events``. Cleared
# on ``/__env__/reset``. Real production envs would write to a JSONL file
# under /var/log and tail-read; in-memory is enough at v1 volumes.
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
    app = FastAPI(title="mantis-crm", docs_url=None, redoc_url=None)

    templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
    app.state.templates = templates
    app.state.admin_token = _admin_token()

    # Resolve the session user once per request + gate everything except
    # ``/``, ``/login``, ``/logout``, ``/oauth/*``, ``/__env__/*``,
    # ``/static/*`` when ``ENV_REQUIRE_AUTH=1``. See #387.
    @app.middleware("http")
    async def _auth_gate(request: Request, call_next):  # noqa: ANN202
        try:
            request.state.current_user = auth.current_user(request)
        except Exception:  # noqa: BLE001 — never block request on auth lookup
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

    # Seed the DB at startup. If you mount a persistent volume, seed
    # only runs on first boot; otherwise it runs every container start.
    @app.on_event("startup")
    def _bootstrap() -> None:
        conn = db.connect()
        # Re-seed only when empty — preserves manual dev state across
        # ``uvicorn --reload``.
        cur = conn.execute("SELECT COUNT(*) FROM contacts")
        if cur.fetchone()[0] == 0:
            seed.seed(conn, seed_val=_seed_val(), fake_now=_now())
            _emit_event("seeded", {"seed": _seed_val(), "now": _now()})

    # Static + templates
    app.mount(
        "/static",
        StaticFiles(directory=str(APP_DIR / "static")),
        name="static",
    )

    # Router registration is split by surface for readability.
    from .routes import auth as auth_router
    from .routes import companies as companies_router
    from .routes import contacts as contacts_router
    from .routes import deals as deals_router
    from .routes import env_admin as env_admin_router
    from .routes import insights as insights_router
    from .routes import notes as notes_router
    from .routes import reports as reports_router
    from .routes import search as search_router
    from .routes import tasks as tasks_router
    from .routes import templates as templates_router

    app.include_router(env_admin_router.router)
    app.include_router(auth_router.router)
    app.include_router(contacts_router.router)
    app.include_router(companies_router.router)
    app.include_router(deals_router.router)
    app.include_router(tasks_router.router)
    app.include_router(notes_router.router)
    app.include_router(templates_router.router)
    app.include_router(insights_router.router)
    app.include_router(search_router.router)
    app.include_router(reports_router.router)

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request) -> Response:
        # Landing page → links to the four main surfaces.
        conn = db.connect()
        contact_count = conn.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
        deal_count = conn.execute("SELECT COUNT(*) FROM deals").fetchone()[0]
        company_count = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                "contact_count": contact_count,
                "deal_count": deal_count,
                "company_count": company_count,
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
