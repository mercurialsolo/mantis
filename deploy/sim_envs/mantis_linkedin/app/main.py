"""mantis-linkedin FastAPI app entrypoint.

Mirrors the mantis-shop / mantis-boattrader shape:

* ``/`` — agent-facing pages (server-rendered HTML).
* ``/__env__/*`` — harness-only, gated on ``X-Env-Admin`` header.
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
        # Fall back to a constant in dev so smoke tests work without
        # explicit env var. Harness always sets one in prod.
        return "dev-admin-token"
    return token


def _now() -> str:
    return os.environ.get("FAKE_NOW") or seed.FAKE_NOW_DEFAULT


def _seed_val() -> int:
    return int(os.environ.get("SEED") or 42)


# ── event log + mutations ─────────────────────────────────────────────


_EVENTS: list[dict[str, Any]] = []
_MUTATIONS: list[dict[str, Any]] = []
_BOOT_TIME = time.time()


def _emit_event(event: str, data: dict[str, Any] | None = None) -> None:
    _EVENTS.append({"ts": time.time(), "event": event, "data": data or {}})


def _emit_mutation(
    *, op: str, target_type: str, target_id: str,
    payload: dict[str, Any] | None = None,
) -> None:
    _MUTATIONS.append({
        "ts": time.time(),
        "op": op,
        "target_type": target_type,
        "target_id": target_id,
        "payload": payload or {},
    })


# ── app factory ──────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(title="mantis-linkedin", docs_url=None, redoc_url=None)

    templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
    app.state.templates = templates
    app.state.admin_token = _admin_token()

    @app.middleware("http")
    async def _auth_gate(request: Request, call_next):  # noqa: ANN202
        try:
            request.state.current_user = auth.current_user(request)
        except Exception:  # noqa: BLE001
            request.state.current_user = None

        if auth.auth_required():
            path = request.url.path
            open_prefixes = ("/login", "/logout", "/signup", "/static/",
                             "/__env__/")
            if path == "/" or path.startswith(open_prefixes):
                pass
            else:
                user = request.state.current_user
                if user is None:
                    return RedirectResponse(
                        f"/login?next={path}", status_code=303,
                    )
        return await call_next(request)

    @app.on_event("startup")
    def _bootstrap() -> None:
        conn = db.connect()
        cur = conn.execute("SELECT COUNT(*) FROM users")
        if cur.fetchone()[0] == 0:
            seed.seed(conn, seed_val=_seed_val(), fake_now=_now())
            _emit_event("seeded", {"seed": _seed_val(), "now": _now()})

    app.mount(
        "/static",
        StaticFiles(directory=str(APP_DIR / "static")),
        name="static",
    )

    # Routers — split by surface.
    from .routes import env_admin as env_admin_router
    from .routes import auth as auth_router
    from .routes import feed as feed_router
    from .routes import profile as profile_router
    from .routes import network as network_router
    from .routes import messaging as messaging_router
    from .routes import jobs as jobs_router

    app.include_router(env_admin_router.router)
    app.include_router(auth_router.router)
    app.include_router(feed_router.router)
    app.include_router(profile_router.router)
    app.include_router(network_router.router)
    app.include_router(messaging_router.router)
    app.include_router(jobs_router.router)

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request) -> Response:
        # The live linkedin.com redirects authed users to /feed/. For
        # anonymous visitors it serves a marketing splash; we render a
        # minimal "home" template that points to login / feed so the
        # CUA can pick either path.
        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                "now": _now(),
            },
        )

    return app


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


def emit_mutation(*, op: str, target_type: str, target_id: str,
                  payload: dict[str, Any] | None = None) -> None:
    _emit_mutation(
        op=op, target_type=target_type, target_id=target_id, payload=payload,
    )


def events_since(ts: float) -> list[dict[str, Any]]:
    return [e for e in _EVENTS if e["ts"] >= ts]


def mutations_since(ts: float) -> list[dict[str, Any]]:
    return [m for m in _MUTATIONS if m["ts"] >= ts]


def clear_events() -> None:
    _EVENTS.clear()
    _MUTATIONS.clear()
