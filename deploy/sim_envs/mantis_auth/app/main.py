"""mantis-auth FastAPI app — entrypoint mounted by Docker + Modal.

The env is a thin SaaS console ("Mantis Console") guarded by a rich auth
wall. All login routes come from the embeddable ``authflow`` package
(see ``authflow/router.py``); this module only wires it up, adds the
product pages behind the wall (``/console``, ``/account``), the mock
mailbox (``/inbox``), and the harness surface (``/__env__/*``).

Auth gate: with ``ENV_REQUIRE_AUTH=1`` (the default for this env), every
route except the landing page, the auth routes, the inbox, the harness
routes, and static assets redirects to ``/login`` when unauthenticated.
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

from . import db, seed
from .authflow import AuthConfig, sessions
from .backend import SqliteAuthBackend

APP_DIR = Path(__file__).parent
ADMIN_TOKEN_ENV = "ENV_ADMIN_TOKEN"
ADMIN_HEADER = "X-Env-Admin"

# Routes reachable while signed out. ``/inbox`` is open by design: the
# magic-link + OTP flows require reading the mailbox *before* a session
# exists.
_OPEN_PREFIXES = ("/login", "/logout", "/auth/", "/inbox",
                  "/__env__/", "/static/")

_EVENTS: list[dict[str, Any]] = []
_BOOT_TIME = time.time()


def _admin_token() -> str:
    token = os.environ.get(ADMIN_TOKEN_ENV, "").strip()
    if not token:
        raise RuntimeError(
            f"{ADMIN_TOKEN_ENV} is required — the harness generates it per run "
            "and passes it to the container as an env var."
        )
    return token


def _now() -> str:
    return os.environ.get("FAKE_NOW") or seed.FAKE_NOW_DEFAULT


def _seed_val() -> int:
    return int(os.environ.get("SEED") or 42)


def _default_layout() -> str:
    return os.environ.get("AUTH_LAYOUT") or "centered"


def _require_auth() -> bool:
    return os.environ.get("ENV_REQUIRE_AUTH", "1").lower() in {"1", "true", "yes"}


def _emit_event(event: str, data: dict[str, Any] | None = None) -> None:
    _EVENTS.append({"ts": time.time(), "event": event, "data": data or {}})


def create_app() -> FastAPI:
    app = FastAPI(title="mantis-auth", docs_url=None, redoc_url=None)

    templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
    app.state.templates = templates
    app.state.admin_token = _admin_token()

    backend = SqliteAuthBackend(now_fn=_now, emit_fn=_emit_event)
    app.state.auth_backend = backend
    app.state.auth_config = AuthConfig(
        templates=templates,
        backend=backend,
        app_name="Mantis Console",
        post_login_redirect="/console",
        default_layout=_default_layout(),
    )

    @app.middleware("http")
    async def _auth_gate(request: Request, call_next):  # noqa: ANN202
        try:
            uid = sessions.session_user_id(request)
            request.state.current_user = backend.lookup_user_by_id(uid) if uid else None
        except Exception:  # noqa: BLE001 — never block a request on auth lookup
            request.state.current_user = None

        if _require_auth():
            path = request.url.path
            if path != "/" and not path.startswith(_OPEN_PREFIXES):
                if request.state.current_user is None:
                    return RedirectResponse(f"/login?next={path}", status_code=303)
        return await call_next(request)

    @app.on_event("startup")
    def _bootstrap() -> None:
        conn = db.connect()
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            seed.seed(conn, seed_val=_seed_val(), fake_now=_now())
            _emit_event("seeded", {"seed": _seed_val(), "now": _now()})

    app.mount("/static",
              StaticFiles(directory=str(APP_DIR / "static")), name="static")

    from .authflow import build_auth_router
    from .routes import console as console_router
    from .routes import env_admin as env_admin_router
    from .routes import inbox as inbox_router

    app.include_router(env_admin_router.router)
    app.include_router(build_auth_router(app.state.auth_config))
    app.include_router(inbox_router.router)
    app.include_router(console_router.router)

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request) -> Response:
        return templates.TemplateResponse(
            "landing.html",
            {"request": request, "config": app.state.auth_config,
             "current_user": getattr(request.state, "current_user", None)},
        )

    return app


app = create_app()


# Helpers exposed to the routes package ---------------------------------


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
