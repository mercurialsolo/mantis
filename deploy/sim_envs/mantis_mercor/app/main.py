"""mantis-mercor FastAPI app — entrypoint for Docker + Modal.

Two route surfaces:

* ``/`` — the marketing + candidate + client surface (server-rendered).
* ``/__env__/*`` — harness-only, gated on ``X-Env-Admin`` header.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import auth, db, seed

APP_DIR = Path(__file__).parent
ADMIN_TOKEN_ENV = "ENV_ADMIN_TOKEN"
ADMIN_HEADER = "X-Env-Admin"


def _admin_token() -> str:
    token = os.environ.get(ADMIN_TOKEN_ENV, "").strip()
    if not token:
        # Fail loudly — the harness ALWAYS sets this. Locally devs may
        # set it themselves; default to a banner if missing.
        raise RuntimeError(
            f"{ADMIN_TOKEN_ENV} is required — harness generates it per run."
        )
    return token


def _now() -> str:
    return os.environ.get("FAKE_NOW") or seed.FAKE_NOW_DEFAULT


def _seed_val() -> int:
    return int(os.environ.get("SEED") or 42)


_EVENTS: list[dict[str, Any]] = []
_BOOT_TIME = time.time()


def _emit_event(event: str, data: dict[str, Any] | None = None) -> None:
    _EVENTS.append({
        "ts": time.time(),
        "event": event,
        "data": data or {},
    })


def _initials_for(name: str) -> str:
    parts = [p for p in (name or "").split() if p]
    if len(parts) >= 3:
        return (parts[0][0] + parts[1][0] + parts[2][0]).upper()
    if len(parts) == 2:
        return (parts[0][0] + parts[1][0] + parts[1][-1]).upper()
    base = (parts[0] if parts else "X").upper()
    return (base + "XX")[:3]


def create_app() -> FastAPI:
    app = FastAPI(title="mantis-mercor", docs_url=None, redoc_url=None)

    templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
    # Make initials helper available in templates.
    templates.env.filters["initials"] = _initials_for
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
            open_prefixes = (
                "/login", "/signup", "/logout", "/__env__/",
                "/static/", "/jobs", "/experts",
            )
            if path == "/" or path.startswith(open_prefixes):
                pass
            elif path.startswith(("/apply/", "/dashboard", "/profile")):
                user = request.state.current_user
                if user is None:
                    return RedirectResponse(
                        f"/login?next={path}", status_code=303,
                    )
        return await call_next(request)

    @app.on_event("startup")
    def _bootstrap() -> None:
        conn = db.connect()
        cur = conn.execute("SELECT COUNT(*) FROM jobs")
        if cur.fetchone()[0] == 0:
            seed.seed(conn, seed_val=_seed_val(), fake_now=_now())
            _emit_event("seeded", {"seed": _seed_val(), "now": _now()})

    app.mount(
        "/static",
        StaticFiles(directory=str(APP_DIR / "static")),
        name="static",
    )

    from .routes import (
        apply as apply_router,
        auth as auth_router,
        dashboard as dashboard_router,
        env_admin as env_admin_router,
        jobs as jobs_router,
        marketing as marketing_router,
        profile as profile_router,
    )

    app.include_router(env_admin_router.router)
    app.include_router(auth_router.router)
    app.include_router(marketing_router.router)
    app.include_router(jobs_router.router)
    app.include_router(apply_router.router)
    app.include_router(dashboard_router.router)
    app.include_router(profile_router.router)

    return app


app = create_app()


# Helpers exposed to the routes package ------------------------------


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


# Apply-draft store — in-memory, per (user_id, job_id) tuple.
# Persists across step submits within one process; cleared on reset.
APPLY_DRAFTS: dict[tuple[str, str], dict[str, Any]] = {}


def get_draft(user_id: str, job_id: str) -> dict[str, Any]:
    return APPLY_DRAFTS.setdefault(
        (user_id, job_id),
        {"headline": "", "skills": "", "hourly_rate": "",
         "resume_text": "", "answers": []},
    )


def set_draft(user_id: str, job_id: str, **updates: Any) -> dict[str, Any]:
    cur = get_draft(user_id, job_id)
    cur.update(updates)
    return cur


def clear_draft(user_id: str, job_id: str) -> None:
    APPLY_DRAFTS.pop((user_id, job_id), None)


def clear_all_drafts() -> None:
    APPLY_DRAFTS.clear()
