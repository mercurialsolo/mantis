"""mantis-shopify FastAPI app — entrypoint.

Surfaces:

* ``/`` — the Shopify Partners back-office mirror (server-rendered).
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
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    base = (parts[0] if parts else "?").upper()
    return (base + "?")[:2]


def _money(cents: int | float | None) -> str:
    if cents is None:
        return "$0.00"
    dollars = (int(cents) / 100.0)
    return f"${dollars:,.2f}"


def _short_date(iso: str) -> str:
    """Format an ISO date like '2026-06-09T09:00:00Z' → 'Jun 9, 2026'."""
    if not iso:
        return ""
    s = iso.replace("Z", "+00:00")
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(s)
        return dt.strftime("%b %-d, %Y")
    except Exception:  # noqa: BLE001
        return iso


def _humanize_last_login(iso: str, now_iso: str) -> str:
    if not iso:
        return "never"
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        nt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return iso
    delta = nt - dt
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 90 * 60:
        return f"about {max(1, seconds // 60)} minutes ago"
    hours = seconds // 3600
    if hours < 24:
        return f"about {hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    if days < 30:
        return f"{days} day{'s' if days != 1 else ''} ago"
    months = days // 30
    if months < 12:
        return f"{months} month{'s' if months != 1 else ''} ago"
    years = months // 12
    return f"over {years} year{'s' if years != 1 else ''} ago"


def create_app() -> FastAPI:
    app = FastAPI(title="mantis-shopify", docs_url=None, redoc_url=None)

    templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
    templates.env.filters["initials"] = _initials_for
    templates.env.filters["money"] = _money
    templates.env.filters["short_date"] = _short_date
    templates.env.globals["humanize_last_login"] = _humanize_last_login
    app.state.templates = templates
    app.state.admin_token = _admin_token()

    @app.middleware("http")
    async def _auth_gate(request: Request, call_next):  # noqa: ANN202
        try:
            request.state.current_user = auth.current_user(request)
        except Exception:  # noqa: BLE001
            request.state.current_user = None

        # Inject the effective owner identity so templates always render
        # the topbar / sidebar shell — Partners is a post-login surface.
        request.state.effective_user = auth.effective_user(request)

        if auth.auth_required():
            path = request.url.path
            open_prefixes = (
                "/login", "/logout", "/__env__/", "/static/",
            )
            if not (path.startswith(open_prefixes)):
                user = request.state.current_user
                if user is None:
                    return RedirectResponse(
                        f"/login?next={path}", status_code=303,
                    )
        return await call_next(request)

    @app.on_event("startup")
    def _bootstrap() -> None:
        conn = db.connect()
        cur = conn.execute("SELECT COUNT(*) FROM partners")
        if cur.fetchone()[0] == 0:
            seed.seed(conn, seed_val=_seed_val(), fake_now=_now())
            _emit_event("seeded", {"seed": _seed_val(), "now": _now()})

    app.mount(
        "/static",
        StaticFiles(directory=str(APP_DIR / "static")),
        name="static",
    )

    from .routes import (
        admin as admin_router,
        auth as auth_router,
        catalogs as catalogs_router,
        directory as directory_router,
        docs as docs_router,
        env_admin as env_admin_router,
        home as home_router,
        payouts as payouts_router,
        pos as pos_router,
        sales as sales_router,
        settings as settings_router,
        stores as stores_router,
        support as support_router,
        team as team_router,
        themes as themes_router,
    )

    app.include_router(env_admin_router.router)
    app.include_router(auth_router.router)
    app.include_router(home_router.router)
    app.include_router(stores_router.router)
    app.include_router(sales_router.router)
    app.include_router(catalogs_router.router)
    app.include_router(themes_router.router)
    app.include_router(directory_router.router)
    app.include_router(pos_router.router)
    app.include_router(docs_router.router)
    app.include_router(support_router.router)
    app.include_router(payouts_router.router)
    app.include_router(team_router.router)
    app.include_router(settings_router.router)
    app.include_router(admin_router.router)

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
