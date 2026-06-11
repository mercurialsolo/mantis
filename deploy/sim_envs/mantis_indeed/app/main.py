"""mantis-indeed FastAPI app — entrypoint mounted by Docker + Modal."""

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
        # In sim env: require the harness to set this. For local dev
        # default to "test" so the import-time check doesn't break
        # tests / smoke runs that forgot to export it.
        token = "test"
        os.environ.setdefault(ADMIN_TOKEN_ENV, token)
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
    app = FastAPI(title="mantis-indeed", docs_url=None, redoc_url=None)

    templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

    # Tiny markdown filter for templates that use `{{ x | md | safe }}`.
    # Handles ##/###/**bold**/bullet lists; otherwise wraps paragraphs.
    def _md(text: str) -> str:
        import html
        import re as _re
        if not text:
            return ""
        out_lines: list[str] = []
        in_ul = False
        for raw in str(text).split("\n"):
            line = raw.rstrip()
            if not line:
                if in_ul:
                    out_lines.append("</ul>")
                    in_ul = False
                out_lines.append("")
                continue
            m_h = _re.match(r"^(#{1,6})\s+(.*)$", line)
            if m_h:
                if in_ul:
                    out_lines.append("</ul>"); in_ul = False
                lvl = len(m_h.group(1))
                out_lines.append(f"<h{lvl}>{html.escape(m_h.group(2))}</h{lvl}>")
                continue
            if line.lstrip().startswith(("- ", "* ")):
                if not in_ul:
                    out_lines.append("<ul>"); in_ul = True
                item = line.lstrip()[2:]
                out_lines.append(f"<li>{html.escape(item)}</li>")
                continue
            if in_ul:
                out_lines.append("</ul>"); in_ul = False
            out_lines.append(f"<p>{html.escape(line)}</p>")
        if in_ul:
            out_lines.append("</ul>")
        body = "\n".join(out_lines)
        # **bold** → <strong>
        body = _re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", body)
        return body

    templates.env.filters["md"] = _md
    app.state.templates = templates
    app.state.admin_token = _admin_token()

    @app.middleware("http")
    async def _auth_gate(request: Request, call_next):
        try:
            request.state.current_user = auth.current_user(request)
        except Exception:
            request.state.current_user = None

        # Always populate an effective_user so the post-login topnav
        # renders the signed-in shell when ENV_REQUIRE_AUTH=0 (default).
        try:
            request.state.effective_user = auth.effective_user(request)
        except Exception:
            request.state.effective_user = None

        if auth.auth_required():
            path = request.url.path
            open_prefixes = (
                "/login", "/logout", "/signup", "/__env__/",
                "/static/", "/jobs", "/viewjob",
            )
            if path == "/" or path.startswith(open_prefixes):
                pass
            elif path.startswith(("/apply/", "/myjobs", "/resumes",
                                  "/employers/")):
                user = request.state.current_user
                if user is None:
                    return RedirectResponse(
                        f"/login?next={path}", status_code=303,
                    )
                if path.startswith("/employers/") and user.get("role") != "employer":
                    return RedirectResponse(
                        "/login?error=employer+only", status_code=303,
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

    from .routes import env_admin as env_admin_router
    from .routes import auth as auth_router
    from .routes import home as home_router
    from .routes import jobs as jobs_router
    from .routes import apply as apply_router
    from .routes import myjobs as myjobs_router
    from .routes import resumes as resumes_router
    from .routes import employers as employers_router

    app.include_router(env_admin_router.router)
    app.include_router(auth_router.router)
    app.include_router(home_router.router)
    app.include_router(jobs_router.router)
    app.include_router(apply_router.router)
    app.include_router(myjobs_router.router)
    app.include_router(resumes_router.router)
    app.include_router(employers_router.router)

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
