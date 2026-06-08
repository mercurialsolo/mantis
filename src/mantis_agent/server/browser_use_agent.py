"""Browser-Use Plane FastAPI app — Playwright-driven (#785, PR 2).

Mirrors `computer_agent.py`'s shape but drives Playwright + Chromium
instead of Xvfb + xdotool. Implements the **base surface** of the
unified `ComputeClient` contract: `session/init`, `session/close`,
`screenshot`, `dispatch`, `health`.

PR 3-4 add the DOM-aware extension endpoints (`state/*`, `tabs/*`,
`links/peek`). They are NOT exposed here yet — handlers gate on the
advertised `dom_aware` capability, so the missing routes would surface
as a 404 (acceptable at scaffold stage).

This module is imported by `deploy/modal/browser_use_plane.py`; running
it directly with `uvicorn` is supported for local debugging.
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from ..gym.browser_use_wire import (
    BrowserUseHealthResponse,
    BrowserUseScreenshotResponse,
    BrowserUseSessionCloseRequest,
    BrowserUseSessionCloseResponse,
    BrowserUseSessionInitRequest,
    BrowserUseSessionInitResponse,
    DispatchActionRequest,
    DispatchActionResponse,
)
from ..gym.compute_contract import Capabilities

logger = logging.getLogger(__name__)


@dataclass
class _Session:
    """Per-session Playwright state.

    A single Modal container handles one session at a time at v1 (matches
    the existing `ComputerAgent` per-profile-lock semantics). PR 3+ may
    relax this once concurrent tabs are wired through.
    """

    token: str
    tenant_id: str
    profile_id: str
    run_id: str
    capabilities: Capabilities
    last_action_ms: int = 0
    playwright: Any = None  # opaque — Playwright runtime handle
    browser: Any = None
    context: Any = None
    page: Any = None
    dispatch_cache: dict[str, DispatchActionResponse] = field(default_factory=dict)


_sessions: dict[str, _Session] = {}


def _require_session(request: Request) -> _Session:
    token = request.headers.get("X-Mantis-Session", "")
    if not token or token not in _sessions:
        raise HTTPException(status_code=401, detail="missing or unknown session token")
    return _sessions[token]


def _start_playwright(req: BrowserUseSessionInitRequest) -> _Session:
    """Lazy-import Playwright + launch a browser.

    Import is deferred so this module remains importable in environments
    without Playwright (e.g. test runners that mock the client).
    """
    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    pw = sync_playwright().start()
    launch_args: dict[str, Any] = {"headless": req.headless}
    if req.proxy_server:
        launch_args["proxy"] = {"server": req.proxy_server}
    if req.profile_dir:
        # Persistent context — profile path is per-tenant on the shared
        # volume. See `docs/reference/browser-use-plane.md` for the
        # layout contract.
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=req.profile_dir,
            viewport={"width": req.viewport[0], "height": req.viewport[1]},
            extra_http_headers=req.extra_http_headers or {},
            **launch_args,
        )
        browser = ctx.browser
        page = ctx.new_page() if not ctx.pages else ctx.pages[0]
    else:
        browser = pw.chromium.launch(**launch_args)
        ctx = browser.new_context(
            viewport={"width": req.viewport[0], "height": req.viewport[1]},
            extra_http_headers=req.extra_http_headers or {},
        )
        page = ctx.new_page()

    if req.start_url and req.start_url != "about:blank":
        page.goto(req.start_url, wait_until="load")

    return _Session(
        token=uuid.uuid4().hex,
        tenant_id=req.tenant_id,
        profile_id=req.profile_id,
        run_id=req.run_id,
        capabilities=Capabilities.for_browser_use_plane(),
        playwright=pw,
        browser=browser,
        context=ctx,
        page=page,
        last_action_ms=int(time.time() * 1000),
    )


def _stop_playwright(sess: _Session) -> None:
    try:
        if sess.page is not None:
            sess.page.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        if sess.context is not None:
            sess.context.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        if sess.browser is not None:
            sess.browser.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        if sess.playwright is not None:
            sess.playwright.stop()
    except Exception:  # noqa: BLE001
        pass


def _dispatch(sess: _Session, req: DispatchActionRequest) -> DispatchActionResponse:
    """Execute one structured action against the bound page.

    Idempotent on `step_id` via a per-session TTL'd-feeling cache. The
    cache is process-local and capped — sufficient for a single-session
    container.
    """
    cached = sess.dispatch_cache.get(req.step_id)
    if cached is not None:
        return DispatchActionResponse(
            ok=cached.ok,
            error=cached.error,
            deduplicated=True,
        )

    page = sess.page
    try:
        if req.kind == "click":
            p = req.params
            page.mouse.click(
                int(p["x"]),
                int(p["y"]),
                button=str(p.get("button", "left")),  # type: ignore[arg-type]
                click_count=int(p.get("click_count", 1)),
            )
        elif req.kind == "key":
            p = req.params
            page.keyboard.press(str(p["key"]))
        elif req.kind == "type":
            p = req.params
            page.keyboard.type(str(p["text"]), delay=int(p.get("delay_ms", 0)))
        elif req.kind == "scroll":
            p = req.params
            page.mouse.wheel(int(p.get("x", 0) or 0), int(p["delta_y"]))
        else:
            return DispatchActionResponse(ok=False, error=f"unknown kind {req.kind!r}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[browser-use] dispatch %s failed: %s", req.kind, exc)
        resp = DispatchActionResponse(ok=False, error=str(exc))
        sess.dispatch_cache[req.step_id] = resp
        return resp

    sess.last_action_ms = int(time.time() * 1000)
    resp = DispatchActionResponse(ok=True)
    sess.dispatch_cache[req.step_id] = resp
    # Keep the cache bounded — simple FIFO.
    if len(sess.dispatch_cache) > 1024:
        for k in list(sess.dispatch_cache.keys())[:128]:
            sess.dispatch_cache.pop(k, None)
    return resp


def build_app() -> FastAPI:
    """Construct the Browser-Use Plane FastAPI app.

    Wrapping construction in a factory keeps the module side-effect-free
    on import — important for both Modal's `add_local_python_source` and
    for the test suite (which imports models without spinning servers).
    """
    app = FastAPI(title="Mantis Browser-Use Plane Agent")

    @app.post("/session/init", response_model=BrowserUseSessionInitResponse)
    def session_init(req: BrowserUseSessionInitRequest) -> BrowserUseSessionInitResponse:
        # Idempotent on run_id — second init with the same run_id returns
        # the existing token.
        for s in _sessions.values():
            if s.run_id == req.run_id and s.tenant_id == req.tenant_id:
                return BrowserUseSessionInitResponse(
                    session_token=s.token,
                    browser_pid=None,
                    capabilities=s.capabilities.as_dict(),
                )
        sess = _start_playwright(req)
        _sessions[sess.token] = sess
        logger.warning(
            "[browser-use] session init tenant=%s profile=%s run=%s token=%s",
            req.tenant_id,
            req.profile_id,
            req.run_id,
            sess.token[:8],
        )
        return BrowserUseSessionInitResponse(
            session_token=sess.token,
            browser_pid=None,
            capabilities=sess.capabilities.as_dict(),
        )

    @app.post("/session/close", response_model=BrowserUseSessionCloseResponse)
    def session_close(req: BrowserUseSessionCloseRequest) -> BrowserUseSessionCloseResponse:
        sess = _sessions.pop(req.session_token, None)
        if sess is None:
            return BrowserUseSessionCloseResponse(closed=False)
        _stop_playwright(sess)
        return BrowserUseSessionCloseResponse(closed=True)

    @app.post("/screenshot", response_model=BrowserUseScreenshotResponse)
    def screenshot(request: Request) -> BrowserUseScreenshotResponse:
        sess = _require_session(request)
        page = sess.page
        png = page.screenshot(type="png")
        width = sess.capabilities.as_dict().get("width") or sess.page.viewport_size["width"]
        height = sess.capabilities.as_dict().get("height") or sess.page.viewport_size["height"]
        scroll_y = int(page.evaluate("window.scrollY"))
        return BrowserUseScreenshotResponse(
            image_b64=base64.b64encode(png).decode("ascii"),
            width=int(width),
            height=int(height),
            scroll_y=scroll_y,
            captured_at_ms=int(time.time() * 1000),
        )

    @app.post("/dispatch", response_model=DispatchActionResponse)
    def dispatch(req: DispatchActionRequest, request: Request) -> DispatchActionResponse:
        sess = _require_session(request)
        return _dispatch(sess, req)

    @app.get("/health", response_model=BrowserUseHealthResponse)
    def health(request: Request) -> BrowserUseHealthResponse:
        token = request.headers.get("X-Mantis-Session", "")
        sess = _sessions.get(token) if token else None
        return BrowserUseHealthResponse(
            ok=True,
            last_action_ms=sess.last_action_ms if sess else None,
            session_token=sess.token if sess else None,
        )

    return app


# Lazy global app object — built on first import for uvicorn convenience.
app = build_app()
