"""`ComputerAgent` — FastAPI service for the Computer Plane (#698, Phase 1).

Speaks the wire contract defined in `mantis_agent.gym.computer_wire`. One
session per container instance: `POST /session/init` brings up Xvfb +
Chrome bound to a `(tenant_id, profile_id, run_id)` triple, subsequent
`/screenshot` / `/xdotool` / `/cdp` calls drive it, `/session/close`
tears it down.

The server is intentionally thin: it delegates Chrome / Xvfb lifecycle
to the existing `XdotoolGymEnv` and only adds the wire-layer concerns
(session binding, `step_id` LRU dedup, base64 PNG transport).

Run locally for tests:

    uvicorn mantis_agent.server.computer_agent:app --host 0.0.0.0 --port 8090

In Modal, the `computer_plane` function in
`deploy/modal/modal_cua_server.py` exposes this via
`@modal.fastapi_endpoint`.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import secrets
import subprocess
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from ..gym.computer_wire import (
    CDPRequest,
    CDPResponse,
    HealthResponse,
    ScreenshotRequest,
    ScreenshotResponse,
    SessionCloseRequest,
    SessionCloseResponse,
    SessionInitRequest,
    SessionInitResponse,
    XdotoolRequest,
    XdotoolResponse,
)

logger = logging.getLogger(__name__)

# Server-side LRU dedup parameters — see wire contract §"Idempotency".
_DEDUP_TTL_SECONDS = 30.0
_DEDUP_CAPACITY = 1000


@dataclass
class _DedupEntry:
    response: XdotoolResponse
    inserted_at: float


@dataclass
class _Session:
    """Per-container state for the single active session.

    Phase 1 ships one session per container instance — the brain plane
    serializes requests against this session via the session token. Phase
    2's pluggable backends may permit multi-tenant containers.
    """

    token: str
    tenant_id: str
    profile_id: str
    run_id: str
    enable_cdp: bool
    viewport: tuple[int, int]
    env: Any  # XdotoolGymEnv
    last_action_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    dedup: OrderedDict[str, _DedupEntry] = field(default_factory=OrderedDict)
    dedup_lock: Lock = field(default_factory=Lock)

    def touch(self) -> None:
        self.last_action_ms = int(time.time() * 1000)

    def lookup_dedup(self, step_id: str) -> XdotoolResponse | None:
        with self.dedup_lock:
            entry = self.dedup.get(step_id)
            if entry is None:
                return None
            if time.time() - entry.inserted_at > _DEDUP_TTL_SECONDS:
                # Stale → forget.
                self.dedup.pop(step_id, None)
                return None
            # Refresh LRU position.
            self.dedup.move_to_end(step_id)
            cached = entry.response.model_copy()
            cached.deduplicated = True
            return cached

    def remember_dedup(self, step_id: str, response: XdotoolResponse) -> None:
        with self.dedup_lock:
            self.dedup[step_id] = _DedupEntry(response=response, inserted_at=time.time())
            self.dedup.move_to_end(step_id)
            while len(self.dedup) > _DEDUP_CAPACITY:
                self.dedup.popitem(last=False)


class ComputerAgentState:
    """Module-level holder so handlers can reach the active session.

    Kept as a small class (vs. a bare module global) so tests can swap an
    instance in / out without leaking state across the test suite.
    """

    def __init__(self) -> None:
        self.session: _Session | None = None
        self.lock = Lock()


_state = ComputerAgentState()


def _state_for_tests() -> ComputerAgentState:
    """Test hook — lets tests reset state between test cases."""
    return _state


def _require_session(session_token: str | None) -> _Session:
    if not session_token:
        raise HTTPException(401, "Missing X-Mantis-Session header")
    s = _state.session
    if s is None or s.token != session_token:
        raise HTTPException(401, "Invalid or expired session token")
    return s


def _new_xdotool_env(req: SessionInitRequest) -> Any:
    """Construct + start the Xvfb / Chrome stack for this session.

    Goes through `make_computer_client(ComputerPlaneConfig())` so the
    server-side wiring uses the same factory the brain plane uses — keeps
    the impl honest about being a `ComputerClient`.
    """
    from ..gym.computer_client import ComputerPlaneConfig, make_computer_client

    profile_dir = f"/data/chrome-profile/{req.tenant_id}__{req.profile_id}"
    env = make_computer_client(
        ComputerPlaneConfig(),
        start_url="about:blank",
        viewport=req.viewport,
        browser="google-chrome",
        proxy_server=req.proxy_server or "",
        profile_dir=profile_dir,
    )
    env.reset(task="computer_plane_session", start_url="about:blank")
    return env


def build_app() -> FastAPI:
    """Build the FastAPI app — factored so Modal + tests can both call it."""
    app = FastAPI(title="Mantis ComputerAgent", version="1.0.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        s = _state.session
        return HealthResponse(
            ok=True,
            last_action_ms=s.last_action_ms if s else None,
            session_token=s.token if s else None,
        )

    @app.post("/session/init", response_model=SessionInitResponse)
    def session_init(req: SessionInitRequest) -> SessionInitResponse:
        with _state.lock:
            s = _state.session
            # Idempotent on run_id — a second init for the same run_id
            # returns the existing token. Different run_id while an
            # active session exists is a 409.
            if s is not None:
                if s.run_id == req.run_id:
                    return SessionInitResponse(
                        session_token=s.token,
                        chrome_pid=getattr(s.env, "_browser_proc", None) and s.env._browser_proc.pid,
                        xvfb_display=s.env._env.get("DISPLAY", ":99"),
                    )
                raise HTTPException(
                    409,
                    f"Active session bound to run_id={s.run_id}; "
                    f"requested run_id={req.run_id}",
                )

            token = secrets.token_urlsafe(24)
            env = _new_xdotool_env(req)
            display = env._env.get("DISPLAY", ":99")
            chrome_pid = env._browser_proc.pid if env._browser_proc else None
            _state.session = _Session(
                token=token,
                tenant_id=req.tenant_id,
                profile_id=req.profile_id,
                run_id=req.run_id,
                enable_cdp=req.enable_cdp,
                viewport=req.viewport,
                env=env,
            )
            return SessionInitResponse(
                session_token=token,
                chrome_pid=chrome_pid,
                xvfb_display=display,
            )

    @app.post("/session/close", response_model=SessionCloseResponse)
    def session_close(req: SessionCloseRequest) -> SessionCloseResponse:
        with _state.lock:
            s = _state.session
            if s is None or s.token != req.session_token:
                return SessionCloseResponse(closed=False)
            try:
                s.env.shutdown()
            except Exception as exc:  # noqa: BLE001 — best-effort teardown
                logger.warning("session_close: shutdown raised: %s", exc)
            _state.session = None
            return SessionCloseResponse(closed=True)

    @app.post("/screenshot", response_model=ScreenshotResponse)
    def screenshot(
        req: ScreenshotRequest,  # noqa: ARG001 — reserved for future fields
        x_mantis_session: str | None = Header(default=None),
    ) -> ScreenshotResponse:
        s = _require_session(x_mantis_session)
        s.touch()
        img = s.env.screenshot()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        scroll_y = 0
        # Best-effort scroll_y read — falls back to 0 when CDP disabled.
        try:
            ok, value = s.env._cdp_call(
                "Runtime.evaluate",
                {"expression": "window.scrollY", "returnByValue": True},
            )
            if ok and isinstance(value, dict):
                result = value.get("result", {})
                if isinstance(result, dict):
                    scroll_y = int(result.get("value", 0) or 0)
        except Exception:  # noqa: BLE001 — diagnostic only
            scroll_y = 0
        return ScreenshotResponse(
            image_b64=base64.b64encode(buf.getvalue()).decode("ascii"),
            width=img.width,
            height=img.height,
            scroll_y=scroll_y,
            captured_at_ms=int(time.time() * 1000),
        )

    @app.post("/xdotool", response_model=XdotoolResponse)
    def xdotool(
        req: XdotoolRequest,
        x_mantis_session: str | None = Header(default=None),
    ) -> XdotoolResponse:
        s = _require_session(x_mantis_session)
        cached = s.lookup_dedup(req.step_id)
        if cached is not None:
            s.touch()
            return cached

        cmd = ["xdotool", *req.argv]
        timeout_s = max(0.5, req.timeout_ms / 1000.0)
        try:
            proc = subprocess.run(
                cmd,
                env=s.env._env,
                capture_output=True,
                timeout=timeout_s,
            )
            response = XdotoolResponse(
                stdout=proc.stdout.decode(errors="replace"),
                stderr=proc.stderr.decode(errors="replace"),
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            response = XdotoolResponse(
                stdout="",
                stderr=f"xdotool timeout after {timeout_s:.1f}s: {exc}",
                returncode=124,
            )
        s.remember_dedup(req.step_id, response)
        s.touch()
        return response

    @app.post("/cdp", response_model=CDPResponse)
    def cdp(
        req: CDPRequest,
        x_mantis_session: str | None = Header(default=None),
    ) -> CDPResponse:
        s = _require_session(x_mantis_session)
        if not s.enable_cdp:
            raise HTTPException(403, "CDP disabled for this session")
        s.touch()
        try:
            ok, value = s.env._cdp_call(
                "Runtime.evaluate",
                {
                    "expression": req.expression,
                    "returnByValue": True,
                    "awaitPromise": req.await_promise,
                },
            )
            import json as _json
            return CDPResponse(
                result_json=_json.dumps(value if value is not None else {}),
                returncode=0 if ok else 1,
            )
        except Exception as exc:  # noqa: BLE001 — bubble as 5xx
            raise HTTPException(500, f"cdp_evaluate failed: {exc}") from exc

    return app


# Module-level app for `uvicorn mantis_agent.server.computer_agent:app`
app = build_app()


def _admin_token() -> str:
    """Token clients must send as `Authorization: Bearer <token>` once
    Phase 1 wires the secret. Phase 0 stub is permissive — defaults to
    empty so local tests can hit the endpoints without auth.
    """
    return os.environ.get("MANTIS_COMPUTER_PLANE_TOKEN", "")
