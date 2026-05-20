"""Modal-compatible viewer — thread-safe, works with synchronous executors.

Unlike viewer.py (async, for local StreamingCUA), this module runs the viewer
server and screen capture in background threads, bridging the sync executor
world to the async MJPEG/SSE viewer.

Screen capture via mss works with Xvfb virtual displays — set DISPLAY=:99
before calling, and the viewer captures exactly what the agent sees.

Usage in a Modal executor:
    from mantis_agent.viewer_modal import modal_viewer

    with modal_viewer() as (event_bus, viewer_url):
        print(f"Viewer: {viewer_url}")
        event_bus.emit({"type": "task_start", "task": "...", "max_steps": 50})
        # ... run agent tasks, call event_bus.emit() for actions ...
        event_bus.emit({"type": "done", "success": True, "summary": "..."})
"""

from __future__ import annotations

import io
import json
import logging
import secrets
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any

from .streamer import ScreenStreamer

logger = logging.getLogger(__name__)


class ModalViewerBus:
    """Thread-safe event bus for bridging sync executors to async SSE.

    Safe to call emit() from any thread — the SSE generator polls
    events_since() from uvicorn's async event loop.
    """

    def __init__(self) -> None:
        self._events: deque[dict] = deque(maxlen=1000)
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        """Push an event (thread-safe, sync-compatible)."""
        event.setdefault("ts", time.time())
        with self._lock:
            self._events.append(event)

    def events_since(self, index: int) -> tuple[list[dict], int]:
        """Get events after a given index. Returns (new_events, new_index)."""
        with self._lock:
            all_events = list(self._events)
        return all_events[index:], len(all_events)


def _start_background(
    port: int = 7860,
    fps: float = 3.0,
    monitor: int = 1,
    *,
    proxy_diag: dict | None = None,
    api_run_id: str = "",
    api_tenant_id: str = "",
) -> tuple[ModalViewerBus, str, callable]:
    """Start capture + server in background threads.

    Returns (event_bus, token, stop_fn).
    """
    import asyncio

    import uvicorn
    from fastapi import FastAPI, Query
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from starlette.exceptions import HTTPException

    token = secrets.token_urlsafe(24)
    event_bus = ModalViewerBus()
    streamer = ScreenStreamer(fps=fps, buffer_size=15, monitor=monitor)

    # ── Screen capture thread ────────────────────────────────────────
    capture_stop = threading.Event()

    def capture_loop() -> None:
        interval = 1.0 / fps
        _logged_error = False
        while not capture_stop.is_set():
            try:
                streamer.capture_once()
            except Exception as e:
                if not _logged_error:
                    import os
                    logger.error(
                        f"Screen capture failed (DISPLAY={os.environ.get('DISPLAY', '<unset>')}): {e}"
                    )
                    print(f"  [viewer] capture error: {e} (DISPLAY={os.environ.get('DISPLAY', '<unset>')})")
                    _logged_error = True
            capture_stop.wait(interval)

    threading.Thread(target=capture_loop, daemon=True, name="viewer-capture").start()

    # ── FastAPI app ──────────────────────────────────────────────────
    app = FastAPI(docs_url=None, redoc_url=None)

    def _auth(t: str | None) -> None:
        if t != token:
            raise HTTPException(401, "Unauthorized")

    @app.get("/", response_class=HTMLResponse)
    async def index(token: str = Query(...)):
        _auth(token)
        from .viewer import VIEWER_HTML

        return VIEWER_HTML.replace("{{TOKEN}}", token)

    @app.get("/stream")
    async def stream(token: str = Query(...)):
        _auth(token)

        async def generate():
            last_idx = -1
            while True:
                frame = streamer.latest
                if frame and frame.index != last_idx:
                    last_idx = frame.index
                    buf = io.BytesIO()
                    frame.image.save(buf, format="JPEG", quality=70)
                    data = buf.getvalue()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(data)).encode() + b"\r\n"
                        b"\r\n" + data + b"\r\n"
                    )
                await asyncio.sleep(0.15)

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/events")
    async def events(token: str = Query(...)):
        _auth(token)

        async def generate():
            idx = 0
            while True:
                new_events, new_idx = event_bus.events_since(idx)
                for ev in new_events:
                    yield f"data: {json.dumps(ev)}\n\n"
                idx = new_idx
                if not new_events:
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/history")
    async def history(token: str = Query(...)):
        _auth(token)
        events_list, _ = event_bus.events_since(0)
        return JSONResponse(events_list)

    # ── #viewer-proxy-diag: surface the actual proxy egress IP /
    # city / region the run is using. Lets operators verify the geo
    # modifier (e.g. ``-cc-us-city-miami``) actually landed instead
    # of trusting the runtime block. ``proxy_diag`` is the
    # ipinfo-probe dict from ``setup_env`` — already computed once
    # at executor startup; this endpoint is a cheap re-serve.
    @app.get("/api/proxy_info")
    async def proxy_info(token: str = Query(...)):
        _auth(token)
        return JSONResponse(proxy_diag or {})

    # ── #viewer-takeover: Pause / Resume controls that proxy to
    # the cua-server API. The browser POSTs here with the viewer
    # token; this endpoint uses the executor's ``MANTIS_API_TOKEN``
    # (env var, server-side only) to call the cua-server API. The
    # API token never reaches the browser.
    def _post_to_cua_api(action: str, reason: str = "") -> tuple[int, dict]:
        """POST {action,run_id,...} to the cua-server API on behalf of
        the viewer user. Returns (http_status, body_dict)."""
        import os as _os
        import requests as _r
        api_token = _os.environ.get("MANTIS_API_TOKEN", "")
        if not api_token:
            return 500, {"detail": "MANTIS_API_TOKEN unset in executor env"}
        if not api_run_id:
            return 400, {"detail": "api_run_id not threaded — viewer can't route action"}
        api_endpoint = _os.environ.get(
            "MANTIS_API_ENDPOINT",
            "https://getmason--mantis-cua-server-api.modal.run",
        )
        body: dict = {"action": action, "run_id": api_run_id}
        if reason:
            body["reason"] = reason
        try:
            r = _r.post(
                f"{api_endpoint}/v1/predict",
                headers={
                    "X-Mantis-Token": api_token,
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=15,
            )
            try:
                return r.status_code, r.json()
            except Exception:
                return r.status_code, {"raw": r.text[:300]}
        except Exception as exc:  # noqa: BLE001
            return 502, {"detail": f"{type(exc).__name__}: {str(exc)[:200]}"}

    @app.post("/api/pause_run")
    async def pause_run(token: str = Query(...)):
        _auth(token)
        status_code, body = _post_to_cua_api("pause", reason="viewer_takeover")
        return JSONResponse(body, status_code=status_code)

    @app.post("/api/resume_run")
    async def resume_run(token: str = Query(...)):
        _auth(token)
        status_code, body = _post_to_cua_api("resume")
        return JSONResponse(body, status_code=status_code)

    @app.get("/api/run_state")
    async def run_state(token: str = Query(...)):
        """Proxy to cua-server ``action=status`` so the viewer can
        poll Pause/Resume button state (paused vs running) without
        the browser needing the API token."""
        _auth(token)
        status_code, body = _post_to_cua_api("status")
        return JSONResponse(body, status_code=status_code)

    # ── Server thread ────────────────────────────────────────────────
    config = uvicorn.Config(
        app, host="0.0.0.0", port=port,
        log_level="warning", access_log=False,
    )
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True, name="viewer-server").start()
    time.sleep(1)  # Let server bind

    def stop() -> None:
        capture_stop.set()
        server.should_exit = True

    return event_bus, token, stop


@contextmanager
def modal_viewer(
    port: int = 7860,
    fps: float = 3.0,
    monitor: int = 1,
    *,
    proxy_diag: dict | None = None,
    api_run_id: str = "",
    api_tenant_id: str = "",
):
    """Context manager: starts viewer + Modal tunnel, yields (event_bus, url).

    The viewer captures the screen (Xvfb or real display) via mss and
    streams it as MJPEG to a public authenticated URL via modal.forward().

    Args:
        port: Local port for the viewer server.
        fps: Screen capture rate (2-5 FPS recommended).
        monitor: Which monitor to capture (1=primary, 0=all).
        proxy_diag: ipinfo egress probe from
            :func:`task_loop.diagnose_proxy_egress`. Surfaced via
            ``/api/proxy_info`` so the viewer header shows the actual
            exit IP / city / region.
        api_run_id: run identifier for the cua-server API, threaded
            through so the viewer's ``/api/pause_run`` /
            ``/api/resume_run`` buttons can route their POST to the
            right run.
        api_tenant_id: tenant for the cua-server API call.

    Yields:
        (event_bus, url): ModalViewerBus for emitting events, and the
        public authenticated URL for the viewer dashboard.
    """
    import modal

    event_bus, token, stop = _start_background(
        port, fps, monitor,
        proxy_diag=proxy_diag or {},
        api_run_id=api_run_id,
        api_tenant_id=api_tenant_id,
    )

    with modal.forward(port) as tunnel:
        url = f"{tunnel.url}?token={token}"
        logger.info(f"Viewer tunnel: {url}")
        print(f"\n  Viewer: {url}\n")
        try:
            yield event_bus, url
        finally:
            # Brief pause so viewer receives final events
            time.sleep(3)
            stop()
