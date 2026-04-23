"""Live streaming viewer for the Mantis CUA agent.

Provides a web-based dashboard where humans can watch the agent's screen
in real-time, see its actions and reasoning, and monitor progress.

Architecture:
- MJPEG stream for live screen feed (zero-latency video in <img> tag)
- Server-Sent Events for agent actions, thinking, and status updates
- Token-based authentication (random token printed to console on start)
- Single self-contained module (HTML/CSS/JS inline, no static files)

Usage:
    mantis --viewer "your task here"
    mantis --viewer --viewer-port 8080 "your task here"
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import secrets
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .streamer import ScreenStreamer

logger = logging.getLogger(__name__)


class ViewerEventBus:
    """Pub/sub for broadcasting agent events to connected viewer clients."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[dict]] = []
        self._history: deque[dict] = deque(maxlen=500)

    def subscribe(self) -> asyncio.Queue[dict]:
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=100)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict]) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    async def emit(self, event: dict[str, Any]) -> None:
        self._history.append(event)
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    @property
    def history(self) -> list[dict]:
        return list(self._history)


def create_viewer_app(
    streamer: ScreenStreamer,
    event_bus: ViewerEventBus,
    token: str,
) -> Any:
    """Create the FastAPI viewer application."""
    from fastapi import FastAPI, Query
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from starlette.exceptions import HTTPException

    app = FastAPI(docs_url=None, redoc_url=None)

    def _auth(t: str | None) -> None:
        if t != token:
            raise HTTPException(401, "Unauthorized")

    @app.get("/", response_class=HTMLResponse)
    async def index(token: str = Query(...)):
        _auth(token)
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
                    frame.image.save(buf, format="JPEG", quality=75)
                    data = buf.getvalue()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(data)).encode() + b"\r\n"
                        b"\r\n" + data + b"\r\n"
                    )
                await asyncio.sleep(0.1)

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/events")
    async def events(token: str = Query(...)):
        _auth(token)

        async def generate():
            q = event_bus.subscribe()
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=15)
                        yield f"data: {json.dumps(event)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                event_bus.unsubscribe(q)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/history")
    async def history(token: str = Query(...)):
        _auth(token)
        return JSONResponse(event_bus.history)

    return app


async def run_with_viewer(
    agent: Any,
    task: str,
    streamer: ScreenStreamer,
    port: int = 7860,
) -> Any:
    """Run the agent with the live viewer server.

    Starts the MJPEG/SSE server, wires up agent event callbacks,
    prints the authenticated URL, and runs the agent to completion.
    """
    import uvicorn

    token = secrets.token_urlsafe(24)
    event_bus = ViewerEventBus()
    app = create_viewer_app(streamer, event_bus, token)

    agent.on_event = event_bus.emit

    # Start capturing early so viewer has frames before agent loop begins
    await streamer.start()

    config = uvicorn.Config(
        app, host="0.0.0.0", port=port,
        log_level="warning", access_log=False,
    )
    server = uvicorn.Server(config)

    url = f"http://localhost:{port}?token={token}"
    logger.info(f"Viewer available at {url}")
    print(f"\n  Viewer: {url}\n")

    server_task = asyncio.create_task(server.serve())
    try:
        result = await agent.run(task)
        # Brief pause so the viewer receives the final 'done' event
        await asyncio.sleep(3)
        return result
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass


# ── Inline HTML/CSS/JS for the viewer dashboard ─────────────────────────────

VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mantis &mdash; Live Viewer</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>&#x1F52D;</text></svg>">
<style>
:root {
    --bg-0: #0d1117;
    --bg-1: #161b22;
    --bg-2: #1c2128;
    --border: #30363d;
    --text-1: #e6edf3;
    --text-2: #8b949e;
    --text-3: #6e7681;
    --green: #3fb950;
    --blue: #58a6ff;
    --amber: #d29922;
    --red: #f85149;
    --purple: #bc8cff;
    --orange: #f0883e;
    --mantis: #00d4aa;
}
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: var(--bg-0);
    color: var(--text-1);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* ── Header ─────────────────────────────────────────────── */
header {
    background: var(--bg-1);
    border-bottom: 1px solid var(--border);
    padding: 10px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    min-height: 48px;
    flex-shrink: 0;
}
.logo {
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 0.5px;
    color: var(--mantis);
    flex-shrink: 0;
}
.task {
    flex: 1;
    font-size: 13px;
    color: var(--text-2);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.stats {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
}
.stat {
    font-size: 13px;
    color: var(--text-2);
    font-variant-numeric: tabular-nums;
}
.stat strong { color: var(--text-1); font-weight: 600; }
.conn-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--red);
    transition: background 0.3s;
    flex-shrink: 0;
}
.conn-dot.connected { background: var(--green); }

/* ── Main layout ────────────────────────────────────────── */
main {
    flex: 1;
    display: flex;
    overflow: hidden;
}

/* Video panel */
.video-panel {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #080b10;
    position: relative;
    overflow: hidden;
    padding: 12px;
}
.video-wrap {
    position: relative;
    display: inline-block;
    line-height: 0;
}
.video-wrap img {
    display: block;
    max-width: 100%;
    max-height: calc(100vh - 100px);
    border-radius: 6px;
    box-shadow: 0 0 40px rgba(0,0,0,0.5);
}
#click-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    border-radius: 6px;
    overflow: hidden;
}
.click-ring {
    position: absolute;
    width: 24px; height: 24px;
    border-radius: 50%;
    border: 2px solid var(--blue);
    background: rgba(88, 166, 255, 0.15);
    transform: translate(-50%, -50%) scale(0.5);
    animation: ring-pulse 1.2s ease-out forwards;
    pointer-events: none;
}
@keyframes ring-pulse {
    0%   { transform: translate(-50%, -50%) scale(0.5); opacity: 1; }
    70%  { opacity: 0.6; }
    100% { transform: translate(-50%, -50%) scale(3); opacity: 0; }
}
.waiting-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-3);
    font-size: 14px;
}
.waiting-overlay.hidden { display: none; }

/* ── Sidebar ────────────────────────────────────────────── */
.sidebar {
    width: 360px;
    background: var(--bg-1);
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
}
.sidebar-header {
    padding: 12px 16px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-3);
    border-bottom: 1px solid var(--border);
}
.action-log {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
}
.action-log::-webkit-scrollbar { width: 6px; }
.action-log::-webkit-scrollbar-track { background: transparent; }
.action-log::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Log entries */
.entry {
    padding: 8px 10px;
    border-radius: 6px;
    margin-bottom: 4px;
    font-size: 13px;
    animation: entry-in 0.3s ease-out;
    line-height: 1.4;
}
@keyframes entry-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.entry-action { background: var(--bg-2); }
.entry-thinking {
    background: transparent;
    border-left: 2px solid var(--border);
    margin-left: 4px;
    padding-left: 12px;
    color: var(--text-3);
    font-style: italic;
    font-size: 12px;
    max-height: 80px;
    overflow: hidden;
}
.entry-done {
    background: rgba(63, 185, 80, 0.1);
    border: 1px solid rgba(63, 185, 80, 0.2);
}
.entry-done.failed {
    background: rgba(248, 81, 73, 0.1);
    border: 1px solid rgba(248, 81, 73, 0.2);
}
.entry-system {
    color: var(--text-3);
    font-size: 12px;
    text-align: center;
    padding: 6px;
}
.step-badge {
    display: inline-block;
    min-width: 22px;
    height: 18px;
    line-height: 18px;
    text-align: center;
    background: var(--border);
    border-radius: 9px;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-2);
    margin-right: 6px;
}
.action-type {
    font-weight: 600;
    font-size: 12px;
    margin-right: 6px;
}
.action-type.click,
.action-type.double_click { color: var(--blue); }
.action-type.type_text     { color: var(--green); }
.action-type.key_press     { color: var(--amber); }
.action-type.scroll        { color: var(--purple); }
.action-type.drag          { color: var(--orange); }
.action-type.wait          { color: var(--text-3); }
.action-type.done          { color: var(--green); }
.params {
    color: var(--text-2);
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
    font-size: 12px;
}
.reasoning {
    margin-top: 4px;
    font-size: 12px;
    color: var(--text-3);
    font-style: italic;
}

/* ── Footer ─────────────────────────────────────────────── */
footer {
    background: var(--bg-1);
    border-top: 1px solid var(--border);
    padding: 6px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 12px;
    color: var(--text-3);
    flex-shrink: 0;
}
.state-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}
.state-pill.idle      { background: rgba(110,118,129,0.15); color: var(--text-3); }
.state-pill.running   { background: rgba(88,166,255,0.15);  color: var(--blue); }
.state-pill.completed { background: rgba(63,185,80,0.15);   color: var(--green); }
.state-pill.failed    { background: rgba(248,81,73,0.15);   color: var(--red); }
.state-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: currentColor;
}
.state-pill.running .state-dot {
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.3; }
}
</style>
</head>
<body>

<header>
    <div class="logo">MANTIS</div>
    <div class="task" id="task">Waiting for task...</div>
    <div class="stats">
        <div class="stat">Step <strong id="step-num">0</strong>/<strong id="step-max">-</strong></div>
        <div class="stat" id="elapsed">0:00</div>
        <div class="conn-dot" id="conn-dot" title="Disconnected"></div>
    </div>
</header>

<main>
    <div class="video-panel">
        <div class="video-wrap">
            <img id="feed" alt="Screen feed">
            <div id="click-overlay"></div>
        </div>
        <div class="waiting-overlay" id="waiting">Connecting to stream...</div>
    </div>
    <aside class="sidebar">
        <div class="sidebar-header">Activity</div>
        <div class="action-log" id="log"></div>
    </aside>
</main>

<footer>
    <div class="state-pill idle" id="state-pill">
        <div class="state-dot"></div>
        <span id="state-text">Idle</span>
    </div>
    <span style="flex:1"></span>
    <span>Mantis CUA Agent</span>
</footer>

<script>
const TOKEN = '{{TOKEN}}';
const $ = id => document.getElementById(id);
const log      = $('log');
const feed     = $('feed');
const waiting  = $('waiting');
const connDot  = $('conn-dot');
const stepNum  = $('step-num');
const stepMax  = $('step-max');
const elapsed  = $('elapsed');
const taskEl   = $('task');
const statePill= $('state-pill');
const stateText= $('state-text');

let screenW = 1920, screenH = 1080;
let startTime = null;
let timer = null;

/* ── MJPEG feed ─────────────────────────────────────────── */
feed.src = '/stream?token=' + TOKEN;
feed.onload = function() { waiting.classList.add('hidden'); };
feed.onerror = function() {
    waiting.textContent = 'Stream disconnected — reconnecting...';
    waiting.classList.remove('hidden');
    setTimeout(function() {
        feed.src = '/stream?token=' + TOKEN + '&t=' + Date.now();
    }, 2000);
};

/* ── SSE events ─────────────────────────────────────────── */
function connectSSE() {
    var es = new EventSource('/events?token=' + TOKEN);
    es.onopen = function() {
        connDot.classList.add('connected');
        connDot.title = 'Connected';
    };
    es.onerror = function() {
        connDot.classList.remove('connected');
        connDot.title = 'Reconnecting...';
    };
    es.onmessage = function(e) { handleEvent(JSON.parse(e.data)); };
}

function handleEvent(ev) {
    switch (ev.type) {
        case 'task_start':
            taskEl.textContent = ev.task;
            stepMax.textContent = ev.max_steps;
            if (ev.screen_size) { screenW = ev.screen_size[0]; screenH = ev.screen_size[1]; }
            setState('running');
            startTime = Date.now();
            if (!timer) timer = setInterval(updateElapsed, 1000);
            addEntry('system', 'Task started');
            break;
        case 'step':
            stepNum.textContent = ev.step;
            break;
        case 'thinking':
            addEntry('thinking', ev.text, ev.step);
            break;
        case 'action':
            addActionEntry(ev);
            if (ev.action_type === 'click' || ev.action_type === 'double_click')
                showClick(ev.params.x, ev.params.y);
            break;
        case 'execution':
            if (!ev.success)
                addEntry('system', 'Failed: ' + (ev.error || 'unknown'));
            break;
        case 'done':
            setState(ev.success ? 'completed' : 'failed');
            addDoneEntry(ev);
            if (timer) { clearInterval(timer); timer = null; }
            break;
    }
}

/* ── Render entries ─────────────────────────────────────── */
function addActionEntry(ev) {
    var d = document.createElement('div');
    d.className = 'entry entry-action';
    d.innerHTML =
        '<span class="step-badge">' + ev.step + '</span>' +
        '<span class="action-type ' + ev.action_type + '">' + ev.action_type + '</span>' +
        '<span class="params">' + fmtParams(ev.action_type, ev.params) + '</span>' +
        (ev.reasoning ? '<div class="reasoning">' + esc(ev.reasoning) + '</div>' : '');
    appendLog(d);
}

function addEntry(type, text, step) {
    var d = document.createElement('div');
    d.className = 'entry entry-' + type;
    d.innerHTML = (step !== undefined ? '<span class="step-badge">' + step + '</span> ' : '') + esc(text);
    appendLog(d);
}

function addDoneEntry(ev) {
    var d = document.createElement('div');
    d.className = 'entry entry-done' + (ev.success ? '' : ' failed');
    d.innerHTML =
        '<strong>' + (ev.success ? 'Completed' : 'Failed') + '</strong>' +
        '<div style="margin-top:4px;font-size:12px;color:var(--text-2)">' + esc(ev.summary || '') + '</div>' +
        '<div style="margin-top:4px;font-size:11px;color:var(--text-3)">' +
            ev.total_steps + ' steps &middot; ' + (ev.total_time || 0).toFixed(1) + 's</div>';
    appendLog(d);
}

function appendLog(el) {
    var atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 60;
    log.appendChild(el);
    if (atBottom) el.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function fmtParams(type, p) {
    if (!p) return '';
    switch (type) {
        case 'click':
        case 'double_click':
            return '(' + p.x + ', ' + p.y + ')' + (p.button && p.button !== 'left' ? ' ' + p.button : '');
        case 'type_text':
            var t = p.text || '';
            return '"' + esc(t.length > 40 ? t.slice(0, 40) + '...' : t) + '"';
        case 'key_press':
            return p.keys || '';
        case 'scroll':
            return p.direction + (p.amount ? ' x' + p.amount : '');
        case 'drag':
            return '(' + p.start_x + ',' + p.start_y + ') -> (' + p.end_x + ',' + p.end_y + ')';
        case 'wait':
            return p.seconds ? p.seconds + 's' : '';
        case 'done':
            return p.success ? 'success' : 'failed';
        default:
            return JSON.stringify(p);
    }
}

/* ── Click overlay ──────────────────────────────────────── */
function showClick(x, y) {
    var rect = feed.getBoundingClientRect();
    if (rect.width === 0) return;
    var dx = (x / screenW) * rect.width;
    var dy = (y / screenH) * rect.height;
    var ring = document.createElement('div');
    ring.className = 'click-ring';
    ring.style.left = dx + 'px';
    ring.style.top  = dy + 'px';
    $('click-overlay').appendChild(ring);
    setTimeout(function() { ring.remove(); }, 1200);
}

/* ── Helpers ────────────────────────────────────────────── */
function setState(s) {
    statePill.className = 'state-pill ' + s;
    stateText.textContent = s.charAt(0).toUpperCase() + s.slice(1);
}

function updateElapsed() {
    if (!startTime) return;
    var secs = Math.floor((Date.now() - startTime) / 1000);
    var m = Math.floor(secs / 60);
    var s = secs % 60;
    elapsed.textContent = m + ':' + String(s).padStart(2, '0');
}

function esc(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

/* ── Init ───────────────────────────────────────────────── */
connectSSE();

// Replay history on reconnect
fetch('/api/history?token=' + TOKEN)
    .then(function(r) { return r.json(); })
    .then(function(events) { events.forEach(handleEvent); })
    .catch(function() {});
</script>
</body>
</html>
"""
