"""Minimal stub env — proves the harness contract end-to-end without a real env.

Implements just enough of the ``/__env__/*`` surface to (a) satisfy the
acceptance criteria of #336 and (b) let the integration layer be tested
without standing up a 50k-row CRM. Uses the stdlib ``http.server`` so the
slim install (just Pillow) can boot the stub — no FastAPI dependency.

Endpoints (mirror docs/envs/SPEC.md §"Shared harness contract"):

* ``GET  /``             — bare HTML "stub env" page (the agent sees this)
* ``GET  /__env__/health``  — ``{"ok": true, "seed": <int>, "now": <iso>}``
* ``POST /__env__/reset``   — resets in-memory state; idempotent
* ``POST /__env__/seed``    — re-seed with body ``{"seed": int}``
* ``POST /__env__/clock``   — re-set clock with body ``{"now": iso8601}``
* ``GET  /__env__/oracle``  — query param ``task_id``; returns canned grade
* ``GET  /__env__/state``   — full in-memory state JSON
* ``GET  /__env__/events``  — newline-delimited events emitted since boot

Admin-token middleware:
  Every ``/__env__/*`` route except ``/__env__/health`` requires the
  ``X-Env-Admin`` header to match ``ENV_ADMIN_TOKEN``. Missing or wrong
  token returns 401 with body ``{"error": "admin token required"}``.
  ``/__env__/health`` is intentionally open so health-poll loops don't
  need the token plumbed in — health is non-sensitive.

The stub returns ``passed=true`` for any ``task_id`` so the integration
layer has a non-trivial oracle response to assert against. Real envs
land their own oracle logic; the stub only exists to prove plumbing.

Run standalone for local testing::

    python -m mantis_agent.sim_envs.stub_app --port 8001 \\
        --admin-token devtoken --seed 42

Or import the handler factory for in-process tests:

    from mantis_agent.sim_envs.stub_app import build_handler
    handler = build_handler(admin_token="t", seed=42, now="2026-01-15T09:00:00Z")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

# Path of the agent-facing landing page. Anything under this prefix is
# the SPA. ``/__env__/*`` is the harness surface and never returned to
# the agent's browser context (the admin-token check enforces this).
AGENT_LANDING_HTML = (
    "<!doctype html><html><head><title>mantis stub env</title></head>"
    "<body><h1>mantis stub env</h1><p>This is a stub. See "
    "<code>docs/envs/SPEC.md</code> for the real contract.</p></body></html>"
)


class _EnvState:
    """In-memory state for the stub. Real envs replace this with a DB."""

    def __init__(self, seed: int, now: str, admin_token: str) -> None:
        self.seed = seed
        self.now = now
        self.admin_token = admin_token
        self.boot_time = time.time()
        # Each entry: {"ts": float, "event": str, "data": dict}
        self.events: list[dict[str, Any]] = []
        # Touched flag so tests can verify reset cleared state.
        self.touched = False
        self._lock = threading.Lock()

    def emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.events.append({
                "ts": time.time(),
                "event": event,
                "data": data or {},
            })

    def reset(self) -> None:
        with self._lock:
            self.events.clear()
            self.touched = False
            self.boot_time = time.time()


def build_handler(
    *,
    admin_token: str,
    seed: int = 42,
    now: str = "2026-01-15T09:00:00Z",
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class closed over a fresh :class:`_EnvState`.

    Returned as a class because :class:`HTTPServer` instantiates one
    handler per request — the state has to live on a closure, not on
    ``self``.
    """

    state = _EnvState(seed=seed, now=now, admin_token=admin_token)

    class StubHandler(BaseHTTPRequestHandler):
        # http.server logs to stderr by default; quiet it down so test
        # output stays readable. Operators wanting verbose logs can set
        # MANTIS_STUB_ENV_VERBOSE=1.
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            if os.environ.get("MANTIS_STUB_ENV_VERBOSE"):
                super().log_message(format, *args)

        # ── helpers ─────────────────────────────────────────────────
        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, status: int, body_str: str) -> None:
            body = body_str.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        def _require_admin(self) -> bool:
            token = self.headers.get("X-Env-Admin", "")
            if token == state.admin_token:
                return True
            self._json(401, {"error": "admin token required"})
            return False

        # ── routing ─────────────────────────────────────────────────
        def do_GET(self) -> None:  # noqa: N802 — http.server contract
            parsed = urlparse(self.path)
            path = parsed.path

            # Health is open by design — health checks can't authenticate.
            if path == "/__env__/health":
                self._json(200, {
                    "ok": True,
                    "seed": state.seed,
                    "now": state.now,
                    "boot_time": state.boot_time,
                })
                return

            if path.startswith("/__env__/"):
                if not self._require_admin():
                    return

            if path == "/__env__/state":
                self._json(200, {
                    "seed": state.seed,
                    "now": state.now,
                    "touched": state.touched,
                    "event_count": len(state.events),
                })
                return

            if path == "/__env__/events":
                qs = parse_qs(parsed.query)
                since = float(qs.get("since", ["0"])[0])
                filtered = [e for e in state.events if e["ts"] >= since]
                self._json(200, {"events": filtered})
                return

            if path == "/__env__/oracle":
                qs = parse_qs(parsed.query)
                task_id = (qs.get("task_id", [""])[0] or "").strip()
                state.emit("oracle_query", {"task_id": task_id})
                # Stub: every task passes. Real envs encode per-task
                # ground-truth assertions over the DB.
                self._json(200, {
                    "passed": True,
                    "score": 1.0,
                    "task_id": task_id,
                    "reasons": ["stub env: every task passes"],
                    "diff": {},
                })
                return

            # Agent-facing surface: anything else returns the landing page.
            self._html(200, AGENT_LANDING_HTML)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path

            if path.startswith("/__env__/"):
                if not self._require_admin():
                    return

            if path == "/__env__/reset":
                state.reset()
                self._json(200, {"ok": True})
                return

            if path == "/__env__/seed":
                body = self._read_json_body()
                new_seed = int(body.get("seed", state.seed))
                state.seed = new_seed
                state.emit("seed_changed", {"seed": new_seed})
                self._json(200, {"ok": True, "seed": new_seed})
                return

            if path == "/__env__/clock":
                body = self._read_json_body()
                new_now = str(body.get("now") or state.now)
                state.now = new_now
                state.emit("clock_changed", {"now": new_now})
                self._json(200, {"ok": True, "now": new_now})
                return

            self._json(404, {"error": f"no route for POST {path}"})

    return StubHandler


def serve(host: str, port: int, *, admin_token: str, seed: int, now: str) -> HTTPServer:
    """Boot a stub-env HTTP server bound to ``host:port`` and return it.

    The server runs in its own thread so the caller (tests, env_up.py)
    can do other work while it serves. Call :meth:`HTTPServer.shutdown`
    to stop cleanly.
    """
    handler_cls = build_handler(admin_token=admin_token, seed=seed, now=now)
    server = HTTPServer((host, port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name=f"stub-env-{port}")
    thread.start()
    logger.info("stub env serving at http://%s:%d", host, port)
    return server


def main() -> None:
    """CLI entrypoint: ``python -m mantis_agent.sim_envs.stub_app ...``."""
    parser = argparse.ArgumentParser(description="mantis stub simulated env")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--admin-token", default=None,
                        help="Admin token for /__env__/*. Default: ENV_ADMIN_TOKEN env var.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--now", default="2026-01-15T09:00:00Z")
    args = parser.parse_args()

    admin_token = args.admin_token or os.environ.get("ENV_ADMIN_TOKEN")
    if not admin_token:
        raise SystemExit("error: --admin-token or ENV_ADMIN_TOKEN must be set")

    server = serve(
        host=args.host,
        port=args.port,
        admin_token=admin_token,
        seed=args.seed,
        now=args.now,
    )
    try:
        # Block forever — Ctrl-C exits.
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
