"""Per-session computer-plane orchestration (Phase 1.5, #846).

The body of the long-lived ``computer_session`` Modal function lives
here so it can be imported into the Modal app module without
inlining hundreds of lines into ``deploy/modal/modal_cua_server.py``,
and so unit tests can drive it through ``runpy``-style scaffolding.

**Lifecycle:**

1. Brain calls ``POST /v1/computer_sessions`` on the router.
2. Router ``.spawn()``s this function with the session payload.
3. This function:
   a. Builds the ``ComputerAgent`` FastAPI app.
   b. Pre-binds the session into the agent's module-level state (so
      the brain doesn't need a separate ``/session/init`` hop — it
      already has a ``session_token`` returned by the router).
   c. Starts uvicorn on port 8090 in a daemon thread.
   d. Opens ``modal.forward(8090)`` to mint a tunnel URL.
   e. Publishes ``{tunnel_url, session_token, status="ready"}`` into
      the shared ``session_dict`` keyed by ``session_id``.
   f. Polls the dict for ``close_requested=True``, with a TTL ceiling.
4. On exit (close OR TTL OR error): tears Chrome down, writes a
   terminal record, returns.

The ``session_dict`` argument is dependency-injected so unit tests can
pass a plain ``dict`` and exercise the loop without Modal.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# Tunnel-URL polling backoff. The router spins on the same dict
# entry; values chosen so a cold-start (~30s) ticks ~30 times before
# we declare it stuck.
_TUNNEL_PUBLISH_RETRY = 0.5  # seconds between dict writes if the entry vanished
_CLOSE_POLL_INTERVAL = 2.0   # seconds between close-sentinel polls inside session


def _bind_session_inline(
    session_token: str,
    init_payload: dict,
) -> tuple[Any, Any]:
    """Stand up Chrome + Xvfb and register the session in
    ``ComputerAgentState`` so subsequent HTTP calls see it.

    Returns ``(env, session)`` for the caller to hold a reference to —
    needed because the FastAPI app reads the module-level ``_state``
    but the env teardown happens via ``env.shutdown()`` on exit.

    Phase 1.5 (#846) note: this path runs a *cold* Chrome on every
    session — unlike Phase 1's pinned ``computer_plane()`` ASGI app
    which has a warm Chrome ready. After env construction we poll
    Chrome's CDP readiness so we never publish the tunnel URL until
    Chrome has actually finished loading ``start_url``. Without this
    the brain's first ``screenshot()`` lands on a blank/transient
    paint and Claude extracts nothing.
    """
    # Imports are inline so this module stays importable in test
    # contexts that don't have FastAPI / the X server libraries.
    from ..gym.computer_wire import SessionInitRequest
    from .computer_agent import _Session, _new_xdotool_env, _state

    req = SessionInitRequest.model_validate(init_payload)
    env = _new_xdotool_env(req)
    # Cold-start headroom — Chrome's first paint after a fresh
    # container boot lags 3-6s past ``env.reset``'s built-in settle
    # (1.5s + 2s = 3.5s). Without this, the brain's first screenshot
    # lands on a blank/transient paint and Claude extracts nothing.
    # The active CDP probe ``_await_chrome_ready`` we tried first
    # interfered with Chrome's network stack on a cold container, so
    # we fall back to a quiet sleep here. Total publish budget then
    # is roughly 3.5s (settle) + 8s (this) + 2s (uvicorn settle) =
    # ~13.5s on first-time spawn; warm containers see the same wait
    # but the brain still gets a fully-painted page.
    time.sleep(8.0)
    session = _Session(
        token=session_token,
        tenant_id=req.tenant_id,
        profile_id=req.profile_id,
        run_id=req.run_id,
        enable_cdp=req.enable_cdp,
        viewport=tuple(req.viewport),  # type: ignore[arg-type]
        env=env,
    )
    with _state.lock:
        _state.session = session
    return env, session


def _await_chrome_ready(env: Any, expected_url: str, *,
                         timeout_seconds: float = 15.0) -> None:
    """Poll Chrome via CDP until it reports the expected URL is
    actively rendering, OR until the timeout expires.

    * ``expected_url`` empty / ``about:blank`` → wait for any
      non-empty URL Chrome reports. Covers the "navigate later via
      xdotool" path.
    * Otherwise → wait for ``current_url``'s host to match the
      expected host (URL slugs may differ from start_url after a
      redirect or hash-fragment normalize).
    * Best-effort — CDP failure / timeout logs a warning and returns
      so a wedged Chrome still surfaces as a router-side timeout
      rather than hanging the orchestrator forever.
    """
    import time as _time
    from urllib.parse import urlparse

    expected_host = ""
    if expected_url:
        try:
            expected_host = (urlparse(expected_url).hostname or "").lower()
        except Exception:  # noqa: BLE001
            expected_host = ""

    deadline = _time.time() + max(1.0, timeout_seconds)
    while _time.time() < deadline:
        try:
            current = (env.current_url or "").lower()
        except Exception:  # noqa: BLE001
            current = ""
        if not expected_host:
            if current and not current.startswith(("about:", "chrome:")):
                return
        else:
            try:
                current_host = (urlparse(current).hostname or "").lower()
            except Exception:  # noqa: BLE001
                current_host = ""
            if current_host == expected_host:
                # Cheap insurance — one more tick past first-paint so
                # the brain's first screenshot is past FOUC/spinners.
                _time.sleep(0.8)
                return
        _time.sleep(0.5)
    logger.warning(
        "session orchestrator: Chrome not ready after %.0fs (expected=%s)",
        timeout_seconds, expected_url[:80] if expected_url else "<empty>",
    )


def _start_uvicorn_in_thread(app: Any, port: int) -> tuple[threading.Thread, Any]:
    """Run uvicorn on ``port`` in a daemon thread.

    Returns the thread and the server handle so callers can stop it
    cleanly on exit. uvicorn import is inline so the module loads
    in test contexts without uvicorn installed.
    """
    import uvicorn

    config = uvicorn.Config(
        app, host="0.0.0.0", port=port, log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="computer-agent-uvicorn")
    thread.start()
    return thread, server


def run_session_loop(
    *,
    session_id: str,
    session_token: str,
    init_payload: dict,
    ttl_seconds: int,
    session_dict: Any,  # modal.Dict in prod, dict in tests
    tunnel_url: str,    # publishes to session_dict[session_id]['tunnel_url']
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.time,
    publish_only: bool = False,
) -> dict:
    """Publish tunnel URL + loop until close or TTL.

    Pure logic — no Modal calls, no Chrome lifecycle. The container-
    level orchestrator (``computer_session`` in modal_cua_server.py)
    wraps Chrome/Xvfb startup around this. Splitting them lets us
    test the publishing + loop logic end-to-end with plain dicts.

    ``publish_only=True`` short-circuits the loop after the initial
    publish — used by unit tests so they don't have to wait for TTL.
    """
    started = now()
    initial_entry = {
        "session_id": session_id,
        "session_token": session_token,
        "tunnel_url": tunnel_url,
        "status": "ready",
        "started_at_ms": int(started * 1000),
        "ttl_seconds": ttl_seconds,
        "tenant_id": init_payload.get("tenant_id", ""),
        "profile_id": init_payload.get("profile_id", ""),
        "run_id": init_payload.get("run_id", ""),
    }
    try:
        session_dict[session_id] = initial_entry
    except Exception as exc:  # noqa: BLE001 — store fault ≠ runtime fault
        logger.warning(
            "session_dict publish failed for %s: %s", session_id, exc,
        )
        # Caller's reaper / brain will time out on the router side.
        return {"session_id": session_id, "status": "publish_failed", "error": str(exc)}

    if publish_only:
        return {"session_id": session_id, "status": "ready"}

    while True:
        elapsed = now() - started
        if elapsed > ttl_seconds:
            _safe_update_entry(session_dict, session_id, {
                "status": "expired",
                "terminal_at_ms": int(now() * 1000),
            })
            return {"session_id": session_id, "status": "expired"}

        entry = _safe_get_entry(session_dict, session_id)
        if isinstance(entry, dict) and entry.get("close_requested"):
            _safe_update_entry(session_dict, session_id, {
                "status": "closed",
                "terminal_at_ms": int(now() * 1000),
            })
            return {"session_id": session_id, "status": "closed"}

        sleep(_CLOSE_POLL_INTERVAL)


# ── small helpers — make session_dict reads/writes resilient ──


def _safe_get_entry(session_dict: Any, session_id: str) -> Optional[dict]:
    try:
        return session_dict.get(session_id)
    except Exception:  # noqa: BLE001
        return None


def _safe_update_entry(session_dict: Any, session_id: str, patch: dict) -> None:
    """Read-modify-write — best-effort. Loses to a concurrent writer
    in the rare case both router and orchestrator race; the close
    path tolerates that (terminal state is sticky in the RunStateStore
    via the #866 first-terminal-wins guard)."""
    try:
        cur = session_dict.get(session_id) or {}
        if not isinstance(cur, dict):
            cur = {}
        cur.update(patch)
        session_dict[session_id] = cur
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "session_dict update failed for %s: %s", session_id, exc,
        )
