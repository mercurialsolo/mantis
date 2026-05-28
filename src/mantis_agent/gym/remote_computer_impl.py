"""`RemoteComputerImpl` — HTTPS client for the Phase 1 computer plane.

Speaks the wire contract defined in `mantis_agent.gym.computer_wire`
against a `ComputerAgent` FastAPI server (typically the
`computer_plane` Modal function defined in
`deploy/modal/modal_cua_server.py`).

The client implements `GymEnvironment` so brain code (`task_loop`, all
step handlers, the runner) need not change — once the factory dispatches
`backend="modal"` here, the rest of the call graph is identical.

Implements:

* On-demand `/session/init` from the first `reset()` call.
* Client-generated `step_id` per logical step. Retries reuse the **same**
  `step_id`; new logical steps mint a new one.
* Transient 5xx retry with exponential backoff (bounded).
* Bounded request timeout matched to the Phase 0 latency budget.

Does NOT implement (deferred to later phases):

* Multipart-binary screenshot transport — base64-in-JSON is fine for
  Phase 1 boattrader-scale workloads. Revisit if p95 screenshot RT
  becomes screenshot-payload-bound.
* Multi-region URL selection — single base URL for v1.
"""

from __future__ import annotations

import base64
import io
import logging
import secrets
import time
import uuid
from typing import Any

import requests
from PIL import Image

from ..actions import Action, ActionType
from .base import GymObservation, GymResult
from .computer_client import ComputerClient
from .computer_wire import (
    CDPRequest,
    CDPResponse,
    ScreenshotRequest,
    ScreenshotResponse,
    SessionCloseRequest,
    SessionInitRequest,
    SessionInitResponse,
    XdotoolRequest,
    XdotoolResponse,
)
from .local_xdotool_impl import LatencyTracker

logger = logging.getLogger(__name__)

# Phase 1 retry budget. The same `step_id` is reused across retries so
# the server's LRU dedup turns a successful-but-disconnected retry into
# a no-op rather than a double click.
_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = (0.25, 1.0)
_DEFAULT_HTTP_TIMEOUT = 10.0


class RemoteComputerImpl(ComputerClient):
    """`ComputerClient` impl over HTTPS.

    The kwarg surface mirrors `XdotoolGymEnv.__init__` so call sites that
    used to construct `XdotoolGymEnv(**kw)` switch to
    `make_computer_client(cfg, **kw)` with no other change.
    """

    def __init__(
        self,
        *,
        base_url: str,
        auth_token: str | None = None,
        enable_cdp: bool = False,
        tenant_id: str | None = None,
        profile_id: str | None = None,
        run_id: str | None = None,
        start_url: str = "about:blank",
        viewport: tuple[int, int] = (1280, 720),
        proxy_server: str = "",
        chrome_flags: list[str] | None = None,
        request_timeout_seconds: float = _DEFAULT_HTTP_TIMEOUT,
        # The following kwargs are accepted for parity with
        # XdotoolGymEnv.__init__ but unused server-side (computer plane
        # manages its own lifecycle).
        browser: str = "google-chrome",  # noqa: ARG002
        display: str | None = None,  # noqa: ARG002
        settle_time: float = 1.5,  # noqa: ARG002
        human_speed: bool = False,  # noqa: ARG002
        profile_dir: str = "/data/chrome-profile",  # noqa: ARG002
        save_screenshots: str = "",  # noqa: ARG002
        cdp_port: int = 9222,  # noqa: ARG002
        reuse_session: bool = False,  # noqa: ARG002
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth_token = auth_token
        self._enable_cdp = enable_cdp
        self._tenant_id = tenant_id or "default"
        self._profile_id = profile_id or "default"
        self._run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        self._start_url = start_url
        self._viewport = viewport
        self._proxy_server = proxy_server
        self._chrome_flags = chrome_flags or []
        self._timeout = request_timeout_seconds
        self._session_token: str | None = None

        self.screenshot_latency = LatencyTracker("screenshot")
        self.xdotool_latency = LatencyTracker("xdotool")

    # ── helpers ─────────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _headers(self, include_session: bool = True) -> dict[str, str]:
        h: dict[str, str] = {}
        if self._auth_token:
            h["Authorization"] = f"Bearer {self._auth_token}"
        if include_session and self._session_token:
            h["X-Mantis-Session"] = self._session_token
        return h

    def _post(
        self,
        path: str,
        json_body: dict[str, Any],
        *,
        include_session: bool = True,
        step_id_for_retry: str | None = None,
    ) -> dict[str, Any]:
        """POST with bounded retry on transient 5xx.

        Successful response → returns the parsed JSON.
        Persistent 5xx / network error → raises after exhausting retries.
        4xx → raises immediately without retry.
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self._url(path),
                    json=json_body,
                    headers=self._headers(include_session=include_session),
                    timeout=self._timeout,
                )
                if resp.status_code < 400:
                    return resp.json()
                if 400 <= resp.status_code < 500:
                    raise RuntimeError(
                        f"computer-plane {path} returned {resp.status_code}: {resp.text[:200]}"
                    )
                last_exc = RuntimeError(
                    f"computer-plane {path} returned {resp.status_code}: {resp.text[:200]}"
                )
            except requests.RequestException as exc:
                last_exc = exc

            if attempt < _MAX_RETRIES:
                backoff = _RETRY_BACKOFF_SECONDS[
                    min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)
                ]
                logger.warning(
                    "computer-plane %s attempt %d/%d failed (%s); retrying in %.2fs%s",
                    path,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    last_exc,
                    backoff,
                    f" (step_id={step_id_for_retry})" if step_id_for_retry else "",
                )
                time.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    # ── wire contract surface (used by tests + advanced callers) ────

    def session_init(self) -> SessionInitResponse:
        req = SessionInitRequest(
            tenant_id=self._tenant_id,
            profile_id=self._profile_id,
            run_id=self._run_id,
            proxy_server=self._proxy_server or None,
            chrome_flags=self._chrome_flags,
            enable_cdp=self._enable_cdp,
            viewport=self._viewport,
        )
        payload = self._post(
            "/session/init", req.model_dump(), include_session=False
        )
        resp = SessionInitResponse.model_validate(payload)
        self._session_token = resp.session_token
        return resp

    def session_close(self) -> None:
        if not self._session_token:
            return
        try:
            self._post(
                "/session/close",
                SessionCloseRequest(session_token=self._session_token).model_dump(),
                include_session=False,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort teardown
            logger.warning("session_close raised: %s", exc)
        finally:
            self._session_token = None

    def remote_screenshot(self) -> ScreenshotResponse:
        if not self._session_token:
            self.session_init()
        t0 = time.perf_counter()
        try:
            payload = self._post(
                "/screenshot", ScreenshotRequest().model_dump()
            )
            return ScreenshotResponse.model_validate(payload)
        finally:
            self.screenshot_latency.record_ms((time.perf_counter() - t0) * 1000.0)

    def remote_xdotool(self, *argv: str, step_id: str | None = None) -> XdotoolResponse:
        if not self._session_token:
            self.session_init()
        sid = step_id or f"step-{secrets.token_hex(8)}"
        req = XdotoolRequest(argv=list(argv), step_id=sid)
        t0 = time.perf_counter()
        try:
            payload = self._post(
                "/xdotool", req.model_dump(), step_id_for_retry=sid
            )
            return XdotoolResponse.model_validate(payload)
        finally:
            self.xdotool_latency.record_ms((time.perf_counter() - t0) * 1000.0)

    def remote_cdp(self, expression: str, *, await_promise: bool = False) -> CDPResponse:
        if not self._enable_cdp:
            raise RuntimeError(
                "RemoteComputerImpl.remote_cdp called but enable_cdp=False"
            )
        if not self._session_token:
            self.session_init()
        sid = f"step-{secrets.token_hex(8)}"
        req = CDPRequest(expression=expression, await_promise=await_promise, step_id=sid)
        payload = self._post("/cdp", req.model_dump())
        return CDPResponse.model_validate(payload)

    # ── GymEnvironment surface ──────────────────────────────────────

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        self.session_init()
        # Use a hard-coded "go to start_url" via xdotool typing? No — the
        # session-init server-side handles the initial navigation via the
        # XdotoolGymEnv.reset path. Just snapshot the current screen.
        return self._screenshot_observation()

    def step(self, action: Action) -> GymResult:
        """Execute an action remotely.

        Translates Mantis action types to xdotool argv via the same scheme
        the local impl uses — but the actual subprocess fires on the
        computer-plane container. A single Mantis action becomes one or
        more `xdotool` calls; each gets its own `step_id` so the server's
        LRU dedup is per-subprocess, not per-Mantis-action.
        """
        for argv in self._action_to_argv(action):
            self.remote_xdotool(*argv)
        if action.action_type == ActionType.WAIT:
            time.sleep(min(action.params.get("seconds", 1.0) or 1.0, 10.0))
        obs = self._screenshot_observation()
        return GymResult(observation=obs, reward=0.0, done=False, info={})

    def close(self) -> None:
        self.session_close()

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._viewport

    def latency_report(self) -> dict[str, dict[str, float | int]]:
        return {
            "screenshot": self.screenshot_latency.summary(),
            "xdotool": self.xdotool_latency.summary(),
        }

    # ── helpers ──────────────────────────────────────────────────────

    def _screenshot_observation(self) -> GymObservation:
        resp = self.remote_screenshot()
        img = Image.open(io.BytesIO(base64.b64decode(resp.image_b64)))
        return GymObservation(
            screenshot=img,
            extras={
                "scroll_y": resp.scroll_y,
                "captured_at_ms": resp.captured_at_ms,
                "width": resp.width,
                "height": resp.height,
            },
        )

    def _action_to_argv(self, action: Action) -> list[list[str]]:
        """Translate Mantis `Action` → ordered list of xdotool argv vectors.

        Returns a list of argv vectors (in dispatch order) because a
        single Mantis action — e.g. `CLICK` — typically maps to two
        `xdotool` calls (mousemove then click). The remote dispatcher
        sends each vector with its own `step_id`.

        Mirrors the subset used by today's `XdotoolGymEnv._execute_action`.
        Anything outside the production subset is logged and dropped
        (step still returns a screenshot, matching local-impl behavior).
        """
        p = action.params or {}
        vw, vh = self._viewport

        if action.action_type == ActionType.CLICK:
            x = int(p.get("x", vw // 2))
            y = int(p.get("y", vh // 2))
            button = p.get("button", "left")
            btn = {"left": "1", "middle": "2", "right": "3"}.get(button, "1")
            return [["mousemove", str(x), str(y)], ["click", btn]]

        if action.action_type == ActionType.DOUBLE_CLICK:
            x = int(p.get("x", vw // 2))
            y = int(p.get("y", vh // 2))
            return [
                ["mousemove", str(x), str(y)],
                ["click", "--repeat", "2", "1"],
            ]

        if action.action_type == ActionType.TYPE:
            text = str(p.get("text") or p.get("content") or "")
            if not text:
                return []
            return [["type", "--delay", "0", text]]

        if action.action_type == ActionType.KEY_PRESS:
            keys = str(p.get("keys") or p.get("key") or "")
            if not keys:
                return []
            return [["key", keys]]

        if action.action_type == ActionType.SCROLL:
            direction = p.get("direction", "down")
            amount = max(0, min(int(p.get("amount", 3) or 0), 40))
            if amount == 0:
                return []
            x = int(p.get("x", vw // 2))
            y = int(p.get("y", vh // 2))
            btn = "4" if direction == "up" else "5"
            return [
                ["mousemove", str(x), str(y)],
                ["click", "--repeat", str(amount), btn],
            ]

        if action.action_type in (ActionType.WAIT, ActionType.DONE):
            return []

        logger.warning(
            "RemoteComputerImpl: unsupported action type %s — screenshot-only step",
            action.action_type,
        )
        return []
