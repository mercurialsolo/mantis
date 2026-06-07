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
    CdpClickAtPointRequest,
    CdpClickAtPointResponse,
    CdpCountPagesResponse,
    CurrentUrlResponse,
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
        # Honored — forwarded to the remote `SessionInitRequest` so the
        # remote Chrome lands at the right page on the right on-volume
        # profile with the brain-supplied header set.
        profile_dir: str | None = None,
        extra_http_headers: dict[str, str] | None = None,
        # The following kwargs are accepted for parity with
        # XdotoolGymEnv.__init__ but unused server-side (computer plane
        # manages its own lifecycle).
        browser: str = "google-chrome",  # noqa: ARG002
        display: str | None = None,  # noqa: ARG002
        settle_time: float = 1.5,  # noqa: ARG002
        human_speed: bool = False,  # noqa: ARG002
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
        self._profile_dir = profile_dir
        self._extra_http_headers = extra_http_headers or None
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

    def _get(self, path: str) -> dict[str, Any]:
        """GET with bounded retry on transient 5xx — mirrors `_post`.

        Used by the read-only proxies (`/current_url`, `/cdp_count_pages`)
        where there's nothing to serialize.
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    self._url(path),
                    headers=self._headers(),
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
            start_url=self._start_url,
            profile_dir=self._profile_dir,
            extra_http_headers=self._extra_http_headers,
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

    def shutdown(self) -> None:
        """Alias for `close` — matches `XdotoolGymEnv.shutdown` which the
        lifecycle code in `run_executor_lifecycle` calls."""
        self.close()

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._viewport

    def latency_report(self) -> dict[str, dict[str, float | int]]:
        return {
            "screenshot": self.screenshot_latency.summary(),
            "xdotool": self.xdotool_latency.summary(),
        }

    # ── XdotoolGymEnv-compatibility surface ─────────────────────────
    #
    # Step handlers and the runner reach past `GymEnvironment` ABC into
    # concrete `XdotoolGymEnv` methods. The migration's correctness
    # rests on these proxies returning the same shapes the local impl
    # does — every divergence becomes a brain-side crash at runtime.

    def screenshot(self) -> Image.Image:
        """PIL Image matching `XdotoolGymEnv.screenshot()` shape.

        Handlers call `env.screenshot()` (not `env._capture()`) for the
        ad-hoc post-action verify; same image bytes as
        `_screenshot_observation()` so callers can mix the two.
        """
        return self._screenshot_observation().screenshot  # type: ignore[return-value]

    def _capture(self) -> GymObservation:
        """Alias for `_screenshot_observation` — some handlers poke at
        the private name directly (mirrors `XdotoolGymEnv._capture`).
        """
        return self._screenshot_observation()

    @property
    def current_url(self) -> str:
        """Active Chrome tab URL via `/current_url`. Empty string when
        no page has loaded yet — never `None`, matching the local impl.
        """
        if not self._session_token:
            self.session_init()
        try:
            payload = self._get("/current_url")
            return CurrentUrlResponse.model_validate(payload).url or ""
        except Exception as exc:  # noqa: BLE001 — observability, never fatal
            logger.warning("RemoteComputerImpl.current_url failed: %s", exc)
            return ""

    def cdp_evaluate(self, expression: str) -> Any:
        """Run a JS expression via the remote `/cdp` Runtime.evaluate.

        Returns the unwrapped value (any JSON-serializable type) or
        `None` on failure — matches `XdotoolGymEnv.cdp_evaluate`'s
        contract. Requires `enable_cdp=True` on this session (raises
        `RuntimeError` otherwise — caller treats absent CDP as
        feature-not-available).
        """
        if not self._enable_cdp:
            raise RuntimeError(
                "RemoteComputerImpl.cdp_evaluate called but enable_cdp=False"
            )
        if not self._session_token:
            self.session_init()
        sid = f"step-{secrets.token_hex(8)}"
        req = CDPRequest(expression=expression, await_promise=False, step_id=sid)
        try:
            payload = self._post("/cdp", req.model_dump())
        except Exception as exc:  # noqa: BLE001
            logger.debug("RemoteComputerImpl.cdp_evaluate raised: %s", exc)
            return None
        resp = CDPResponse.model_validate(payload)
        if resp.returncode != 0:
            return None
        import json as _json
        try:
            value = _json.loads(resp.result_json or "{}")
        except _json.JSONDecodeError:
            return None
        # `Runtime.evaluate` returns `{result: {type, value}}`; unwrap
        # to mirror `XdotoolGymEnv.cdp_evaluate`.
        return (value.get("result") or {}).get("value") if isinstance(value, dict) else None

    def cdp_click_at_point(self, x: int, y: int) -> bool:
        """SoM-anchored click via remote `/cdp_click_at_point`.

        Returns the server's `ok` flag — `True` iff an element was found
        at (x, y) and the synthetic click was dispatched.
        """
        return self._remote_click(x, y, via_pointer=False)

    def cdp_click_via_pointer(self, x: int, y: int) -> bool:
        """Real-pointer click via `Input.dispatchMouseEvent`.

        Same endpoint as `cdp_click_at_point` with `via_pointer=True`;
        needed when sites gate on `isTrusted=true` (#audit batch).
        """
        return self._remote_click(x, y, via_pointer=True)

    def _remote_click(self, x: int, y: int, *, via_pointer: bool) -> bool:
        if not self._enable_cdp:
            raise RuntimeError(
                "RemoteComputerImpl click via CDP requires enable_cdp=True"
            )
        if not self._session_token:
            self.session_init()
        sid = f"step-{secrets.token_hex(8)}"
        req = CdpClickAtPointRequest(
            x=int(x), y=int(y), via_pointer=via_pointer, step_id=sid,
        )
        try:
            payload = self._post(
                "/cdp_click_at_point", req.model_dump(), step_id_for_retry=sid,
            )
        except Exception as exc:  # noqa: BLE001 — fall back like the local impl
            logger.debug("RemoteComputerImpl click failed: %s", exc)
            return False
        return bool(CdpClickAtPointResponse.model_validate(payload).ok)

    def _chrome_offset_px(self) -> int:
        """Chrome chrome-offset (outerHeight - innerHeight) via CDP.

        Used by callers translating screen coords to viewport coords.
        Returns 0 when CDP is unavailable, matching the local impl.
        """
        try:
            value = self.cdp_evaluate(
                "Math.max(0, window.outerHeight - window.innerHeight)"
            )
        except Exception:  # noqa: BLE001
            return 0
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    def cdp_count_pages(self) -> int:
        """Number of open Chrome `type=page` tabs whose URL isn't a
        system page — proxies `/cdp_count_pages`. Returns 0 on failure.
        """
        if not self._session_token:
            self.session_init()
        try:
            payload = self._get("/cdp_count_pages")
            return int(CdpCountPagesResponse.model_validate(payload).count)
        except Exception as exc:  # noqa: BLE001
            logger.debug("RemoteComputerImpl.cdp_count_pages failed: %s", exc)
            return 0

    def cdp_history_back(self, *, settle_seconds: float = 1.5) -> bool:
        """Navigate back via `window.history.back()` over CDP — verifies
        the URL changed within `settle_seconds`. Mirrors
        `XdotoolGymEnv.cdp_history_back`.
        """
        if not self._enable_cdp:
            return False
        url_before = self.current_url or ""
        try:
            self.cdp_evaluate("window.history.back()")
        except Exception as exc:  # noqa: BLE001
            logger.debug("cdp_history_back dispatch failed: %s", exc)
            return False
        deadline = time.time() + max(0.1, settle_seconds)
        while time.time() < deadline:
            time.sleep(0.1)
            try:
                url_after = self.current_url or ""
            except Exception:  # noqa: BLE001
                url_after = ""
            if url_after and url_after != url_before:
                return True
        return False

    def capture_browser_state(self) -> dict[str, Any]:
        """Stub matching `XdotoolGymEnv.capture_browser_state` shape.

        Full browser-state capture is brain-side bookkeeping; the
        remote doesn't own it. Returning `{}` keeps call sites that
        spread the result safe.
        """
        return {}

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
