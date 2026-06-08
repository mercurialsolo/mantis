"""HTTPS client for the Browser-Use Plane Modal function (#785, PR 2).

Implements the **base surface** of the unified `ComputeClient` contract
(session_init, session_close, screenshot, dispatch_action, health) over
HTTPS against a Browser-Use Plane FastAPI host.

PR 3-4 add the DOM-aware extension methods (`state.*`, `tabs.*`,
`links.*`). They are kept off this class until then so PR 2 ships a
small, testable scaffold.

The class deliberately mirrors the surface shape of
`RemoteComputerImpl` (the Computer-Plane HTTPS client) so that a future
brain-plane refactor can swap them via the unified factory
(`compute_factory.make_compute_client`).
"""

from __future__ import annotations

import base64
import logging
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import requests
from PIL import Image

from .base import GymObservation, GymResult
from .browser_use_wire import (
    BrowserUseHealthResponse,
    BrowserUseScreenshotResponse,
    BrowserUseSessionCloseRequest,
    BrowserUseSessionInitRequest,
    BrowserUseSessionInitResponse,
    DispatchActionRequest,
    DispatchActionResponse,
    LinksPeekTargetRequest,
    LinksPeekTargetResponse,
    StateClipboardResponse,
    StateCurrentUrlResponse,
    StateFocusedElementResponse,
    StatePageLoadResponse,
    StateSafeBackRequest,
    StateSafeBackResponse,
    StateTabsResponse,
    TabsActivateRequest,
    TabsActivateResponse,
    TabsCloseRequest,
    TabsCloseResponse,
    TabsOpenInNewRequest,
    TabsOpenInNewResponse,
)
from .compute_contract import Capabilities
from .computer_client import ComputerClient

logger = logging.getLogger(__name__)


@dataclass
class _ClientState:
    session_token: str | None = None
    last_capabilities: Capabilities | None = None
    last_url: str = ""


class BrowserUsePlaneClient(ComputerClient):
    """`ComputeClient` impl for the Browser-Use Plane Modal function.

    Mirrors `RemoteComputerImpl` shape so the brain plane is plane-agnostic
    at the call site. Capabilities are read from `session/init` and made
    available via `capabilities()` for downstream `CapabilityAllowlist`
    enforcement (`compute_contract.CapabilityAllowlist`).
    """

    def __init__(
        self,
        *,
        base_url: str,
        auth_token: str | None = None,
        tenant_id: str = "default",
        profile_id: str = "default",
        run_id: str | None = None,
        proxy_server: str | None = None,
        start_url: str = "about:blank",
        profile_dir: str | None = None,
        viewport: tuple[int, int] = (1280, 720),
        extra_http_headers: dict[str, str] | None = None,
        headless: bool = True,
        request_timeout_s: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth_token = auth_token
        self._tenant_id = tenant_id
        self._profile_id = profile_id
        self._run_id = run_id or uuid.uuid4().hex
        self._proxy_server = proxy_server
        self._start_url = start_url
        self._profile_dir = profile_dir
        self._viewport = viewport
        self._extra_http_headers = extra_http_headers
        self._headless = headless
        self._timeout = request_timeout_s
        self._http = requests.Session()
        self._state = _ClientState()

    # ── ComputeClient base surface ────────────────────────────────

    def reset(self, task: str, **_kw: Any) -> GymObservation:
        """Open or reuse a session; return the initial screenshot.

        Mirrors `GymEnvironment.reset`. The first call performs
        `session/init`; subsequent calls reuse the bound session and
        re-screenshot.
        """
        if self._state.session_token is None:
            self._session_init()
        png = self._screenshot_bytes()
        img = Image.open(BytesIO(png))
        return GymObservation(screenshot=img, extras={"url": self._state.last_url})

    def step(self, action: Any) -> GymResult:
        """Dispatch a structured action; return new screenshot.

        Mantis `Action` types map to `DispatchActionRequest` kinds. The
        translation is deliberately minimal in PR 2 — full action-type
        coverage lands when PR 3-4 wire handlers through.
        """
        kind, params = self._translate_action(action)
        self._dispatch(kind=kind, params=params)
        png = self._screenshot_bytes()
        img = Image.open(BytesIO(png))
        return GymResult(
            observation=GymObservation(screenshot=img, extras={"url": self._state.last_url}),
            reward=0.0,
            done=False,
            info={},
        )

    def close(self) -> None:
        if self._state.session_token is None:
            return
        try:
            self._session_close()
        finally:
            self._http.close()
            self._state = _ClientState()

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._viewport

    # ── ComputeClient extras ──────────────────────────────────────

    def capabilities(self) -> Capabilities:
        """Last-advertised capabilities. Falls back to the Browser-Use
        Plane default if `session/init` has not run."""
        if self._state.last_capabilities is not None:
            return self._state.last_capabilities
        return Capabilities.for_browser_use_plane()

    # ── HTTP plumbing ─────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self._auth_token:
            h["Authorization"] = f"Bearer {self._auth_token}"
        if self._state.session_token:
            h["X-Mantis-Session"] = self._state.session_token
        return h

    def _session_init(self) -> None:
        req = BrowserUseSessionInitRequest(
            tenant_id=self._tenant_id,
            profile_id=self._profile_id,
            run_id=self._run_id,
            proxy_server=self._proxy_server,
            viewport=self._viewport,
            start_url=self._start_url,
            profile_dir=self._profile_dir,
            extra_http_headers=self._extra_http_headers,
            headless=self._headless,
        )
        resp = self._http.post(
            f"{self._base_url}/session/init",
            json=req.model_dump(),
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        parsed = BrowserUseSessionInitResponse.model_validate(resp.json())
        self._state.session_token = parsed.session_token
        self._state.last_capabilities = parsed.resolved_capabilities()
        logger.warning(
            "[browser-use] session init ok token=%s caps=%s",
            parsed.session_token[:8],
            self._state.last_capabilities.as_dict(),
        )

    def _session_close(self) -> None:
        assert self._state.session_token is not None
        req = BrowserUseSessionCloseRequest(session_token=self._state.session_token)
        try:
            self._http.post(
                f"{self._base_url}/session/close",
                json=req.model_dump(),
                headers=self._headers(),
                timeout=self._timeout,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort teardown
            logger.warning("[browser-use] session close failed: %s", exc)

    def _screenshot_bytes(self) -> bytes:
        resp = self._http.post(
            f"{self._base_url}/screenshot",
            json={"format": "png"},
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        parsed = BrowserUseScreenshotResponse.model_validate(resp.json())
        return base64.b64decode(parsed.image_b64)

    def _dispatch(self, *, kind: str, params: dict[str, Any]) -> None:
        step_id = uuid.uuid4().hex
        req = DispatchActionRequest(kind=kind, params=params, step_id=step_id)
        resp = self._http.post(
            f"{self._base_url}/dispatch",
            json=req.model_dump(),
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        parsed = DispatchActionResponse.model_validate(resp.json())
        if not parsed.ok:
            raise RuntimeError(f"dispatch {kind} failed: {parsed.error}")

    def health(self) -> BrowserUseHealthResponse:
        resp = self._http.get(
            f"{self._base_url}/health", headers=self._headers(), timeout=self._timeout
        )
        resp.raise_for_status()
        return BrowserUseHealthResponse.model_validate(resp.json())

    # ── state.* extensions (#778) ─────────────────────────────────
    # Implements `SupportsBrowserState`. Capability-gated behind
    # `dom_aware`; handlers MUST enforce the capability allowlist before
    # calling these. The client itself does not enforce (single
    # responsibility) — fail-loud lives in the handler layer.

    def state_current_url(self) -> str:
        self._ensure_session()
        resp = self._http.get(
            f"{self._base_url}/state/current_url",
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        parsed = StateCurrentUrlResponse.model_validate(resp.json())
        self._state.last_url = parsed.url
        return parsed.url

    def state_tabs(self) -> list[dict[str, Any]]:
        self._ensure_session()
        resp = self._http.get(
            f"{self._base_url}/state/tabs",
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        parsed = StateTabsResponse.model_validate(resp.json())
        return [t.model_dump() for t in parsed.tabs]

    def state_focused_element(self) -> dict[str, Any] | None:
        self._ensure_session()
        resp = self._http.get(
            f"{self._base_url}/state/focused_element",
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        parsed = StateFocusedElementResponse.model_validate(resp.json())
        if parsed.element is None:
            return None
        return parsed.element.model_dump()

    def state_clipboard(self) -> str:
        self._ensure_session()
        resp = self._http.get(
            f"{self._base_url}/state/clipboard",
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return StateClipboardResponse.model_validate(resp.json()).text

    def state_page_load(self) -> dict[str, Any]:
        self._ensure_session()
        resp = self._http.get(
            f"{self._base_url}/state/page_load",
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return StatePageLoadResponse.model_validate(resp.json()).model_dump()

    def state_safe_back(
        self, pinned_origin: str | None = None
    ) -> dict[str, Any]:
        self._ensure_session()
        step_id = uuid.uuid4().hex
        req = StateSafeBackRequest(pinned_origin=pinned_origin, step_id=step_id)
        resp = self._http.post(
            f"{self._base_url}/state/safe_back",
            json=req.model_dump(),
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        parsed = StateSafeBackResponse.model_validate(resp.json())
        if parsed.popped and parsed.new_url:
            self._state.last_url = parsed.new_url
        return parsed.model_dump()

    def _ensure_session(self) -> None:
        """Lazy session init for state.* reads that happen before any
        screenshot/dispatch. Mirrors `reset()`'s implicit init path."""
        if self._state.session_token is None:
            self._session_init()

    # ── tabs.* extensions (#779) ─────────────────────────────────
    # Implements `SupportsTabs`. Capability-gated behind `dom_aware`.

    def tabs_open_in_new(
        self,
        url: str | None = None,
        *,
        via_selector: str | None = None,
    ) -> str:
        """Open a new tab; return its server-side `tab_id`.

        Pass `url` to navigate directly. Pass `via_selector` to open
        the anchor at that CSS selector in a new tab (modifier-aware
        click — handles `target=_blank` and `Cmd/Ctrl-click`).
        """
        self._ensure_session()
        step_id = uuid.uuid4().hex
        req = TabsOpenInNewRequest(
            url=url, via_selector=via_selector, step_id=step_id
        )
        resp = self._http.post(
            f"{self._base_url}/tabs/open_in_new",
            json=req.model_dump(),
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        parsed = TabsOpenInNewResponse.model_validate(resp.json())
        return parsed.tab_id

    def tabs_close(self, tab_id: str) -> None:
        self._ensure_session()
        step_id = uuid.uuid4().hex
        req = TabsCloseRequest(tab_id=tab_id, step_id=step_id)
        resp = self._http.post(
            f"{self._base_url}/tabs/close",
            json=req.model_dump(),
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        TabsCloseResponse.model_validate(resp.json())

    def tabs_activate(self, tab_id: str) -> None:
        self._ensure_session()
        step_id = uuid.uuid4().hex
        req = TabsActivateRequest(tab_id=tab_id, step_id=step_id)
        resp = self._http.post(
            f"{self._base_url}/tabs/activate",
            json=req.model_dump(),
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        parsed = TabsActivateResponse.model_validate(resp.json())
        if parsed.activated and parsed.url:
            self._state.last_url = parsed.url

    # ── links.peek_target (#780) ─────────────────────────────────
    # Implements `SupportsLinkPeek`. Capability-gated behind `dom_aware`.

    def links_peek_target(self, selector_or_bbox: Any) -> str | None:
        """Read anchor `href` without clicking.

        `selector_or_bbox` is either a CSS selector string or a 4-tuple
        `(x1, y1, x2, y2)` (vision bbox). The server walks up to the
        nearest anchor element from a bbox centroid hit.
        """
        self._ensure_session()
        selector: str | None = None
        bbox: list[int] | None = None
        if isinstance(selector_or_bbox, str):
            selector = selector_or_bbox
        elif isinstance(selector_or_bbox, (list, tuple)) and len(selector_or_bbox) == 4:
            bbox = [int(v) for v in selector_or_bbox]
        else:
            raise TypeError(
                f"links_peek_target expected str | 4-tuple; got {type(selector_or_bbox)}"
            )
        req = LinksPeekTargetRequest(selector=selector, bbox=bbox)
        resp = self._http.post(
            f"{self._base_url}/links/peek_target",
            json=req.model_dump(),
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        parsed = LinksPeekTargetResponse.model_validate(resp.json())
        return parsed.href

    def links_peek_target_full(
        self, selector_or_bbox: Any
    ) -> dict[str, Any]:
        """Full projection — `{href, target, tag}` — for handlers that
        need the `target` attribute (`_blank`/`_self`) to decide
        new-tab vs current-tab semantics.
        """
        self._ensure_session()
        selector: str | None = None
        bbox: list[int] | None = None
        if isinstance(selector_or_bbox, str):
            selector = selector_or_bbox
        elif isinstance(selector_or_bbox, (list, tuple)) and len(selector_or_bbox) == 4:
            bbox = [int(v) for v in selector_or_bbox]
        else:
            raise TypeError(
                f"links_peek_target_full expected str | 4-tuple; got {type(selector_or_bbox)}"
            )
        req = LinksPeekTargetRequest(selector=selector, bbox=bbox)
        resp = self._http.post(
            f"{self._base_url}/links/peek_target",
            json=req.model_dump(),
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return LinksPeekTargetResponse.model_validate(resp.json()).model_dump()

    # ── Action translation ────────────────────────────────────────

    def _translate_action(self, action: Any) -> tuple[str, dict[str, Any]]:
        """Map a Mantis `Action` to a Browser-Use dispatch verb.

        Kept minimal — PR 3-4 expand coverage. The cases below cover the
        common shapes used by today's step handlers. Unknown shapes raise
        so silent fall-through doesn't mask wire-contract bugs.
        """
        from ..actions import Action

        if not isinstance(action, Action):
            raise TypeError(f"BrowserUsePlaneClient.step expected Action, got {type(action)}")

        # Mantis Action is a tagged union — fields populated depend on type.
        atype = getattr(action, "type", None) or getattr(action, "action_type", None)
        if atype == "click":
            return "click", {
                "x": int(action.x),
                "y": int(action.y),
                "button": getattr(action, "button", "left"),
                "click_count": int(getattr(action, "click_count", 1)),
            }
        if atype == "key":
            return "key", {"key": str(getattr(action, "key", "") or action.text or "")}
        if atype == "type":
            return "type", {
                "text": str(action.text),
                "delay_ms": int(getattr(action, "delay_ms", 0)),
            }
        if atype == "scroll":
            return "scroll", {
                "delta_y": int(getattr(action, "delta_y", 0)),
                "x": int(getattr(action, "x", 0) or 0),
                "y": int(getattr(action, "y", 0) or 0),
            }
        raise NotImplementedError(
            f"BrowserUsePlaneClient does not yet handle Action type={atype!r}; "
            "extend `_translate_action` and add a test in "
            "tests/test_browser_use_plane_client.py."
        )
