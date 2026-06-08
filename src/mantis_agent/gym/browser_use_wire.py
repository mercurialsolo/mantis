"""Pydantic wire models for the Browser-Use Plane RPC (#785).

Browser-Use Plane diverges from Computer Plane on the *dispatch* surface:
where Computer Plane sends raw `xdotool argv`, Browser-Use Plane sends
structured action verbs (`click`, `key`, `type`, `scroll`) that map to
Playwright's `page.mouse` / `keyboard` primitives. The base lifecycle
(session_init, session_close, screenshot, health) mirrors the Computer
Plane shape so the umbrella `ComputeClient` contract (#785) is honored.

DOM-aware extensions (`state.*`, `tabs.*`, `links.*`) land in PR 3-4 —
they are NOT in this module yet. Computer Plane's wire surface does not
gain them; pure-CUA executors carry a `CapabilityAllowlist` that
excludes `dom_aware`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .compute_contract import Capabilities, ComputeBackend


class BrowserUseSessionInitRequest(BaseModel):
    """Bind a Browser-Use Plane session to `(tenant_id, profile_id, run_id)`.

    Profile is stored as a Playwright `userDataDir` blob under
    `/data/browser-use-profile/<tenant>__<profile_id>/`. Storage layout
    differs from Computer Plane (which uses Chrome's `--user-data-dir`);
    see `docs/reference/browser-use-plane.md` for the per-plane storage
    contract.
    """

    tenant_id: str
    profile_id: str
    run_id: str
    proxy_server: str | None = None
    viewport: tuple[int, int] = (1280, 720)
    start_url: str = "about:blank"
    profile_dir: str | None = None
    extra_http_headers: dict[str, str] | None = None
    # Browser-Use Plane has no Xvfb. Headless mode is the default; set
    # False only when debugging locally.
    headless: bool = True


class BrowserUseSessionInitResponse(BaseModel):
    """Mirrors Computer Plane's `SessionInitResponse` shape with one diff:
    no `xvfb_display` (Browser-Use Plane runs headless). Capabilities
    advertise `dom_aware=True`."""

    session_token: str
    browser_pid: int | None = None
    capabilities: dict[str, object] | None = None

    def resolved_capabilities(self) -> Capabilities:
        if self.capabilities is None:
            return Capabilities.for_browser_use_plane()
        backend_value = self.capabilities.get(
            "backend", ComputeBackend.BROWSER_USE_PLANE.value
        )
        try:
            backend = ComputeBackend(backend_value)
        except ValueError:
            backend = ComputeBackend.BROWSER_USE_PLANE
        return Capabilities(
            dom_aware=bool(self.capabilities.get("dom_aware", True)),
            stealth=bool(self.capabilities.get("stealth", False)),
            supports_cdp=bool(self.capabilities.get("supports_cdp", True)),
            backend=backend,
        )


class BrowserUseSessionCloseRequest(BaseModel):
    session_token: str


class BrowserUseSessionCloseResponse(BaseModel):
    closed: bool


class BrowserUseScreenshotResponse(BaseModel):
    image_b64: str
    width: int
    height: int
    scroll_y: int
    captured_at_ms: int


class DispatchActionRequest(BaseModel):
    """Uniform structured action verb — the Browser-Use Plane analogue of
    Computer Plane's `xdotool argv`.

    `kind` discriminates:
      - `click` — `{x, y, button, click_count}`
      - `key` — `{key}` (Playwright key names: `Enter`, `Escape`, `Tab`, ...)
      - `type` — `{text, delay_ms}` (`delay_ms` is per-character pace)
      - `scroll` — `{delta_y, x, y}` (CDP `Input.dispatchMouseEvent` with `wheel`)

    `step_id` enables idempotency per the unified contract; the server
    keeps a TTL'd LRU and returns `deduplicated=true` on retry.
    """

    kind: Literal["click", "key", "type", "scroll"]
    params: dict[str, object] = Field(default_factory=dict)
    step_id: str
    timeout_ms: int = 5000


class DispatchActionResponse(BaseModel):
    ok: bool
    error: str | None = None
    deduplicated: bool = False


class BrowserUseHealthResponse(BaseModel):
    ok: bool
    last_action_ms: int | None = None
    session_token: str | None = None
