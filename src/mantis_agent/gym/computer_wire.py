"""Pydantic wire models for the Computer Plane RPC.

These models are the canonical contract between the brain plane and any
computer-plane backend (Modal computer-plane function in Phase 1, E2B
Desktop / Daytona in Phase 2). They are defined here in Phase 0 — before
any HTTP hop exists — so that the contract is locked early and the Phase
1 client + server can be implemented against a single source of truth.

The surface is intentionally CUA-pure: `screenshot` + `xdotool(argv)` are
required; `cdp_evaluate` / `cdp_click_at_point` are opt-in per executor
(`enable_cdp=false` default). No DOM-aware verbs. See
`feedback_browser_infra_rpc_surface.md`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SessionInitRequest(BaseModel):
    """Bind a `ComputerAgent` instance to a `(tenant_id, profile_id, run_id)`.

    Idempotent on `run_id`: a second init for the same `run_id` returns the
    existing session token. Different `run_id` against an active session is
    a 409.
    """

    tenant_id: str
    profile_id: str
    run_id: str
    proxy_server: str | None = None
    chrome_flags: list[str] = Field(default_factory=list)
    enable_cdp: bool = False
    viewport: tuple[int, int] = (1280, 720)


class SessionInitResponse(BaseModel):
    session_token: str
    chrome_pid: int | None = None
    xvfb_display: str


class SessionCloseRequest(BaseModel):
    session_token: str


class SessionCloseResponse(BaseModel):
    closed: bool


class ScreenshotRequest(BaseModel):
    """Empty body for v1. Reserved for future params (region, format)."""

    format: Literal["png"] = "png"


class ScreenshotResponse(BaseModel):
    image_b64: str
    width: int
    height: int
    scroll_y: int
    captured_at_ms: int


class XdotoolRequest(BaseModel):
    """`step_id` is client-generated and unique per logical step.

    Retries reuse the **same** `step_id`. The `ComputerAgent` server-side
    LRU returns the cached response with `deduplicated=true` on repeats.
    """

    argv: list[str]
    step_id: str
    timeout_ms: int = 5000


class XdotoolResponse(BaseModel):
    stdout: str
    stderr: str
    returncode: int
    deduplicated: bool = False


class CDPRequest(BaseModel):
    """Opt-in CDP escape hatch — action-dispatch only by convention.

    Never used to derive grounding. See `feedback_cua_no_dom_access.md` and
    `feedback_cua_cdp_post_action_verify.md`.
    """

    expression: str
    await_promise: bool = False
    step_id: str


class CDPResponse(BaseModel):
    result_json: str
    returncode: int = 0


class HealthResponse(BaseModel):
    ok: bool
    last_action_ms: int | None = None
    session_token: str | None = None
