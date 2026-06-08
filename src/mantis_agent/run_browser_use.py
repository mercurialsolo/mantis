"""`run_browser_use` — brain-plane executor for Browser-Use Plane (#785, PR 2).

Mirrors `_run_claude_executor`'s shape but routes the brain through
`BrowserUsePlaneClient` instead of the Computer Plane (`XdotoolGymEnv` /
`RemoteComputerImpl`). Stays headless / Playwright-driven; no Xvfb, no
xdotool, no local proxy subprocess.

PR 2 scaffold scope:

- Constructs the client + configures `CapabilityAllowlist.browser_use()`
  for the executor.
- Drives the existing `run_executor_lifecycle` with a Browser-Use plane
  env so the rest of the task-loop pipeline (Augur emission, retry,
  step recovery) keeps working without per-plane forks.
- NOT yet wired into `modal_cua_server.py`'s `cua_model` dispatch
  table — that lands once PR 3 / PR 4 add the DOM-aware extensions and
  there's an HTTP API contract worth exposing. Callers in PR 2 must
  invoke this function directly (e.g. from a smoke script).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from .gym.browser_use_plane_client import BrowserUsePlaneClient
from .gym.compute_contract import (
    Capabilities,
    CapabilityAllowlist,
    CapabilityNotAllowed,
    ComputeBackend,
)

logger = logging.getLogger(__name__)


EXECUTOR_NAME = "run_browser_use"


def make_browser_use_allowlist() -> CapabilityAllowlist:
    """Capability allowlist for the `run_browser_use` executor.

    Admits `dom_aware` + `supports_cdp` extensions; pure-CUA executors
    use `CapabilityAllowlist.pure_cua()` instead. This is the enforcement
    seam — handlers that consume DOM-aware extensions cross-check this
    allowlist before each call (see `compute_contract.CapabilityAllowlist`
    and the PR 3 / PR 4 handlers).
    """
    return CapabilityAllowlist.browser_use(executor=EXECUTOR_NAME)


def resolve_browser_use_base_url() -> str:
    """Resolve the Modal function URL for the Browser-Use Plane.

    Reads `MANTIS_BROWSER_USE_URL` first (operator override / local
    dev); falls back to looking the function up via the Modal SDK by
    app + function name. Empty string is returned if neither path
    succeeds — callers must raise (the executor cannot run without a
    backing host).
    """
    explicit = (os.environ.get("MANTIS_BROWSER_USE_URL") or "").strip()
    if explicit:
        return explicit
    try:
        import modal  # noqa: PLC0415

        fn = modal.Function.from_name("mantis-browser-use", "browser_use")
        return fn.get_web_url()  # type: ignore[no-any-return]
    except Exception as exc:  # noqa: BLE001 — best-effort resolution
        logger.warning(
            "[browser-use] cannot resolve URL via Modal SDK: %s; "
            "set MANTIS_BROWSER_USE_URL in the secret",
            exc,
        )
        return ""


def make_browser_use_client(
    *,
    tenant_id: str = "default",
    profile_id: str = "default",
    run_id: str | None = None,
    proxy_server: str | None = None,
    start_url: str = "about:blank",
    profile_dir: str | None = None,
    viewport: tuple[int, int] = (1280, 720),
    extra_http_headers: dict[str, str] | None = None,
    base_url: str | None = None,
    auth_token: str | None = None,
) -> tuple[BrowserUsePlaneClient, Capabilities]:
    """Construct a `BrowserUsePlaneClient` + return the expected `Capabilities`.

    The capabilities returned are the *expected* / declared shape (before
    `session/init` actually runs). After init, the client's
    `capabilities()` reflects what the server advertised — and the
    allowlist enforcement step in handlers re-checks it.
    """
    resolved_url = base_url or resolve_browser_use_base_url()
    if not resolved_url:
        raise RuntimeError(
            "Browser-Use Plane URL is not configured. Set "
            "MANTIS_BROWSER_USE_URL or deploy the Modal app "
            "'mantis-browser-use' via `modal deploy "
            "deploy/modal/browser_use_plane.py`."
        )
    client = BrowserUsePlaneClient(
        base_url=resolved_url,
        auth_token=auth_token,
        tenant_id=tenant_id,
        profile_id=profile_id,
        run_id=run_id,
        proxy_server=proxy_server,
        start_url=start_url,
        profile_dir=profile_dir,
        viewport=viewport,
        extra_http_headers=extra_http_headers,
    )
    return client, Capabilities.for_browser_use_plane()


def _validate_executor_compat(allowlist: CapabilityAllowlist, advertised: Capabilities) -> None:
    """Fail fast at session start if the executor's allowlist doesn't
    permit the advertised capabilities it intends to use.

    For `run_browser_use`, that means `dom_aware` must be on the
    allowlist AND the server must advertise it. Mismatched runs raise
    `CapabilityNotAllowed` here, not mid-plan.
    """
    if advertised.dom_aware and not allowlist.allows("dom_aware"):
        raise CapabilityNotAllowed("dom_aware", executor=allowlist.executor)
    if not advertised.dom_aware:
        logger.warning(
            "[browser-use] server advertised dom_aware=False — likely "
            "misconfigured. Expected Browser-Use Plane to advertise "
            "dom_aware=True for the %s executor.",
            allowlist.executor,
        )


def run_browser_use_executor(
    task_file_contents: str,
    *,
    claude_model: str = "claude-sonnet-4-6",
    max_steps: int = 30,
    frames_per_inference: int = 2,
    thinking_budget: int = 2048,
    base_url: str | None = None,
    auth_token: str | None = None,
    profile_dir: str | None = None,
    **_extra: Any,
) -> dict[str, Any]:
    """Execute a task suite against the Browser-Use Plane.

    Mirror of `_run_claude_executor` in shape. Currently uses
    `ClaudeBrain` for vision-grounded actions (PR 3 may add a brain
    variant that consumes DOM-aware reads via `state.*` / `tabs.*`).

    Scaffold-only at PR 2: not wired into modal_cua_server.py's
    cua_model dispatch table. Invoke directly via the Modal SDK or a
    smoke script.
    """
    from .brain_claude import ClaudeBrain
    from .task_loop import TaskLoopConfig, run_executor_lifecycle, setup_viewer

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    t0 = time.time()

    task_suite = json.loads(task_file_contents)
    session_name = task_suite.get("session_name", "browser_use")

    brain = ClaudeBrain(
        model=claude_model,
        max_tokens=4096,
        thinking_budget=thinking_budget,
        screen_size=(1280, 720),
    )
    brain.load()

    client, expected_caps = make_browser_use_client(
        tenant_id=task_suite.get("_tenant_id", "default"),
        profile_id=task_suite.get("_profile_id", "default"),
        run_id=run_id,
        proxy_server=task_suite.get("_proxy_server"),
        start_url=task_suite.get("base_url", "about:blank"),
        profile_dir=profile_dir,
        extra_http_headers=task_suite.get("_browser_extra_headers"),
        base_url=base_url,
        auth_token=auth_token,
    )

    allowlist = make_browser_use_allowlist()
    _validate_executor_compat(allowlist, expected_caps)

    viewer_ctx, viewer_event_bus, _viewer_url = setup_viewer(False, proxy_diag={"disabled": True})

    config = TaskLoopConfig(
        run_id=run_id,
        session_name=session_name,
        model_name=f"BrowserUse-Claude ({claude_model})",
        results_prefix="browser_use",
        brain=brain,
        env=client,
        max_steps=max_steps,
        frames_per_inference=frames_per_inference,
        viewer_event_bus=viewer_event_bus,
        on_task_complete=None,
        volume_commit=None,
        summary_extras={
            "compute_backend": ComputeBackend.BROWSER_USE_PLANE.value,
            "capability_allowlist": sorted(allowlist.allowed),
        },
    )
    return run_executor_lifecycle(
        task_suite,
        config,
        proxy_proc=None,
        viewer_ctx=viewer_ctx,
        t0=t0,
    )
