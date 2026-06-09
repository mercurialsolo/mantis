"""Browser-Use Plane Modal deployment (#785, PR 2).

A standalone Modal app — separate from `mantis-cua-server` (Computer
Plane) per the epic decision (#785): "two planes on separate hosts."
Deploying Browser-Use Plane is independent of redeploying the brain or
Computer Plane.

Image: Microsoft Playwright Python image (Chromium + Playwright +
Python) with `mantis_agent` + FastAPI added. Headless by default; no
Xvfb, no xdotool. CPU-only — no GPU, no Chromium-via-Xvfb stealth
stack.

Endpoints come from `mantis_agent.server.browser_use_agent.build_app`
— see `docs/reference/browser-use-plane.md` for the wire contract.

Deploy::

    modal deploy deploy/modal/browser_use_plane.py

After deploy, resolve the web URL with::

    modal.Function.from_name('mantis-browser-use', 'browser_use').get_web_url()

and pass it to `BrowserUsePlaneClient(base_url=...)` or
`make_compute_client(ComputeBackend.BROWSER_USE_PLANE,
browser_use_base_url=...)`.
"""

from __future__ import annotations


import modal

APP_NAME = "mantis-browser-use"
app = modal.App(APP_NAME)

# Shared with Computer Plane / brain plane for cross-plane profile reads
# (deferred to a follow-up — at v1 profiles are per-plane; see #785).
# Mounting the same volume here doesn't enable shared profiles by itself;
# the storage paths under it differ (`/data/browser-use-profile/...`).
vol = modal.Volume.from_name("osworld-data", create_if_missing=True)

# `mcr.microsoft.com/playwright-python` ships Chromium + the Python
# Playwright SDK preinstalled. Smaller than building from scratch, and
# bundles the exact Chromium build the SDK version expects.
browser_use_image = (
    modal.Image.from_registry(
        "mcr.microsoft.com/playwright/python:v1.49.0-jammy",
        add_python=None,  # base image ships Python 3.x
    )
    .run_commands(
        # Match Computer Plane's locale + TZ posture so screenshots /
        # date-format-sensitive sites behave identically.
        "sed -i 's/# en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen && locale-gen || true",
        "ln -sf /usr/share/zoneinfo/America/New_York /etc/localtime || true",
    )
    .env({
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "TZ": "America/New_York",
    })
    .pip_install(
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2",
        "pillow",
        "httpx",
        # The mantis_agent gym package transitively imports
        # BrowserUsePlaneClient which uses `requests`. Even though the
        # server doesn't need it at runtime, the import path is
        # evaluated on container start.
        "requests>=2.28",
        # #826: patchright — drop-in patched Playwright that strips
        # automation tells at the binary level. Default-preferred over
        # vanilla Playwright. Falls back to playwright.sync_api when
        # MANTIS_BROWSER_USE_DRIVER=playwright is set.
        "patchright>=1.49",
    )
    .run_commands(
        # patchright ships its own Chromium build; the playwright
        # browsers we got from the base image are also fine. Make sure
        # at least one stack is initialized so the first request
        # doesn't pay a 60 s download.
        "python -m patchright install chromium || python -m playwright install chromium",
    )
    .add_local_python_source("mantis_agent")
)


@app.function(
    image=browser_use_image,
    volumes={"/data": vol},
    # No secrets at v1 — Browser-Use Plane's base surface doesn't read
    # tenant credentials. Proxy / auth-header configuration arrives via
    # /session/init body. Add Secret.from_dotenv() here if a future
    # version needs env-var access (e.g. PRIVATEPROXY_* shared with the
    # brain plane).
    secrets=[],
    timeout=14400,  # 4 hours — match brain executor timeouts
    memory=4096,
    cpu=2.0,
    scaledown_window=600,
    # Single-session-per-container at v1; matches Computer Plane's
    # pinned-container semantics (see modal_cua_server.py:2812-2813 for
    # the rationale). Per-session fan-out is a Phase-2 concern.
    min_containers=1,
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def browser_use():
    """Browser-Use Plane RPC server — Playwright + Chromium over HTTPS.

    Base surface only at v1 — `session/init`, `session/close`,
    `screenshot`, `dispatch`, `health`. DOM-aware extensions
    (`state/*`, `tabs/*`, `links/peek`) land with PR 3-4 of #785.
    """
    from mantis_agent.server.browser_use_agent import build_app
    return build_app()
