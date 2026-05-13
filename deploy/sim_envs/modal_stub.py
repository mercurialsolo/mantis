"""Modal stub env — deploys the in-package stub as a Modal app.

This is the reference deploy for #336. Real envs land their own file in
this directory; they all follow the same shape:

1. Define a Modal image carrying the env's runtime + dependencies.
2. Mount a Modal Secret holding ``ENV_ADMIN_TOKEN``.
3. Expose an ASGI route that serves both the agent UI (``/``) and the
   harness surface (``/__env__/*``).

For the stub the ASGI app is built on top of stdlib ``http.server``
wrapped to look like an ASGI callable, so no extra dependency lands in
the image.

## Deploy

    uv run modal deploy deploy/sim_envs/modal_stub.py

That gives you a Modal app named ``mantis-sim-env-stub`` whose ``web``
function is reachable from the local CLI:

    uv run mantis plan run plans/_stub_env/T01_stub_passthrough.json \\
        --env stub --runtime modal --endpoint <BRAIN_URL>

## Required Modal Secret

Mount a Modal Secret named ``mantis-sim-env-stub-secrets`` containing
``ENV_ADMIN_TOKEN=<a-random-token>``. The same value must be exported
locally as ``MANTIS_STUB_ENV_ADMIN_TOKEN`` so the local CLI can sign
admin calls against the deployed stub.
"""

from __future__ import annotations

import os

import modal

APP_NAME = "mantis-sim-env-stub"

app = modal.App(APP_NAME)

stub_image = (
    modal.Image.debian_slim(python_version="3.11")
    # The stub uses only stdlib; we pull the mantis_agent package in so
    # the deploy file can reach the handler factory.
    .pip_install("pillow>=10.0")
    .add_local_python_source("mantis_agent")
)

secret = modal.Secret.from_name(
    "mantis-sim-env-stub-secrets",
    required_keys=["ENV_ADMIN_TOKEN"],
)


def _build_asgi_app():
    """Build an ASGI app that wraps the stdlib stub handler.

    The mantis stub uses ``http.server`` for the slim install path. To
    host it on Modal we need an ASGI surface. Rather than re-implement
    routing in two places we wrap the stub via a tiny ASGI ↔ WSGI bridge
    using the stdlib + ``a2wsgi`` equivalent — but only stdlib is
    available in the slim image, so the bridge is hand-rolled here.
    """
    # Modal's ``asgi_app`` decorator expects a callable conforming to
    # the ASGI spec. We use Starlette's ASGI primitives indirectly via
    # a minimal manual implementation: parse the HTTP request, dispatch
    # to a per-request stdlib handler, return the response.
    #
    # In practice we just delegate to a Starlette app to avoid the
    # 200-line stdlib dance. Starlette ships with Modal's slim image
    # via uvicorn (which depends on it), so this is free.
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse, Response
    from starlette.routing import Route

    from mantis_agent.sim_envs.stub_app import AGENT_LANDING_HTML, _EnvState

    admin_token = os.environ.get("ENV_ADMIN_TOKEN", "")
    if not admin_token:
        raise RuntimeError(
            "ENV_ADMIN_TOKEN not set — the Modal stub requires the "
            "mantis-sim-env-stub-secrets secret to be mounted."
        )

    seed = int(os.environ.get("SEED") or 42)
    now = os.environ.get("FAKE_NOW") or "2026-01-15T09:00:00Z"
    state = _EnvState(seed=seed, now=now, admin_token=admin_token)

    def _require_admin(request: Request) -> Response | None:
        if request.headers.get("x-env-admin", "") != state.admin_token:
            return JSONResponse({"error": "admin token required"}, status_code=401)
        return None

    async def health(_request: Request) -> Response:
        return JSONResponse({
            "ok": True,
            "seed": state.seed,
            "now": state.now,
            "boot_time": state.boot_time,
        })

    async def reset(request: Request) -> Response:
        err = _require_admin(request)
        if err is not None:
            return err
        state.reset()
        return JSONResponse({"ok": True})

    async def seed_route(request: Request) -> Response:
        err = _require_admin(request)
        if err is not None:
            return err
        body = await request.json() if await request.body() else {}
        new_seed = int(body.get("seed", state.seed))
        state.seed = new_seed
        state.emit("seed_changed", {"seed": new_seed})
        return JSONResponse({"ok": True, "seed": new_seed})

    async def clock_route(request: Request) -> Response:
        err = _require_admin(request)
        if err is not None:
            return err
        body = await request.json() if await request.body() else {}
        new_now = str(body.get("now") or state.now)
        state.now = new_now
        state.emit("clock_changed", {"now": new_now})
        return JSONResponse({"ok": True, "now": new_now})

    async def oracle(request: Request) -> Response:
        err = _require_admin(request)
        if err is not None:
            return err
        task_id = (request.query_params.get("task_id") or "").strip()
        state.emit("oracle_query", {"task_id": task_id})
        return JSONResponse({
            "passed": True,
            "score": 1.0,
            "task_id": task_id,
            "reasons": ["stub env: every task passes"],
            "diff": {},
        })

    async def state_dump(request: Request) -> Response:
        err = _require_admin(request)
        if err is not None:
            return err
        return JSONResponse({
            "seed": state.seed,
            "now": state.now,
            "touched": state.touched,
            "event_count": len(state.events),
        })

    async def events_route(request: Request) -> Response:
        err = _require_admin(request)
        if err is not None:
            return err
        since = float(request.query_params.get("since") or 0)
        filtered = [e for e in state.events if e["ts"] >= since]
        return JSONResponse({"events": filtered})

    async def landing(_request: Request) -> Response:
        return HTMLResponse(AGENT_LANDING_HTML)

    return Starlette(routes=[
        Route("/__env__/health", health, methods=["GET"]),
        Route("/__env__/reset", reset, methods=["POST"]),
        Route("/__env__/seed", seed_route, methods=["POST"]),
        Route("/__env__/clock", clock_route, methods=["POST"]),
        Route("/__env__/oracle", oracle, methods=["GET"]),
        Route("/__env__/state", state_dump, methods=["GET"]),
        Route("/__env__/events", events_route, methods=["GET"]),
        Route("/{rest:path}", landing, methods=["GET"]),
    ])


@app.function(
    image=stub_image,
    secrets=[secret],
    min_containers=0,
    max_containers=4,
    timeout=300,
)
@modal.asgi_app()
def web():
    return _build_asgi_app()
