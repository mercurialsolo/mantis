"""FastAPI app + route handlers for the Baseten workload.

The single FastAPI ``app`` lives here. Decorators throughout this file
register the public routes:

- ``GET /health``, ``GET /v1/health`` — liveness
- ``GET /metrics`` — Prometheus
- ``GET /v1/models`` — model list
- ``GET /v1/runs/{run_id}/video`` — recorded screencast
- ``POST /v1/chat/completions`` — OpenAI-compatible proxy to the in-pod llama.cpp
- ``POST /v1/predict``, ``POST /predict`` — main run dispatch

Heavy lifting lives elsewhere:

- :class:`~.runtime.BasetenCUARuntime` — model + run lifecycle (the
  module-level ``runtime`` singleton is owned here; the class lives in
  :mod:`.runtime` so it stays focused).
- :mod:`.middleware` — auth / secret resolvers exposed as FastAPI deps.
- :mod:`.paths` — per-tenant path resolvers.
- :mod:`.logging_setup` — JSON log formatter + ``DetachedRunLogHandler``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests

try:
    from fastapi import Depends, FastAPI, HTTPException, Request, Response
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool
except ImportError as exc:  # pragma: no cover - container-only deps
    raise ImportError(
        "mantis_agent.baseten_server requires fastapi + uvicorn. "
        "Install via: pip install -e '.[server]'  (or run inside the "
        "Baseten Truss image, which provisions them in build_commands)."
    ) from exc

from .. import metrics as mantis_metrics
from ..api_schemas import (
    MAX_COST_USD,
    MAX_RUNTIME_MINUTES,
    PredictRequest,
    PureCUARequest,
    assert_hosts_allowed,
    extract_navigate_hosts,
)
from ..idempotency import get_idempotency_cache
from ..rate_limit import get_rate_limiter
from ..server_utils import (
    build_proxy_config,
    parse_lead_row,
    plan_signature_from_steps,
    safe_state_key,
    start_local_proxy,
    utc_now,
    wait_for_openai_server,
    write_leads_csv,
)
from ..tenant_auth import TenantConfig
from ..webhooks import WebhookPayload, deliver_webhook_async
from .logging_setup import (
    DetachedRunLogHandler,
    JsonLogFormatter,
    configure_logging,
)
from .middleware import (
    load_secret_environment,
    read_secret,
    require_mantis_token,
    require_run_scope,
    resolve_anthropic_key,
)
from .paths import (
    data_root,
    find_gguf,
    find_mmproj,
    first_existing,
    new_run_id,
    repo_root,
    tenant_chrome_profile,
    tenant_root,
    tenant_state_key,
)
from .runtime import BasetenCUARuntime


configure_logging()
logger = logging.getLogger("mantis_agent.baseten_server")


# ── Module-level singletons ─────────────────────────────────────────────────

# Interactive API docs at /docs (Swagger UI) and /redoc are on by default
# so self-hosters get a browsable surface for free. Set
# ``MANTIS_ENABLE_DOCS_UI=0`` on production tenant fleets that don't want
# the UI exposed publicly. ``/openapi.json`` stays on regardless — it's
# what client SDKs and IDE plugins consume.
_DOCS_UI_ENABLED = os.environ.get("MANTIS_ENABLE_DOCS_UI", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

from .. import __version__ as _mantis_version  # noqa: E402 — kept near app config

app = FastAPI(
    title="Mantis CUA",
    description=(
        "Perception-reasoning-action agent for computer use. Drives a "
        "real browser via Holo3 + Claude. See the docs at "
        "https://mercurialsolo.github.io/mantis/ — this surface is the "
        "live HTTP API. Auth: `X-Mantis-Token` per tenant, plus "
        "`Authorization: Api-Key …` when fronted by a Baseten gateway."
    ),
    version=_mantis_version,
    docs_url="/docs" if _DOCS_UI_ENABLED else None,
    redoc_url="/redoc" if _DOCS_UI_ENABLED else None,
)
runtime = BasetenCUARuntime()


# ── Middleware: standard X-RateLimit-* response headers (#275) ──────────────
# Each rate-limited route handler stashes the limit / remaining / reset
# triple on ``request.state.rate_limit_headers`` right after consulting
# the limiter; this middleware copies them onto the response so callers
# can throttle proactively instead of crashing into 429s. Routes that
# don't rate-limit (``/health``, ``/v1/version``, ``/v1/models``,
# ``/metrics``, ``/v1/runs/{id}/video``) simply don't set the attribute
# and the middleware is a no-op.

import time as _time  # noqa: E402 — kept near the middleware it powers


@app.middleware("http")
async def _attach_rate_limit_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
    response = await call_next(request)
    headers = getattr(request.state, "rate_limit_headers", None)
    if headers:
        for name, value in headers.items():
            response.headers[name] = str(value)
    return response


def _stash_rate_limit_headers(
    request: Request, decision: Any, tenant_limit: int,
) -> None:
    """Record the rate-limit triple on ``request.state`` for the middleware.

    Called from every route that consults the rate limiter, regardless of
    whether the decision was allow or deny — denied requests still
    benefit from the headers (alongside the existing ``Retry-After`` on
    429s).
    """
    limit = decision.limit or tenant_limit or 0
    if not limit:
        return  # rate-limit disabled for this tenant
    remaining = max(0, int(decision.rate_remaining))
    # Unix timestamp when the bucket will be back at full capacity.
    reset_ts = int(_time.time() + max(0.0, decision.reset_after_seconds))
    request.state.rate_limit_headers = {
        "X-RateLimit-Limit": limit,
        "X-RateLimit-Remaining": remaining,
        "X-RateLimit-Reset": reset_ts,
    }


# ── Backwards-compat aliases (one-minor-release deprecation) ────────────────
# Inside this file, route handlers were carved out of the original single-file
# module that used single-underscore names for everything. The handlers below
# still call ``_require_run_scope``, ``_resolve_anthropic_key``, etc.; aliasing
# the canonical sibling names keeps the move purely a rename of where each
# helper *lives*, not what the handler bodies look like.

_JsonLogFormatter = JsonLogFormatter
_configure_logging = configure_logging
_DetachedRunLogHandler = DetachedRunLogHandler

_require_mantis_token = require_mantis_token
_require_run_scope = require_run_scope
_read_secret = read_secret
_resolve_anthropic_key = resolve_anthropic_key
_load_secret_environment = load_secret_environment

_data_root = data_root
_tenant_root = tenant_root
_tenant_state_key = tenant_state_key
_tenant_chrome_profile = tenant_chrome_profile
_repo_root = repo_root
_new_run_id = new_run_id
_first_existing = first_existing
_find_gguf = find_gguf
_find_mmproj = find_mmproj

_safe_state_key = safe_state_key
_utc_now = utc_now
_parse_lead_row = parse_lead_row
_write_leads_csv = write_leads_csv
_plan_signature_from_steps = plan_signature_from_steps
_wait_for_openai_server = wait_for_openai_server
_start_local_proxy = start_local_proxy
_build_proxy_config = build_proxy_config


# ── Routes ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup() -> None:
    runtime.load()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": runtime.loaded, "model": runtime.model_kind}


@app.get("/v1/health")
def health_v1() -> dict[str, Any]:
    """Versioned alias for /health.

    The unversioned /health endpoint is what platform liveness probes target;
    /v1/health is the same payload available under the public API path.
    """
    return {"ok": runtime.loaded, "model": runtime.model_kind}


@app.get("/v1/version")
def version_info() -> dict[str, Any]:
    """Runtime version snapshot — useful when multiple deployments serve
    different builds and a client needs to pin behavior to a specific one.

    ``git_sha`` and ``build_time`` are populated by the deploy pipeline via
    ``MANTIS_GIT_SHA`` / ``MANTIS_BUILD_TIME`` env vars (empty strings when
    running outside a build context). No auth required — version info is
    safe to expose alongside ``/health``.
    """
    return {
        "version": _mantis_version,
        "model": runtime.model_kind,
        "ready": runtime.loaded,
        "git_sha": os.environ.get("MANTIS_GIT_SHA", ""),
        "build_time": os.environ.get("MANTIS_BUILD_TIME", ""),
    }


@app.get("/v1/runs/{run_id}/video")
def get_run_video(
    run_id: str,
    request: Request,
    tenant: TenantConfig = Depends(_require_mantis_token),
) -> Any:
    """Download the screencast for a run.

    Default: serves the **polished** version (title card + captioned run
    footage + outro card). Pass ``?raw=1`` to fetch the raw screencast
    without overlays.

    Resolves to the per-tenant run dir; returns 404 if no recording exists
    (recording wasn't requested or ffmpeg failed). Auth requires a valid
    token but not specifically the ``run`` scope.
    """
    from fastapi.responses import FileResponse
    from mantis_agent.recorder import content_type_for

    raw_only = request.query_params.get("raw", "").lower() in {"1", "true", "yes"}

    safe_run_id = safe_state_key(run_id)
    tenant_dir = _data_root() / "tenants" / safe_state_key(tenant.tenant_id)
    runs_dir = tenant_dir / "runs" / safe_run_id

    # Prefer polished by default; fall back to raw if polished is missing
    # (e.g., ffmpeg compose failed). ?raw=1 skips polished entirely.
    prefixes = ("recording",) if raw_only else ("recording_polished", "recording")
    for prefix in prefixes:
        for fmt in ("mp4", "webm", "gif"):
            candidate = runs_dir / f"{prefix}.{fmt}"
            if candidate.exists() and candidate.stat().st_size > 0:
                return FileResponse(
                    candidate,
                    media_type=content_type_for(fmt),  # type: ignore[arg-type]
                    filename=f"{safe_run_id}.{fmt}",
                )
    raise HTTPException(
        status_code=404,
        detail="no recording for this run "
        "(record_video=true on /v1/predict required)",
    )


@app.get("/metrics")
def metrics_endpoint() -> Any:
    """Prometheus scrape endpoint.

    Returns 503 if prometheus_client isn't installed in the container.
    """
    if not mantis_metrics.is_available():
        raise HTTPException(status_code=503, detail="prometheus_client not installed")
    return Response(
        content=mantis_metrics.render_text(),
        media_type=mantis_metrics.CONTENT_TYPE_LATEST,
    )


@app.get("/v1/models")
def models() -> dict[str, Any]:
    """OpenAI-compatible model listing.

    Public so clients can discover the model id before sending requests.
    Auth is enforced on the inference path itself (/v1/chat/completions).
    """
    return {
        "object": "list",
        "data": [
            {
                "id": runtime.model_kind,
                "object": "model",
                "owned_by": "mantis",
            }
        ],
    }


# Sentinel headers the upstream llama.cpp shouldn't see. We strip them so the
# Mantis-side auth credential never reaches the inference layer.
_PROXY_DROP_HEADERS = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "x-mantis-token",
    "authorization",
    "cookie",
}


@app.post("/v1/chat/completions")
async def chat_completions_proxy(
    request: Request,
    tenant: TenantConfig = Depends(_require_run_scope),
) -> Any:
    """Auth-gated reverse proxy to the in-pod llama.cpp Holo3 server.

    Designed for OpenAI-compat clients (the host integration's BrainHolo3 client,
    direct `openai.OpenAI(...)` callers) that want to use Holo3 inference
    without the full /predict orchestrator.

    What this endpoint does:
      • Validates ``X-Mantis-Token`` and resolves the tenant (must have
        ``run`` scope).
      • Strips Mantis-side auth headers before forwarding so the upstream
        llama.cpp never sees them.
      • Forwards the JSON body verbatim to the in-pod llama.cpp server at
        ``MANTIS_LLAMA_PORT``.
      • Passes upstream status codes and JSON bodies through.
    """
    body = await request.body()
    upstream_port = os.environ.get("MANTIS_LLAMA_PORT", "18080")
    upstream = f"http://127.0.0.1:{upstream_port}/v1/chat/completions"

    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _PROXY_DROP_HEADERS
    }
    headers["Content-Type"] = "application/json"

    logger.info(
        "v1/chat/completions tenant=%s upstream=%s bytes=%d",
        tenant.tenant_id,
        upstream,
        len(body),
    )

    try:
        r = await run_in_threadpool(
            requests.post,
            upstream,
            data=body,
            headers=headers,
            timeout=180,
        )
    except requests.RequestException as exc:
        mantis_metrics.CHAT_COMPLETIONS.labels(
            tenant_id=tenant.tenant_id, outcome="upstream_error"
        ).inc()
        logger.exception("v1/chat/completions upstream error tenant=%s", tenant.tenant_id)
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

    mantis_metrics.CHAT_COMPLETIONS.labels(
        tenant_id=tenant.tenant_id,
        outcome="ok" if 200 <= r.status_code < 300 else f"status_{r.status_code // 100}xx",
    ).inc()

    try:
        payload = r.json()
    except ValueError:
        # Upstream returned non-JSON (rare; usually means it crashed). Surface
        # the raw text so callers can debug.
        return JSONResponse(
            content={"error": {"message": r.text[:1000], "type": "upstream_error"}},
            status_code=r.status_code if r.status_code >= 400 else 502,
        )
    return JSONResponse(content=payload, status_code=r.status_code)


async def _handle_predict(
    request: Request, tenant: TenantConfig
) -> dict[str, Any]:
    """Shared handler for /predict and /v1/predict.

    Tier-1 + Tier-2 pipeline:

    1. Pydantic validation, global cap clamp.
    2. Per-tenant cap clamp.
    3. State-key + Chrome-profile namespacing per tenant.
    4. Per-tenant Anthropic key resolution.
    5. (Tier-2) URL allowlist enforcement on the plan.
    6. (Tier-2) Idempotency-key cache lookup.
    7. (Tier-2) Rate-limit token consumption.
    8. (Tier-2) Concurrency-slot acquisition (released in finally).
    9. (Tier-2) Webhook callback registered with the runtime if requested.
    10. Forward to the runtime.
    """
    try:
        raw = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="request body must be JSON") from exc
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")

    try:
        req = PredictRequest.model_validate(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid request: {exc}") from exc

    payload = req.model_dump(exclude_none=True)
    payload["max_cost"] = min(
        float(payload.get("max_cost", MAX_COST_USD)),
        tenant.max_cost_per_run,
    )
    payload["max_time_minutes"] = min(
        int(payload.get("max_time_minutes", MAX_RUNTIME_MINUTES)),
        tenant.max_time_minutes_per_run,
    )
    payload["state_key"] = _tenant_state_key(tenant, payload.get("state_key"))

    os.environ["ANTHROPIC_API_KEY"] = _resolve_anthropic_key(tenant)
    os.environ["MANTIS_TENANT_ID"] = tenant.tenant_id
    profile_dir = _tenant_chrome_profile(tenant, payload["state_key"])
    os.environ["MANTIS_CHROME_PROFILE_DIR"] = str(profile_dir)

    is_run_mode = req.action is None

    # ── Tier-2: URL allowlist ────────────────────────────────────────────
    if is_run_mode and tenant.allowed_domains:
        plan_obj: Any = None
        if req.task_suite is not None:
            plan_obj = req.task_suite
        elif req.task_file_contents:
            try:
                plan_obj = json.loads(req.task_file_contents)
            except json.JSONDecodeError:
                plan_obj = None
        if plan_obj is not None:
            try:
                hosts = extract_navigate_hosts(plan_obj)
                assert_hosts_allowed(hosts, tenant.is_domain_allowed)
            except PermissionError as exc:
                mantis_metrics.PREDICT_REQUESTS.labels(
                    tenant_id=tenant.tenant_id, mode="run", outcome="denied_allowlist"
                ).inc()
                raise HTTPException(status_code=403, detail=str(exc)) from exc

    # ── Tier-2: Idempotency-key cache ────────────────────────────────────
    idempotency_key = request.headers.get("Idempotency-Key", "").strip()
    if is_run_mode and idempotency_key:
        cached = get_idempotency_cache().get(tenant.tenant_id, idempotency_key)
        if cached is not None:
            logger.info(
                "predict idempotency-hit tenant=%s key=%s run_id=%s",
                tenant.tenant_id, idempotency_key, cached.run_id,
            )
            mantis_metrics.PREDICT_REQUESTS.labels(
                tenant_id=tenant.tenant_id, mode="run", outcome="idempotent_hit"
            ).inc()
            return cached.response

    # ── Tier-2: Rate limit (token bucket) — applied to run-mode only ─────
    limiter = get_rate_limiter()
    if is_run_mode:
        rate_decision = limiter.try_consume_rate_token(
            tenant.tenant_id, tenant.rate_limit_per_minute
        )
        _stash_rate_limit_headers(
            request, rate_decision, tenant.rate_limit_per_minute,
        )
        if not rate_decision.allowed:
            mantis_metrics.RATE_LIMIT_REJECTIONS.labels(
                tenant_id=tenant.tenant_id, kind="rate"
            ).inc()
            mantis_metrics.PREDICT_REQUESTS.labels(
                tenant_id=tenant.tenant_id, mode="run", outcome="rate_limited"
            ).inc()
            raise HTTPException(
                status_code=429,
                detail=rate_decision.reason,
                headers={"Retry-After": str(int(rate_decision.retry_after_seconds) + 1)},
            )

    # ── Tier-2: Concurrency slot ─────────────────────────────────────────
    concurrency_acquired = False
    if is_run_mode:
        decision = limiter.try_acquire_concurrency_slot(
            tenant.tenant_id, tenant.max_concurrent_runs
        )
        if not decision.allowed:
            mantis_metrics.RATE_LIMIT_REJECTIONS.labels(
                tenant_id=tenant.tenant_id, kind="concurrent"
            ).inc()
            mantis_metrics.PREDICT_REQUESTS.labels(
                tenant_id=tenant.tenant_id, mode="run", outcome="rate_limited"
            ).inc()
            raise HTTPException(
                status_code=429,
                detail=decision.reason,
                headers={"Retry-After": str(int(decision.retry_after_seconds) + 1)},
            )
        concurrency_acquired = True
        mantis_metrics.CONCURRENT_RUNS.labels(tenant_id=tenant.tenant_id).set(
            decision.concurrent
        )

    # ── Tier-2: Webhook URL — caller may override the tenant default ─────
    webhook_url = (
        raw.get("callback_url") or tenant.webhook_url or ""
    ).strip()
    if webhook_url:
        payload["_webhook_url"] = webhook_url
        payload["_webhook_secret_name"] = tenant.webhook_secret_name
        payload["_tenant_id"] = tenant.tenant_id

    logger.info(
        "predict tenant=%s scope=run state_key=%s detached=%s action=%s",
        tenant.tenant_id,
        payload["state_key"],
        payload.get("detached", True),
        req.action or "run",
    )

    try:
        response = await run_in_threadpool(runtime.run, payload)
        mode = req.action or "run"
        mantis_metrics.PREDICT_REQUESTS.labels(
            tenant_id=tenant.tenant_id, mode=mode, outcome="ok"
        ).inc()
        if is_run_mode and idempotency_key and isinstance(response, dict):
            get_idempotency_cache().store(
                tenant.tenant_id, idempotency_key, response.get("run_id", ""), response
            )
        if is_run_mode and webhook_url and isinstance(response, dict):
            run_id = response.get("run_id", "")
            if response.get("status") in {"succeeded", "failed", "cancelled"}:
                deliver_webhook_async(
                    webhook_url,
                    WebhookPayload(
                        run_id=run_id,
                        tenant_id=tenant.tenant_id,
                        status=str(response.get("status", "")),
                        summary=response.get("summary") or {},
                    ),
                    secret_name=tenant.webhook_secret_name,
                )
        return response
    except ValueError as exc:
        mantis_metrics.PREDICT_REQUESTS.labels(
            tenant_id=tenant.tenant_id, mode="run", outcome="bad_request"
        ).inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        mantis_metrics.PREDICT_REQUESTS.labels(
            tenant_id=tenant.tenant_id, mode=req.action or "run", outcome="not_found"
        ).inc()
        run_id = str(payload.get("run_id") or "")
        detail = f"unknown run_id: {run_id}" if run_id else "run not found"
        raise HTTPException(status_code=404, detail=detail) from exc
    except HTTPException:
        raise
    except Exception as exc:
        mantis_metrics.PREDICT_REQUESTS.labels(
            tenant_id=tenant.tenant_id, mode="run", outcome="error"
        ).inc()
        logger.exception("predict failed tenant=%s", tenant.tenant_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if concurrency_acquired:
            limiter.release_concurrency_slot(tenant.tenant_id)
            mantis_metrics.CONCURRENT_RUNS.labels(tenant_id=tenant.tenant_id).set(
                limiter.get_concurrent(tenant.tenant_id)
            )


@app.post("/v1/predict")
async def predict_v1(
    request: Request,
    tenant: TenantConfig = Depends(_require_run_scope),
) -> dict[str, Any]:
    """Tier-1 multi-tenant /predict. Validated, per-tenant capped and isolated.

    Requires the ``run`` scope on the tenant token — read-only keys
    (e.g. observers with only ``status`` / ``result`` access) are
    rejected with 403. Polling actions (``status``/``result``/``logs``/
    ``cancel``) flow through the same handler, so any caller that needs
    to poll a run also needs ``run`` scope.
    """
    return await _handle_predict(request, tenant)


@app.post("/predict")
async def predict(
    request: Request,
    tenant: TenantConfig = Depends(_require_run_scope),
) -> dict[str, Any]:
    """Backwards-compat alias for /v1/predict.

    Kept indefinitely for callers built against the v1.0 deployment shape.
    Identical behavior to /v1/predict, including the ``run`` scope check.
    """
    return await _handle_predict(request, tenant)


@app.post("/v1/cua")
async def cua_v1(
    request: Request,
    tenant: TenantConfig = Depends(_require_run_scope),
) -> dict[str, Any]:
    """Pure CUA loop — brain ↔ XdotoolGymEnv only, no Claude.

    Mantis is a pure pass-through for the configured brain (Holo3):

    * No ``PlanDecomposer`` — the instruction is handed verbatim.
    * No ``ClaudeGrounding`` — click coords come straight from the brain.
    * No ``ClaudeExtractor`` — no post-step verification.

    Action surface available to the brain (executed by xdotool against
    the headed Chrome inside Xvfb): ``click``, ``double_click``,
    ``type_text``, ``key_press``, ``scroll``, ``drag``, ``wait``,
    ``done``. Same per-tenant auth / cap / allowlist / rate-limit /
    concurrency / webhook plumbing as ``/v1/predict``.
    """
    try:
        raw = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="request body must be JSON") from exc
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")

    try:
        req = PureCUARequest.model_validate(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid request: {exc}") from exc

    payload = req.model_dump(exclude_none=True)
    payload["max_cost"] = min(
        float(payload.get("max_cost", MAX_COST_USD)),
        tenant.max_cost_per_run,
    )
    payload["max_time_minutes"] = min(
        int(payload.get("max_time_minutes", MAX_RUNTIME_MINUTES)),
        tenant.max_time_minutes_per_run,
    )
    payload["state_key"] = _tenant_state_key(tenant, payload.get("state_key"))

    os.environ["ANTHROPIC_API_KEY"] = _resolve_anthropic_key(tenant)
    os.environ["MANTIS_TENANT_ID"] = tenant.tenant_id
    profile_dir = _tenant_chrome_profile(tenant, payload["state_key"])
    os.environ["MANTIS_CHROME_PROFILE_DIR"] = str(profile_dir)

    # URL allowlist: enforce on start_url + any URL embedded in the
    # instruction. Mirrors /predict's per-tenant gate.
    if tenant.allowed_domains:
        plan_obj = {
            "base_url": payload.get("start_url", ""),
            "tasks": [{"intent": payload["instruction"]}],
        }
        try:
            hosts = extract_navigate_hosts(plan_obj)
            assert_hosts_allowed(hosts, tenant.is_domain_allowed)
        except PermissionError as exc:
            mantis_metrics.PREDICT_REQUESTS.labels(
                tenant_id=tenant.tenant_id, mode="cua", outcome="denied_allowlist"
            ).inc()
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    # Rate limit + concurrency — same plumbing as /predict run mode.
    limiter = get_rate_limiter()
    rate_decision = limiter.try_consume_rate_token(
        tenant.tenant_id, tenant.rate_limit_per_minute
    )
    _stash_rate_limit_headers(
        request, rate_decision, tenant.rate_limit_per_minute,
    )
    if not rate_decision.allowed:
        mantis_metrics.RATE_LIMIT_REJECTIONS.labels(
            tenant_id=tenant.tenant_id, kind="rate"
        ).inc()
        mantis_metrics.PREDICT_REQUESTS.labels(
            tenant_id=tenant.tenant_id, mode="cua", outcome="rate_limited"
        ).inc()
        raise HTTPException(
            status_code=429,
            detail=rate_decision.reason,
            headers={"Retry-After": str(int(rate_decision.retry_after_seconds) + 1)},
        )

    decision = limiter.try_acquire_concurrency_slot(
        tenant.tenant_id, tenant.max_concurrent_runs
    )
    if not decision.allowed:
        mantis_metrics.RATE_LIMIT_REJECTIONS.labels(
            tenant_id=tenant.tenant_id, kind="concurrent"
        ).inc()
        mantis_metrics.PREDICT_REQUESTS.labels(
            tenant_id=tenant.tenant_id, mode="cua", outcome="rate_limited"
        ).inc()
        raise HTTPException(
            status_code=429,
            detail=decision.reason,
            headers={"Retry-After": str(int(decision.retry_after_seconds) + 1)},
        )
    mantis_metrics.CONCURRENT_RUNS.labels(tenant_id=tenant.tenant_id).set(
        decision.concurrent
    )

    logger.info(
        "cua tenant=%s state_key=%s detached=%s start_url=%s",
        tenant.tenant_id,
        payload["state_key"],
        payload.get("detached", False),
        payload.get("start_url", ""),
    )

    try:
        response = await run_in_threadpool(runtime.run_pure_cua, payload)
        mantis_metrics.PREDICT_REQUESTS.labels(
            tenant_id=tenant.tenant_id, mode="cua", outcome="ok"
        ).inc()
        return response
    except ValueError as exc:
        mantis_metrics.PREDICT_REQUESTS.labels(
            tenant_id=tenant.tenant_id, mode="cua", outcome="bad_request"
        ).inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        mantis_metrics.PREDICT_REQUESTS.labels(
            tenant_id=tenant.tenant_id, mode="cua", outcome="error"
        ).inc()
        logger.exception("cua failed tenant=%s", tenant.tenant_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        limiter.release_concurrency_slot(tenant.tenant_id)
        mantis_metrics.CONCURRENT_RUNS.labels(tenant_id=tenant.tenant_id).set(
            limiter.get_concurrent(tenant.tenant_id)
        )
