"""Framework-agnostic predict-payload prep + per-profile lock helpers (#342).

The Baseten ASGI app and the Modal ASGI app both need the same pre-spawn
work: validate the JSON, clamp the per-tenant caps, resolve
``profile_id`` / ``workflow_id`` / legacy ``state_key`` (#341), enforce
the tenant URL allowlist, and write a tenant-scoped lockfile so two
concurrent submissions for the same ``profile_id`` get a 409 instead of
silently corrupting Chrome's user-data-dir.

This module raises :class:`DispatchError` instead of ``HTTPException``
so it stays usable from contexts that don't depend on FastAPI.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..api_schemas import (
    MAX_COST_USD,
    MAX_RUNTIME_MINUTES,
    PredictRequest,
    PureCUARequest,
    assert_hosts_allowed,
    extract_navigate_hosts,
)
from ..baseten_server.paths import (
    tenant_chrome_profile,
    tenant_lock_path,
    tenant_profile_id,
    tenant_state_key,
    tenant_workflow_id,
)
from ..baseten_server.middleware import resolve_anthropic_key
from ..tenant_auth import TenantConfig


class DispatchError(Exception):
    """Validation / authorization failure surfaced from the prep step.

    ``status_code`` maps to HTTP cleanly; FastAPI shells re-raise as
    ``HTTPException(status_code, detail)``.
    """

    def __init__(self, status_code: int, detail: str, *, run_id: str = "") -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.run_id = run_id


def _resolve_identity(payload: dict[str, Any], tenant: TenantConfig) -> None:
    """In-place: set ``profile_id``, ``workflow_id``, ``state_key`` (#341)."""
    caller_state_key = payload.get("state_key")
    caller_profile_id = payload.get("profile_id")
    caller_workflow_id = payload.get("workflow_id")
    if caller_profile_id or caller_workflow_id:
        payload["profile_id"] = tenant_profile_id(tenant, caller_profile_id)
        payload["workflow_id"] = tenant_workflow_id(tenant, caller_workflow_id)
    else:
        legacy = tenant_state_key(tenant, caller_state_key)
        payload["profile_id"] = legacy
        payload["workflow_id"] = legacy
    payload["state_key"] = payload["workflow_id"]


def _check_allowlist(plan_obj: Any, tenant: TenantConfig) -> None:
    if not tenant.allowed_domains:
        return
    hosts = extract_navigate_hosts(plan_obj)
    try:
        assert_hosts_allowed(hosts, tenant.is_domain_allowed)
    except PermissionError as exc:
        raise DispatchError(403, str(exc)) from exc


def prepare_predict_payload(raw: dict[str, Any], tenant: TenantConfig) -> dict[str, Any]:
    """Validate + normalize a ``/v1/predict`` request body.

    Returns the prepared payload dict with caps clamped, identity
    resolved + tenant-prefixed, and chrome-profile env vars set. The
    returned dict carries ``profile_id``, ``workflow_id`` and (legacy)
    ``state_key`` so downstream code that reads any of the three keeps
    working.

    Raises :class:`DispatchError` on validation, allowlist, or
    permission failures.
    """
    if not isinstance(raw, dict):
        raise DispatchError(400, "request body must be a JSON object")

    try:
        req = PredictRequest.model_validate(raw)
    except Exception as exc:
        raise DispatchError(400, f"invalid request: {exc}") from exc

    payload = req.model_dump(exclude_none=True)
    payload["max_cost"] = min(
        float(payload.get("max_cost", MAX_COST_USD)),
        tenant.max_cost_per_run,
    )
    payload["max_time_minutes"] = min(
        int(payload.get("max_time_minutes", MAX_RUNTIME_MINUTES)),
        tenant.max_time_minutes_per_run,
    )
    _resolve_identity(payload, tenant)

    os.environ["ANTHROPIC_API_KEY"] = resolve_anthropic_key(tenant)
    os.environ["MANTIS_TENANT_ID"] = tenant.tenant_id
    profile_dir = tenant_chrome_profile(tenant, payload["profile_id"])
    os.environ["MANTIS_CHROME_PROFILE_DIR"] = str(profile_dir)

    # Tier-2 URL allowlist — enforce on whatever pre-built plan shape
    # the caller submitted. ``micro`` / ``plan_text`` decomposition
    # happens downstream and is gated separately.
    if req.action is None:
        plan_obj: Any = None
        if req.task_suite is not None:
            plan_obj = req.task_suite
        elif req.task_file_contents:
            try:
                plan_obj = json.loads(req.task_file_contents)
            except json.JSONDecodeError:
                plan_obj = None
        if plan_obj is not None:
            _check_allowlist(plan_obj, tenant)

    return payload


def prepare_cua_payload(raw: dict[str, Any], tenant: TenantConfig) -> dict[str, Any]:
    """Validate + normalize a ``/v1/cua`` request body.

    Same shape as :func:`prepare_predict_payload`, but routed through
    :class:`PureCUARequest`. The allowlist gate checks the ``start_url``
    plus any URL embedded in ``instruction``.
    """
    if not isinstance(raw, dict):
        raise DispatchError(400, "request body must be a JSON object")

    try:
        req = PureCUARequest.model_validate(raw)
    except Exception as exc:
        raise DispatchError(400, f"invalid request: {exc}") from exc

    payload = req.model_dump(exclude_none=True)
    payload["max_cost"] = min(
        float(payload.get("max_cost", MAX_COST_USD)),
        tenant.max_cost_per_run,
    )
    payload["max_time_minutes"] = min(
        int(payload.get("max_time_minutes", MAX_RUNTIME_MINUTES)),
        tenant.max_time_minutes_per_run,
    )
    _resolve_identity(payload, tenant)

    os.environ["ANTHROPIC_API_KEY"] = resolve_anthropic_key(tenant)
    os.environ["MANTIS_TENANT_ID"] = tenant.tenant_id
    profile_dir = tenant_chrome_profile(tenant, payload["profile_id"])
    os.environ["MANTIS_CHROME_PROFILE_DIR"] = str(profile_dir)

    plan_obj = {
        "base_url": payload.get("start_url", ""),
        "tasks": [{"intent": payload["instruction"]}],
    }
    _check_allowlist(plan_obj, tenant)

    return payload


# ── Profile lock (#342) ───────────────────────────────────────────


def acquire_profile_lock(
    tenant: TenantConfig,
    profile_id: str,
    run_id: str,
    *,
    stale_after_seconds: int = 4 * 3600,
) -> bool:
    """Attempt to acquire the per-tenant per-profile lock.

    Returns ``True`` on success. Returns ``False`` if the lock is held
    by a still-fresh run (caller should surface a 409 and include the
    held ``run_id`` from :func:`read_profile_lock`).

    Stale locks (older than ``stale_after_seconds``, default 4h) are
    treated as released and overwritten — covers the case where an
    executor crashed without writing terminal status.
    """
    path = tenant_lock_path(tenant, profile_id)
    if path.exists():
        try:
            age = (
                datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
            )
        except OSError:
            age = 0.0
        if age < stale_after_seconds:
            try:
                held = path.read_text(encoding="utf-8").strip().splitlines()[0]
            except OSError:
                held = ""
            if held and held != run_id:
                return False
    _write_lock(path, run_id)
    return True


def read_profile_lock(tenant: TenantConfig, profile_id: str) -> str:
    """Return the ``run_id`` currently holding the lock, or empty string."""
    path = tenant_lock_path(tenant, profile_id)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip().splitlines()[0]
    except OSError:
        return ""


def release_profile_lock(tenant: TenantConfig, profile_id: str) -> None:
    """Idempotent — missing file is treated as success."""
    path = tenant_lock_path(tenant, profile_id)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        # Best-effort. A failed unlink turns into a stale-lock
        # next-acquire path, not a correctness bug.
        pass


def _write_lock(path: Path, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    path.write_text(f"{run_id}\n{now}\n", encoding="utf-8")
