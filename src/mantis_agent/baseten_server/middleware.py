"""Auth + secret middleware for the Baseten CUA workload.

``X-Mantis-Token`` is the container-level auth header (deliberately
distinct from Baseten's gateway-side ``Authorization: Api-Key`` so the
two layers do not collide). Tenant resolution and Anthropic key lookup
both happen here.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, Header, HTTPException

from ..tenant_auth import DEFAULT_TENANT, TenantConfig, get_key_store


SECRET_ENV_MAP = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "proxy_url": "PROXY_URL",
    "proxy_user": "PROXY_USER",
    "proxy_pass": "PROXY_PASS",
    "hf_access_token": "HF_TOKEN",
    "mantis_api_token": "MANTIS_API_TOKEN",
}


def require_mantis_token(
    x_mantis_token: str | None = Header(default=None, alias="X-Mantis-Token"),
) -> TenantConfig:
    """Container-level auth → resolved TenantConfig.

    Uses a custom header (``X-Mantis-Token``) instead of ``Authorization: Bearer``
    so it does not collide with Baseten's gateway auth, which sends
    ``Authorization: Api-Key <baseten_key>`` to the container.

    Backwards-compat: if MANTIS_TENANT_KEYS_PATH is unset and MANTIS_API_TOKEN
    matches, returns DEFAULT_TENANT (single-tenant mode). Multi-tenant mode is
    enabled by mounting a JSON keys file and setting MANTIS_TENANT_KEYS_PATH.
    """
    store = get_key_store()
    if not store.is_multi_tenant and not os.environ.get("MANTIS_API_TOKEN", "").strip():
        raise HTTPException(status_code=503, detail="server auth not configured")
    if not x_mantis_token:
        raise HTTPException(status_code=401, detail="missing X-Mantis-Token header")
    tenant = store.resolve(x_mantis_token)
    if tenant is None:
        raise HTTPException(status_code=401, detail="invalid X-Mantis-Token")
    return tenant


def require_run_scope(tenant: TenantConfig = Depends(require_mantis_token)) -> TenantConfig:
    if not tenant.has_scope("run"):
        raise HTTPException(status_code=403, detail="tenant lacks 'run' scope")
    return tenant


def read_secret(name: str) -> str:
    path = Path("/secrets") / name
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def resolve_anthropic_key(tenant: TenantConfig) -> str:
    """Return the Anthropic key this tenant should use.

    Reads from the secret named by the tenant's ``anthropic_secret_name``
    (each tenant can have its own Anthropic billing). Falls back to the
    legacy ``ANTHROPIC_API_KEY`` env var if the per-tenant secret isn't
    present on disk.
    """
    name = tenant.anthropic_secret_name or DEFAULT_TENANT.anthropic_secret_name
    value = read_secret(name)
    if value:
        return value
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def load_secret_environment() -> None:
    for secret_name, env_name in SECRET_ENV_MAP.items():
        if os.environ.get(env_name):
            continue
        value = read_secret(secret_name)
        if value:
            os.environ[env_name] = value
