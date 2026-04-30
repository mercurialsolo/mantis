"""Multi-tenant API-key auth + per-tenant config resolution.

Backwards-compat: if MANTIS_TENANT_KEYS_PATH is unset, falls back to a
single-tenant mode using MANTIS_API_TOKEN (the v1 deployment shape). New
deployments mount a JSON keys file and get per-tenant isolation.

JSON shape (cleartext token → tenant config):

    {
      "tenant_keys": {
        "<x-mantis-token-value>": {
          "tenant_id": "tenant_a",
          "scope": ["run", "status"],
          "max_concurrent_runs": 3,
          "max_cost_per_run": 5.0,
          "max_time_minutes_per_run": 30,
          "anthropic_secret_name": "anthropic_api_key_tenant_a",
          "allowed_domains": ["*.boattrader.com", "crm.example.com"]
        }
      }
    }

The file should be mounted as a Baseten/EKS/GKE secret volume; this module
re-reads it on each request so token rotation is hot (no pod restart).
"""

from __future__ import annotations

import dataclasses
import hmac
import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger("mantis_agent.tenant_auth")


# ── Tenant config shape ─────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class TenantConfig:
    """Resolved tenant settings for a single request."""

    tenant_id: str
    scopes: tuple[str, ...] = ("run", "status", "result", "logs")
    max_concurrent_runs: int = 5
    max_cost_per_run: float = 25.0
    max_time_minutes_per_run: int = 60
    rate_limit_per_minute: int = 30
    anthropic_secret_name: str = "anthropic_api_key"
    allowed_domains: tuple[str, ...] = ()  # empty = no allowlist (legacy)
    webhook_url: str = ""  # if set, server POSTs run-completion notifications here
    webhook_secret_name: str = ""  # secret name for HMAC-signing webhook bodies

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def is_domain_allowed(self, host: str) -> bool:
        """Return True if `host` matches any allowed-domain pattern.

        Supports exact match and `*.example.com` wildcard. Empty allowlist
        means no domain restriction (legacy single-tenant mode).
        """
        if not self.allowed_domains:
            return True
        host = host.lower().strip()
        for pattern in self.allowed_domains:
            pattern = pattern.lower().strip()
            if pattern == host:
                return True
            if pattern.startswith("*.") and host.endswith(pattern[1:]):
                return True
        return False


DEFAULT_TENANT = TenantConfig(
    tenant_id="default",
    scopes=("run", "status", "result", "logs"),
    max_concurrent_runs=5,
    max_cost_per_run=25.0,
    max_time_minutes_per_run=60,
    rate_limit_per_minute=30,
    anthropic_secret_name="anthropic_api_key",
    allowed_domains=(),
)


# ── Key store with lazy-reload ──────────────────────────────────────────────
class TenantKeyStore:
    """File-backed key store. Reads MANTIS_TENANT_KEYS_PATH on each lookup
    (with 5s cache) so secret rotation doesn't need a pod restart."""

    _CACHE_TTL_SECONDS = 5.0

    def __init__(self, path: str | None = None) -> None:
        self._path = path or os.environ.get("MANTIS_TENANT_KEYS_PATH", "")
        self._lock = threading.Lock()
        self._cache: dict[str, TenantConfig] = {}
        self._cache_loaded_at: float = 0.0
        self._fallback_token = os.environ.get("MANTIS_API_TOKEN", "").strip()

    @property
    def is_multi_tenant(self) -> bool:
        return bool(self._path)

    def _load(self) -> dict[str, TenantConfig]:
        """Read + parse the keys file. Empty dict if missing/invalid."""
        if not self._path:
            return {}
        try:
            data = json.loads(Path(self._path).read_text())
        except FileNotFoundError:
            logger.warning("tenant keys file not found at %s", self._path)
            return {}
        except json.JSONDecodeError as exc:
            logger.error("tenant keys file invalid JSON: %s", exc)
            return {}

        out: dict[str, TenantConfig] = {}
        for token, raw in (data.get("tenant_keys") or {}).items():
            if not isinstance(raw, dict) or not raw.get("tenant_id"):
                continue
            out[token] = TenantConfig(
                tenant_id=str(raw["tenant_id"]),
                scopes=tuple(raw.get("scopes") or DEFAULT_TENANT.scopes),
                max_concurrent_runs=int(
                    raw.get("max_concurrent_runs", DEFAULT_TENANT.max_concurrent_runs)
                ),
                max_cost_per_run=float(
                    raw.get("max_cost_per_run", DEFAULT_TENANT.max_cost_per_run)
                ),
                max_time_minutes_per_run=int(
                    raw.get("max_time_minutes_per_run", DEFAULT_TENANT.max_time_minutes_per_run)
                ),
                rate_limit_per_minute=int(
                    raw.get("rate_limit_per_minute", DEFAULT_TENANT.rate_limit_per_minute)
                ),
                anthropic_secret_name=str(
                    raw.get("anthropic_secret_name", DEFAULT_TENANT.anthropic_secret_name)
                ),
                allowed_domains=tuple(raw.get("allowed_domains") or ()),
                webhook_url=str(raw.get("webhook_url") or ""),
                webhook_secret_name=str(raw.get("webhook_secret_name") or ""),
            )
        return out

    def _refresh(self) -> dict[str, TenantConfig]:
        with self._lock:
            now = time.time()
            if now - self._cache_loaded_at > self._CACHE_TTL_SECONDS:
                self._cache = self._load()
                self._cache_loaded_at = now
            return self._cache

    def resolve(self, presented_token: str) -> TenantConfig | None:
        """Return TenantConfig for a presented X-Mantis-Token, or None.

        Uses constant-time compare to avoid timing oracles. Falls back to
        single-tenant mode (DEFAULT_TENANT) if no keys file is configured
        but the legacy MANTIS_API_TOKEN matches.
        """
        if not presented_token:
            return None

        # Multi-tenant path
        if self.is_multi_tenant:
            store = self._refresh()
            for token, cfg in store.items():
                if hmac.compare_digest(presented_token, token):
                    return cfg
            return None

        # Single-tenant fallback
        if self._fallback_token and hmac.compare_digest(presented_token, self._fallback_token):
            return DEFAULT_TENANT
        return None


# Module-level singleton — server reuses this across requests.
_KEY_STORE: TenantKeyStore | None = None


def get_key_store() -> TenantKeyStore:
    global _KEY_STORE
    if _KEY_STORE is None:
        _KEY_STORE = TenantKeyStore()
    return _KEY_STORE


def reset_key_store() -> None:
    """Test helper: forget the singleton so MANTIS_TENANT_KEYS_PATH /
    MANTIS_API_TOKEN env changes are picked up."""
    global _KEY_STORE
    _KEY_STORE = None
