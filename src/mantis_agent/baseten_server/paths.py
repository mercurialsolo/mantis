"""Path helpers for the Baseten CUA workload.

Resolves the per-tenant data subtree, run IDs, and on-disk model files.
Pure filesystem helpers — no FastAPI / runtime dependencies.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..server_utils import safe_state_key
from ..tenant_auth import TenantConfig


def data_root() -> Path:
    """Top-level data dir. Per-tenant subdirs live under this."""
    root = Path(os.environ.get("MANTIS_DATA_DIR", "/workspace/mantis-data"))
    root.mkdir(parents=True, exist_ok=True)
    for child in ("results", "runs", "screenshots", "checkpoints", "chrome-profile", "tenants"):
        (root / child).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MANTIS_DEBUG_DIR", str(root / "screenshots" / "claude_debug"))
    return root


def tenant_root(tenant: TenantConfig) -> Path:
    """Per-tenant subtree of the data volume. Caller cannot escape this prefix.

    Layout::

      /workspace/mantis-data/tenants/<tenant_id>/
        ├── runs/<run_id>/{status,result,leads,events}
        ├── checkpoints/<state_key>.json
        ├── chrome-profile/<state_key>/
        └── screenshots/<run_id>/
    """
    root = data_root() / "tenants" / safe_state_key(tenant.tenant_id)
    for child in ("runs", "checkpoints", "chrome-profile", "screenshots"):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def tenant_state_key(tenant: TenantConfig, caller_state_key: str | None) -> str:
    """Server-namespaced state key. Caller's value is sanitized + prefixed."""
    base = safe_state_key(caller_state_key or "default")
    return f"{safe_state_key(tenant.tenant_id)}__{base}"


def tenant_chrome_profile(tenant: TenantConfig, state_key: str) -> Path:
    """Per-tenant, per-state-key Chrome profile dir."""
    profile = tenant_root(tenant) / "chrome-profile" / safe_state_key(state_key)
    profile.mkdir(parents=True, exist_ok=True)
    return profile


def repo_root() -> Path:
    return Path(os.environ.get("MANTIS_REPO_ROOT", "/workspace/cua-agent"))


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("none of these paths exist: " + ", ".join(str(p) for p in paths))


def find_gguf(model_dir: Path, preferred: str = "") -> Path:
    if preferred:
        return first_existing([Path(preferred), model_dir / preferred])

    candidates = [
        path
        for path in model_dir.glob("*.gguf")
        if "mmproj" not in path.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"no model GGUF found in {model_dir}")

    def rank(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        if "q8_0" in name:
            return (0, name)
        if "q4_k_m" in name:
            return (1, name)
        return (2, name)

    return sorted(candidates, key=rank)[0]


def find_mmproj(model_dir: Path, preferred: str = "") -> Path | None:
    if preferred:
        path = Path(preferred)
        return path if path.exists() else model_dir / preferred
    candidates = sorted(model_dir.glob("*mmproj*.gguf"))
    return candidates[0] if candidates else None
