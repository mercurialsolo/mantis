"""Tenant-scoped runtime recipe persistence (#809).

Code-shipped recipes (``mantis_agent.recipes.<name>``) require a fork
and redeploy to add or change. Runtime recipes live on the shared
volume at ``/data/tenants/<tenant>/recipes/<name>.json`` and can be
registered, listed, fetched, and deleted via HTTP — no redeploy.

The store is intentionally small:

- One JSON file per recipe under the tenant's recipes dir.
- ``ExtractionSchema.from_dict`` validates the payload at write time
  so malformed schemas fail at registration, not at extract time.
- Lookup precedence: runtime (tenant-scoped) → code-shipped (global).
  A tenant can therefore override a shipped recipe by name without
  touching anyone else's behaviour.

This module is the persistence layer; the HTTP routes that surface it
live in the deploy-side server modules (``modal_cua_server.py``,
``baseten_server/routes.py``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..extraction import ExtractionSchema
from ..server_utils import safe_state_key


# Recipe names must be filesystem-safe and reasonably short. The
# regex is intentionally narrower than ``safe_state_key`` would
# tolerate — runtime recipes are user-named, so we want predictable
# slug-style names rather than arbitrary tenant ids.
_NAME_MAX_LEN = 64
_VALID_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
)


class RuntimeRecipeError(ValueError):
    """Raised on invalid recipe name or malformed schema payload."""


def _data_root() -> Path:
    return Path(os.environ.get("MANTIS_DATA_DIR", "/data"))


def _tenant_recipes_dir(tenant_id: str) -> Path:
    return _data_root() / "tenants" / safe_state_key(tenant_id) / "recipes"


def _validate_name(name: str) -> str:
    """Reject names with path separators, dots, or unexpected chars.

    Returns the validated name unchanged so callers can use it in the
    file path without re-sanitising.
    """
    if not isinstance(name, str) or not name:
        raise RuntimeRecipeError("recipe name is required")
    if len(name) > _NAME_MAX_LEN:
        raise RuntimeRecipeError(
            f"recipe name too long (max {_NAME_MAX_LEN} chars)",
        )
    if any(c not in _VALID_NAME_CHARS for c in name):
        raise RuntimeRecipeError(
            "recipe name may only contain letters, digits, hyphen, underscore",
        )
    # Defence in depth: also reject names that look like path traversal
    # attempts even though the charset already excludes them.
    if name.startswith("-") or name in {".", ".."}:
        raise RuntimeRecipeError("invalid recipe name")
    return name


def register(
    tenant_id: str, name: str, schema_payload: dict
) -> dict:
    """Write a runtime recipe to the tenant's recipes dir.

    The payload must be in the shape accepted by
    ``ExtractionSchema.from_dict``. Validation happens at write time —
    malformed payloads raise :class:`RuntimeRecipeError` before any
    file is touched.

    Returns the persisted shape (the schema payload plus the name).
    Overwrites any existing recipe with the same name.
    """
    name = _validate_name(name)
    if not isinstance(schema_payload, dict):
        raise RuntimeRecipeError("schema must be a dict")
    try:
        ExtractionSchema.from_dict(schema_payload)
    except (ValueError, TypeError) as exc:
        raise RuntimeRecipeError(f"invalid schema: {exc}") from exc

    target_dir = _tenant_recipes_dir(tenant_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{name}.json"
    body = {"name": name, "schema": schema_payload}
    target.write_text(json.dumps(body, indent=2))
    return body


def get(tenant_id: str, name: str) -> dict | None:
    """Return the persisted ``{name, schema}`` blob or None if missing."""
    name = _validate_name(name)
    target = _tenant_recipes_dir(tenant_id) / f"{name}.json"
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_schema(tenant_id: str, name: str) -> ExtractionSchema | None:
    """Resolve a tenant-scoped runtime recipe to an ``ExtractionSchema``.

    Returns ``None`` if no runtime recipe by that name exists. The
    caller is expected to fall back to ``mantis_agent.recipes.load_schema``
    for code-shipped recipes.
    """
    body = get(tenant_id, name)
    if body is None:
        return None
    payload = body.get("schema")
    if not isinstance(payload, dict):
        return None
    try:
        return ExtractionSchema.from_dict(payload)
    except (ValueError, TypeError):
        return None


def list_recipes(tenant_id: str) -> list[dict]:
    """List runtime recipes for the tenant.

    Returns a list of summary dicts (``{"name": str}``) — full schema
    bodies are not echoed back to keep list responses cheap. Use
    :func:`get` to fetch a single recipe's full body.
    """
    out: list[dict] = []
    target_dir = _tenant_recipes_dir(tenant_id)
    if not target_dir.exists():
        return out
    for path in sorted(target_dir.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            continue
        name = path.stem
        try:
            _validate_name(name)
        except RuntimeRecipeError:
            # A leftover file with an unexpected name — skip silently.
            continue
        out.append({"name": name})
    return out


def delete(tenant_id: str, name: str) -> bool:
    """Delete a tenant runtime recipe. Returns True if a file was removed."""
    name = _validate_name(name)
    target = _tenant_recipes_dir(tenant_id) / f"{name}.json"
    if not target.exists():
        return False
    try:
        target.unlink()
        return True
    except OSError:
        return False


__all__ = [
    "RuntimeRecipeError",
    "delete",
    "get",
    "list_recipes",
    "load_schema",
    "register",
]
