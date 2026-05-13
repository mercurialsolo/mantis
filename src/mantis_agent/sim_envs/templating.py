"""Plan URL templating — substitute ``{{ENV_URL}}`` with the live env URL.

Plans under ``plans/<env>/`` carry a placeholder in the first navigate
step rather than a hardcoded ``http://localhost:8001`` so the same plan
JSON works against a local Docker env, a Modal env, or (one day) an
e2b env with no edits.

The substitution is a pure string replace inside the plan payload right
before the runner sees it. We touch every string-shaped field on every
step (intent + params values) — placeholders that show up in plain
intents like ``"Navigate to {{ENV_URL}}/contacts"`` work the same way as
the canonical ``params.url`` slot.

Out of scope: more elaborate templating (Jinja, dotted lookups). Plans
should stay declarative JSON; if a future env needs N variables we
extend the placeholder set, not the templating engine.
"""

from __future__ import annotations

from typing import Any

ENV_URL_PLACEHOLDER = "{{ENV_URL}}"


def _replace_in(value: Any, env_url: str) -> Any:
    """Recursively replace ``ENV_URL_PLACEHOLDER`` in strings inside a JSON-shaped value."""
    if isinstance(value, str):
        if ENV_URL_PLACEHOLDER in value:
            return value.replace(ENV_URL_PLACEHOLDER, env_url)
        return value
    if isinstance(value, list):
        return [_replace_in(item, env_url) for item in value]
    if isinstance(value, dict):
        return {k: _replace_in(v, env_url) for k, v in value.items()}
    return value


def substitute_env_url(plan_payload: dict[str, Any], env_url: str) -> dict[str, Any]:
    """Return a copy of ``plan_payload`` with ``{{ENV_URL}}`` swapped for ``env_url``.

    Trailing slash on ``env_url`` is stripped — plans usually template
    ``{{ENV_URL}}/some/path`` so we don't want a double slash.
    """
    base = env_url.rstrip("/")
    return _replace_in(plan_payload, base)
