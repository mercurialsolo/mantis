"""HTTP client for sim-env oracle + mutations endpoints.

Companion to :mod:`mantis_agent.gym.grading`. ``grade_run`` already wraps
the terminal ``GET /__env__/oracle?task_id=<id>`` call; this module adds
the per-step counterpart — ``GET /__env__/mutations`` — used as a cheap
verifier signal for training rewards.

Why a separate module from ``gym.grading``: terminal grading happens
once per run and lives near the CLI; mutation polling happens during
training inside the reward path and lives next to the env-session glue.
Keeping them apart means a future Modal reward worker can import
``sim_envs.oracle_client`` without dragging the grading + CLI surface in.

The functions never raise. Network failures populate an ``error`` key on
the returned dict so the caller can decide whether to fall back to a
vision-based verifier or just skip the reward contribution. This shape
matches the rest of the sim-envs runtime: best-effort, observable, no
exceptions across the boundary.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def fetch_mutations(
    url: str,
    admin_token: str,
    *,
    since_id: int = 0,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Hit ``GET <url>/__env__/mutations[?since=<id>]`` and return the parsed body.

    Args:
        url: env base URL (no trailing slash required).
        admin_token: value for the ``X-Env-Admin`` header.
        since_id: when > 0, only return entries with ``id`` > since_id.
            Mantis envs use a monotonically increasing integer id per
            mutation, so the caller stores the last-seen id between
            polls and passes it back here.
        timeout_s: socket timeout.

    Returns:
        On success: ``{"mutations": [<entry>, ...]}`` exactly as the env
        returned. Each entry has at least ``id`` (int), ``operation``
        (str), ``target_type`` (str), ``target_id`` (str), and
        ``payload`` (dict).

        On failure: ``{"mutations": [], "error": "<reason>"}``. The
        ``mutations`` key is always present so callers can treat the
        response uniformly.
    """
    if not url:
        return {"mutations": [], "error": "url is empty"}
    if not admin_token:
        return {"mutations": [], "error": "admin_token is empty"}

    base = url.rstrip("/")
    suffix = f"?since={int(since_id)}" if since_id > 0 else ""
    full = f"{base}/__env__/mutations{suffix}"
    req = Request(full, headers={"X-Env-Admin": admin_token})
    try:
        with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 — env URL only
            body = resp.read().decode("utf-8")
            payload = json.loads(body)
    except HTTPError as exc:
        return {"mutations": [], "error": f"HTTP {exc.code}: {exc.reason}"}
    except (URLError, ConnectionError, OSError, TimeoutError) as exc:
        return {"mutations": [], "error": f"network: {exc!r}"}
    except json.JSONDecodeError as exc:
        return {"mutations": [], "error": f"non-JSON body: {exc!r}"}

    if not isinstance(payload, dict):
        return {"mutations": [], "error": f"non-dict payload: {type(payload).__name__}"}

    mutations = payload.get("mutations")
    if not isinstance(mutations, list):
        return {"mutations": [], "error": "no mutations list in payload"}

    return {"mutations": mutations}


def last_mutation_id(mutations: list[dict[str, Any]]) -> int:
    """Return the highest ``id`` in a mutations list, or 0 if empty.

    The caller uses this to advance its ``since_id`` between polls.
    Mutation ids are 1-indexed and strictly increasing, so taking the
    max is sufficient — no need to sort.
    """
    if not mutations:
        return 0
    return max(int(m.get("id") or 0) for m in mutations)


__all__ = ["fetch_mutations", "last_mutation_id"]
