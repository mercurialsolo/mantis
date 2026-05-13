"""Simulated environments (#331 / harness #336).

This package contains the *integration layer* that lets `mantis plan run`
talk to a self-hosted, deterministic web env — the same contract whether
the env runs locally (Docker / subprocess) or on Modal.

Public surface:

- :class:`RuntimeBackend` — start/health/url/events/stop protocol every
  backend implements.
- :class:`RuntimeHandle` — opaque handle returned by ``start`` and
  threaded back into ``stop`` / ``fetch_events``.
- :func:`get_backend` — resolves a backend by name (``local`` / ``modal``
  / ``e2b``). ``e2b`` raises ``NotImplementedError``; the slot is
  reserved so a future PR can land it without touching plans / CLI.
- :func:`substitute_env_url` — templates ``{{ENV_URL}}`` into a plan
  payload before the runner sees it.

Env images are not in this package. They live under ``deploy/sim_envs/``
(Modal apps) and the per-env Dockerfiles. This package only contains the
glue that boots them and reaches them over HTTP.

Adding a new backend is a one-file change: implement ``RuntimeBackend``,
register it in ``registry.py``. Nothing in ``MicroPlanRunner``,
``SiteConfig``, the planner, or the executor needs to change.
"""

from __future__ import annotations

from .registry import get_backend, list_backends
from .runtime import RuntimeBackend, RuntimeHandle
from .templating import ENV_URL_PLACEHOLDER, substitute_env_url

__all__ = [
    "ENV_URL_PLACEHOLDER",
    "RuntimeBackend",
    "RuntimeHandle",
    "get_backend",
    "list_backends",
    "substitute_env_url",
]
