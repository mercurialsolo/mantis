"""CLI integration glue — boots, grades, and tears down a sim env around a plan run.

Pulled out of :mod:`mantis_agent.cli` so the main CLI module stays
readable. Public surface:

* :class:`EnvSession` — context-manager wrapping ``backend.start`` +
  ``backend.wait_healthy`` and ``backend.stop`` (skip on ``keep=True``).
* :func:`detect_default_runtime` — picks ``modal`` when running on
  Modal, ``local`` otherwise. Caller can override with ``--runtime``.

Why a session class and not a free function: the cleanup happens on
multiple exit paths (success, runner exception, env health failure
mid-run). A context manager makes the cleanup contract explicit and the
caller's life simpler.
"""

from __future__ import annotations

import logging
import os
from contextlib import AbstractContextManager
from typing import Any

from .registry import get_backend
from .runtime import RuntimeBackend, RuntimeHandle

logger = logging.getLogger(__name__)


def detect_default_runtime() -> str:
    """Pick a sensible default for ``--runtime`` based on the environment.

    Heuristic: if the process is already running on Modal (``MODAL_TASK_ID``
    or ``MODAL_FUNCTION_NAME`` set) we default to ``modal``. Otherwise
    ``local``. The CLI's explicit ``--runtime`` flag always wins; this
    is only consulted when the user didn't pass it.
    """
    if os.environ.get("MODAL_TASK_ID") or os.environ.get("MODAL_FUNCTION_NAME"):
        return "modal"
    return "local"


class EnvSession(AbstractContextManager["EnvSession"]):
    """Context manager that owns an env's lifecycle for one plan run.

    Usage::

        with EnvSession("stub", runtime="local", seed=42) as session:
            url = session.url
            plan = template(plan, url)
            runner.run(plan)
            grading = grade_run(url, session.admin_token, task_id)

    Mantras:

    * ``__exit__`` calls ``backend.stop`` unless ``keep=True``.
    * ``backend.stop`` is idempotent; ``__exit__`` is safe to call twice.
    * If ``start`` succeeds but ``wait_healthy`` fails, we tear down
      before re-raising so we don't leak a half-booted container.
    """

    def __init__(
        self,
        env_name: str,
        *,
        runtime: str,
        seed: int = 42,
        now: str = "2026-01-15T09:00:00Z",
        keep: bool = False,
        health_timeout_s: float | None = None,
    ) -> None:
        self.env_name = env_name
        self.runtime = runtime
        self.seed = seed
        self.now = now
        self.keep = keep
        self.health_timeout_s = health_timeout_s
        self._backend: RuntimeBackend | None = None
        self._handle: RuntimeHandle | None = None

    @property
    def url(self) -> str:
        if self._handle is None:
            raise RuntimeError("EnvSession.url accessed before __enter__")
        return self._handle.url

    @property
    def admin_token(self) -> str:
        if self._handle is None:
            raise RuntimeError("EnvSession.admin_token accessed before __enter__")
        return self._handle.admin_token

    @property
    def handle(self) -> RuntimeHandle:
        if self._handle is None:
            raise RuntimeError("EnvSession.handle accessed before __enter__")
        return self._handle

    @property
    def backend(self) -> RuntimeBackend:
        if self._backend is None:
            raise RuntimeError("EnvSession.backend accessed before __enter__")
        return self._backend

    def __enter__(self) -> "EnvSession":
        backend = get_backend(self.runtime)
        self._backend = backend
        handle = backend.start(self.env_name, seed=self.seed, now=self.now)
        self._handle = handle
        try:
            if self.health_timeout_s is not None:
                backend.wait_healthy(handle, timeout_s=self.health_timeout_s)
            else:
                backend.wait_healthy(handle)
        except Exception:
            # Half-booted env — tear down before propagating so we don't
            # leak a hanging container / subprocess. ``stop`` is idempotent.
            try:
                backend.stop(handle)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.exception("env stop failed during health-wait error")
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.keep:
            logger.info(
                "EnvSession.keep=True — env %s left running at %s",
                self.env_name, self.url if self._handle else "?",
            )
            return
        if self._backend is None or self._handle is None:
            return
        try:
            self._backend.stop(self._handle)
        except Exception:  # noqa: BLE001 — exit must not raise
            logger.exception("env stop failed in EnvSession.__exit__")
