"""RuntimeBackend protocol — pluggable hosting for simulated envs.

Every backend implements the same five operations:

* ``start(env_name, seed, now, admin_token)`` — boot a fresh env instance
  with the supplied seed + frozen clock, return a :class:`RuntimeHandle`
  the caller threads back into the other operations.
* ``wait_healthy(handle, timeout_s)`` — block until ``/__env__/health``
  returns 200 or raise :class:`TimeoutError`.
* ``get_url(handle)`` — agent-facing base URL, e.g.
  ``http://localhost:8001`` for the local backend, ``https://<id>-myenv-fn.modal.run``
  for the Modal backend.
* ``fetch_events(handle, since)`` — pull the env's ``events.jsonl`` (or
  whatever equivalent the backend exposes) since the supplied wall-clock
  timestamp. Returns a list of dicts; merging into the agent trace is
  the caller's job (see :mod:`mantis_agent.gym.trace_exporter`).
* ``stop(handle)`` — tear down the instance. Idempotent: calling twice
  with the same handle should not raise.

The protocol is intentionally narrow. Anything beyond these five ops
(grading, URL templating, plan dispatch, batch runs) is backend-agnostic
and lives in higher layers.

The handle is a plain dataclass rather than an opaque ID because some
backends (Modal) need to remember sub-objects (the deployed app handle,
the per-run suffix, etc.) for ``stop`` and ``fetch_events``. Storing
them on the handle keeps the call sites clean and the backend stateless
between calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class RuntimeHandle:
    """Opaque-ish handle returned by ``RuntimeBackend.start``.

    Most fields are populated by the backend and only meaningful to
    that backend. ``url`` and ``admin_token`` are the two callers always
    consume — everything else is bookkeeping.
    """

    env_name: str
    url: str
    admin_token: str
    backend: str

    # Backend-specific state. The local backend stuffs a process /
    # container id; the Modal backend stuffs the per-run app name and
    # function handle. Treat as private to the backend that produced it.
    extra: dict[str, Any] = field(default_factory=dict)

    # Unix epoch seconds — the moment ``start`` completed. ``fetch_events``
    # uses this as the default ``since`` when the caller doesn't pass one.
    started_at: float = 0.0


@runtime_checkable
class RuntimeBackend(Protocol):
    """The five-method contract every hosting backend implements.

    Implementations live in:

    * :mod:`mantis_agent.sim_envs.local` — Docker (or stub subprocess)
    * :mod:`mantis_agent.sim_envs.modal_backend` — one ``modal.App`` per env
    * :mod:`mantis_agent.sim_envs.e2b` — stub raising NotImplementedError
    """

    name: str

    def start(
        self,
        env_name: str,
        *,
        seed: int = 42,
        now: str = "2026-01-15T09:00:00Z",
        admin_token: str | None = None,
    ) -> RuntimeHandle:
        ...

    def wait_healthy(self, handle: RuntimeHandle, *, timeout_s: float = 30.0) -> None:
        ...

    def get_url(self, handle: RuntimeHandle) -> str:
        ...

    def fetch_events(
        self,
        handle: RuntimeHandle,
        *,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def stop(self, handle: RuntimeHandle) -> None:
        ...
