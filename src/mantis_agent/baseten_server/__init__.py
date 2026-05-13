"""Baseten workload server for Mantis CUA.

Public surface (re-exported here for backwards compatibility — the
single-file ``mantis_agent.baseten_server`` was split into a package in
PR #107):

- :data:`app` — the FastAPI application that Baseten Truss / Modal mount
- :class:`BasetenCUARuntime` — the model + run lifecycle singleton

External callers — both the deployed Baseten Truss config and the test
suite — import via ``from mantis_agent.baseten_server import app``.
That import path is preserved.

Imports are deferred via ``__getattr__`` so that lightweight consumers
that only need :mod:`baseten_server.middleware` or
:mod:`baseten_server.paths` (e.g. the Modal HTTP endpoint added in
#342) don't pay the cost of importing the full GPU / Chrome runtime
chain.
"""

from __future__ import annotations

__all__ = ["app", "runtime", "BasetenCUARuntime"]


def __getattr__(name: str):  # PEP 562
    if name in {"app", "runtime"}:
        from .routes import app, runtime
        return {"app": app, "runtime": runtime}[name]
    if name == "BasetenCUARuntime":
        from .runtime import BasetenCUARuntime
        return BasetenCUARuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
