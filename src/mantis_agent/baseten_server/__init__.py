"""Baseten workload server for Mantis CUA.

Public surface (re-exported here for backwards compatibility — the
single-file ``mantis_agent.baseten_server`` was split into a package in
PR #107):

- :data:`app` — the FastAPI application that Baseten Truss / Modal mount
- :class:`BasetenCUARuntime` — the model + run lifecycle singleton

External callers — both the deployed Baseten Truss config and the test
suite — import via ``from mantis_agent.baseten_server import app``.
That import path is preserved.
"""

from __future__ import annotations

from .routes import app, runtime
from .runtime import BasetenCUARuntime

__all__ = ["app", "runtime", "BasetenCUARuntime"]
