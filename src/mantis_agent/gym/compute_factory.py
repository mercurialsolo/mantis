"""Umbrella factory for the unified `ComputeClient` contract (#785, PR 2).

Dispatches between the two compute planes based on the resolved
`ComputeBackend`:

- `ComputeBackend.COMPUTER_PLANE` → `make_computer_client(...)`
  (existing factory in `computer_client.py`).
- `ComputeBackend.BROWSER_USE_PLANE` → `BrowserUsePlaneClient(...)`
  (new client in `browser_use_plane_client.py`).

`make_compute_client` is the single brain-plane entry point — call sites
don't need to know which plane they're on. The selected plane's client
is constructed; capabilities advertised at `session_init` are then
checked against a per-executor `CapabilityAllowlist` (enforcement seam
is one layer up in the executor; see `run_browser_use.py`).

Backwards compat: existing call sites that construct
`make_computer_client(...)` directly keep working — this module sits
alongside, not in place.
"""

from __future__ import annotations

from typing import Any

from .browser_use_plane_client import BrowserUsePlaneClient
from .compute_contract import Capabilities, ComputeBackend
from .computer_client import (
    ComputerClient,
    ComputerPlaneConfig,
    make_computer_client,
)


def make_compute_client(
    compute_backend: ComputeBackend | str = ComputeBackend.COMPUTER_PLANE,
    *,
    computer_plane_cfg: ComputerPlaneConfig | None = None,
    browser_use_base_url: str | None = None,
    browser_use_auth_token: str | None = None,
    **env_kwargs: Any,
) -> ComputerClient:
    """Build a `ComputeClient` for the resolved compute plane.

    `compute_backend` accepts either a `ComputeBackend` enum or its
    string value (which is what `plan.runtime.compute_backend` carries on
    the wire). Default is `COMPUTER_PLANE` — pure-CUA stays the path
    when no plan-level or submission-time override is set.

    Pass `computer_plane_cfg` for Computer Plane runs;
    `browser_use_base_url` for Browser-Use Plane runs. Remaining
    `env_kwargs` are forwarded to the underlying client (e.g.
    `start_url`, `viewport`, `proxy_server`).
    """
    backend = (
        compute_backend
        if isinstance(compute_backend, ComputeBackend)
        else ComputeBackend(compute_backend)
    )

    if backend is ComputeBackend.COMPUTER_PLANE:
        return make_computer_client(computer_plane_cfg, **env_kwargs)

    if backend is ComputeBackend.BROWSER_USE_PLANE:
        if not browser_use_base_url:
            raise ValueError(
                "ComputeBackend.BROWSER_USE_PLANE requires browser_use_base_url "
                "(URL of the deployed Browser-Use Plane Modal function)."
            )
        return BrowserUsePlaneClient(
            base_url=browser_use_base_url,
            auth_token=browser_use_auth_token,
            **env_kwargs,
        )

    raise ValueError(f"unknown ComputeBackend: {backend!r}")


def default_capabilities_for(backend: ComputeBackend | str) -> Capabilities:
    """Capabilities a backend advertises before `session_init`.

    Used by call sites that need a Capabilities snapshot pre-session —
    e.g. setting up the per-executor `CapabilityAllowlist` based on
    which plane was resolved.
    """
    b = backend if isinstance(backend, ComputeBackend) else ComputeBackend(backend)
    if b is ComputeBackend.COMPUTER_PLANE:
        return Capabilities.for_computer_plane()
    if b is ComputeBackend.BROWSER_USE_PLANE:
        return Capabilities.for_browser_use_plane()
    raise ValueError(f"unknown ComputeBackend: {backend!r}")
