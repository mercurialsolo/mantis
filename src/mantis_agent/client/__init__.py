"""Typed Python client for the Mantis HTTP API (#267).

Replaces the hand-rolled ``submit() / poll() / fetch_result()`` boilerplate
documented in ``docs/client/index.md`` with a typed wrapper around the
five end-user-facing endpoints:

- ``POST /v1/predict`` — submit a plan, plus the ``status / result / logs
  / cancel`` action variants for an existing ``run_id``.
- ``POST /v1/cua`` — pure CUA pass-through (no Claude decomposition).
- ``GET /v1/runs/{run_id}/video`` — fetch the screencast bytes.
- ``GET /v1/health`` — liveness probe.

The surface is deliberately small. Three example usage shapes:

.. code-block:: python

    from mantis_agent.client import MantisClient, PredictRequest

    # 1. fire-and-poll
    client = MantisClient.from_env()
    handle = client.predict(PredictRequest(micro="plans/example/extract_listings.json",
                                           state_key="marketplace-prod"))
    final = client.wait_for_completion(handle.run_id)
    result = client.result(handle.run_id)

    # 2. submit + wait + result in one call
    result = client.run_to_completion(
        PredictRequest(micro="plans/example/extract_listings.json",
                       state_key="marketplace-prod", max_cost=2),
    )

    # 3. pure CUA pass-through
    response = client.cua_run("Open https://example.com and click the docs link")

Install with ``pip install mantis-agent[client]`` — pulls only ``requests``
+ ``pydantic``, no FastAPI / torch / Pillow / Holo3.
"""

from __future__ import annotations

from mantis_agent.api_schemas import (
    DetachedRunHandle,
    PredictRequest,
    PureCUARequest,
    RunStatus,
)

from .client import MantisClient
from .errors import (
    MantisAPIError,
    MantisAuthError,
    MantisError,
    MantisRateLimitError,
    MantisRunFailed,
    MantisTimeoutError,
)

__all__ = [
    "MantisClient",
    # Request / response types re-exported from api_schemas
    "PredictRequest",
    "PureCUARequest",
    "RunStatus",
    "DetachedRunHandle",
    # Error hierarchy
    "MantisError",
    "MantisAPIError",
    "MantisAuthError",
    "MantisRateLimitError",
    "MantisTimeoutError",
    "MantisRunFailed",
]
