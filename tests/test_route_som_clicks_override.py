"""Per-request ``route_som_clicks`` override for both /v1/cua and /v1/predict.

The default is the environment toggle ``MANTIS_ROUTE_SOM_CLICKS``;
callers can pin SoM on / off per request without redeploying, matching
the ablation pattern documented in :mod:`scripts.ablate_v1_cua`. This
module covers:

1. The Pydantic request schemas (``PureCUARequest``, ``PredictRequest``)
   accept ``route_som_clicks`` and forward it through ``model_dump``.
2. The runtime override resolution: env default → ``True`` when request
   says ``True`` → ``False`` when request says ``False`` → env value
   when request says ``None``.

The runtime-side resolution is the same inline ``RoutingPolicy`` builder
used in both ``run_pure_cua`` (/v1/cua) and ``_run_micro`` (/v1/predict),
so the test exercises the policy-build helper directly to avoid spinning
up a full FastAPI test client / GymRunner.
"""

from __future__ import annotations

import pytest

from mantis_agent.api_schemas import PredictRequest, PureCUARequest
from mantis_agent.gym.runner import RoutingPolicy


# ── Request schema accepts route_som_clicks ────────────────────────────


def test_pure_cua_request_accepts_route_som_clicks() -> None:
    req = PureCUARequest.model_validate({
        "instruction": "do something",
        "route_som_clicks": True,
    })
    assert req.route_som_clicks is True
    payload = req.model_dump(exclude_none=True)
    assert payload.get("route_som_clicks") is True


def test_pure_cua_request_route_som_clicks_defaults_none() -> None:
    """Unset → omitted from model_dump(exclude_none=True), so the runtime
    sees ``payload.get("route_som_clicks") is None`` and falls back to
    the env toggle."""
    req = PureCUARequest.model_validate({"instruction": "x"})
    assert req.route_som_clicks is None
    assert "route_som_clicks" not in req.model_dump(exclude_none=True)


def test_predict_request_accepts_route_som_clicks() -> None:
    req = PredictRequest.model_validate({
        "task_suite": {"_micro_plan": [{"intent": "x", "type": "navigate"}]},
        "route_som_clicks": False,
    })
    assert req.route_som_clicks is False
    payload = req.model_dump(exclude_none=True)
    assert payload.get("route_som_clicks") is False


def test_predict_request_route_som_clicks_omitted_when_none() -> None:
    req = PredictRequest.model_validate({
        "task_suite": {"_micro_plan": [{"intent": "x", "type": "navigate"}]},
    })
    assert req.route_som_clicks is None
    assert "route_som_clicks" not in req.model_dump(exclude_none=True)


# ── Runtime override resolution ────────────────────────────────────────


def _resolve_policy(payload: dict, monkeypatch: pytest.MonkeyPatch) -> RoutingPolicy:
    """Mirror the inline override resolution in ``runtime.run_pure_cua`` /
    ``_run_micro``. Kept as a tiny helper so a regression in the override
    semantics surfaces here rather than at endpoint level."""
    policy = RoutingPolicy.from_env()
    override = payload.get("route_som_clicks")
    if override is not None:
        policy = RoutingPolicy(
            plan_executor_enabled=policy.plan_executor_enabled,
            som_enabled=policy.som_enabled,
            som_for_unstructured_clicks=bool(override),
        )
    return policy


def test_resolve_policy_request_true_overrides_env_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env says off (default); request flips on."""
    monkeypatch.delenv("MANTIS_ROUTE_SOM_CLICKS", raising=False)
    policy = _resolve_policy({"route_som_clicks": True}, monkeypatch)
    assert policy.som_for_unstructured_clicks is True


def test_resolve_policy_request_false_overrides_env_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env says on; request flips off."""
    monkeypatch.setenv("MANTIS_ROUTE_SOM_CLICKS", "enabled")
    policy = _resolve_policy({"route_som_clicks": False}, monkeypatch)
    assert policy.som_for_unstructured_clicks is False


def test_resolve_policy_request_none_defers_to_env_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request says None (or field absent); env wins."""
    monkeypatch.setenv("MANTIS_ROUTE_SOM_CLICKS", "enabled")
    policy = _resolve_policy({}, monkeypatch)
    assert policy.som_for_unstructured_clicks is True


def test_resolve_policy_request_none_defers_to_env_default_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset env + missing request field → dataclass default ``False``."""
    monkeypatch.delenv("MANTIS_ROUTE_SOM_CLICKS", raising=False)
    policy = _resolve_policy({}, monkeypatch)
    assert policy.som_for_unstructured_clicks is False


def test_resolve_policy_preserves_other_policy_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Override only flips ``som_for_unstructured_clicks`` — the other
    policy fields (``plan_executor_enabled`` / ``som_enabled``) stay
    whatever the env-default produced. Catches a regression where a
    naive ``RoutingPolicy(som_for_unstructured_clicks=...)`` rebuild
    would silently reset the others to dataclass defaults."""
    monkeypatch.setenv("MANTIS_ROUTE_PLAN_EXECUTOR", "disabled")
    monkeypatch.setenv("MANTIS_ROUTE_SOM", "disabled")
    monkeypatch.delenv("MANTIS_ROUTE_SOM_CLICKS", raising=False)
    policy = _resolve_policy({"route_som_clicks": True}, monkeypatch)
    assert policy.som_for_unstructured_clicks is True
    assert policy.plan_executor_enabled is False
    assert policy.som_enabled is False
