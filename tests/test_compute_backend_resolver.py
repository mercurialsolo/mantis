"""Tests for `compute_backend` precedence (#785, PR 2).

Plan > submission > default(`computer_plane`).
"""

from __future__ import annotations

from mantis_agent.gym.compute_backend_resolver import resolve_compute_backend
from mantis_agent.gym.compute_contract import ComputeBackend


def test_default_is_computer_plane():
    assert resolve_compute_backend() is ComputeBackend.COMPUTER_PLANE


def test_default_when_plan_has_no_runtime_block():
    plan = {"steps": [{"intent": "x", "type": "navigate"}]}
    assert resolve_compute_backend(plan=plan) is ComputeBackend.COMPUTER_PLANE


def test_submission_overrides_default():
    assert (
        resolve_compute_backend(submission_value="browser_use_plane")
        is ComputeBackend.BROWSER_USE_PLANE
    )


def test_plan_overrides_submission():
    plan = {"runtime": {"compute_backend": "browser_use_plane"}}
    # Submission says computer_plane but plan wins.
    assert (
        resolve_compute_backend(plan=plan, submission_value="computer_plane")
        is ComputeBackend.BROWSER_USE_PLANE
    )


def test_plan_overrides_submission_inverse():
    plan = {"runtime": {"compute_backend": "computer_plane"}}
    assert (
        resolve_compute_backend(plan=plan, submission_value="browser_use_plane")
        is ComputeBackend.COMPUTER_PLANE
    )


def test_unknown_plan_value_falls_through_to_submission():
    # Forward-compat: an unrecognized backend label in the plan should
    # fall through to the next layer rather than raise.
    plan = {"runtime": {"compute_backend": "future_plane_xyz"}}
    assert (
        resolve_compute_backend(plan=plan, submission_value="browser_use_plane")
        is ComputeBackend.BROWSER_USE_PLANE
    )


def test_unknown_submission_falls_through_to_default():
    assert (
        resolve_compute_backend(submission_value="future_plane_xyz")
        is ComputeBackend.COMPUTER_PLANE
    )


def test_enum_values_accepted_directly():
    assert (
        resolve_compute_backend(submission_value=ComputeBackend.BROWSER_USE_PLANE)
        is ComputeBackend.BROWSER_USE_PLANE
    )


def test_legacy_flat_array_plan_does_not_crash():
    # Old shape: just a list of steps, no runtime block.
    plan = [{"intent": "x", "type": "navigate"}]
    assert resolve_compute_backend(plan=plan) is ComputeBackend.COMPUTER_PLANE  # type: ignore[arg-type]


def test_compute_factory_dispatches_to_browser_use():
    from mantis_agent.gym.compute_factory import make_compute_client

    client = make_compute_client(
        ComputeBackend.BROWSER_USE_PLANE,
        browser_use_base_url="https://browser-use.test",
    )
    # Smoke — the factory returned a BrowserUsePlaneClient instance.
    from mantis_agent.gym.browser_use_plane_client import BrowserUsePlaneClient

    assert isinstance(client, BrowserUsePlaneClient)


def test_compute_factory_requires_url_for_browser_use():
    from mantis_agent.gym.compute_factory import make_compute_client

    import pytest

    with pytest.raises(ValueError, match="browser_use_base_url"):
        make_compute_client(ComputeBackend.BROWSER_USE_PLANE)


def test_default_capabilities_for_each_backend():
    from mantis_agent.gym.compute_factory import default_capabilities_for

    pure = default_capabilities_for(ComputeBackend.COMPUTER_PLANE)
    bu = default_capabilities_for(ComputeBackend.BROWSER_USE_PLANE)
    assert pure.dom_aware is False
    assert bu.dom_aware is True
