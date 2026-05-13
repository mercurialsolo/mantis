"""Tests for the Modal runtime backend.

We mock the ``modal`` SDK so the suite runs without a Modal account.
Two paths covered:

* ``stub`` env name short-circuits to the long-lived ``mantis-sim-env-stub``
  app (no per-run deploy).
* Other env names look up ``mantis-sim-env-<env>-<suffix>`` and fail
  clearly when the function isn't deployed.

We also test the e2b backend stub raises ``NotImplementedError`` with a
clear message (so a half-finished landing of #336.5 is caught).
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

from mantis_agent.sim_envs.e2b import E2BBackend
from mantis_agent.sim_envs.registry import get_backend, list_backends


def _install_fake_modal(get_web_url: str = "https://fake-modal.run") -> types.ModuleType:
    """Build a minimal fake ``modal`` module and register it in sys.modules."""
    fake = types.ModuleType("modal")

    class FakeFunction:
        @classmethod
        def from_name(cls, app_name: str, fn_name: str):
            f = cls()
            f.app_name = app_name
            f.fn_name = fn_name
            return f

        def get_web_url(self) -> str:
            return get_web_url

    fake.Function = FakeFunction
    sys.modules["modal"] = fake
    return fake


def test_list_backends_returns_three():
    assert list_backends() == ["local", "modal", "e2b"]


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("nonexistent")


def test_e2b_backend_start_raises_not_implemented():
    backend = E2BBackend()
    with pytest.raises(NotImplementedError) as ei:
        backend.start("stub")
    assert "v1" in str(ei.value)


def test_modal_backend_stub_env_uses_shared_app(monkeypatch):
    _install_fake_modal("https://fake-stub.modal.run")
    from mantis_agent.sim_envs.modal_backend import ModalBackend

    backend = ModalBackend()
    handle = backend.start("stub", admin_token="t1")
    assert handle.url == "https://fake-stub.modal.run"
    assert handle.extra["app_name"] == "mantis-sim-env-stub"
    assert handle.backend == "modal"
    # No-op stop on the shared stub app.
    backend.stop(handle)


def test_modal_backend_missing_env_raises_clear_error(monkeypatch):
    fake = _install_fake_modal()

    class _Boom:
        @classmethod
        def from_name(cls, *_a, **_kw):
            raise RuntimeError("function not found")

    fake.Function = _Boom

    from mantis_agent.sim_envs.modal_backend import ModalBackend

    backend = ModalBackend()
    with pytest.raises(RuntimeError) as ei:
        backend.start("mantis-crm")
    msg = str(ei.value)
    # The clear error mentions deploy path + suggests --runtime local fallback
    assert "deploy/sim_envs" in msg
    assert "--runtime local" in msg


def test_modal_backend_missing_sdk(monkeypatch):
    """No ``modal`` module on path → start() raises a friendly error."""
    monkeypatch.delitem(sys.modules, "modal", raising=False)
    # ``import modal`` inside the backend should hit ImportError.
    with mock.patch.dict(
        sys.modules,
        {"modal": None},  # type: ignore[dict-item]
    ):
        from mantis_agent.sim_envs.modal_backend import ModalBackend

        backend = ModalBackend()
        with pytest.raises(RuntimeError) as ei:
            backend.start("stub")
        assert "modal SDK not installed" in str(ei.value)
