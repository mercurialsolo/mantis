"""Unit tests for the Phase 2 E2B + Daytona computer-plane backends.

The impls provision a remote sandbox at construction. These tests
inject fake provider SDKs so we can exercise every branch (success,
template/snapshot errors, /health timeout, teardown) without
hitting a real cloud account.

Real-provider integration tests will land in a follow-up gated on
``E2B_API_KEY`` / ``DAYTONA_API_KEY`` env vars + a published image
SHA. The unit tests here are the always-green CI baseline.
"""

from __future__ import annotations

from typing import Any

import pytest


# ── Fake provider SDKs ────────────────────────────────────────────────


class _FakeE2BSandbox:
    """Stand-in for ``e2b.Sandbox``."""

    def __init__(self, template: str, api_key: str, host: str = "fake-host.e2b.dev"):
        self.template = template
        self.api_key = api_key
        self._host = host
        self.closed = False

    def get_host(self, port: int) -> str:
        return self._host

    def close(self) -> None:
        self.closed = True


class _FakeE2BModule:
    """Stand-in for the ``e2b`` package."""

    Sandbox = _FakeE2BSandbox


class _FakeDaytonaPreviewLink:
    def __init__(self, url: str, token: str = "fake-token") -> None:
        self.url = url
        self.token = token


class _FakeDaytonaSandbox:
    def __init__(self, snapshot: str, host: str = "fake-host.daytona.io"):
        self.snapshot = snapshot
        self._host = host
        self.deleted = False

    def get_preview_link(self, port: int) -> _FakeDaytonaPreviewLink:
        return _FakeDaytonaPreviewLink(f"https://{self._host}")

    def delete(self) -> None:
        self.deleted = True


class _FakeDaytonaClient:
    def __init__(self, config: Any) -> None:
        self.config = config

    def create(self, snapshot: str) -> _FakeDaytonaSandbox:
        return _FakeDaytonaSandbox(snapshot=snapshot)


class _FakeDaytonaConfig:
    def __init__(self, *, api_key: str, server_url: str) -> None:
        self.api_key = api_key
        self.server_url = server_url


class _FakeDaytonaModule:
    Daytona = _FakeDaytonaClient
    DaytonaConfig = _FakeDaytonaConfig


# ── Fake HTTP that the /health probe hits ────────────────────────────


def _install_fake_requests(monkeypatch, ok: bool = True) -> None:
    """Make ``requests.get`` return a 200 health probe immediately."""
    import requests

    class _Resp:
        def __init__(self, code: int) -> None:
            self.status_code = code

    def _fake_get(url: str, **kw: Any) -> _Resp:
        if not ok:
            raise requests.ConnectionError("simulated connect refused")
        return _Resp(200)

    monkeypatch.setattr(requests, "get", _fake_get)


# ── E2B happy path + error paths ──────────────────────────────────────


def test_e2b_constructs_with_fake_sdk_and_health_ready(monkeypatch) -> None:
    """Happy path: sandbox provisions, /health is up, super().__init__ runs."""
    from mantis_agent.gym.e2b_impl import E2BComputerImpl

    _install_fake_requests(monkeypatch, ok=True)
    impl = E2BComputerImpl(
        api_key="x", template="t", sdk_module=_FakeE2BModule,
        startup_timeout_seconds=2.0,
        tenant_id="acme", profile_id="p", run_id="r",
    )
    # RemoteComputerImpl sets _base_url after super().__init__.
    assert impl._base_url == "https://fake-host.e2b.dev"
    assert impl._sandbox is not None


def test_e2b_health_timeout_tears_down_sandbox(monkeypatch) -> None:
    """If /health never answers we MUST close the sandbox so we don't
    bill the operator for idle compute."""
    from mantis_agent.gym.e2b_impl import E2BComputerImpl

    _install_fake_requests(monkeypatch, ok=False)
    # Speed up the deadline + sleep so the test stays fast.
    import mantis_agent.gym.e2b_impl as e2b_mod
    monkeypatch.setattr(e2b_mod.time, "sleep", lambda *a, **kw: None)

    sandbox = _FakeE2BSandbox(template="t", api_key="x")

    class _CapturingModule:
        @staticmethod
        def Sandbox(template: str, api_key: str) -> _FakeE2BSandbox:
            return sandbox

    with pytest.raises(TimeoutError):
        E2BComputerImpl(
            api_key="x", template="t", sdk_module=_CapturingModule,
            startup_timeout_seconds=0.1,
        )
    # Sandbox MUST have been closed by the teardown path.
    assert sandbox.closed is True


def test_e2b_missing_api_key_raises_value_error(monkeypatch) -> None:
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    from mantis_agent.gym.e2b_impl import E2BComputerImpl

    with pytest.raises(ValueError, match="E2B API key"):
        E2BComputerImpl(sdk_module=_FakeE2BModule)


def test_e2b_uses_env_template_when_unset(monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_E2B_TEMPLATE", "env-template")
    _install_fake_requests(monkeypatch, ok=True)
    from mantis_agent.gym.e2b_impl import E2BComputerImpl

    captured_template = {"v": ""}

    class _CapturingModule:
        @staticmethod
        def Sandbox(template: str, api_key: str) -> _FakeE2BSandbox:
            captured_template["v"] = template
            return _FakeE2BSandbox(template=template, api_key=api_key)

    E2BComputerImpl(
        api_key="x", sdk_module=_CapturingModule,
        startup_timeout_seconds=2.0,
    )
    assert captured_template["v"] == "env-template"


def test_e2b_provision_failure_wraps_underlying_error(monkeypatch) -> None:
    from mantis_agent.gym.e2b_impl import E2BComputerImpl

    class _FailingModule:
        @staticmethod
        def Sandbox(template: str, api_key: str) -> Any:
            raise RuntimeError("simulated template-not-found")

    with pytest.raises(RuntimeError, match="provisioning failed"):
        E2BComputerImpl(
            api_key="x", template="t", sdk_module=_FailingModule,
        )


def test_e2b_close_tears_down_sandbox(monkeypatch) -> None:
    from mantis_agent.gym.e2b_impl import E2BComputerImpl

    _install_fake_requests(monkeypatch, ok=True)
    impl = E2BComputerImpl(
        api_key="x", template="t", sdk_module=_FakeE2BModule,
        startup_timeout_seconds=2.0,
    )
    sandbox = impl._sandbox
    # Skip the upstream session-close call (RemoteComputerImpl.close
    # would try to talk to the fake URL).
    monkeypatch.setattr(
        "mantis_agent.gym.remote_computer_impl.RemoteComputerImpl.close",
        lambda self: None,
    )
    impl.close()
    assert sandbox.closed is True
    assert impl._sandbox is None


# ── Daytona happy path + error paths ──────────────────────────────────


def test_daytona_constructs_with_fake_sdk_and_skip_preview_header(
    monkeypatch,
) -> None:
    from mantis_agent.gym.daytona_impl import DaytonaComputerImpl

    _install_fake_requests(monkeypatch, ok=True)
    impl = DaytonaComputerImpl(
        api_key="x", snapshot="snap-1",
        sdk_module=_FakeDaytonaModule,
        startup_timeout_seconds=2.0,
        tenant_id="acme", profile_id="p", run_id="r",
    )
    assert impl._base_url == "https://fake-host.daytona.io"
    # Both bypass headers should be on every wire call:
    # - skip-preview defeats the consent interstitial
    # - preview-token defeats the auth0 wall on the preview URL
    assert impl._extra_http_headers is not None
    assert (
        impl._extra_http_headers.get("X-Daytona-Skip-Preview-Warning")
        == "true"
    )
    assert (
        impl._extra_http_headers.get("X-Daytona-Preview-Token")
        == "fake-token"
    )


def test_daytona_health_timeout_tears_down(monkeypatch) -> None:
    from mantis_agent.gym.daytona_impl import DaytonaComputerImpl

    _install_fake_requests(monkeypatch, ok=False)
    import mantis_agent.gym.daytona_impl as d_mod
    monkeypatch.setattr(d_mod.time, "sleep", lambda *a, **kw: None)

    sandbox = _FakeDaytonaSandbox(snapshot="snap-1")
    client = _FakeDaytonaClient(config=None)
    client.create = lambda snapshot: sandbox  # type: ignore[assignment]

    class _CapturingModule:
        @staticmethod
        def Daytona(config: Any) -> _FakeDaytonaClient:
            return client

        DaytonaConfig = _FakeDaytonaConfig

    with pytest.raises(TimeoutError):
        DaytonaComputerImpl(
            api_key="x", snapshot="snap-1",
            sdk_module=_CapturingModule,
            startup_timeout_seconds=0.1,
        )
    assert sandbox.deleted is True


def test_daytona_missing_snapshot_raises_value_error(monkeypatch) -> None:
    monkeypatch.delenv("MANTIS_DAYTONA_SNAPSHOT", raising=False)
    from mantis_agent.gym.daytona_impl import DaytonaComputerImpl

    with pytest.raises(ValueError, match="snapshot id"):
        DaytonaComputerImpl(api_key="x", sdk_module=_FakeDaytonaModule)


def test_daytona_uses_env_snapshot_when_unset(monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_DAYTONA_SNAPSHOT", "env-snap")
    _install_fake_requests(monkeypatch, ok=True)
    from mantis_agent.gym.daytona_impl import DaytonaComputerImpl

    captured: dict = {"v": ""}

    class _CapturingModule:
        @staticmethod
        def Daytona(config: Any) -> _FakeDaytonaClient:
            client = _FakeDaytonaClient(config=config)

            def _create(snapshot: str) -> _FakeDaytonaSandbox:
                captured["v"] = snapshot
                return _FakeDaytonaSandbox(snapshot=snapshot)

            client.create = _create  # type: ignore[assignment]
            return client

        DaytonaConfig = _FakeDaytonaConfig

    DaytonaComputerImpl(
        api_key="x", sdk_module=_CapturingModule,
        startup_timeout_seconds=2.0,
    )
    assert captured["v"] == "env-snap"


def test_daytona_close_tears_down(monkeypatch) -> None:
    from mantis_agent.gym.daytona_impl import DaytonaComputerImpl

    _install_fake_requests(monkeypatch, ok=True)
    impl = DaytonaComputerImpl(
        api_key="x", snapshot="snap-1",
        sdk_module=_FakeDaytonaModule,
        startup_timeout_seconds=2.0,
    )
    sandbox = impl._sandbox
    monkeypatch.setattr(
        "mantis_agent.gym.remote_computer_impl.RemoteComputerImpl.close",
        lambda self: None,
    )
    impl.close()
    assert sandbox.deleted is True
    assert impl._sandbox is None


# ── Factory wiring ────────────────────────────────────────────────────


def test_make_computer_client_e2b_routes_to_e2b_impl(monkeypatch) -> None:
    """``make_computer_client(cfg=backend='e2b')`` instantiates
    E2BComputerImpl. We monkeypatch the impl so the test doesn't need
    an actual SDK or HTTP probe."""
    from mantis_agent.gym import computer_client as cc

    captured: dict = {}

    class _StubImpl:
        def __init__(self, **kw: Any) -> None:
            captured.update(kw)

    monkeypatch.setattr(
        "mantis_agent.gym.e2b_impl.E2BComputerImpl", _StubImpl,
    )
    cfg = cc.ComputerPlaneConfig(backend="e2b")
    impl = cc.make_computer_client(cfg, tenant_id="acme")
    assert isinstance(impl, _StubImpl)
    assert captured.get("tenant_id") == "acme"


def test_make_computer_client_daytona_routes_to_daytona_impl(monkeypatch) -> None:
    from mantis_agent.gym import computer_client as cc

    captured: dict = {}

    class _StubImpl:
        def __init__(self, **kw: Any) -> None:
            captured.update(kw)

    monkeypatch.setattr(
        "mantis_agent.gym.daytona_impl.DaytonaComputerImpl", _StubImpl,
    )
    cfg = cc.ComputerPlaneConfig(backend="daytona")
    impl = cc.make_computer_client(cfg, profile_id="p")
    assert isinstance(impl, _StubImpl)
    assert captured.get("profile_id") == "p"
