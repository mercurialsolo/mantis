"""Tests for ``SessionRoutedComputerImpl`` (Phase 1.5, #846, PR 3).

Drives the wrapper against a stubbed ``SessionRouterClient`` so the
construct → mint-session → delegate flow is exercised without Modal,
without HTTP, without Xvfb / Chrome.
"""

from __future__ import annotations

from typing import Any

import pytest

from mantis_agent.gym.computer_client import (
    ComputerPlaneConfig,
    make_computer_client,
)
from mantis_agent.gym.session_routed_impl import SessionRoutedComputerImpl
from mantis_agent.session_wire import (
    SessionCloseResponse,
    SessionCreateResponse,
    SessionRouterError,
    SessionUnreachableError,
)


class _StubRouterClient:
    """Records calls + serves canned responses."""

    def __init__(
        self,
        create_response: SessionCreateResponse | Exception | None = None,
    ) -> None:
        self.create_calls: list[Any] = []
        self.close_calls: list[tuple[str, str]] = []
        self._create_response = create_response or SessionCreateResponse(
            session_id="sess_stub",
            base_url="https://stub.modal.run",
            session_token="tok_stub",
            expires_at_ms=2_000_000_000_000,
            sandbox_id="fc-stub",
        )

    def create_session(self, req):
        self.create_calls.append(req)
        if isinstance(self._create_response, Exception):
            raise self._create_response
        return self._create_response

    def close_session(self, session_id: str, *, reason: str = "brain_closed", quiet: bool = True):
        self.close_calls.append((session_id, reason))
        return SessionCloseResponse(closed=True, terminal_state="closed")


# ── Construction wires session_token via pre_minted path ─────────────


def test_construct_mints_session_and_pre_seeds_token() -> None:
    client = _StubRouterClient()
    impl = SessionRoutedComputerImpl(
        router_url="https://api.modal.run",
        auth_token="my-token",
        tenant_id="t1",
        profile_id="alice",
        run_id="r1",
        router_client=client,
    )
    # Router was called once with the right identity.
    assert len(client.create_calls) == 1
    req = client.create_calls[0]
    assert req.tenant_id == "t1"
    assert req.profile_id == "alice"
    assert req.run_id == "r1"
    # base_url + token bubbled through to the underlying impl.
    assert impl._base_url == "https://stub.modal.run"  # noqa: SLF001
    assert impl._session_token == "tok_stub"  # noqa: SLF001
    # session_id exposed for log correlation.
    assert impl.session_id == "sess_stub"
    assert impl.sandbox_id == "fc-stub"


def test_construct_propagates_session_create_failure() -> None:
    """Router unreachable → SessionUnreachableError bubbles up.
    make_computer_client callers decide whether to fall back to the
    pinned computer_plane URL."""
    client = _StubRouterClient(
        create_response=SessionUnreachableError("router timeout"),
    )
    with pytest.raises(SessionUnreachableError):
        SessionRoutedComputerImpl(
            router_url="https://api.modal.run",
            auth_token="my-token",
            tenant_id="t", profile_id="p", run_id="r",
            router_client=client,
        )


# ── close() teardown calls the router then super().close() ──────────


def test_close_calls_router_then_super_close() -> None:
    client = _StubRouterClient()
    impl = SessionRoutedComputerImpl(
        router_url="https://x", auth_token="t",
        tenant_id="t1", profile_id="p", run_id="r",
        router_client=client,
    )
    # Patch super().close — we just want to verify ordering, not test
    # the network teardown.
    super_called = {"hit": False}

    def fake_super_close(self) -> None:
        super_called["hit"] = True

    SessionRoutedComputerImpl.__mro__[1].close = fake_super_close  # type: ignore[assignment]
    impl.close()
    assert client.close_calls == [("sess_stub", "brain_closed")]
    assert super_called["hit"]


def test_close_swallows_router_failure() -> None:
    """A router-side failure on close must not prevent the underlying
    super().close() — otherwise we leak the session-token cache + the
    brain's teardown surface gets noisy."""

    class _Boom(_StubRouterClient):
        def close_session(self, *args, **kwargs):
            raise SessionRouterError("router 500")

    impl = SessionRoutedComputerImpl(
        router_url="https://x", auth_token="t",
        tenant_id="t1", profile_id="p", run_id="r",
        router_client=_Boom(),
    )
    super_called = {"hit": False}

    def fake_super_close(self) -> None:
        super_called["hit"] = True

    SessionRoutedComputerImpl.__mro__[1].close = fake_super_close  # type: ignore[assignment]
    impl.close()
    assert super_called["hit"]


# ── make_computer_client dispatches to the session-routed path ──────


def test_make_computer_client_routes_to_session_when_router_url_set(monkeypatch) -> None:
    """The factory must pick SessionRoutedComputerImpl when
    ``session_router_url`` is set; otherwise fall back to the pinned
    RemoteComputerImpl path."""
    captured = {}

    class _StubImpl:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "mantis_agent.gym.session_routed_impl.SessionRoutedComputerImpl",
        _StubImpl,
    )

    cfg = ComputerPlaneConfig(
        backend="modal",
        session_router_url="https://api.modal.run",
        session_router_auth_token="my-token",
        session_ttl_seconds=1800,
    )
    out = make_computer_client(
        cfg,
        tenant_id="t1", profile_id="p", run_id="r",
        start_url="https://example.com",
    )
    assert isinstance(out, _StubImpl)
    assert captured["router_url"] == "https://api.modal.run"
    assert captured["auth_token"] == "my-token"
    assert captured["ttl_seconds"] == 1800
    assert captured["tenant_id"] == "t1"
    assert captured["start_url"] == "https://example.com"


def test_make_computer_client_falls_back_to_remote_when_no_router_url(monkeypatch) -> None:
    """No router URL set → falls through to the existing
    RemoteComputerImpl path (Phase 1 behaviour preserved)."""
    captured = {}

    class _StubRemote:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "mantis_agent.gym.remote_computer_impl.RemoteComputerImpl",
        _StubRemote,
    )
    cfg = ComputerPlaneConfig(
        backend="modal",
        remote_base_url="https://pinned-computer-plane.modal.run",
        remote_auth_token="some-bearer",
    )
    out = make_computer_client(cfg, tenant_id="t1", profile_id="p", run_id="r")
    assert isinstance(out, _StubRemote)
    assert captured["base_url"] == "https://pinned-computer-plane.modal.run"


def test_make_computer_client_session_router_requires_token() -> None:
    cfg = ComputerPlaneConfig(
        backend="modal",
        session_router_url="https://api.modal.run",
        # No token.
    )
    with pytest.raises(ValueError, match="session_router_auth_token"):
        make_computer_client(cfg, tenant_id="t", profile_id="p", run_id="r")


# ── RemoteComputerImpl pre_minted_session_token short-circuit ───────


def test_remote_computer_impl_skips_session_init_when_token_pre_minted(monkeypatch) -> None:
    """When pre_minted_session_token is set, session_init() returns the
    cached token instead of POSTing /session/init."""
    from mantis_agent.gym.remote_computer_impl import RemoteComputerImpl

    impl = RemoteComputerImpl(
        base_url="https://x",
        tenant_id="t", profile_id="p", run_id="r",
        pre_minted_session_token="cached_tok",
    )
    # Prove that no _post is needed — by patching it to raise on call.
    monkeypatch.setattr(
        impl, "_post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not POST")),
    )
    resp = impl.session_init()
    assert resp.session_token == "cached_tok"
