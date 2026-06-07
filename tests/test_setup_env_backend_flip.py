"""Tests for the `MANTIS_COMPUTER_PLANE_BACKEND` flip plumbed through
`task_loop.setup_env`.

Phase 1 (#698) acceptance requires the flip be a one-secret edit with
no code change on the call sites. These tests pin three properties:

  1. Default config returns a `LocalXdotoolImpl` — production today
     keeps working unchanged when the env var is absent.
  2. `backend="modal"` returns a `RemoteComputerImpl` and forwards
     `start_url` / `profile_dir` / `extra_http_headers` into the
     impl so the remote `SessionInitRequest` carries them.
  3. The local Xvfb subprocess is NOT spawned when the env runs in a
     remote backend — its display lives on the computer-plane
     container, so a brain-side `Xvfb` would clobber `DISPLAY` and
     also waste a CPU.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from mantis_agent.gym.computer_client import ComputerPlaneConfig
from mantis_agent.gym.local_xdotool_impl import LocalXdotoolImpl
from mantis_agent.gym.remote_computer_impl import RemoteComputerImpl
from mantis_agent.task_loop import _computer_plane_config_from_env, setup_env


@pytest.fixture(autouse=True)
def _disable_proxy_egress_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe hits ipinfo.io over HTTP — short-circuit so tests are
    network-free."""
    monkeypatch.setattr(
        "mantis_agent.task_loop.diagnose_proxy_egress",
        lambda *_a, **_kw: {"disabled": True},
    )


@pytest.fixture(autouse=True)
def _no_real_xvfb(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Capture every `subprocess.Popen` invocation from setup_env so
    tests can assert whether Xvfb was actually spawned."""
    spawns: list[list[str]] = []

    class _FakeProc:
        pass

    def _popen(argv: list[str], **_kw: Any) -> _FakeProc:
        spawns.append(argv)
        return _FakeProc()

    monkeypatch.setattr(
        "mantis_agent.task_loop.subprocess.Popen", _popen
    )
    return spawns


@pytest.fixture()
def _local_no_chrome(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop `LocalXdotoolImpl.__init__` from spawning Chrome under tests
    that exercise the `local` factory branch — only the type matters
    for the assertions."""
    monkeypatch.setattr(
        "mantis_agent.task_loop.build_proxy_config", lambda **_kw: None
    )
    monkeypatch.setattr(
        "mantis_agent.task_loop.resolve_proxy_server", lambda _p: ("", None)
    )


# ── env-var → config ────────────────────────────────────────────────


def test_config_from_env_defaults_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANTIS_COMPUTER_PLANE_BACKEND", raising=False)
    cfg = _computer_plane_config_from_env()
    assert cfg.backend == "local"
    assert cfg.remote_base_url is None


def test_config_from_env_modal_reads_url_and_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANTIS_COMPUTER_PLANE_BACKEND", "modal")
    monkeypatch.setenv("MANTIS_COMPUTER_PLANE_URL", "https://cp.example/")
    monkeypatch.setenv("MANTIS_COMPUTER_PLANE_TOKEN", "tok-xyz")
    monkeypatch.setenv("MANTIS_COMPUTER_PLANE_ENABLE_CDP", "1")
    cfg = _computer_plane_config_from_env()
    assert cfg.backend == "modal"
    assert cfg.remote_base_url == "https://cp.example/"
    assert cfg.remote_auth_token == "tok-xyz"
    assert cfg.enable_cdp is True


def test_config_from_env_unknown_backend_falls_back_to_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANTIS_COMPUTER_PLANE_BACKEND", "lambda")
    cfg = _computer_plane_config_from_env()
    assert cfg.backend == "local"


# ── setup_env: default path ─────────────────────────────────────────


def test_setup_env_default_returns_local_xdotool_impl(
    monkeypatch: pytest.MonkeyPatch,
    _local_no_chrome: None,
    _no_real_xvfb: list[list[str]],
) -> None:
    monkeypatch.delenv("MANTIS_COMPUTER_PLANE_BACKEND", raising=False)

    env, _proxy, _diag = setup_env(
        base_url="https://example.com",
        run_id="r1",
        session_name="sn",
        proxy_disabled=True,
        display=":99",
        start_xvfb=True,
    )
    assert isinstance(env, LocalXdotoolImpl)
    # Default = local backend → Xvfb spawned by setup_env.
    assert any(argv[0] == "Xvfb" for argv in _no_real_xvfb)


# ── setup_env: modal-backend path ───────────────────────────────────


def test_setup_env_modal_backend_returns_remote_computer_impl(
    monkeypatch: pytest.MonkeyPatch,
    _no_real_xvfb: list[list[str]],
) -> None:
    monkeypatch.setattr(
        "mantis_agent.task_loop.build_proxy_config", lambda **_kw: None
    )
    monkeypatch.setattr(
        "mantis_agent.task_loop.resolve_proxy_server", lambda _p: ("", None)
    )
    cfg = ComputerPlaneConfig(
        backend="modal",
        remote_base_url="https://cp.example",
    )

    env, _proxy, _diag = setup_env(
        base_url="https://target.example",
        run_id="r1",
        session_name="sn",
        proxy_disabled=True,
        display=":99",
        start_xvfb=True,
        computer_plane_config=cfg,
    )
    assert isinstance(env, RemoteComputerImpl)
    assert env._start_url == "https://target.example"
    # CRITICAL: remote backend must NOT spawn a brain-side Xvfb. The
    # remote container owns its own display; clobbering DISPLAY locally
    # would also break colocated executors during the migration window.
    assert not any(argv[0] == "Xvfb" for argv in _no_real_xvfb)


def test_setup_env_modal_backend_forwards_session_identity(
    monkeypatch: pytest.MonkeyPatch,
    _no_real_xvfb: list[list[str]],
) -> None:
    """`RemoteComputerImpl` binds its session to (tenant, profile, run).
    setup_env must surface the brain-side identifiers so the remote can
    honor the per-profile lock and the run-scoped idempotency.
    """
    monkeypatch.setattr(
        "mantis_agent.task_loop.build_proxy_config", lambda **_kw: None
    )
    monkeypatch.setattr(
        "mantis_agent.task_loop.resolve_proxy_server", lambda _p: ("", None)
    )
    cfg = ComputerPlaneConfig(
        backend="modal", remote_base_url="https://cp.example"
    )
    env, _proxy, _diag = setup_env(
        base_url="https://x",
        run_id="r-42",
        session_name="claude_cua",
        proxy_disabled=True,
        computer_plane_config=cfg,
    )
    assert isinstance(env, RemoteComputerImpl)
    assert env._run_id == "r-42"
    assert env._tenant_id == "claude_cua"
    assert env._profile_id == "claude_cua"


def test_setup_env_modal_backend_forwards_extra_http_headers(
    monkeypatch: pytest.MonkeyPatch,
    _no_real_xvfb: list[list[str]],
) -> None:
    monkeypatch.setattr(
        "mantis_agent.task_loop.build_proxy_config", lambda **_kw: None
    )
    monkeypatch.setattr(
        "mantis_agent.task_loop.resolve_proxy_server", lambda _p: ("", None)
    )
    cfg = ComputerPlaneConfig(
        backend="modal", remote_base_url="https://cp.example"
    )
    env, _proxy, _diag = setup_env(
        base_url="https://x",
        run_id="r1",
        session_name="sn",
        proxy_disabled=True,
        extra_http_headers={"X-Daytona-Skip-Preview-Warning": "true"},
        computer_plane_config=cfg,
    )
    assert isinstance(env, RemoteComputerImpl)
    assert env._extra_http_headers == {
        "X-Daytona-Skip-Preview-Warning": "true",
    }


def test_setup_env_modal_backend_via_env_var_alone(
    monkeypatch: pytest.MonkeyPatch,
    _no_real_xvfb: list[list[str]],
) -> None:
    """The Phase-1 rollback story rests on this: a single secret edit
    flips behavior without touching call sites."""
    monkeypatch.setattr(
        "mantis_agent.task_loop.build_proxy_config", lambda **_kw: None
    )
    monkeypatch.setattr(
        "mantis_agent.task_loop.resolve_proxy_server", lambda _p: ("", None)
    )
    monkeypatch.setenv("MANTIS_COMPUTER_PLANE_BACKEND", "modal")
    monkeypatch.setenv("MANTIS_COMPUTER_PLANE_URL", "https://cp.example")

    env, _proxy, _diag = setup_env(
        base_url="https://x",
        run_id="r1",
        session_name="sn",
        proxy_disabled=True,
        display=":99",
        start_xvfb=True,
    )
    assert isinstance(env, RemoteComputerImpl)
    assert env._base_url == "https://cp.example"
