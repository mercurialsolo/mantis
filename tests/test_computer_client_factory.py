"""Tests for the Computer Plane Phase 0 seam (#697).

Locks the public surface of ``make_computer_client``: factory dispatch,
config-driven backend selection, kwargs pass-through to the underlying
``XdotoolGymEnv``, latency tracker behavior, and the marker hierarchy
that lets brain-side code ``isinstance(env, ComputerClient)``.
"""

from __future__ import annotations

from typing import Any

import pytest

from mantis_agent.gym.base import GymEnvironment
from mantis_agent.gym.computer_client import (
    ComputerClient,
    ComputerPlaneConfig,
    make_computer_client,
)
from mantis_agent.gym.local_xdotool_impl import LatencyTracker, LocalXdotoolImpl
from mantis_agent.gym.xdotool_env import XdotoolGymEnv


# ── marker hierarchy ──────────────────────────────────────────────────


def test_local_impl_is_a_computer_client_and_gym_environment() -> None:
    """Phase 0 invariant: brain code can type-check ``ComputerClient`` and
    still feed it into any code path that expects ``GymEnvironment``."""
    assert issubclass(LocalXdotoolImpl, ComputerClient)
    assert issubclass(LocalXdotoolImpl, GymEnvironment)
    assert issubclass(LocalXdotoolImpl, XdotoolGymEnv)


def test_computer_client_is_a_gym_environment() -> None:
    """The marker base is itself a ``GymEnvironment`` so all impls
    inherit the gym contract."""
    assert issubclass(ComputerClient, GymEnvironment)


# ── factory dispatch ──────────────────────────────────────────────────


def test_default_config_returns_local_impl(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ComputerPlaneConfig()`` defaults to ``backend='local'`` and the
    factory returns a ``LocalXdotoolImpl``."""
    captured: dict[str, Any] = {}

    class _StubLocal:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "mantis_agent.gym.local_xdotool_impl.LocalXdotoolImpl", _StubLocal
    )

    env = make_computer_client(ComputerPlaneConfig(), start_url="https://example.com")
    assert isinstance(env, _StubLocal)
    assert captured["start_url"] == "https://example.com"


def test_no_config_defaults_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing no config behaves like ``ComputerPlaneConfig()``."""
    monkeypatch.setattr(
        "mantis_agent.gym.local_xdotool_impl.LocalXdotoolImpl",
        lambda **_: object(),
    )
    assert make_computer_client() is not None


def test_modal_backend_requires_remote_base_url() -> None:
    cfg = ComputerPlaneConfig(backend="modal")
    with pytest.raises(ValueError, match="requires remote_base_url"):
        make_computer_client(cfg)


def test_e2b_backend_not_yet_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Phase 2"):
        make_computer_client(ComputerPlaneConfig(backend="e2b"))


def test_daytona_backend_not_yet_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Phase 2"):
        make_computer_client(ComputerPlaneConfig(backend="daytona"))


def test_unknown_backend_raises() -> None:
    cfg = ComputerPlaneConfig.__new__(ComputerPlaneConfig)
    cfg.backend = "alien"  # type: ignore[assignment]
    cfg.remote_base_url = None
    cfg.remote_auth_token = None
    cfg.enable_cdp = False
    cfg.per_executor_overrides = {}
    with pytest.raises(ValueError, match="unknown ComputerPlaneConfig.backend"):
        make_computer_client(cfg)


# ── per-executor overrides ────────────────────────────────────────────


def test_resolve_for_executor_swaps_backend() -> None:
    cfg = ComputerPlaneConfig(
        backend="local",
        per_executor_overrides={"run_claude_cua": "modal"},
        remote_base_url="https://stub",
    )
    resolved = cfg.resolve_for_executor("run_claude_cua")
    assert resolved.backend == "modal"
    assert resolved.remote_base_url == "https://stub"
    # Original is unchanged.
    assert cfg.backend == "local"


def test_resolve_for_executor_returns_self_when_no_override() -> None:
    cfg = ComputerPlaneConfig(backend="local")
    assert cfg.resolve_for_executor("run_holo3") is cfg
    assert cfg.resolve_for_executor(None) is cfg


def test_resolve_for_executor_returns_self_when_override_matches() -> None:
    cfg = ComputerPlaneConfig(
        backend="local",
        per_executor_overrides={"run_holo3": "local"},
    )
    # Same backend → return self (no allocation).
    assert cfg.resolve_for_executor("run_holo3") is cfg


# ── latency tracker ───────────────────────────────────────────────────


def test_latency_tracker_summary_shape() -> None:
    tracker = LatencyTracker("screenshot")
    for ms in [10.0, 20.0, 30.0, 40.0, 50.0]:
        tracker.record_ms(ms)
    summary = tracker.summary()
    assert summary["name"] == "screenshot"
    assert summary["count"] == 5
    # Index math: p50 → idx int(5 * 0.5) = 2 → sorted[2] = 30.0
    assert summary["p50_ms"] == 30.0
    assert summary["p99_ms"] == 50.0
    assert summary["max_ms"] == 50.0
    assert summary["mean_ms"] == 30.0


def test_latency_tracker_empty_returns_count_zero() -> None:
    tracker = LatencyTracker("xdotool")
    summary = tracker.summary()
    assert summary == {"name": "xdotool", "count": 0}


def test_latency_tracker_bounded_capacity() -> None:
    tracker = LatencyTracker("x", capacity=3)
    for ms in [1.0, 2.0, 3.0, 4.0, 5.0]:
        tracker.record_ms(ms)
    summary = tracker.summary()
    assert summary["count"] == 3
    # Only the last 3 (3, 4, 5) survive.
    assert summary["max_ms"] == 5.0


# ── instrumentation ───────────────────────────────────────────────────


def test_local_impl_records_screenshot_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The screenshot wrapper times the underlying ``_screenshot``."""

    def _faux(self: Any) -> str:
        return "img"  # type: ignore[return-value]

    monkeypatch.setattr(XdotoolGymEnv, "_screenshot", _faux)
    env = LocalXdotoolImpl()
    result = env._screenshot()

    assert result == "img"
    summary = env.screenshot_latency.summary()
    assert summary["count"] == 1
    assert summary["p50_ms"] >= 0.0


def test_local_impl_records_xdotool_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The xdotool wrapper times each subprocess invocation."""
    calls: list[tuple[str, ...]] = []

    def _faux(self: Any, *args: str) -> None:
        calls.append(args)

    monkeypatch.setattr(XdotoolGymEnv, "_xdotool", _faux)
    env = LocalXdotoolImpl()
    env._xdotool("mousemove", "100", "200")
    env._xdotool("click", "1")

    assert calls == [("mousemove", "100", "200"), ("click", "1")]
    summary = env.xdotool_latency.summary()
    assert summary["count"] == 2


def test_latency_report_returns_both_trackers() -> None:
    env = object.__new__(LocalXdotoolImpl)
    env.screenshot_latency = LatencyTracker("screenshot")
    env.xdotool_latency = LatencyTracker("xdotool")
    env.screenshot_latency.record_ms(5.0)
    env.xdotool_latency.record_ms(0.5)

    report = env.latency_report()
    assert set(report.keys()) == {"screenshot", "xdotool"}
    assert report["screenshot"]["count"] == 1
    assert report["xdotool"]["count"] == 1


# ── kwarg pass-through ────────────────────────────────────────────────


def test_local_impl_forwards_kwargs_to_xdotool_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LocalXdotoolImpl must inherit reuse_session, viewport, etc."""
    env = LocalXdotoolImpl(reuse_session=True, viewport=(1024, 768))
    assert env._reuse_session is True
    assert env._viewport == (1024, 768)
    # Both trackers initialized.
    assert env.screenshot_latency.summary() == {"name": "screenshot", "count": 0}
    assert env.xdotool_latency.summary() == {"name": "xdotool", "count": 0}
