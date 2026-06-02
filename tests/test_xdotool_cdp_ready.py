"""Regression tests for the CDP-readiness poll that replaced the blind
``time.sleep(3)`` after Chrome launch.

The pre-fix bug: ``_launch_browser`` slept a fixed 3s then immediately ran the
CDP stealth-inject + persistent-header-session calls. On a cold Modal
container Chrome can take >3s to bind ``--remote-debugging-port``, so the
header seam's ``urlopen('/json/list')`` got ECONNREFUSED, the seam never
opened, the Daytona consent cookie was dropped, and the boattrader sim env
rendered its consent overlay — which the click handler's find_all pre-scan
reads as a blocked page → ``page_blocked`` halt → 0 leads. Surfaced on the
BT02 frozen smoke 2026-06-01 (both tasks scored 0.0 / fail at $0 cost).

The fix: ``_wait_for_cdp_ready`` polls ``/json/version`` until it answers (up
to a deadline) before the CDP calls, so a slow cold-start is absorbed instead
of raced against a fixed sleep.
"""

from __future__ import annotations

import urllib.error

import pytest

from mantis_agent.gym import xdotool_env
from mantis_agent.gym.xdotool_env import XdotoolGymEnv


class _FakeResp:
    """Minimal stand-in for the ``urlopen`` context-manager response."""

    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _make_env(cdp_port: int = 9222) -> XdotoolGymEnv:
    env = XdotoolGymEnv.__new__(XdotoolGymEnv)
    env._cdp_port = cdp_port
    return env


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    """Drive ``time`` from a fake clock so the poll loop never really sleeps.

    ``sleep(s)`` advances the clock by ``s``; ``time()`` reads it. This makes
    the deadline arithmetic deterministic and instant.
    """
    clock = {"t": 1000.0}
    monkeypatch.setattr(xdotool_env.time, "time", lambda: clock["t"])
    monkeypatch.setattr(
        xdotool_env.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s)
    )
    return clock


def test_cdp_ready_returns_true_immediately_when_port_up(
    fake_clock: dict[str, float], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm path: the port answers on the first probe."""
    calls = {"n": 0}

    def _urlopen(url: str, timeout: float = 2):
        calls["n"] += 1
        assert "/json/version" in url
        return _FakeResp(200)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)

    assert _make_env()._wait_for_cdp_ready() is True
    assert calls["n"] == 1


def test_cdp_ready_polls_through_cold_start_refusals(
    fake_clock: dict[str, float], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The race scenario: the port refuses twice, then comes up — the poll
    absorbs the cold-start instead of giving up like the fixed sleep did."""
    calls = {"n": 0}

    def _urlopen(url: str, timeout: float = 2):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError(ConnectionRefusedError(111, "refused"))
        return _FakeResp(200)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)

    assert _make_env()._wait_for_cdp_ready() is True
    assert calls["n"] == 3  # two refusals absorbed, third succeeded


def test_cdp_ready_gives_up_after_deadline(
    fake_clock: dict[str, float], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the port never comes up, return False (best-effort) rather than
    block forever — the caller's CDP calls are independently best-effort."""

    def _urlopen(url: str, timeout: float = 2):
        raise urllib.error.URLError(ConnectionRefusedError(111, "refused"))

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)

    assert _make_env()._wait_for_cdp_ready(deadline_s=3.0) is False
