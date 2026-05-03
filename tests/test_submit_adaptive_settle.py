"""Tests for the adaptive submit settle.

Surfaced by the staffcrm v6 verify (run 20260503_132147_0eabfcc4):
the runner correctly typed the credentials and clicked Sign In, but
the fixed 2.5s settle wasn't long enough for the CRM's login redirect.
The state-change verifier (#150) correctly demoted the step to fail —
but each retry burned the same too-short window.

The fix is adaptive polling: wait up to 8s, breaking out early as
soon as the URL changes. Fast logins finish in 1-2s and pay almost
no penalty; slow logins get the full budget. Pure-observational —
polls the env's CDP-backed ``current_url``, no LLM call.

These tests exercise the polling logic in isolation via a fake env
(no Modal, no Xvfb).
"""

from __future__ import annotations

import time
from typing import Any

from mantis_agent.gym.micro_runner import MicroPlanRunner


# ── Fakes ───────────────────────────────────────────────────────────────


class _FakeEnvWithUrl:
    """Stand-in for XdotoolGymEnv that exposes a controllable current_url."""

    def __init__(self, urls: list[str]) -> None:
        # Each successive read returns the next URL in the list.
        self._urls = list(urls)
        self.reads = 0

    @property
    def current_url(self) -> str:
        if self.reads < len(self._urls):
            url = self._urls[self.reads]
        else:
            url = self._urls[-1] if self._urls else ""
        self.reads += 1
        return url


class _FakeEnvNoUrl:
    """Stand-in env that doesn't expose current_url at all."""


class _FakeEnvRaising:
    """Stand-in env where current_url raises (simulates CDP unreachable)."""

    @property
    def current_url(self) -> str:
        raise RuntimeError("CDP unreachable")


def _runner_with_env(env: Any) -> MicroPlanRunner:
    """Build a MicroPlanRunner with just enough state for the settle helpers."""
    runner = MicroPlanRunner.__new__(MicroPlanRunner)
    runner.env = env
    return runner


# ── _best_effort_current_url ──────────────────────────────────────────


def test_best_effort_url_returns_value_when_env_exposes_it() -> None:
    runner = _runner_with_env(_FakeEnvWithUrl(["https://x.test/login"]))
    assert runner._best_effort_current_url() == "https://x.test/login"


def test_best_effort_url_returns_empty_when_env_lacks_attr() -> None:
    runner = _runner_with_env(_FakeEnvNoUrl())
    assert runner._best_effort_current_url() == ""


def test_best_effort_url_returns_empty_on_raise() -> None:
    runner = _runner_with_env(_FakeEnvRaising())
    assert runner._best_effort_current_url() == ""


# ── _adaptive_submit_settle ──────────────────────────────────────────


def test_settle_breaks_early_on_url_change() -> None:
    """The login pattern: URL was /login, becomes /dashboard mid-settle."""
    env = _FakeEnvWithUrl([
        "https://x.test/login",       # poll 1: still on login
        "https://x.test/dashboard",   # poll 2: navigated! exit
    ])
    runner = _runner_with_env(env)
    t0 = time.monotonic()
    elapsed = runner._adaptive_submit_settle(url_before="https://x.test/login")
    real_elapsed = time.monotonic() - t0
    # Should have stopped well before the 8s max.
    assert elapsed < 4.0
    assert real_elapsed < 4.0


def test_settle_pays_full_budget_when_url_never_changes() -> None:
    env = _FakeEnvWithUrl(["https://x.test/login"] * 100)
    runner = _runner_with_env(env)
    t0 = time.monotonic()
    elapsed = runner._adaptive_submit_settle(url_before="https://x.test/login")
    real_elapsed = time.monotonic() - t0
    # Returns the max budget (with the min-settle floor applied).
    assert elapsed >= MicroPlanRunner._SUBMIT_SETTLE_MAX_SECONDS - 1
    assert real_elapsed >= MicroPlanRunner._SUBMIT_SETTLE_MAX_SECONDS - 1


def test_settle_always_waits_minimum() -> None:
    """Even if the URL changes immediately, we wait at least the min so
    the page has DOM time before the next find_form_target call."""
    env = _FakeEnvWithUrl(["https://x.test/already-changed"] * 100)
    runner = _runner_with_env(env)
    t0 = time.monotonic()
    runner._adaptive_submit_settle(url_before="https://x.test/login")
    real_elapsed = time.monotonic() - t0
    # Min settle floor is enforced.
    assert real_elapsed >= MicroPlanRunner._SUBMIT_SETTLE_MIN_SECONDS - 0.05


def test_settle_handles_env_without_current_url() -> None:
    """No URL accessor → polling can't detect change → falls through to
    full budget. Doesn't raise."""
    runner = _runner_with_env(_FakeEnvNoUrl())
    elapsed = runner._adaptive_submit_settle(url_before="https://x.test/login")
    assert elapsed >= MicroPlanRunner._SUBMIT_SETTLE_MAX_SECONDS - 1


def test_settle_handles_empty_url_before() -> None:
    """If we couldn't capture URL pre-click, we can't detect change.
    Pay the full budget. Don't raise."""
    env = _FakeEnvWithUrl(["https://x.test/some-page"])
    runner = _runner_with_env(env)
    elapsed = runner._adaptive_submit_settle(url_before="")
    assert elapsed >= MicroPlanRunner._SUBMIT_SETTLE_MAX_SECONDS - 1


def test_settle_returns_elapsed_seconds_for_cost_meter() -> None:
    """The returned float is consumed by the cost meter — must be a
    realistic non-negative duration."""
    env = _FakeEnvWithUrl([
        "https://x.test/login",
        "https://x.test/dashboard",
    ])
    runner = _runner_with_env(env)
    elapsed = runner._adaptive_submit_settle(url_before="https://x.test/login")
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


# ── Constants are sane ───────────────────────────────────────────────


def test_settle_constants_are_reasonable() -> None:
    """Sanity: max > min, both positive, poll interval smaller than min."""
    assert MicroPlanRunner._SUBMIT_SETTLE_MAX_SECONDS > MicroPlanRunner._SUBMIT_SETTLE_MIN_SECONDS
    assert MicroPlanRunner._SUBMIT_SETTLE_MIN_SECONDS > 0
    assert MicroPlanRunner._SUBMIT_SETTLE_POLL_SECONDS < MicroPlanRunner._SUBMIT_SETTLE_MIN_SECONDS
    # Max budget should cover a slow CRM login (~5s) plus headroom.
    assert MicroPlanRunner._SUBMIT_SETTLE_MAX_SECONDS >= 5.0
