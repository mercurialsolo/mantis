"""Unit tests for ``mantis_agent.gym.pointer_retry``.

The helper extracted from ``ClaudeGuidedFormHandler`` (PR #447 Fix B)
so the click handler and any future handler can share the trust-gate-
bypass logic.

The gate conditions are well-defined: ``url_before`` non-empty + final
URL equals ``url_before`` + executor was SoM + env exposes
``cdp_click_via_pointer``. Each test pins one or more of these
conditions so a future regression in the gate logic is caught
immediately.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mantis_agent.gym.pointer_retry import pointer_retry_if_unchanged


class _FakeRunner:
    """Minimal runner stub: URL history + cost dict + settle stub."""

    def __init__(self, urls: list[str]) -> None:
        self._urls = list(urls)
        self.costs: dict[str, float] = {"gpu_seconds": 0, "gpu_steps": 0}
        self.settle_calls: list[str] = []

    def _best_effort_current_url(self) -> str:
        return self._urls.pop(0) if self._urls else ""

    def _adaptive_submit_settle(self, *, url_before: str) -> float:
        self.settle_calls.append(url_before)
        return 0.5


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the stabilization-window sleeps in tests."""
    monkeypatch.setattr("mantis_agent.gym.pointer_retry.time.sleep", lambda *_: None)


def test_pointer_retry_fires_when_som_click_did_not_navigate() -> None:
    """Canonical case: SoM click + URL unchanged → retry via real-pointer."""
    runner = _FakeRunner(urls=[
        "https://app.example/leads",  # url_after_click (immediate poll)
        "https://app.example/leads",  # final_url (after stabilization)
        "https://app.example/leads/42",  # post-retry URL
    ])
    env = MagicMock()
    env.cdp_click_via_pointer.return_value = True

    result = pointer_retry_if_unchanged(
        env, runner, 100, 200,
        url_before="https://app.example/leads",
        executor_backend="som",
        log_prefix="[test]",
    )

    env.cdp_click_via_pointer.assert_called_once_with(100, 200)
    assert result == "https://app.example/leads/42"
    # Cost-accounting: one extra settle + one extra gpu_step billed
    assert runner.costs["gpu_steps"] == 1
    assert runner.costs["gpu_seconds"] == 0.5
    assert runner.settle_calls == ["https://app.example/leads"]


def test_pointer_retry_skipped_when_url_changed_after_click() -> None:
    """If the URL DID change, the click actually navigated → no retry."""
    runner = _FakeRunner(urls=[
        "https://app.example/leads/42",  # immediate poll: URL changed
        "https://app.example/leads/42",  # final URL: still changed
    ])
    env = MagicMock()

    result = pointer_retry_if_unchanged(
        env, runner, 100, 200,
        url_before="https://app.example/leads",
        executor_backend="som",
        log_prefix="[test]",
    )

    env.cdp_click_via_pointer.assert_not_called()
    assert result == "https://app.example/leads/42"
    # No extra settle billed when retry doesn't fire
    assert runner.costs["gpu_steps"] == 0


def test_pointer_retry_skipped_when_executor_was_vision() -> None:
    """xdotool / vision-path clicks are already real events; no retry needed."""
    runner = _FakeRunner(urls=[
        "https://app.example/leads",
        "https://app.example/leads",
    ])
    env = MagicMock()

    result = pointer_retry_if_unchanged(
        env, runner, 100, 200,
        url_before="https://app.example/leads",
        executor_backend="vision",  # ← not SoM
        log_prefix="[test]",
    )

    env.cdp_click_via_pointer.assert_not_called()
    assert result == "https://app.example/leads"


def test_pointer_retry_skipped_when_url_before_empty() -> None:
    """No baseline URL → can't decide if click navigated → no retry."""
    runner = _FakeRunner(urls=["", ""])
    env = MagicMock()

    result = pointer_retry_if_unchanged(
        env, runner, 100, 200,
        url_before="",
        executor_backend="som",
        log_prefix="[test]",
    )

    env.cdp_click_via_pointer.assert_not_called()
    # Empty url_before is preserved through the function
    assert result == ""


def test_pointer_retry_skipped_when_env_lacks_cdp_click_via_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If env doesn't expose the real-pointer method, gate fails closed."""
    runner = _FakeRunner(urls=[
        "https://app.example/leads",
        "https://app.example/leads",
    ])
    env = MagicMock(spec=[])  # no methods auto-created

    result = pointer_retry_if_unchanged(
        env, runner, 100, 200,
        url_before="https://app.example/leads",
        executor_backend="som",
        log_prefix="[test]",
    )

    # No exception raised; returns final URL
    assert result == "https://app.example/leads"


def test_pointer_retry_handles_url_bounce_between_polls() -> None:
    """If the immediate post-click URL is transient (page mid-navigation),
    re-read after the stabilization window — the gate evaluates against
    the FINAL URL, not the transient one.
    """
    runner = _FakeRunner(urls=[
        "https://app.example/leads?status=Contacted",  # transient
        "https://app.example/leads",  # final (bounced back to base)
        "https://app.example/leads",  # post-retry settle
    ])
    env = MagicMock()
    env.cdp_click_via_pointer.return_value = True

    result = pointer_retry_if_unchanged(
        env, runner, 100, 200,
        url_before="https://app.example/leads",
        executor_backend="som",
        log_prefix="[test]",
    )

    # Bounce detected → retry SHOULD have fired (final URL == url_before)
    env.cdp_click_via_pointer.assert_called_once()
    assert result == "https://app.example/leads"


def test_pointer_retry_blank_final_url_keeps_immediate_read() -> None:
    """If the second URL poll returns blank (transient read failure),
    don't treat blank as a navigation. Keep the first poll's value.
    """
    runner = _FakeRunner(urls=[
        "https://app.example/leads",  # immediate poll
        "",                            # second poll: blank (failure)
        "https://app.example/leads",  # post-retry settle
    ])
    env = MagicMock()
    env.cdp_click_via_pointer.return_value = True

    result = pointer_retry_if_unchanged(
        env, runner, 100, 200,
        url_before="https://app.example/leads",
        executor_backend="som",
        log_prefix="[test]",
    )

    # Retry SHOULD still fire because url_after_click == url_before
    # (blank was discarded, keeping the immediate read of "leads")
    env.cdp_click_via_pointer.assert_called_once()
    assert result == "https://app.example/leads"


def test_pointer_retry_cdp_dispatch_failure_returns_pre_retry_url() -> None:
    """If cdp_click_via_pointer returns False or raises, no settle is
    billed and the returned URL is the pre-retry stabilized URL.
    """
    runner = _FakeRunner(urls=[
        "https://app.example/leads",
        "https://app.example/leads",
    ])
    env = MagicMock()
    env.cdp_click_via_pointer.return_value = False  # dispatch declined

    result = pointer_retry_if_unchanged(
        env, runner, 100, 200,
        url_before="https://app.example/leads",
        executor_backend="som",
        log_prefix="[test]",
    )

    env.cdp_click_via_pointer.assert_called_once()
    # No extra cost billed (dispatch failed)
    assert runner.costs["gpu_steps"] == 0
    # Returned URL is the pre-retry stabilized URL
    assert result == "https://app.example/leads"


def test_pointer_retry_cdp_dispatch_exception_does_not_propagate() -> None:
    """An exception in cdp_click_via_pointer is caught and logged at debug;
    function returns gracefully with the pre-retry URL.
    """
    runner = _FakeRunner(urls=[
        "https://app.example/leads",
        "https://app.example/leads",
    ])
    env = MagicMock()
    env.cdp_click_via_pointer.side_effect = RuntimeError("CDP unreachable")

    result = pointer_retry_if_unchanged(
        env, runner, 100, 200,
        url_before="https://app.example/leads",
        executor_backend="som",
        log_prefix="[test]",
    )

    assert result == "https://app.example/leads"
    assert runner.costs["gpu_steps"] == 0


def test_pointer_retry_runner_without_url_reader_returns_blank() -> None:
    """A runner stub that doesn't expose ``_best_effort_current_url``
    is treated as having no URL info; gate fails closed.
    """
    runner = MagicMock(spec=["costs"])  # no URL reader
    runner.costs = {"gpu_seconds": 0, "gpu_steps": 0}
    env = MagicMock()

    result = pointer_retry_if_unchanged(
        env, runner, 100, 200,
        url_before="https://app.example/leads",
        executor_backend="som",
        log_prefix="[test]",
    )

    # url_after_click was "" (no reader); url_before was "/leads"
    # → gate fails (mismatch) → no retry
    env.cdp_click_via_pointer.assert_not_called()
    assert result == ""
