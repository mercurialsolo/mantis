"""Tests for the runtime ``pause_on_captcha`` field (#570).

Promotes ``MANTIS_PAUSE_ON_CAPTCHA`` from a deploy-wide env var to a
per-submission runtime field. A single submission can opt out of the
30-min cf_challenge auto-pause loop (PR #555) without touching the
Modal Secret — useful for CI / verify reruns where a real human won't
solve the CF widget in time anyway.

Covers:

1. ``is_captcha_autopause_enabled(override=...)`` resolution order
2. ``MicroPlanRunner.pause_on_captcha`` stored as-passed
3. ``build_micro_suite`` round-trips through ``_pause_on_captcha``
4. ``merge_runtime`` accepts the field
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from mantis_agent.gym import external_pause
from mantis_agent.gym.micro_runner import MicroPlanRunner
from mantis_agent.server_utils import build_micro_suite, merge_runtime


# ── is_captcha_autopause_enabled resolution ────────────────────────


def test_override_false_disables_regardless_of_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_PAUSE_ON_CAPTCHA", "1")
    assert external_pause.is_captcha_autopause_enabled(override=False) is False


def test_override_true_enables_regardless_of_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_PAUSE_ON_CAPTCHA", "0")
    assert external_pause.is_captcha_autopause_enabled(override=True) is True


def test_override_none_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_PAUSE_ON_CAPTCHA", "0")
    assert external_pause.is_captcha_autopause_enabled(override=None) is False
    monkeypatch.setenv("MANTIS_PAUSE_ON_CAPTCHA", "1")
    assert external_pause.is_captcha_autopause_enabled(override=None) is True


def test_override_none_defaults_on_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANTIS_PAUSE_ON_CAPTCHA", raising=False)
    assert external_pause.is_captcha_autopause_enabled() is True
    assert external_pause.is_captcha_autopause_enabled(override=None) is True


def test_backward_compat_no_kwarg_still_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The legacy call-without-kwarg shape must keep working — nothing
    # outside MicroPlanRunner threads the override yet.
    monkeypatch.setenv("MANTIS_PAUSE_ON_CAPTCHA", "false")
    assert external_pause.is_captcha_autopause_enabled() is False


# ── MicroPlanRunner storage ────────────────────────────────────────


def _runner(**kwargs) -> MicroPlanRunner:
    return MicroPlanRunner(brain=MagicMock(), env=MagicMock(), **kwargs)


def test_runner_default_pause_on_captcha_is_none() -> None:
    # None preserves the env-var fallback path — must not be coerced
    # to True/False at construction.
    runner = _runner()
    assert runner.pause_on_captcha is None


def test_runner_explicit_false_stored() -> None:
    runner = _runner(pause_on_captcha=False)
    assert runner.pause_on_captcha is False


def test_runner_explicit_true_stored() -> None:
    runner = _runner(pause_on_captcha=True)
    assert runner.pause_on_captcha is True


# ── build_micro_suite round-trip ───────────────────────────────────


def test_suite_omits_pause_on_captcha_when_caller_passes_none() -> None:
    suite = build_micro_suite([], "example.com")
    assert "_pause_on_captcha" not in suite


def test_suite_persists_explicit_false() -> None:
    suite = build_micro_suite([], "example.com", pause_on_captcha=False)
    assert suite["_pause_on_captcha"] is False


def test_suite_persists_explicit_true() -> None:
    suite = build_micro_suite([], "example.com", pause_on_captcha=True)
    assert suite["_pause_on_captcha"] is True


# ── merge_runtime accepts pause_on_captcha ─────────────────────────


def test_merge_runtime_threads_pause_on_captcha() -> None:
    merged = merge_runtime({"pause_on_captcha": False})
    assert merged["pause_on_captcha"] is False


def test_merge_runtime_override_wins_over_plan_default() -> None:
    merged = merge_runtime(
        {"pause_on_captcha": True},
        pause_on_captcha=False,
    )
    assert merged["pause_on_captcha"] is False
