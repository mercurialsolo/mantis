"""Tests for #541 external pause + auto-pause-on-captcha (human takeover).

Two trigger paths:

1. External pause — API container writes pause_request.json (via
   action=pause HTTP); runner blocks in wait_while_paused while
   keeping Chrome + noVNC viewer alive.
2. Auto-pause-on-captcha — runner self-triggers the same sentinel
   when a step fails with failure_class='cf_challenge' (gated by
   MANTIS_PAUSE_ON_CAPTCHA, default-on).

Plus a fix to the failure_class aggregation: cause-level classes
(cf_challenge / proxy_failed / http_*xx / nav_timeout) always win
over symptom-level classes (brain_loop_exhausted, no_state_change)
when both are observed in a run's results.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mantis_agent.gym import external_pause


# ── external_pause module ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_pause_path():
    """Reset the module-level sentinel path between tests."""
    original = external_pause._REQUEST_PATH
    external_pause._REQUEST_PATH = None
    yield
    external_pause._REQUEST_PATH = original


def test_is_pause_requested_returns_false_when_no_path_wired():
    """Without ``init_paths``, all helpers no-op — runner proceeds
    normally (local CLI / tests have no Modal volume to write to)."""
    assert external_pause._REQUEST_PATH is None
    assert external_pause.is_pause_requested() is False
    assert external_pause.read_pause_reason() == ""
    assert external_pause.request_pause("x") is False
    assert external_pause.clear_pause_request() is False
    assert external_pause.wait_while_paused(max_seconds=1) == "not_paused"


def test_init_paths_then_request_and_clear(tmp_path: Path):
    """Roundtrip: init → request → is_pause_requested → read_reason
    → clear → not requested anymore."""
    external_pause.init_paths(tmp_path / "pause.json")
    assert external_pause.is_pause_requested() is False
    assert external_pause.request_pause("test_reason") is True
    assert external_pause.is_pause_requested() is True
    assert external_pause.read_pause_reason() == "test_reason"
    assert external_pause.clear_pause_request() is True
    assert external_pause.is_pause_requested() is False


def test_wait_while_paused_returns_resumed_when_sentinel_cleared(tmp_path: Path):
    """The whole point: wait_while_paused blocks until the sentinel
    disappears, then returns 'resumed'. Use a short timeout + a
    background-thread clear to simulate action=resume."""
    import threading
    external_pause.init_paths(tmp_path / "pause.json")
    external_pause.request_pause("test")

    def _clear_after(delay):
        time.sleep(delay)
        external_pause.clear_pause_request()

    t = threading.Thread(target=_clear_after, args=(0.5,), daemon=True)
    t.start()
    result = external_pause.wait_while_paused(max_seconds=5, poll_seconds=0.1)
    assert result == "resumed"


def test_wait_while_paused_returns_timeout(tmp_path: Path):
    """Bounded wait — must not block forever. After max_seconds the
    helper clears the sentinel itself and returns 'timeout'."""
    external_pause.init_paths(tmp_path / "pause.json")
    external_pause.request_pause("hang")
    result = external_pause.wait_while_paused(max_seconds=1, poll_seconds=0.1)
    assert result == "timeout"
    # Sentinel self-cleared so next iteration doesn't re-pause.
    assert external_pause.is_pause_requested() is False


def test_wait_while_paused_immediate_return_when_no_sentinel(tmp_path: Path):
    """Hot-path optimization: if sentinel isn't set when called,
    return immediately (no sleep, no log spam)."""
    external_pause.init_paths(tmp_path / "pause.json")
    assert external_pause.wait_while_paused(max_seconds=10) == "not_paused"


def test_is_pause_requested_calls_reload_cb_only_when_cached_exists(tmp_path: Path):
    """Volume-staleness defence: when the cached stat says the
    sentinel exists, ``is_pause_requested`` invokes the reload
    callback wired through ``init_paths`` and re-stats. Without
    cached-exists, the reload is skipped — we don't pay the
    reload tax on every poll of a healthy run."""
    reload_calls: list[int] = []
    sentinel = tmp_path / "pause.json"
    external_pause.init_paths(
        sentinel, reload_cb=lambda: reload_calls.append(1),
    )

    # Cached state: sentinel does NOT exist → no reload call.
    assert external_pause.is_pause_requested() is False
    assert reload_calls == []

    # Write sentinel: cached state will now say "exists" → reload fires
    # then re-stat confirms exists.
    external_pause.request_pause(reason="external")
    assert external_pause.is_pause_requested() is True
    assert len(reload_calls) == 1

    # Delete sentinel between calls. The next call's first stat still
    # sees the file (we just wrote it locally), reload fires, second
    # stat may or may not see the deletion depending on filesystem
    # semantics — for the local-FS test we just confirm reload was
    # invoked when the cached path-exists check passed.
    sentinel.unlink()
    assert external_pause.is_pause_requested() is False
    # No reload call on this iteration — first stat returned False
    # so the reload short-circuit didn't fire.
    assert len(reload_calls) == 1


def test_init_paths_reload_cb_defaults_to_noop(tmp_path: Path):
    """Legacy callers that don't pass ``reload_cb`` get the default
    no-op so existing test code + local-CLI runs keep working
    without changes."""
    # Reset module state via fresh init_paths without reload_cb.
    external_pause.init_paths(tmp_path / "pause.json")
    # Cycle through write / read / delete — should not raise.
    external_pause.request_pause()
    assert external_pause.is_pause_requested() is True
    external_pause.clear_pause_request()
    assert external_pause.is_pause_requested() is False


def test_is_pause_requested_swallows_reload_cb_exception(tmp_path: Path):
    """A failing reload_cb (e.g. transient network error talking to
    the Modal Volume service) must NOT block ``is_pause_requested``
    from returning a stat answer — the reload is best-effort."""
    def _boom():
        raise RuntimeError("modal volume unavailable")
    external_pause.init_paths(tmp_path / "pause.json", reload_cb=_boom)
    external_pause.request_pause()
    # Should not raise even though reload_cb does.
    assert external_pause.is_pause_requested() is True


def test_is_captcha_autopause_enabled_default_true(monkeypatch):
    monkeypatch.delenv("MANTIS_PAUSE_ON_CAPTCHA", raising=False)
    assert external_pause.is_captcha_autopause_enabled() is True


def test_is_captcha_autopause_disabled_via_env(monkeypatch):
    for v in ["0", "false", "no", "off"]:
        monkeypatch.setenv("MANTIS_PAUSE_ON_CAPTCHA", v)
        assert external_pause.is_captcha_autopause_enabled() is False, (
            f"{v!r} should disable"
        )


# ── _maybe_auto_pause_on_captcha wiring in RunExecutor ───────────────────


def test_runner_auto_pauses_on_cf_challenge_step_result(
    tmp_path: Path, monkeypatch,
):
    """When _emit_augur_step's adjacent helper sees a step result
    with failure_class='cf_challenge', it must write the pause
    sentinel AND block in-place via wait_while_paused.

    Mocks wait_while_paused to return immediately so the test
    doesn't block on the 30-min sentinel timeout. Captures the
    sentinel state at the moment wait_while_paused is invoked
    (the sentinel is cleared on wait return when timeout, but
    we care that it was written before the wait)."""
    from mantis_agent.gym.run_executor import RunExecutor

    external_pause.init_paths(tmp_path / "pause.json")
    monkeypatch.delenv("MANTIS_PAUSE_ON_CAPTCHA", raising=False)

    # Mock wait_while_paused — capture sentinel state at call time,
    # then return immediately (no blocking, no auto-clear).
    sentinel_at_wait: dict = {}
    def _spy_wait(*args, **kwargs):
        sentinel_at_wait["was_requested"] = external_pause.is_pause_requested()
        sentinel_at_wait["reason"] = external_pause.read_pause_reason()
        return "resumed"
    monkeypatch.setattr(external_pause, "wait_while_paused", _spy_wait)

    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = MagicMock()

    sr = MagicMock()
    sr.step_index = 2
    sr.failure_class = "cf_challenge"
    executor._maybe_auto_pause_on_captcha(sr)

    # Sentinel was written before the wait kicked in.
    assert sentinel_at_wait.get("was_requested") is True, (
        "Sentinel must be written before wait_while_paused is called"
    )
    assert sentinel_at_wait.get("reason") == "cf_challenge_human_takeover"


def test_runner_does_not_auto_pause_on_other_failures(tmp_path: Path, monkeypatch):
    """Auto-pause is targeted at cf_challenge — other failure classes
    should NOT trigger pause (they have their own recovery paths)."""
    from mantis_agent.gym.run_executor import RunExecutor

    external_pause.init_paths(tmp_path / "pause.json")
    monkeypatch.delenv("MANTIS_PAUSE_ON_CAPTCHA", raising=False)

    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = MagicMock()

    for fc in ("no_state_change", "brain_loop_exhausted", "selector_miss", ""):
        sr = MagicMock(step_index=2, failure_class=fc)
        executor._maybe_auto_pause_on_captcha(sr)
        assert external_pause.is_pause_requested() is False, (
            f"failure_class={fc!r} must NOT trigger auto-pause"
        )


def test_runner_skips_auto_pause_when_env_disabled(tmp_path: Path, monkeypatch):
    """MANTIS_PAUSE_ON_CAPTCHA=0 → cf_challenge falls back to legacy
    halt behavior (no pause sentinel)."""
    from mantis_agent.gym.run_executor import RunExecutor

    external_pause.init_paths(tmp_path / "pause.json")
    monkeypatch.setenv("MANTIS_PAUSE_ON_CAPTCHA", "0")

    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = MagicMock()
    sr = MagicMock(step_index=2, failure_class="cf_challenge")
    executor._maybe_auto_pause_on_captcha(sr)
    assert external_pause.is_pause_requested() is False


# ── _pick_primary_failure_class + _emit_augur_failure_class_tag ──────────


def test_pick_primary_failure_class_prefers_cause_over_symptom():
    """cf_challenge (cause) on step 2 must win over brain_loop_exhausted
    (symptom) on step 3 — the legacy 'first non-empty' behavior was
    surfacing the symptom and hiding the cause."""
    from mantis_agent.gym.run_executor import RunExecutor

    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = MagicMock()

    results = [
        MagicMock(failure_class=""),                       # step 1 success
        MagicMock(failure_class="cf_challenge"),           # step 2 cause
        MagicMock(failure_class="brain_loop_exhausted"),   # step 3 symptom
    ]
    primary = executor._pick_primary_failure_class(results)
    assert primary == "cf_challenge"


def test_pick_primary_failure_class_falls_back_to_first_when_no_cause():
    """When no cause-level class is observed, fall back to legacy
    'first non-empty' behavior."""
    from mantis_agent.gym.run_executor import RunExecutor

    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = MagicMock()

    results = [
        MagicMock(failure_class=""),
        MagicMock(failure_class="brain_loop_exhausted"),
        MagicMock(failure_class="no_state_change"),
    ]
    assert executor._pick_primary_failure_class(results) == "brain_loop_exhausted"


def test_pick_primary_failure_class_returns_empty_when_all_clean():
    """All steps succeeded → empty string (caller emits no tag)."""
    from mantis_agent.gym.run_executor import RunExecutor

    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = MagicMock()

    results = [MagicMock(failure_class=""), MagicMock(failure_class="")]
    assert executor._pick_primary_failure_class(results) == ""


def test_pick_primary_failure_class_recognizes_all_cause_classes():
    """The cause set must include every documented cause-level
    class — defending against future regression where a new class
    is added to failure_class.py but not to the cause set."""
    from mantis_agent.gym.run_executor import RunExecutor

    cause_set = RunExecutor._CAUSE_LEVEL_FAILURE_CLASSES
    # Spot-check the headline ones
    for cls in ("cf_challenge", "http_4xx", "http_5xx", "nav_timeout",
                "wrong_target", "budget_exceeded"):
        assert cls in cause_set, f"{cls!r} missing from cause-level set"


def test_emit_augur_failure_class_tag_emits_cause_and_symptom(monkeypatch):
    """End-to-end: when cause + symptom both present, emit
    failure_class=<cause> AND failure_class_symptom=<symptom> so
    the dashboard can show both signals."""
    from mantis_agent.gym.run_executor import RunExecutor

    executor = RunExecutor.__new__(RunExecutor)
    augur = MagicMock()
    augur.active = True
    runner = MagicMock(_augur=augur, _final_halt_reason="")
    executor.parent = runner

    results = [
        MagicMock(failure_class=""),
        MagicMock(failure_class="cf_challenge"),
        MagicMock(failure_class="brain_loop_exhausted"),
    ]
    executor._emit_augur_failure_class_tag(results)

    tag_calls = {c.args[0]: c.args[1] for c in augur.add_tag.call_args_list}
    assert tag_calls.get("failure_class") == "cf_challenge"
    assert tag_calls.get("failure_class_symptom") == "brain_loop_exhausted"


def test_emit_augur_failure_class_tag_skips_symptom_when_same_as_cause():
    """When cause == final-step failure, don't emit redundant symptom tag."""
    from mantis_agent.gym.run_executor import RunExecutor

    executor = RunExecutor.__new__(RunExecutor)
    augur = MagicMock()
    augur.active = True
    runner = MagicMock(_augur=augur, _final_halt_reason="")
    executor.parent = runner

    results = [
        MagicMock(failure_class=""),
        MagicMock(failure_class="cf_challenge"),  # only fc, cause==symptom
    ]
    executor._emit_augur_failure_class_tag(results)

    tag_keys = {c.args[0] for c in augur.add_tag.call_args_list}
    assert "failure_class" in tag_keys
    assert "failure_class_symptom" not in tag_keys
