"""cua-issues 2026-06-29 (S02) — done-gate honesty.

The audit found the runner force-accepted a `done(success=True)` the
done-gate had already rejected `max_done_rejections` times (S02), and that
honest `done(success=False)` bails (A02 "stuck", L07 ERR_TUNNEL) were also
reported `succeeded`. Both now get a non-"done" termination_reason
("done_unverified" / "done_failed") so the status mapping reports halted.

These pin the status consequence via `_run_status_from_result`.
"""

from __future__ import annotations

from types import SimpleNamespace

from mantis_agent.task_loop import _run_status_from_result


def _r(reason: str, success: bool = False, paused: bool = False) -> SimpleNamespace:
    return SimpleNamespace(termination_reason=reason, success=success, paused=paused)


def test_verified_done_succeeds():
    assert _run_status_from_result(_r("done", success=True)) == "succeeded"


def test_done_failed_is_halted():
    # Agent explicitly bailed (done(success=false)).
    assert _run_status_from_result(_r("done_failed", success=False)) == "halted"


def test_done_unverified_is_halted():
    # Force-accepted after the done-gate hit its rejection cap (S02).
    assert _run_status_from_result(_r("done_unverified", success=False)) == "halted"


def test_loop_and_max_steps_halted():
    assert _run_status_from_result(_r("loop")) == "halted"
    assert _run_status_from_result(_r("max_steps")) == "halted"


def test_env_done_with_reward_succeeds():
    assert _run_status_from_result(_r("env_done", success=True)) == "succeeded"
