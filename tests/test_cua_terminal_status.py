"""cua-issues 2026-06-29 — /v1/cua status honesty.

The frame-by-frame audit found `/v1/cua` reported `succeeded` on 43/43 runs
while only 4 actually worked (91% false-success). Root cause: the cua
executor's result envelope (`run_executor_lifecycle`) carried no
`terminal_status`, so the Modal/Baseten wire mapping hit its
`else → succeeded` default for every run regardless of loop/max_steps.

These tests pin the new honest mapping: `derive_terminal_status` returns
"completed" only when every task deliberately finished (done/env_done),
"paused" when any task paused, and "halted" for loop/max_steps/empty.
"""

from __future__ import annotations

from mantis_agent.task_loop import derive_terminal_status


def _td(reason: str) -> dict:
    return {"task_id": "t", "termination_reason": reason}


def test_all_done_is_completed():
    assert derive_terminal_status([_td("done")]) == "completed"
    assert derive_terminal_status([_td("done"), _td("env_done")]) == "completed"


def test_loop_is_halted():
    assert derive_terminal_status([_td("loop")]) == "halted"


def test_max_steps_is_halted():
    assert derive_terminal_status([_td("max_steps")]) == "halted"


def test_mixed_done_and_loop_is_halted():
    # One task looped → the whole run is not a clean completion.
    assert derive_terminal_status([_td("done"), _td("loop")]) == "halted"


def test_paused_wins():
    assert derive_terminal_status([_td("paused")]) == "paused"
    assert derive_terminal_status([_td("done"), _td("paused")]) == "paused"


def test_empty_is_halted():
    # No task ran → not a success.
    assert derive_terminal_status([]) == "halted"


def test_missing_reason_is_halted():
    assert derive_terminal_status([{"task_id": "t"}]) == "halted"
    assert derive_terminal_status([_td("")]) == "halted"
