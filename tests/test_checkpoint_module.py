"""Regression tests for #115 — RunCheckpoint et al moved to gym/checkpoint.py.

Verifies that the new canonical module is the single source of truth AND
that the legacy micro_runner import path still re-exports every symbol so
external callers don't break.
"""

from __future__ import annotations

from pathlib import Path


def test_canonical_module_exports_all_persisted_types() -> None:
    from mantis_agent.gym import checkpoint as cp

    expected = {
        "StepResult",
        "RunCheckpoint",
        "REVERSE_ACTIONS",
        "PauseRequested",
        "PauseState",
        "RunnerResult",
    }
    assert expected.issubset(set(cp.__all__))
    for name in expected:
        assert hasattr(cp, name), name


def test_legacy_micro_runner_path_reexports_same_objects() -> None:
    """External callers importing from micro_runner must keep working."""
    from mantis_agent.gym import checkpoint as cp
    from mantis_agent.gym import micro_runner as mr

    for name in (
        "StepResult",
        "RunCheckpoint",
        "REVERSE_ACTIONS",
        "PauseRequested",
        "PauseState",
        "RunnerResult",
    ):
        assert getattr(mr, name) is getattr(cp, name), (
            f"{name} differs between canonical and legacy import paths"
        )


def test_run_checkpoint_round_trip(tmp_path: Path) -> None:
    """Save+load preserves every persisted field including new ones (#127)."""
    from mantis_agent.gym.checkpoint import RunCheckpoint

    cp = RunCheckpoint(
        run_key="abc",
        plan_signature="sig",
        session_name="s",
        step_index=7,
        page=2,
        seen_urls=["https://x.test/a"],
        prompt_versions={"holo3_system": "deadbeef"},
        costs={"gpu_seconds": 12.0, "claude_extract": 3},
    )
    target = tmp_path / "cp.json"
    cp.save(str(target))
    loaded = RunCheckpoint.load(str(target))
    assert loaded is not None
    assert loaded.run_key == "abc"
    assert loaded.step_index == 7
    assert loaded.seen_urls == ["https://x.test/a"]
    assert loaded.prompt_versions == {"holo3_system": "deadbeef"}
    assert loaded.costs == {"gpu_seconds": 12.0, "claude_extract": 3}


def test_step_result_excludes_observability_extras_from_dict() -> None:
    from mantis_agent.gym.checkpoint import StepResult

    sr = StepResult(
        step_index=1,
        intent="click",
        success=True,
        screenshot_png=b"PNGDATA",
    )
    serialized = sr.to_dict()
    assert "screenshot_png" not in serialized
    assert "last_action" not in serialized
    assert serialized["step_index"] == 1


def test_pause_requested_carries_prompt_and_extras() -> None:
    from mantis_agent.gym.checkpoint import PauseRequested

    exc = PauseRequested(prompt="approve?", reason="user_input", custom_field=42)
    assert exc.prompt == "approve?"
    assert exc.reason == "user_input"
    assert exc.extras == {"custom_field": 42}


def test_pause_state_round_trips_through_dict() -> None:
    from mantis_agent.gym.checkpoint import PauseState

    ps = PauseState(
        run_key="r",
        pending_tool="ask_user",
        pending_arguments={"prompt": "go?"},
        prompt="go?",
    )
    payload = ps.to_dict()
    restored = PauseState.from_dict(payload)
    assert restored == ps
