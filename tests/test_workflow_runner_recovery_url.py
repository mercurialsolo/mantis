from __future__ import annotations

from types import SimpleNamespace

from mantis_agent.gym import workflow_runner
from mantis_agent.gym.workflow_runner import LoopConfig, WorkflowRunner


class _Env:
    current_url = "https://www.boattrader.com/boats/state-fl/city-miami/zip-33101/"


def test_workflow_runner_anchors_recovery_to_current_url_when_start_url_missing() -> None:
    runner = WorkflowRunner(
        brain=object(),
        env=_Env(),
        loop_config=LoopConfig(
            iteration_intent="Process the next listing.",
            pagination_intent="Click Next.",
        ),
        start_url="",
    )

    assert runner.start_url == "https://www.boattrader.com/boats/state-fl/city-miami/zip-33101/"


def test_loop_iteration_uses_claude_micro_fallback_then_resumes(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeGymRunner:
        def __init__(self, *, brain, max_steps, **_kwargs):
            self.brain = brain
            self.max_steps = max_steps

        def run(self, **kwargs):
            calls.append(
                {
                    "brain": self.brain,
                    "task": kwargs["task"],
                    "task_id": kwargs["task_id"],
                    "start_url": kwargs.get("start_url", ""),
                    "max_steps": self.max_steps,
                }
            )
            success = self.brain == "claude" or "resume_after_claude" in kwargs["task_id"]
            return SimpleNamespace(
                success=success,
                termination_reason="done" if success else "loop",
                trajectory=[],
                total_steps=1,
            )

    monkeypatch.setattr(workflow_runner, "GymRunner", FakeGymRunner)

    runner = WorkflowRunner(
        brain="holo3",
        fallback_brain="claude",
        fallback_label="claude",
        fallback_micro_retries=1,
        fallback_micro_max_steps=4,
        env=_Env(),
        loop_config=LoopConfig(
            iteration_intent="Process listing.",
            pagination_intent="Click Next.",
            max_retries_per_iteration=1,
            max_steps_per_iteration=10,
        ),
        start_url="",
    )

    result = runner._run_iteration("Process listing.", "iter_1")

    assert result.success is True
    assert calls[0] == {
        "brain": "holo3",
        "task": "Process listing.",
        "task_id": "iter_1",
        "start_url": "https://www.boattrader.com/boats/state-fl/city-miami/zip-33101/",
        "max_steps": 10,
    }
    assert calls[1]["brain"] == "claude"
    assert "larger extraction loop" in str(calls[1]["task"])
    assert calls[1]["task_id"] == "iter_1_claude_micro_fallback_1"
    assert calls[1]["start_url"] == ""
    assert calls[1]["max_steps"] == 4
    assert calls[2]["brain"] == "holo3"
    assert calls[2]["task_id"] == "iter_1_resume_after_claude_1"
    assert calls[2]["start_url"] == ""
