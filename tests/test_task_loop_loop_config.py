from __future__ import annotations

from mantis_agent import task_loop
from mantis_agent.gym import workflow_runner


def test_loop_task_passes_extended_loop_config(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakeWorkflowRunner:
        def __init__(self, **kwargs):
            captured["loop_config"] = kwargs["loop_config"]

        def run_loop(self):
            return []

    monkeypatch.setattr(workflow_runner, "WorkflowRunner", FakeWorkflowRunner)

    config = task_loop.TaskLoopConfig(
        run_id="run",
        session_name="session",
        model_name="model",
        results_prefix="test",
        brain=object(),
        env=object(),
        max_steps=90,
        results_dir=str(tmp_path),
    )
    tasks = [
        {
            "task_id": "extract",
            "intent": "extract listings",
            "loop": {
                "pagination_intent": "click next",
                "max_iterations": 20,
                "max_pages": 3,
                "max_steps_per_iteration": 45,
                "max_retries_per_iteration": 4,
                "max_steps_pagination": 35,
            },
        }
    ]

    task_loop.run_task_loop(tasks, config)

    loop_config = captured["loop_config"]
    assert loop_config.max_iterations == 20
    assert loop_config.max_pages == 3
    assert loop_config.max_steps_per_iteration == 45
    assert loop_config.max_retries_per_iteration == 4
    assert loop_config.max_steps_pagination == 35
