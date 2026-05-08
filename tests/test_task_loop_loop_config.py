from __future__ import annotations

from types import SimpleNamespace

from mantis_agent import task_loop
from mantis_agent.gym import runner as runner_module
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


def test_standard_task_uses_fallback_brain_after_primary_failure(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeEnv:
        current_url = "https://example.test/"

    class FakeGymRunner:
        def __init__(self, *, brain, **_kwargs):
            self.brain = brain
            self.max_steps = _kwargs["max_steps"]

        def run(self, **kwargs):
            calls.append(
                {
                    "brain": self.brain,
                    "task": kwargs["task"],
                    "start_url": kwargs.get("start_url", ""),
                    "max_steps": self.max_steps,
                }
            )
            return SimpleNamespace(
                success=self.brain == "claude",
                total_steps=3,
                termination_reason="done" if self.brain == "claude" else "loop",
                trajectory=[],
            )

    monkeypatch.setattr(runner_module, "GymRunner", FakeGymRunner)

    config = task_loop.TaskLoopConfig(
        run_id="run",
        session_name="session",
        model_name="model",
        results_prefix="test",
        brain="holo3",
        fallback_brain="claude",
        fallback_label="claude",
        env=FakeEnv(),
        max_steps=12,
        results_dir=str(tmp_path),
    )
    tasks = [
        {
            "task_id": "fill_filters",
            "intent": "fill filters",
            "fallback_intent": "click the visible search button only",
            "fallback_max_steps": 5,
        }
    ]

    scores, details = task_loop.run_task_loop(tasks, config)

    assert calls == [
        {
            "brain": "holo3",
            "task": "fill filters",
            "start_url": "",
            "max_steps": 12,
        },
        {
            "brain": "claude",
            "task": "click the visible search button only",
            "start_url": "",
            "max_steps": 5,
        },
    ]
    assert scores == [1.0]
    assert details[0]["success"] is True
