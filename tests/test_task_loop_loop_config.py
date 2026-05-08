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
                    "task_id": kwargs["task_id"],
                    "start_url": kwargs.get("start_url", ""),
                    "max_steps": self.max_steps,
                }
            )
            success = self.brain == "claude" or "resume_after_claude" in kwargs["task_id"]
            return SimpleNamespace(
                success=success,
                total_steps=3,
                termination_reason="done" if success else "loop",
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
        fallback_micro_retries=1,
        fallback_micro_max_steps=5,
        env=FakeEnv(),
        max_steps=12,
        results_dir=str(tmp_path),
    )
    tasks = [{"task_id": "fill_filters", "intent": "fill filters"}]

    scores, details = task_loop.run_task_loop(tasks, config)

    assert calls[0] == {
        "brain": "holo3",
        "task": "fill filters",
        "task_id": "fill_filters",
        "start_url": "",
        "max_steps": 12,
    }
    assert calls[1]["brain"] == "claude"
    assert "single stuck browser-control micro-step" in str(calls[1]["task"])
    assert "fill filters" in str(calls[1]["task"])
    assert calls[1]["task_id"] == "fill_filters_claude_micro_fallback_1"
    assert calls[1]["start_url"] == ""
    assert calls[1]["max_steps"] == 5
    assert calls[2] == {
        "brain": "holo3",
        "task": "fill filters",
        "task_id": "fill_filters_resume_after_claude_1",
        "start_url": "",
        "max_steps": 12,
    }
    assert scores == [1.0]
    assert details[0]["success"] is True


def test_failed_standard_task_stops_loop_by_default(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    class FakeEnv:
        current_url = "https://example.test/"

    class FakeGymRunner:
        def __init__(self, *, brain, **_kwargs):
            self.brain = brain

        def run(self, **kwargs):
            calls.append(kwargs["task_id"])
            return SimpleNamespace(
                success=False,
                total_steps=2,
                termination_reason="loop",
                trajectory=[],
            )

    monkeypatch.setattr(runner_module, "GymRunner", FakeGymRunner)

    config = task_loop.TaskLoopConfig(
        run_id="run",
        session_name="session",
        model_name="model",
        results_prefix="test",
        brain="holo3",
        env=FakeEnv(),
        results_dir=str(tmp_path),
    )
    tasks = [
        {"task_id": "submit", "intent": "click submit"},
        {"task_id": "extract", "intent": "extract records"},
    ]

    scores, details = task_loop.run_task_loop(tasks, config)

    assert calls == ["submit"]
    assert scores == [0.0]
    assert [d["task_id"] for d in details] == ["submit"]


def test_standard_task_uses_small_default_step_cap(monkeypatch, tmp_path) -> None:
    captured: list[int] = []

    class FakeEnv:
        current_url = "https://example.test/"

    class FakeGymRunner:
        def __init__(self, *, max_steps, **_kwargs):
            captured.append(max_steps)

        def run(self, **_kwargs):
            return SimpleNamespace(
                success=True,
                total_steps=1,
                termination_reason="done",
                trajectory=[],
            )

    monkeypatch.setattr(runner_module, "GymRunner", FakeGymRunner)

    config = task_loop.TaskLoopConfig(
        run_id="run",
        session_name="session",
        model_name="model",
        results_prefix="test",
        brain="holo3",
        env=FakeEnv(),
        max_steps=90,
        standard_task_max_steps=15,
        results_dir=str(tmp_path),
    )

    task_loop.run_task_loop([{"task_id": "submit", "intent": "click submit"}], config)

    assert captured == [15]
