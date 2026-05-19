from __future__ import annotations

from types import SimpleNamespace

import pytest

from mantis_agent import task_loop
from mantis_agent.gym import runner as runner_module
from mantis_agent.gym import workflow_runner


def test_loop_task_passes_extended_loop_config(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakeWorkflowRunner:
        def __init__(self, **kwargs):
            captured["loop_config"] = kwargs["loop_config"]
            captured["fallback_brain"] = kwargs["fallback_brain"]
            captured["fallback_label"] = kwargs["fallback_label"]
            captured["fallback_micro_retries"] = kwargs["fallback_micro_retries"]
            captured["fallback_micro_max_steps"] = kwargs["fallback_micro_max_steps"]

        def run_loop(self):
            return []

    monkeypatch.setattr(workflow_runner, "WorkflowRunner", FakeWorkflowRunner)

    config = task_loop.TaskLoopConfig(
        run_id="run",
        session_name="session",
        model_name="model",
        results_prefix="test",
        brain=object(),
        fallback_brain="claude",
        fallback_label="claude",
        fallback_micro_retries=3,
        fallback_micro_max_steps=7,
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
    assert captured["fallback_brain"] == "claude"
    assert captured["fallback_label"] == "claude"
    assert captured["fallback_micro_retries"] == 3
    assert captured["fallback_micro_max_steps"] == 7


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


# ── #350: per-task CostMeter snapshot/delta ─────────────────────────────


def test_standard_task_attaches_per_task_costs(monkeypatch, tmp_path) -> None:
    """Each standard task's detail dict carries a ``costs`` entry
    summed across primary + fallback + resume runners."""
    from mantis_agent.gym.cost_meter import CostMeter

    class FakeEnv:
        current_url = "https://example.test/"

    class FakeGymRunner:
        # Each instance starts with its own fresh meter, then we
        # pre-seed it so the run() call returns a known cost shape.
        def __init__(self, *, brain, **_kwargs):
            self.brain = brain
            self.cost_meter = CostMeter()
            # 4 brain steps for primary, 2 for any non-primary runner.
            self._seed_steps = 4 if brain == "holo3" else 2

        def run(self, **_kwargs):
            self.cost_meter.costs["gpu_steps"] += self._seed_steps
            self.cost_meter.costs["gpu_seconds"] += float(self._seed_steps)
            return SimpleNamespace(
                success=True,
                total_steps=self._seed_steps,
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
        results_dir=str(tmp_path),
    )

    scores, details = task_loop.run_task_loop(
        [
            {"task_id": "submit_1", "intent": "click submit"},
            {"task_id": "submit_2", "intent": "click submit"},
        ],
        config,
    )

    assert scores == [1.0, 1.0]
    # Each task gets its own cost dict — meters don't leak across tasks.
    for d in details:
        assert "costs" in d
        assert d["costs"]["gpu_steps"] == 4
        assert d["costs"]["gpu_seconds"] == pytest.approx(4.0)
        assert d["costs"]["total"] == pytest.approx(d["costs"]["gpu"])
        assert d["costs"]["gpu"] > 0


def test_standard_task_costs_sum_fallback_and_resume_runners(
    monkeypatch, tmp_path
) -> None:
    """Primary failure → micro-fallback + resume both bill into the
    same per-task costs entry (no cost gets dropped on the floor)."""
    from mantis_agent.gym.cost_meter import CostMeter

    class FakeEnv:
        current_url = "https://example.test/"

    class FakeGymRunner:
        def __init__(self, *, brain, **_kwargs):
            self.brain = brain
            self.cost_meter = CostMeter()
            self._seed = {"holo3": 3, "claude": 5}.get(brain, 1)

        def run(self, **kwargs):
            self.cost_meter.costs["gpu_steps"] += self._seed
            self.cost_meter.costs["gpu_seconds"] += float(self._seed)
            success = self.brain == "claude" or "resume_after" in kwargs["task_id"]
            return SimpleNamespace(
                success=success,
                total_steps=self._seed,
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
        results_dir=str(tmp_path),
    )

    _scores, details = task_loop.run_task_loop(
        [{"task_id": "fill_filters", "intent": "fill filters"}],
        config,
    )

    # primary holo3 (3) + claude micro-fallback (5) + resume holo3 (3) = 11
    assert details[0]["costs"]["gpu_steps"] == 11
    assert details[0]["costs"]["gpu_seconds"] == pytest.approx(11.0)


def test_failed_task_still_carries_costs(monkeypatch, tmp_path) -> None:
    """A primary-only failure must still attribute the spend it
    incurred (failures cost real money too)."""
    from mantis_agent.gym.cost_meter import CostMeter

    class FakeEnv:
        current_url = "https://example.test/"

    class FakeGymRunner:
        def __init__(self, **_kwargs):
            self.cost_meter = CostMeter()

        def run(self, **_kwargs):
            self.cost_meter.costs["gpu_steps"] += 2
            self.cost_meter.costs["gpu_seconds"] += 6.0
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

    _scores, details = task_loop.run_task_loop(
        [{"task_id": "submit", "intent": "click submit"}],
        config,
    )

    assert details[0]["success"] is False
    assert details[0]["costs"]["gpu_steps"] == 2
    assert details[0]["costs"]["gpu_seconds"] == pytest.approx(6.0)
