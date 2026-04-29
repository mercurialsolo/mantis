"""Tests for training/rollout_collector.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.runner import RunResult, TrajectoryStep
from mantis_agent.rewards import BoatTraderReward

from training.rollout_collector import (  # noqa: E402
    RolloutCollector,
    Task,
    filter_rollouts,
)


class StubRunner:
    """Minimal runner stand-in: writes fixture screenshots, returns canned RunResult."""

    def __init__(self, summary: str, success: bool = True, terminal_reward: float = 1.0):
        self.summary = summary
        self.success = success
        self.terminal_reward = terminal_reward

    def run(self, **kwargs: Any) -> RunResult:
        cd = kwargs.get("capture_dir")
        if cd:
            cd = Path(cd)
            cd.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                Image.new("RGB", (8, 8), "red").save(cd / f"{i:04d}.png")
        traj = [
            TrajectoryStep(
                step=1, action=Action(ActionType.CLICK, {"x": 1, "y": 2}),
                thinking="open listing", reward=0.1, done=False,
                inference_time=0.1, reward_components={"format": 0.1},
            ),
            TrajectoryStep(
                step=2,
                action=Action(ActionType.DONE,
                              {"success": self.success, "summary": self.summary}),
                thinking="extracted", reward=0.0, done=True, inference_time=0.1,
            ),
        ]
        return RunResult(
            task=kwargs["task"], task_id=kwargs["task_id"], success=self.success,
            total_reward=0.1 + self.terminal_reward, total_steps=2, total_time=0.5,
            trajectory=traj, termination_reason="done",
            terminal_reward=self.terminal_reward,
            reward_components={"gate_passed": self.terminal_reward}
                if self.terminal_reward > 0 else {"gate_failed": 0.0},
        )


GOOD_SUMMARY = "2018 Sea Ray 240 $42,500 https://www.boattrader.com/boat/x/"
BAD_SUMMARY = "couldn't find listing"


def test_collector_writes_jsonl_and_screenshots(tmp_path: Path) -> None:
    out = tmp_path / "rollouts"
    collector = RolloutCollector(
        runner_factory=lambda: StubRunner(GOOD_SUMMARY),
        reward_fn=BoatTraderReward(),
        output_dir=out,
    )
    records = collector.collect(Task(task="t", task_id="bt"), n=3)

    assert len(records) == 3
    rows = [json.loads(line) for line in (out / "rollouts.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert all(r["task_id"] == "bt" for r in rows)
    # Distinct seeds → distinct rollout_ids
    assert len({r["rollout_id"] for r in rows}) == 3

    # 3 screenshots × 3 rollouts = 9 PNGs
    pngs = list((out / "screenshots").rglob("*.png"))
    assert len(pngs) == 9


def test_collector_terminal_reward_recorded(tmp_path: Path) -> None:
    out = tmp_path / "rollouts"
    collector = RolloutCollector(
        runner_factory=lambda: StubRunner(GOOD_SUMMARY, terminal_reward=1.0),
        reward_fn=None,  # stub returns canned terminal reward regardless
        output_dir=out,
    )
    records = collector.collect(Task(task="t", task_id="bt"), n=1)
    row = json.loads((out / "rollouts.jsonl").read_text().splitlines()[0])
    assert row["terminal_reward"] == 1.0
    assert row["reward_components"] == {"gate_passed": 1.0}
    assert records[0].success


def test_collector_disables_screenshots(tmp_path: Path) -> None:
    out = tmp_path / "rollouts"
    collector = RolloutCollector(
        runner_factory=lambda: StubRunner(GOOD_SUMMARY),
        output_dir=out,
        save_screenshots=False,
    )
    collector.collect(Task(task="t", task_id="bt"), n=1)

    pngs = list((out / "screenshots").rglob("*.png")) if (out / "screenshots").exists() else []
    assert pngs == []
    row = json.loads((out / "rollouts.jsonl").read_text().splitlines()[0])
    assert "screenshot_path" not in row["trajectory"][0]


def test_filter_rollouts_keeps_only_above_threshold(tmp_path: Path) -> None:
    out = tmp_path / "rollouts"
    # Mix successful and failed rollouts.
    good = RolloutCollector(
        runner_factory=lambda: StubRunner(GOOD_SUMMARY, terminal_reward=1.0),
        output_dir=out,
    )
    bad = RolloutCollector(
        runner_factory=lambda: StubRunner(BAD_SUMMARY, success=False, terminal_reward=0.0),
        output_dir=out,
    )
    good.collect(Task(task="t", task_id="good"), n=2)
    bad.collect(Task(task="t", task_id="bad"), n=2)

    filtered = out / "filtered.jsonl"
    kept = filter_rollouts(out / "rollouts.jsonl", filtered, min_terminal_reward=1.0)
    assert kept == 2
    rows = [json.loads(line) for line in filtered.read_text().splitlines()]
    assert all(r["task_id"] == "good" for r in rows)


def test_collector_continues_on_per_rollout_failure(tmp_path: Path) -> None:
    """A crashing runner should not abort the rest of the batch."""
    out = tmp_path / "rollouts"
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        if calls["n"] == 2:
            class Boom:
                def run(self, **_: Any) -> RunResult:
                    raise RuntimeError("simulated crash")
            return Boom()
        return StubRunner(GOOD_SUMMARY)

    collector = RolloutCollector(runner_factory=factory, output_dir=out)
    records = collector.collect(Task(task="t", task_id="bt"), n=3)
    # 1st and 3rd succeed; 2nd crashes.
    assert len(records) == 2
