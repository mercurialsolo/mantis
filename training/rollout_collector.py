"""Sample rollouts from a Mantis brain + GymEnvironment.

A rollout is one full episode: the runner drives the brain to either DONE
or max_steps, and we record the trajectory + screenshots + reward
breakdown. Output is JSONL, one row per rollout, with screenshots saved
to a sibling directory keyed by rollout_id.

Schema is compatible with `training/convert_claude_trajectories.py` so
filtered-successful rollouts can feed straight back into SFT.

Typical usage (sequential, for ad-hoc collection):

    collector = RolloutCollector(
        runner_factory=lambda: GymRunner(brain=Holo3Brain(), env=PlaywrightGymEnv()),
        reward_fn=BoatTraderReward(),
        output_dir=Path("training/data/rollouts/bt_run1"),
    )
    for task in tasks:
        collector.collect(task, n=8)

Modal-based parallel collection is a separate concern — wrap this class
in a `modal.App.function` that takes a task slice per worker.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from PIL import Image as PILImage  # noqa: F401
    from mantis_agent.gym.runner import GymRunner, RunResult
    from mantis_agent.rewards.base import RewardFn


@dataclass
class Task:
    """A single task to roll out.

    Mirrors what `GymRunner.run()` consumes — the collector just relays.
    """

    task: str
    task_id: str = "default"
    seed: int | None = None
    plan: Any = None
    plan_steps: str | None = None
    plan_inputs: dict[str, str] | None = None
    start_url: str | None = None
    ground_truth: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutRecord:
    """Serialised rollout record. One row per rollout in the JSONL output."""

    rollout_id: str
    task: str
    task_id: str
    seed: int | None
    success: bool
    termination_reason: str
    total_reward: float
    terminal_reward: float
    reward_components: dict[str, float]
    total_steps: int
    total_time: float
    ground_truth: dict[str, Any] | None
    metadata: dict[str, Any]
    trajectory: list[dict[str, Any]]


def _action_to_dict(action: Any) -> dict[str, Any]:
    return {
        "type": action.action_type.value,
        "params": dict(action.params),
        "reasoning": action.reasoning or "",
    }


def _save_screenshot(img: Any, path: Path) -> None:
    """Persist a PIL image as PNG. No-op if img is None (env-less tests)."""
    if img is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


class RolloutCollector:
    """Drives a `GymRunner` and persists every rollout to disk.

    Args:
        runner_factory: zero-arg callable returning a fresh `GymRunner`.
            A new runner per rollout keeps the browser session clean and
            avoids leaked state across episodes.
        reward_fn: optional `RewardFn`. When provided, every step gets a
            per-step reward and the episode gets a terminal reward. Only
            successful rollouts (reward >= success_threshold) are useful for
            rejection-sampled SFT.
        output_dir: directory to write rollouts.jsonl + screenshots/.
        save_screenshots: if False, only the trajectory metadata is saved
            (much smaller files; loses image data so SFT can't reuse).
        success_threshold: terminal reward threshold for `success_only` mode.
    """

    def __init__(
        self,
        runner_factory: Callable[[], "GymRunner"],
        reward_fn: "RewardFn | None" = None,
        output_dir: Path | str = "training/data/rollouts",
        save_screenshots: bool = True,
        success_threshold: float = 1.0,
    ):
        self.runner_factory = runner_factory
        self.reward_fn = reward_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_screenshots = save_screenshots
        self.success_threshold = success_threshold
        self.jsonl_path = self.output_dir / "rollouts.jsonl"
        self.screenshots_root = self.output_dir / "screenshots"

    def collect(
        self,
        task: Task,
        n: int = 1,
        seed_offset: int = 0,
    ) -> list[RolloutRecord]:
        """Run the task `n` times and persist each rollout.

        Args:
            task: the task to roll out.
            n: number of rollouts (typically 4–16 for GRPO grouping).
            seed_offset: added to `i` when generating per-rollout seeds.
                Lets caller distinguish runs across resumes.

        Returns:
            List of recorded rollouts (also appended to rollouts.jsonl).
        """
        records: list[RolloutRecord] = []
        for i in range(n):
            seed = (task.seed if task.seed is not None else 0) + seed_offset + i
            rollout_id = f"{task.task_id}_seed{seed}_run{i}"
            logger.info("rollout %s (n=%d/%d)", rollout_id, i + 1, n)

            try:
                record = self._run_one(task, rollout_id, seed)
            except Exception as exc:
                logger.exception("rollout %s crashed: %s", rollout_id, exc)
                continue

            self._append_jsonl(record)
            records.append(record)
        return records

    def collect_many(
        self,
        tasks: Iterable[Task],
        n_per_task: int = 1,
    ) -> list[RolloutRecord]:
        """Collect rollouts for a sequence of tasks. Sequential."""
        all_records: list[RolloutRecord] = []
        for task in tasks:
            all_records.extend(self.collect(task, n=n_per_task))
        return all_records

    # ── Internals ──────────────────────────────────────────────────────

    def _run_one(self, task: Task, rollout_id: str, seed: int) -> RolloutRecord:
        runner = self.runner_factory()
        screenshot_dir = self.screenshots_root / rollout_id
        t0 = time.time()
        result: RunResult = runner.run(
            task=task.task,
            task_id=task.task_id,
            seed=seed,
            plan=task.plan,
            plan_steps=task.plan_steps,
            plan_inputs=task.plan_inputs,
            start_url=task.start_url,
            reward_fn=self.reward_fn,
            ground_truth=task.ground_truth,
            capture_dir=screenshot_dir if self.save_screenshots else None,
        )
        elapsed = time.time() - t0

        traj_serialised: list[dict[str, Any]] = []
        for tstep in result.trajectory:
            entry: dict[str, Any] = {
                "step": tstep.step,
                "action": _action_to_dict(tstep.action),
                "thinking": tstep.thinking,
                "reward": tstep.reward,
                "reward_components": dict(tstep.reward_components),
                "feedback": tstep.feedback,
                "inference_time": tstep.inference_time,
                "done": tstep.done,
            }
            if self.save_screenshots:
                # The (screenshot, action) SFT pair is the frame the model
                # SAW (frame step-1) and the action it then took. Frame 0
                # is the post-reset observation that drives action at step 1.
                obs_idx = max(tstep.step - 1, 0)
                entry["screenshot_path"] = str(
                    (screenshot_dir / f"{obs_idx:04d}.png").relative_to(self.output_dir)
                )
            traj_serialised.append(entry)

        record = RolloutRecord(
            rollout_id=rollout_id,
            task=task.task,
            task_id=task.task_id,
            seed=seed,
            success=result.success,
            termination_reason=result.termination_reason,
            total_reward=result.total_reward,
            terminal_reward=result.terminal_reward,
            reward_components=dict(result.reward_components),
            total_steps=result.total_steps,
            total_time=elapsed,
            ground_truth=task.ground_truth,
            metadata=dict(task.metadata),
            trajectory=traj_serialised,
        )
        return record

    def _append_jsonl(self, record: RolloutRecord) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), default=str) + "\n")


def filter_rollouts(
    jsonl_path: Path | str,
    output_path: Path | str,
    min_terminal_reward: float = 1.0,
    require_success: bool = True,
) -> int:
    """Filter rollouts.jsonl → only keep ones above the reward threshold.

    Returns the number of rollouts kept. Output JSONL has the same shape
    as input (downstream tooling decides how to reshape into SFT format).
    """
    jsonl_path = Path(jsonl_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with jsonl_path.open() as src, output_path.open("w") as dst:
        for line in src:
            row = json.loads(line)
            if require_success and not row.get("success"):
                continue
            if row.get("terminal_reward", 0.0) < min_terminal_reward:
                continue
            dst.write(line)
            kept += 1
    return kept
