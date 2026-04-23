#!/usr/bin/env python3
"""Run Mantis CUA against gym-anything environments.

Usage:
    # Local gym-anything environment
    python run_gym_anything.py \
        --env-dir environments/vwa_classifieds \
        --task-id task_0 \
        --backend llamacpp \
        --model-url http://localhost:8080/v1

    # Remote gym-anything environment
    python run_gym_anything.py \
        --env-dir environments/vwa_classifieds \
        --task-id task_0 \
        --remote-url http://gym-master:5800 \
        --backend llamacpp

    # Run a range of tasks
    python run_gym_anything.py \
        --env-dir environments/vwa_classifieds \
        --task-range 0-10 \
        --backend llamacpp \
        --result-dir results/classifieds
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_gym_anything")


def parse_task_range(task_range: str) -> list[str]:
    """Parse a task range like '0-10' or '0,3,5' into task_id strings."""
    if "-" in task_range and "," not in task_range:
        start, end = task_range.split("-", 1)
        return [f"task_{i}" for i in range(int(start), int(end) + 1)]
    return [f"task_{t.strip()}" for t in task_range.split(",")]


def create_brain(args: argparse.Namespace):
    """Create the appropriate brain backend."""
    if args.backend == "llamacpp":
        from mantis_agent.brain_llamacpp import LlamaCppBrain
        brain = LlamaCppBrain(
            base_url=args.model_url,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        from mantis_agent.brain import Gemma4Brain
        brain = Gemma4Brain(
            model_name=args.model,
            enable_thinking=True,
            max_new_tokens=args.max_tokens,
            quantize_4bit=args.quantize_4bit,
        )
    brain.load()
    return brain


def load_task_configs(env_dir: str, task_ids: list[str]) -> dict[str, dict]:
    """Load task configurations from a gym-anything env directory.

    Looks for task.json in each task subfolder, or falls back to a
    tasks.json manifest in the env root.
    """
    env_path = Path(env_dir)
    configs = {}

    # Try per-task directories first (gym-anything native format)
    tasks_dir = env_path / "tasks"
    if tasks_dir.is_dir():
        for task_id in task_ids:
            task_json = tasks_dir / task_id / "task.json"
            if task_json.exists():
                with open(task_json) as f:
                    configs[task_id] = json.load(f)
            else:
                logger.warning(f"Task config not found: {task_json}")
        return configs

    # Fall back to manifest file
    manifest = env_path / "tasks.json"
    if manifest.exists():
        with open(manifest) as f:
            all_tasks = json.load(f)
        for task_id in task_ids:
            idx = task_id.replace("task_", "")
            for t in all_tasks:
                if str(t.get("task_id")) == idx:
                    configs[task_id] = t
                    break
        return configs

    logger.warning(f"No task configs found in {env_dir}")
    return {tid: {} for tid in task_ids}


def run_single_task(brain, env, task_id: str, task_config: dict, args) -> dict:
    """Run a single task and return the result dict."""
    from mantis_agent.gym.runner import GymRunner

    task_instruction = task_config.get("intent", task_config.get("task", f"Complete task {task_id}"))

    runner = GymRunner(
        brain=brain,
        env=env,
        max_steps=args.max_steps,
        frames_per_inference=args.frames_per_inference,
    )

    result = runner.run(
        task=task_instruction,
        task_id=task_id,
        seed=args.seed,
    )

    return {
        "task_id": task_id,
        "task": task_instruction,
        "success": result.success,
        "reward": result.total_reward,
        "steps": result.total_steps,
        "time": result.total_time,
        "termination_reason": result.termination_reason,
        "trajectory": [
            {
                "step": s.step,
                "action": str(s.action),
                "thinking": s.thinking[:500] if s.thinking else "",
                "reward": s.reward,
                "inference_time": s.inference_time,
            }
            for s in result.trajectory
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Mantis CUA against gym-anything environments"
    )

    # Environment
    parser.add_argument("--env-dir", required=True, help="Path to gym-anything environment directory")
    parser.add_argument("--task-id", help="Single task ID (e.g., task_0)")
    parser.add_argument("--task-range", help="Task range (e.g., 0-10 or 0,3,5)")
    parser.add_argument("--remote-url", help="gym-anything master server URL for remote execution")
    parser.add_argument("--runner-type", help="Runner type override (docker, qemu, avf)")
    parser.add_argument("--resolution", default="1920x1080", help="Screen resolution WxH")

    # Brain
    parser.add_argument("--backend", choices=["llamacpp", "transformers"], default="llamacpp")
    parser.add_argument("--model", default="gemma-4", help="Model name/path")
    parser.add_argument("--model-url", default="http://localhost:8080/v1", help="llama-server URL")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--quantize-4bit", action="store_true")

    # Runner
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--frames-per-inference", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)

    # Output
    parser.add_argument("--result-dir", default="results/gym_anything")

    args = parser.parse_args()

    # Parse resolution
    w, h = args.resolution.split("x")
    resolution = (int(w), int(h))

    # Determine tasks to run
    if args.task_id:
        task_ids = [args.task_id]
    elif args.task_range:
        task_ids = parse_task_range(args.task_range)
    else:
        parser.error("Either --task-id or --task-range is required")
        return

    # Load task configs
    task_configs = load_task_configs(args.env_dir, task_ids)

    # Create brain
    logger.info(f"Loading brain: {args.backend} ({args.model})")
    brain = create_brain(args)

    # Create environment adapter
    from mantis_agent.gym.gym_anything import GymAnythingAdapter
    env = GymAnythingAdapter(
        env_dir=args.env_dir,
        remote_url=args.remote_url,
        runner=args.runner_type,
        resolution=resolution,
    )

    # Run tasks
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    successes = 0

    for i, task_id in enumerate(task_ids):
        config = task_configs.get(task_id, {})
        intent = config.get("intent", f"Task {task_id}")
        logger.info(f"\n{'='*60}")
        logger.info(f"Task {i+1}/{len(task_ids)}: {task_id} — {intent}")
        logger.info(f"{'='*60}")

        try:
            result = run_single_task(brain, env, task_id, config, args)
            all_results.append(result)
            if result["success"]:
                successes += 1
            logger.info(
                f"Result: {'PASS' if result['success'] else 'FAIL'} "
                f"(reward={result['reward']}, steps={result['steps']}, "
                f"time={result['time']:.1f}s, reason={result['termination_reason']})"
            )
        except Exception as e:
            logger.error(f"Task {task_id} failed with error: {e}", exc_info=True)
            all_results.append({
                "task_id": task_id,
                "success": False,
                "error": str(e),
            })

    # Write results
    summary = {
        "total_tasks": len(task_ids),
        "successes": successes,
        "success_rate": successes / len(task_ids) if task_ids else 0,
        "env_dir": args.env_dir,
        "backend": args.backend,
        "model": args.model,
        "max_steps": args.max_steps,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results,
    }

    output_file = result_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY: {successes}/{len(task_ids)} passed ({summary['success_rate']:.1%})")
    logger.info(f"Results saved to {output_file}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
