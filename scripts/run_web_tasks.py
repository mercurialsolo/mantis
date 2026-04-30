#!/usr/bin/env python3
"""Run Mantis CUA against live web application tasks.

Drives a Playwright browser against any URL — CRMs, SaaS tools, internal
apps. Supports session persistence so login tasks save cookies that
subsequent tasks reuse automatically.

Usage:
    # Run all tasks in a task file (login first, then authenticated tasks)
    python run_web_tasks.py --tasks tasks/crm/sample.json

    # Run a single task by ID
    python run_web_tasks.py --tasks tasks/crm/sample.json --task-id update_lead_industry

    # Run with visible browser (non-headless)
    python run_web_tasks.py --tasks tasks/crm/sample.json --headed

    # Override backend
    python run_web_tasks.py --tasks tasks/crm/sample.json --backend llamacpp --model-url http://localhost:8080/v1
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
logger = logging.getLogger("run_web_tasks")


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


def run_task(brain, env, task_config: dict, session_name: str, args) -> dict:
    """Run a single web task."""
    from mantis_agent.gym.runner import GymRunner

    task_id = task_config["task_id"]
    intent = task_config["intent"]

    # Restore session if task requires authentication
    if task_config.get("require_session") and env.has_session(session_name):
        env.load_session(session_name)
        logger.info(f"Restored session '{session_name}' for task {task_id}")

    runner = GymRunner(
        brain=brain,
        env=env,
        max_steps=args.max_steps,
        frames_per_inference=args.frames_per_inference,
    )

    result = runner.run(
        task=intent,
        task_id=task_id,
        seed=args.seed,
    )

    # Save session if this task is a login task
    if task_config.get("save_session") and result.success:
        saved_path = env.save_session(session_name)
        logger.info(f"Session saved after login: {saved_path}")
    elif task_config.get("save_session"):
        # Save session even on non-explicit-success — the brain might not
        # signal done(success=True) but the login may have actually worked.
        # Check if we're past the login page by looking at the URL.
        current_url = env.current_url
        if current_url and "login" not in current_url.lower():
            saved_path = env.save_session(session_name)
            logger.info(f"Session saved (URL suggests login succeeded): {saved_path}")

    return {
        "task_id": task_id,
        "intent": intent,
        "success": result.success,
        "reward": result.total_reward,
        "steps": result.total_steps,
        "time": result.total_time,
        "termination_reason": result.termination_reason,
        "final_url": env.current_url,
        "trajectory": [
            {
                "step": s.step,
                "action": str(s.action),
                "thinking": s.thinking[:500] if s.thinking else "",
                "inference_time": s.inference_time,
            }
            for s in result.trajectory
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Run Mantis CUA against live web tasks")

    parser.add_argument("--tasks", required=True, help="Path to task config JSON file")
    parser.add_argument("--task-id", help="Run a single task by ID (skip others)")
    parser.add_argument("--headed", action="store_true", help="Run browser in visible mode")
    parser.add_argument("--browser", choices=["chromium", "firefox", "webkit"], default="chromium")
    parser.add_argument("--viewport", default="1280x720", help="Browser viewport WxH")
    parser.add_argument("--session-dir", default=".sessions", help="Directory for session state files")
    parser.add_argument("--settle-time", type=float, default=1.5, help="Seconds to wait after each action")

    # Brain
    parser.add_argument("--backend", choices=["llamacpp", "transformers"], default="llamacpp")
    parser.add_argument("--model", default="gemma-4", help="Model name/path")
    parser.add_argument("--model-url", default="http://localhost:8080/v1", help="llama-server URL")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--quantize-4bit", action="store_true")

    # Runner
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--frames-per-inference", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)

    # Output
    parser.add_argument("--result-dir", default="results/web_tasks")

    args = parser.parse_args()

    # Load task file
    task_file = Path(args.tasks)
    with open(task_file) as f:
        task_suite = json.load(f)

    session_name = task_suite.get("session_name", task_file.stem)
    base_url = task_suite.get("base_url", "")
    tasks = task_suite.get("tasks", [])

    # Filter to single task if specified
    if args.task_id:
        tasks = [t for t in tasks if t["task_id"] == args.task_id]
        if not tasks:
            logger.error(f"Task '{args.task_id}' not found in {task_file}")
            return

    # Parse viewport
    vw, vh = args.viewport.split("x")
    viewport = (int(vw), int(vh))

    # Create brain
    logger.info(f"Loading brain: {args.backend} ({args.model})")
    brain = create_brain(args)

    # Create environment
    from mantis_agent.gym.playwright_env import PlaywrightGymEnv
    env = PlaywrightGymEnv(
        start_url=base_url,
        viewport=viewport,
        headless=not args.headed,
        browser_type=args.browser,
        session_dir=args.session_dir,
        settle_time=args.settle_time,
    )

    # Ensure login task runs first if we don't have a session yet
    login_tasks = [t for t in tasks if t.get("save_session")]
    other_tasks = [t for t in tasks if not t.get("save_session")]

    if not env.has_session(session_name) and login_tasks:
        ordered_tasks = login_tasks + other_tasks
    else:
        ordered_tasks = tasks

    # Run tasks
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    successes = 0

    for i, task_config in enumerate(ordered_tasks):
        task_id = task_config["task_id"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Task {i+1}/{len(ordered_tasks)}: {task_id}")
        logger.info(f"Intent: {task_config['intent'][:120]}")
        logger.info(f"{'='*60}")

        try:
            result = run_task(brain, env, task_config, session_name, args)
            all_results.append(result)
            if result["success"]:
                successes += 1
            status = "PASS" if result["success"] else "FAIL"
            logger.info(
                f"Result: {status} "
                f"(steps={result['steps']}, time={result['time']:.1f}s, "
                f"reason={result['termination_reason']})"
            )
        except Exception as e:
            logger.error(f"Task {task_id} errored: {e}", exc_info=True)
            all_results.append({
                "task_id": task_id,
                "success": False,
                "error": str(e),
            })

    env.close()

    # Write results
    summary = {
        "task_suite": str(task_file),
        "session_name": session_name,
        "total_tasks": len(ordered_tasks),
        "successes": successes,
        "success_rate": successes / len(ordered_tasks) if ordered_tasks else 0,
        "backend": args.backend,
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results,
    }

    output_file = result_dir / f"{session_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY: {successes}/{len(ordered_tasks)} passed ({summary['success_rate']:.1%})")
    logger.info(f"Results: {output_file}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
