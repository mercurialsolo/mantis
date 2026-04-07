"""Run OSWorld-Verified evaluation via HUD platform.

Uses HUD's `hud eval` CLI with the `openai_compatible` agent type,
pointing at our Gemma4 llama-server. HUD handles:
- VM provisioning (Ubuntu desktop with apps)
- Screenshot capture and delivery
- Action execution (click, type, scroll)
- Post-task evaluation and scoring

We just provide the brain (Gemma4 via llama-server).

Usage:
    # Start llama-server first:
    llama-server -hf ggml-org/gemma-4-E4B-it-GGUF --port 8080 -ngl 99 -c 4096

    # Run single task (debug):
    python -m mantis_agent.hud_runner

    # Run full benchmark:
    python -m mantis_agent.hud_runner --full

    # Run remotely on HUD (no local compute needed):
    python -m mantis_agent.hud_runner --full --remote
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def write_eval_config(
    base_url: str = "http://localhost:8080/v1",
    model: str = "gemma-4",
    max_steps: int = 15,
    max_concurrent: int = 5,
):
    """Write .hud_eval.toml for OSWorld evaluation."""
    config = f"""\
[eval]
source = "OSWorld-Verified"
agent = "openai_compatible"
max_concurrent = {max_concurrent}
max_steps = {max_steps}
auto_respond = true
quiet = true

[openai_compatible]
model = "{model}"
base_url = "{base_url}"
"""
    with open(".hud_eval.toml", "w") as f:
        f.write(config)
    logger.info("Wrote .hud_eval.toml")


def run_eval(
    full: bool = False,
    remote: bool = False,
    base_url: str = "http://localhost:8080/v1",
    model: str = "gemma-4",
    max_steps: int = 15,
    max_concurrent: int = 5,
    task_ids: str | None = None,
    verbose: bool = False,
    yes: bool = True,
):
    """Run OSWorld-Verified evaluation via HUD CLI."""

    cmd = [
        "hud", "eval", "OSWorld-Verified", "openai_compatible",
        "--config", f"base_url={base_url}",
        "--model", model,
        "--max-steps", str(max_steps),
        "--max-concurrent", str(max_concurrent),
    ]

    if full:
        cmd.append("--full")
    if remote:
        cmd.append("--remote")
    if task_ids:
        cmd.extend(["--task-ids", task_ids])
    if verbose:
        cmd.append("-vv")
    if yes:
        cmd.append("-y")

    logger.info(f"Running: {' '.join(cmd)}")
    print(f"\n{'='*60}")
    print(f"  OSWorld-Verified × Gemma4 Evaluation")
    print(f"  Model: {model}")
    print(f"  Endpoint: {base_url}")
    print(f"  Mode: {'full (367 tasks)' if full else 'single task (debug)'}")
    print(f"  Execution: {'remote (HUD infra)' if remote else 'local'}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, timeout=7200 if full else 600)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run OSWorld-Verified evaluation via HUD"
    )
    parser.add_argument("--base-url", default="http://localhost:8080/v1",
                        help="llama-server endpoint")
    parser.add_argument("--model", default="gemma-4",
                        help="Model name for the endpoint")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--full", action="store_true",
                        help="Run full 367-task benchmark")
    parser.add_argument("--remote", action="store_true",
                        help="Run on HUD infrastructure (no local compute)")
    parser.add_argument("--task-ids", type=str, default=None,
                        help="Comma-separated task IDs to run")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    sys.exit(run_eval(
        full=args.full,
        remote=args.remote,
        base_url=args.base_url,
        model=args.model,
        max_steps=args.max_steps,
        max_concurrent=args.max_concurrent,
        task_ids=args.task_ids,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()
