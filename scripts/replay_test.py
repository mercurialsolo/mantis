#!/usr/bin/env python3
"""Replay cached screenshots to test prompts without a browser.

Usage:
    # Download screenshots from Modal volume:
    python replay_test.py download [session_name]

    # Test a prompt against cached screenshots:
    python replay_test.py test --prompt "Click the first listing" --dir outputs/screenshots/run_001

    # Run full extraction test with replay env:
    python replay_test.py run --dir outputs/screenshots/run_001 --max-steps 10

    # List available screenshot sets:
    python replay_test.py list
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


def cmd_list(args):
    """List screenshot sets on Modal volume."""
    result = subprocess.run(
        [".venv/bin/modal", "volume", "ls", "osworld-data", "screenshots/"],
        capture_output=True, text=True,
    )
    print(result.stdout or "No screenshots found. Run an extraction first.")


def cmd_download(args):
    """Download screenshots from Modal volume."""
    session = args.session or ""
    local_dir = f"outputs/screenshots/{session}" if session else "outputs/screenshots"
    os.makedirs(local_dir, exist_ok=True)

    if session:
        remote = f"screenshots/{session}/"
    else:
        # Download latest
        result = subprocess.run(
            [".venv/bin/modal", "volume", "ls", "osworld-data", "screenshots/"],
            capture_output=True, text=True,
        )
        dirs = [line.strip().rstrip("/") for line in result.stdout.split("\n") if line.strip()]
        if not dirs:
            print("No screenshots on volume.")
            return
        session = dirs[0]
        remote = f"screenshots/{session}/"
        local_dir = f"outputs/screenshots/{session}"
        os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading {remote} → {local_dir}/")
    subprocess.run(
        [".venv/bin/modal", "volume", "get", "osworld-data", remote, local_dir + "/"],
    )
    count = len([f for f in os.listdir(local_dir) if f.endswith(".png")])
    print(f"Downloaded {count} screenshots to {local_dir}/")


def cmd_test(args):
    """Test a prompt against a single screenshot or directory."""
    from mantis_agent.brain_llamacpp import LlamaCppBrain
    from mantis_agent.gym.replay_env import test_prompt, test_prompt_batch

    brain = LlamaCppBrain(
        base_url=args.brain_url,
        model="gemma4-cua",
        max_tokens=512,
        temperature=0.0,
        use_tool_calling=True,
    )

    if os.path.isfile(args.dir):
        # Single screenshot
        result = test_prompt(brain, args.dir, args.prompt)
        print(f"Action: {result['action']}")
        print(f"Type:   {result['action_type']}")
        print(f"Params: {result['params']}")
        print(f"Think:  {result['thinking'][:200]}")
    else:
        # Directory of screenshots
        results = test_prompt_batch(brain, args.dir, args.prompt, max_screenshots=args.max)
        for r in results:
            print(f"  {r['file']:12s} → {r['action_type']:12s} {r['params']}")


def cmd_run(args):
    """Run full extraction with replay env."""
    from mantis_agent.brain_llamacpp import LlamaCppBrain
    from mantis_agent.gym.replay_env import ReplayGymEnv
    from mantis_agent.gym.runner import GymRunner

    brain = LlamaCppBrain(
        base_url=args.brain_url,
        model="gemma4-cua",
        max_tokens=512,
        temperature=0.0,
        use_tool_calling=True,
    )

    env = ReplayGymEnv(args.dir, loop=True)
    runner = GymRunner(brain=brain, env=env, max_steps=args.max_steps)

    prompt = args.prompt or open("tasks/boattrader/dynamic_production.json").read()
    # Use the s2_extract intent
    task_suite = json.loads(prompt) if prompt.startswith("{") else None
    if task_suite:
        for t in task_suite.get("tasks", []):
            if "extract" in t.get("task_id", ""):
                prompt = t["intent"].replace("{ORDINAL}", "first")
                break

    result = runner.run(task=prompt, task_id="replay_test")
    print(f"\nResult: {result.termination_reason} ({result.total_steps} steps)")
    for step in result.trajectory:
        print(f"  Step {step.step}: {step.action} → {step.feedback[:60] if step.feedback else ''}")


def main():
    parser = argparse.ArgumentParser(description="Replay cached screenshots for CUA testing")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List screenshot sets on Modal volume")

    dl = sub.add_parser("download", help="Download screenshots from Modal")
    dl.add_argument("session", nargs="?", help="Session name (default: latest)")

    test = sub.add_parser("test", help="Test prompt against screenshots")
    test.add_argument("--dir", required=True, help="Screenshot file or directory")
    test.add_argument("--prompt", required=True, help="Prompt to test")
    test.add_argument("--brain-url", default="http://localhost:8080/v1", help="Brain server URL")
    test.add_argument("--max", type=int, default=10, help="Max screenshots to test")

    run = sub.add_parser("run", help="Run full extraction with replay env")
    run.add_argument("--dir", required=True, help="Screenshot directory")
    run.add_argument("--prompt", default="", help="Custom prompt (default: from task file)")
    run.add_argument("--brain-url", default="http://localhost:8080/v1", help="Brain server URL")
    run.add_argument("--max-steps", type=int, default=20, help="Max steps")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return

    {"list": cmd_list, "download": cmd_download, "test": cmd_test, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
