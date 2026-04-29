#!/usr/bin/env python3
"""Convert collected rollouts → Holo3 SFT chat format.

Reads `rollouts.jsonl` produced by `rollout_collector.py` and emits the
same chat-format JSONL that `train_holo3_distill.py` consumes — so a
filtered batch of successful rollouts can be appended to
`training/data/holo3_distill_train.jsonl` for the next SFT iteration.

This is the engine behind rejection-sampled self-distillation:

  1. Sample N rollouts per task (rollout_collector.py)
  2. Filter to high-reward ones (filter_rollouts in rollout_collector.py)
  3. Convert each kept rollout into per-step chat samples (this script)
  4. Append to existing distill training file
  5. Re-run train_holo3_distill.py

Each rollout step → one chat sample:
  system: HOLO3_SYSTEM (verbatim from convert_claude_trajectories.py)
  human:  "<image>\nTask: ...\nScreen size: 1280x720 pixels"
  gpt:    "<brief reasoning>\n<action_text>"

Usage:
    # Filter + convert in one shot:
    python training/convert_rollouts.py \\
        --rollouts training/data/rollouts/bt_run1/rollouts.jsonl \\
        --output  training/data/rollouts/bt_run1/distill.jsonl \\
        --min-reward 1.0 \\
        --screenshots-root training/data/rollouts/bt_run1

    # Append to existing distill set:
    cat training/data/rollouts/bt_run1/distill.jsonl \\
        >> training/data/holo3_distill_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Reuse the same prompt/action conversion as the Claude→Holo3 path so all
# distill samples share a single chat format. Sibling-script import: the
# training/ directory is added to sys.path on demand.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from convert_claude_trajectories import (  # type: ignore[import-not-found]  # noqa: E402
    HOLO3_SYSTEM,
    claude_action_to_holo3,
    thinking_to_brief,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def rollout_to_samples(
    rollout: dict[str, Any],
    screenshots_root: Path,
    screen_size: tuple[int, int] = (1280, 720),
) -> list[dict[str, Any]]:
    """Turn one rollout dict into a list of per-step SFT chat samples.

    Skips steps without a valid action (empty key_press, unparseable type)
    and steps whose screenshot file is missing on disk. Failed rollouts
    are *not* skipped here — caller should filter beforehand.
    """
    samples: list[dict[str, Any]] = []
    intent = rollout.get("task", "")
    if len(intent) > 500:
        intent = intent[:500] + "..."

    for step in rollout.get("trajectory", []):
        action = step.get("action") or {}
        action_text = claude_action_to_holo3(
            action.get("type", ""), action.get("params") or {},
        )
        if action_text is None:
            continue

        # Resolve screenshot — paths in rollouts.jsonl are recorded
        # relative to the collector's output_dir.
        rel = step.get("screenshot_path")
        if not rel:
            continue
        screenshot_path = (screenshots_root / rel).resolve()
        if not screenshot_path.exists():
            logger.debug("missing screenshot at step %s: %s", step.get("step"), screenshot_path)
            continue

        thinking = step.get("thinking") or action.get("reasoning") or ""
        brief = thinking_to_brief(thinking)
        assistant = f"{brief}\n{action_text}" if brief else action_text

        samples.append({
            "conversations": [
                {"from": "system", "value": HOLO3_SYSTEM},
                {
                    "from": "human",
                    "value": f"<image>\nTask: {intent}\nScreen size: {screen_size[0]}x{screen_size[1]} pixels",
                },
                {"from": "gpt", "value": assistant},
            ],
            "image": str(screenshot_path),
            "metadata": {
                "source": "rejection_sampled_rollout",
                "rollout_id": rollout.get("rollout_id", ""),
                "task_id": rollout.get("task_id", ""),
                "step": step.get("step"),
                "action_type": action.get("type"),
                "terminal_reward": rollout.get("terminal_reward", 0.0),
            },
        })
    return samples


def convert_file(
    rollouts_jsonl: Path,
    output_jsonl: Path,
    screenshots_root: Path,
    min_terminal_reward: float = 1.0,
    require_success: bool = True,
    screen_size: tuple[int, int] = (1280, 720),
) -> tuple[int, int]:
    """Convert + filter in one pass. Returns (rollouts_kept, samples_written)."""
    rollouts_kept = 0
    samples_written = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with rollouts_jsonl.open() as src, output_jsonl.open("w") as dst:
        for line in src:
            row = json.loads(line)
            if require_success and not row.get("success"):
                continue
            if row.get("terminal_reward", 0.0) < min_terminal_reward:
                continue
            rollouts_kept += 1
            for sample in rollout_to_samples(row, screenshots_root, screen_size):
                dst.write(json.dumps(sample) + "\n")
                samples_written += 1

    return rollouts_kept, samples_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True, help="rollouts.jsonl from RolloutCollector")
    parser.add_argument("--output", required=True, help="Output Holo3 chat-format JSONL")
    parser.add_argument(
        "--screenshots-root",
        default="",
        help="Directory the screenshot_paths are relative to. "
        "Defaults to the rollouts.jsonl's parent directory.",
    )
    parser.add_argument("--min-reward", type=float, default=1.0,
                        help="Minimum terminal_reward to keep a rollout")
    parser.add_argument("--include-failed", action="store_true",
                        help="Don't filter on success flag (useful for hard-negative mining)")
    parser.add_argument("--screen-w", type=int, default=1280)
    parser.add_argument("--screen-h", type=int, default=720)
    args = parser.parse_args()

    rollouts = Path(args.rollouts)
    if not rollouts.exists():
        logger.error("not found: %s", rollouts)
        sys.exit(1)
    screenshots_root = Path(args.screenshots_root) if args.screenshots_root else rollouts.parent
    output = Path(args.output)

    kept, samples = convert_file(
        rollouts_jsonl=rollouts,
        output_jsonl=output,
        screenshots_root=screenshots_root,
        min_terminal_reward=args.min_reward,
        require_success=not args.include_failed,
        screen_size=(args.screen_w, args.screen_h),
    )
    logger.info("kept %d rollouts → %d SFT samples in %s", kept, samples, output)


if __name__ == "__main__":
    main()
