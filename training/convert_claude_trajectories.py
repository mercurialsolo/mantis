#!/usr/bin/env python3
"""Convert Claude CUA trajectories to Holo3 fine-tuning format.

Takes Claude's trajectory JSONL (action + thinking per step) paired with
screenshots from the Modal volume, and outputs training data in Qwen3.5
chat format for QLoRA fine-tuning of Holo3.

Claude and Holo3 both use 1280x720 viewport — coordinates are directly
transferable. But we teach the model VISUAL grounding (screenshot → action),
not memorized coordinates.

Input:
  - Trajectory JSONL from Modal volume (claude_trajectories_*.jsonl)
  - Screenshots from Modal volume (screenshots/bt_extract_*/NNNN.png)

Output:
  - JSONL with conversations in Qwen3.5 chat format:
    system: CUA instructions
    user: [screenshot] + task description
    assistant: reasoning + action call

Training format matches Holo3's brain_holo3.py SYSTEM_PROMPT so the
fine-tuned model outputs the same format the parser expects.

Usage:
    # Download data from Modal volume first:
    mkdir -p training/data/claude_distill
    modal volume get osworld-data results/claude_trajectories_*.jsonl training/data/claude_distill/
    modal volume get osworld-data screenshots/bt_extract_20260418_022726/ training/data/claude_distill/screenshots/

    # Convert:
    python training/convert_claude_trajectories.py \
        --trajectories training/data/claude_distill/claude_trajectories_bt_extract_20260418_022726.jsonl \
        --screenshots training/data/claude_distill/screenshots/ \
        --output training/data/holo3_distill_train.jsonl \
        --successful-only
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from io import BytesIO
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Holo3 system prompt — must match brain_holo3.py exactly
HOLO3_SYSTEM = """\
You are a computer use agent. You observe screenshots and perform actions to complete tasks.

RESPONSE FORMAT — Every response must follow this structure:
1. One brief sentence of reasoning (what you see and plan to do)
2. One action call

ACTIONS — use exactly one per response:
click(x=<int>, y=<int>)
type_text(text="<string>")
key_press(keys="<string>")
scroll(direction="down", amount=5)
wait(seconds=2)
done(success=true, summary="<detailed result>")
done(success=false, summary="<reason>")

RULES:
- Coordinates are absolute screen pixels. Aim for the CENTER of elements.
- Click input fields ONCE to focus, then type_text(). Use key_press(keys="tab") between fields.
- key_press(keys="alt+left") to go back in browser.
- scroll(direction="down", amount=5) to reveal content below.
- When extracting data, include ALL details in the done() summary: Year, Make, Model, Price, Phone (or "none"), Type, URL.
- NEVER repeat the same action 3 times. Try something different.
- NEVER just describe what you plan to do — you MUST output an action call.
- If stuck for 5+ actions, call done(success=false, summary="stuck: <what happened>").\
"""


def claude_action_to_holo3(action_type: str, params: dict) -> str | None:
    """Convert Claude's action format to Holo3's text action format.

    Returns None for actions that should be filtered out (empty key_press, etc.)
    """
    if action_type == "click":
        x = params.get("x", 0)
        y = params.get("y", 0)
        return f'click(x={x}, y={y})'

    if action_type == "double_click":
        x = params.get("x", 0)
        y = params.get("y", 0)
        return f'double_click(x={x}, y={y})'

    if action_type == "type_text" or action_type == "type":
        text = params.get("text", "")
        if not text:
            return None
        return f'type_text(text="{text}")'

    if action_type == "key_press":
        keys = params.get("keys", params.get("key", ""))
        if not keys:
            return None  # Filter empty key_press (Claude's screenshot requests)
        return f'key_press(keys="{keys}")'

    if action_type == "scroll":
        direction = params.get("direction", "down")
        amount = params.get("amount", 3)
        return f'scroll(direction="{direction}", amount={amount})'

    if action_type == "wait":
        seconds = params.get("seconds", 1.0)
        return f'wait(seconds={seconds})'

    if action_type == "done":
        success = str(params.get("success", False)).lower()
        summary = params.get("summary", "")
        return f'done(success={success}, summary="{summary}")'

    return None


def thinking_to_brief(thinking: str) -> str:
    """Condense Claude's verbose thinking to a brief 1-sentence reasoning.

    Holo3 should output brief reasoning, not Claude's long chain-of-thought.
    Extract the key observation and intent.
    """
    if not thinking:
        return ""

    # Take first 2 sentences max
    sentences = thinking.replace("\n", " ").split(". ")
    brief = ". ".join(sentences[:2]).strip()
    if brief and not brief.endswith("."):
        brief += "."

    # Cap at 200 chars
    if len(brief) > 200:
        brief = brief[:197] + "..."

    return brief


def image_to_base64(image_path: str) -> str:
    """Read image file and return base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def convert_trajectory(
    trajectory: dict,
    screenshots_dir: Path,
    step_offset: int = 0,
    task_intent: str = "",
) -> list[dict]:
    """Convert one Claude trajectory to Holo3 training samples.

    Each step becomes one training example:
      user: [screenshot] + task
      assistant: brief reasoning + action

    Args:
        trajectory: One entry from the JSONL
        screenshots_dir: Directory with step-numbered PNGs
        step_offset: Screenshot numbering offset (for multi-iteration runs)
        task_intent: The task description

    Returns:
        List of conversation dicts ready for JSONL output
    """
    samples = []
    intent = task_intent or trajectory.get("intent", "")

    # Truncate intent to avoid bloating context (keep first 500 chars)
    if len(intent) > 500:
        intent = intent[:500] + "..."

    for step_data in trajectory.get("trajectory", []):
        step_num = step_data["step"]
        action_type = step_data["action_type"]
        params = step_data["action_params"]
        thinking = step_data.get("thinking", "")

        # Convert action to Holo3 format
        action_text = claude_action_to_holo3(action_type, params)
        if action_text is None:
            continue  # Skip empty/invalid actions

        # Find matching screenshot
        # Screenshots are numbered from the env's step counter
        screenshot_num = step_offset + step_num
        screenshot_path = screenshots_dir / f"{screenshot_num:04d}.png"

        if not screenshot_path.exists():
            # Try without offset
            screenshot_path = screenshots_dir / f"{step_num:04d}.png"
            if not screenshot_path.exists():
                logger.debug(f"No screenshot for step {step_num} (tried {screenshot_num:04d}.png)")
                continue

        # Build training sample
        brief_reasoning = thinking_to_brief(thinking)
        assistant_response = f"{brief_reasoning}\n{action_text}" if brief_reasoning else action_text

        # Qwen3.5 conversation format (compatible with Holo3)
        sample = {
            "conversations": [
                {
                    "from": "system",
                    "value": HOLO3_SYSTEM,
                },
                {
                    "from": "human",
                    "value": f"<image>\nTask: {intent}\nScreen size: 1280x720 pixels",
                },
                {
                    "from": "gpt",
                    "value": assistant_response,
                },
            ],
            "image": str(screenshot_path),
            "metadata": {
                "source": "claude_distillation",
                "task_id": trajectory.get("task_id", ""),
                "step": step_num,
                "action_type": action_type,
                "success": trajectory.get("success", False),
            },
        }
        samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Convert Claude trajectories to Holo3 training format")
    parser.add_argument("--trajectories", required=True, help="Path to Claude trajectory JSONL")
    parser.add_argument("--screenshots", required=True, help="Directory with step-numbered screenshots")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--successful-only", action="store_true", help="Only include successful trajectories")
    parser.add_argument("--include-failed", action="store_true", help="Include failed trajectories too (for negative examples)")
    parser.add_argument("--max-steps-per-trajectory", type=int, default=0, help="Max steps per trajectory (0=all)")
    parser.add_argument("--task-intent", default="", help="Override task intent for all samples")
    args = parser.parse_args()

    trajectories_path = Path(args.trajectories)
    screenshots_dir = Path(args.screenshots)
    output_path = Path(args.output)

    if not trajectories_path.exists():
        logger.error(f"Trajectories not found: {trajectories_path}")
        sys.exit(1)
    if not screenshots_dir.exists():
        logger.error(f"Screenshots dir not found: {screenshots_dir}")
        sys.exit(1)

    # Count available screenshots
    png_files = sorted(screenshots_dir.glob("*.png"))
    logger.info(f"Found {len(png_files)} screenshots in {screenshots_dir}")

    # Load trajectories
    trajectories = []
    with open(trajectories_path) as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("{"):
                trajectories.append(json.loads(line))

    logger.info(f"Loaded {len(trajectories)} trajectories")

    # Filter
    if args.successful_only:
        trajectories = [t for t in trajectories if t.get("success")]
        logger.info(f"After success filter: {len(trajectories)} trajectories")

    # Convert
    all_samples = []
    step_offset = 0  # Track cumulative step offset across iterations

    for traj in trajectories:
        samples = convert_trajectory(
            traj,
            screenshots_dir,
            step_offset=step_offset,
            task_intent=args.task_intent,
        )
        all_samples.extend(samples)

        # Advance offset by the number of steps in this trajectory
        step_offset += traj.get("steps", 0)
        logger.info(f"  {traj['task_id']}: {len(samples)} training samples (offset now {step_offset})")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    logger.info(f"\nOutput: {output_path}")
    logger.info(f"Total training samples: {len(all_samples)}")
    logger.info(f"From {len(trajectories)} trajectories")

    # Stats
    action_counts: dict[str, int] = {}
    for s in all_samples:
        atype = s["metadata"]["action_type"]
        action_counts[atype] = action_counts.get(atype, 0) + 1

    logger.info("Action distribution:")
    for atype, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {atype}: {count}")


if __name__ == "__main__":
    main()
