#!/usr/bin/env python3
"""Convert AgentNet dataset to Gemma4 tool-calling format for CUA fine-tuning.

AgentNet format:
  - Trajectories with screenshot + pyautogui code per step
  - Relative coordinates (0.0-1.0)
  - Rich CoT: observation, thought, action, code, reflection

Gemma4 target format:
  - Multi-turn conversations with tool definitions
  - Tool calls using special tokens (<|tool_call>, call:, etc.)
  - Image content as base64 in user messages

The key conversion:
  pyautogui.click(x=0.163, y=0.271) → <|tool_call>call:click{"x":209,"y":195}<tool_call|>
  pyautogui.write('hello')          → <|tool_call>call:type_text{"text":"hello"}<tool_call|>
  pyautogui.hotkey('ctrl','c')      → <|tool_call>call:key_press{"keys":"ctrl+c"}<tool_call|>

Usage:
    # Convert AgentNet to Gemma4 format
    python training/convert_agentnet.py \
        --input /path/to/AgentNet/meta_data_merged.jsonl \
        --images /path/to/AgentNet/images/ \
        --output training/data/gemma4_cua_train.jsonl \
        --screen-width 1280 --screen-height 720 \
        --max-tasks 5000

    # Convert only completed tasks with high alignment
    python training/convert_agentnet.py \
        --input /path/to/AgentNet/meta_data_merged.jsonl \
        --images /path/to/AgentNet/images/ \
        --output training/data/gemma4_cua_train.jsonl \
        --min-alignment 7 --completed-only
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Gemma4 CUA tool definitions (same as actions.py TOOLS)
GEMMA4_TOOLS = [
    {"name": "click", "parameters": {"x": "int", "y": "int", "button": "str"}},
    {"name": "double_click", "parameters": {"x": "int", "y": "int"}},
    {"name": "type_text", "parameters": {"text": "str"}},
    {"name": "key_press", "parameters": {"keys": "str"}},
    {"name": "scroll", "parameters": {"direction": "str", "amount": "int"}},
    {"name": "drag", "parameters": {"start_x": "int", "start_y": "int", "end_x": "int", "end_y": "int"}},
    {"name": "wait", "parameters": {"seconds": "float"}},
    {"name": "done", "parameters": {"success": "bool", "summary": "str"}},
]

SYSTEM_PROMPT = """\
You are a computer use agent. You observe the screen and perform actions to complete tasks.

You receive a screenshot of the current screen state.

Your job:
1. OBSERVE the current screen state carefully
2. REASON step by step about what to do next
3. CALL exactly one tool to perform the next action

# Core rules
- Coordinates are absolute screen pixels. Aim for the CENTER of the target element.
- Execute ONE action per turn. After each action, observe the result before acting again.

# Form filling
- To fill a form: click the input field ONCE to focus it, then call type_text() with the value.
- Do NOT click an input field multiple times. One click focuses it — then immediately type.
- After typing, move to the next field: click the next input, or press key_press('tab').
- To submit a form: press key_press('enter') — this is the most reliable method.

# Avoiding loops
- NEVER repeat the same action more than twice. If it doesn't work, try a different approach.

# Completion
- When the task is complete, call done(success=true, summary="...").
- If stuck after multiple attempts, call done(success=false, summary="...").\
"""


def parse_pyautogui_code(code: str, screen_w: int, screen_h: int) -> dict | None:
    """Convert AgentNet's pyautogui code to Gemma4 tool call.

    AgentNet uses relative coordinates (0.0-1.0). We convert to absolute pixels.
    """
    if not code or not code.strip():
        return None

    code = code.strip()

    # click/doubleClick/rightClick(x=0.163, y=0.271) — relative coords
    click_match = re.search(r'(?:double|right|triple|middle)?[Cc]lick\((?:x=)?([\d.]+),\s*(?:y=)?([\d.]+)\)', code)
    if click_match:
        rx, ry = float(click_match.group(1)), float(click_match.group(2))
        # Relative coords (0-1) → absolute pixels
        if rx <= 1.0 and ry <= 1.0:
            x, y = int(rx * screen_w), int(ry * screen_h)
        else:
            x, y = int(rx), int(ry)
        button = "right" if "rightClick" in code or "button='right'" in code else "left"
        if "doubleClick" in code or "double_click" in code:
            return {"name": "double_click", "arguments": {"x": x, "y": y}}
        return {"name": "click", "arguments": {"x": x, "y": y, "button": button}}

    # write('text') or typewrite('text')
    write_match = re.search(r"(?:write|typewrite)\(['\"](.+?)['\"]\)", code)
    if write_match:
        return {"name": "type_text", "arguments": {"text": write_match.group(1)}}

    # hotkey('key1', 'key2')
    hotkey_match = re.search(r"hotkey\((.+?)\)", code)
    if hotkey_match:
        raw = hotkey_match.group(1).strip("[]")
        keys = [k.strip().strip("'\"") for k in raw.split(",")]
        return {"name": "key_press", "arguments": {"keys": "+".join(keys)}}

    # press('key')
    press_match = re.search(r"press\(['\"](.+?)['\"]\)", code)
    if press_match:
        return {"name": "key_press", "arguments": {"keys": press_match.group(1)}}

    # scroll(amount)
    scroll_match = re.search(r'scroll\((-?\d+)\)', code)
    if scroll_match:
        amount = int(scroll_match.group(1))
        direction = "up" if amount > 0 else "down"
        return {"name": "scroll", "arguments": {"direction": direction, "amount": abs(amount)}}

    # dragTo(x, y)
    drag_match = re.search(r'dragTo\((?:x=)?([\d.]+),\s*(?:y=)?([\d.]+)\)', code)
    if drag_match:
        ex, ey = float(drag_match.group(1)), float(drag_match.group(2))
        if ex <= 1.0 and ey <= 1.0:
            ex, ey = int(ex * screen_w), int(ey * screen_h)
        return {"name": "drag", "arguments": {"start_x": 0, "start_y": 0, "end_x": int(ex), "end_y": int(ey)}}

    # moveTo — treat as wait (no visible action)
    if "moveTo" in code:
        return {"name": "wait", "arguments": {"seconds": 0.5}}

    # terminate
    if "terminate" in code:
        return {"name": "done", "arguments": {"success": True, "summary": "Task completed"}}

    return None


def convert_trajectory(task: dict, images_dir: str, screen_w: int, screen_h: int) -> list[dict] | None:
    """Convert one AgentNet trajectory to Gemma4 multi-turn format.

    Returns a list of messages (system, user, assistant turns).
    """
    traj = task.get("traj", [])
    if not traj:
        return None

    instruction = task.get("instruction", "") or task.get("natural_language_task", "")
    if not instruction:
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    for step in traj:
        value = step.get("value", {})
        code = value.get("code", "")
        image_file = step.get("image", "")

        # Parse pyautogui code to tool call
        tool_call = parse_pyautogui_code(code, screen_w, screen_h)
        if tool_call is None:
            continue

        # Build user turn with screenshot + task
        user_content = []

        # Add image if available
        if image_file and images_dir:
            image_path = os.path.join(images_dir, image_file)
            if os.path.exists(image_path):
                user_content.append({
                    "type": "image",
                    "path": image_path,
                })

        # Task context
        context = f"Task: {instruction}"
        if step.get("index", 0) > 0:
            context += f"\nStep {step['index'] + 1}: Continue executing the task."
        user_content.append({"type": "text", "text": context})

        messages.append({"role": "user", "content": user_content})

        # Build assistant turn with thinking + tool call
        thinking = ""
        if value.get("thought"):
            thinking = value["thought"]
        if value.get("observation"):
            thinking = f"Observation: {value['observation']}\n{thinking}"

        # Gemma4 tool call format
        assistant_content = ""
        if thinking:
            assistant_content += f"{thinking}\n\n"
        assistant_content += f"Action: {tool_call['name']}({json.dumps(tool_call['arguments'])})"

        messages.append({
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": [tool_call],
        })

    if len(messages) <= 1:  # Only system message
        return None

    return messages


def convert_to_training_format(messages: list[dict]) -> dict:
    """Convert multi-turn messages to the format expected by Unsloth/TRL.

    Output format (ShareGPT-style):
    {
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."},
            ...
        ]
    }
    """
    conversations = []
    for msg in messages:
        role_map = {"system": "system", "user": "human", "assistant": "gpt"}
        from_role = role_map.get(msg["role"], msg["role"])

        if isinstance(msg.get("content"), list):
            # Multi-modal content — extract text parts, note image paths
            text_parts = []
            for part in msg["content"]:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif part.get("type") == "image":
                        text_parts.append(f"<image:{part.get('path', 'screenshot')}>")
                elif isinstance(part, str):
                    text_parts.append(part)
            value = "\n".join(text_parts)
        else:
            value = msg.get("content", "")

        conversations.append({"from": from_role, "value": value})

    return {"conversations": conversations}


def main():
    parser = argparse.ArgumentParser(description="Convert AgentNet to Gemma4 CUA training format")
    parser.add_argument("--input", required=True, help="Path to AgentNet meta_data_merged.jsonl")
    parser.add_argument("--images", default="", help="Path to AgentNet images directory")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--screen-width", type=int, default=1280)
    parser.add_argument("--screen-height", type=int, default=720)
    parser.add_argument("--max-tasks", type=int, default=0, help="Limit number of tasks (0=all)")
    parser.add_argument("--min-alignment", type=int, default=5, help="Minimum alignment score")
    parser.add_argument("--completed-only", action="store_true", help="Only include completed tasks")
    parser.add_argument("--min-steps", type=int, default=2, help="Minimum trajectory steps")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum trajectory steps")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stats
    total_tasks = 0
    converted = 0
    skipped_alignment = 0
    skipped_incomplete = 0
    skipped_short = 0
    skipped_parse = 0
    total_steps = 0

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            total_tasks += 1

            if args.max_tasks > 0 and converted >= args.max_tasks:
                break

            try:
                task = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Filters
            alignment = task.get("alignment_score")
            # Skip alignment filter if field is missing/None
            if alignment is not None and args.min_alignment > 0:
                try:
                    if int(alignment) < args.min_alignment:
                        skipped_alignment += 1
                        continue
                except (ValueError, TypeError):
                    pass  # Field exists but not numeric — don't filter

            if args.completed_only and not task.get("task_completed", False):
                skipped_incomplete += 1
                continue

            traj = task.get("traj", [])
            if len(traj) < args.min_steps:
                skipped_short += 1
                continue
            if len(traj) > args.max_steps:
                traj = traj[:args.max_steps]
                task["traj"] = traj

            # Convert
            messages = convert_trajectory(task, args.images, args.screen_width, args.screen_height)
            if messages is None:
                skipped_parse += 1
                continue

            training_example = convert_to_training_format(messages)
            f_out.write(json.dumps(training_example) + "\n")
            converted += 1
            total_steps += len(traj)

            if converted % 500 == 0:
                logger.info(f"Converted {converted} tasks ({total_steps} steps)...")

    logger.info(f"\n{'='*50}")
    logger.info(f"Conversion complete")
    logger.info(f"  Total tasks read:      {total_tasks}")
    logger.info(f"  Converted:             {converted}")
    logger.info(f"  Total steps:           {total_steps}")
    logger.info(f"  Skipped (alignment):   {skipped_alignment}")
    logger.info(f"  Skipped (incomplete):  {skipped_incomplete}")
    logger.info(f"  Skipped (too short):   {skipped_short}")
    logger.info(f"  Skipped (parse fail):  {skipped_parse}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
