"""Action types and Gemma4 tool schemas for the CUA agent.

Actions are the atomic operations the agent can perform on the computer.
Gemma4's native function-calling produces these directly — no parsing needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    TYPE = "type_text"
    KEY_PRESS = "key_press"
    SCROLL = "scroll"
    DRAG = "drag"
    WAIT = "wait"
    DONE = "done"


@dataclass
class Action:
    action_type: ActionType
    params: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    def __str__(self) -> str:
        return f"{self.action_type.value}({self.params})" + (
            f" # {self.reasoning}" if self.reasoning else ""
        )


# ── Gemma4 function-calling tool definitions ──────────────────────────────────
# These are passed to the model via processor.apply_chat_template(tools=TOOLS).
# Gemma4 outputs structured JSON matching these schemas — one model, one pass.

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": (
                "Click at a screen location. Use the bounding box center of the target element. "
                "Coordinates are absolute pixels relative to the screen."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate in pixels"},
                    "y": {"type": "integer", "description": "Y coordinate in pixels"},
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button to click (default: left)",
                    },
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "double_click",
            "description": "Double-click at a screen location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate"},
                    "y": {"type": "integer", "description": "Y coordinate"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": (
                "Type text using the keyboard. The text is typed character by character "
                "into whatever element currently has focus."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to type"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "key_press",
            "description": (
                "Press a keyboard key or key combination. "
                "Examples: 'enter', 'tab', 'escape', 'command+c', 'ctrl+a', 'shift+tab'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "string",
                        "description": "Key or key combination (e.g. 'enter', 'command+v')",
                    },
                },
                "required": ["keys"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll at the current mouse position or a specific location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Number of scroll clicks (default: 3)",
                    },
                    "x": {"type": "integer", "description": "Optional X coordinate to scroll at"},
                    "y": {"type": "integer", "description": "Optional Y coordinate to scroll at"},
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drag",
            "description": "Drag from one screen position to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_x": {"type": "integer"},
                    "start_y": {"type": "integer"},
                    "end_x": {"type": "integer"},
                    "end_y": {"type": "integer"},
                },
                "required": ["start_x", "start_y", "end_x", "end_y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": (
                "Wait and observe the screen. Use this when an action is in progress "
                "(loading, animation) and you need to see the result before acting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "number",
                        "description": "Seconds to wait (default: 1.0)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "Signal that the task is complete. Call this when you have achieved the goal "
                "or determined it cannot be completed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean", "description": "Whether the task succeeded"},
                    "summary": {"type": "string", "description": "Brief summary of what was done"},
                },
                "required": ["success", "summary"],
            },
        },
    },
]


def parse_tool_call(name: str, arguments: dict[str, Any], reasoning: str = "") -> Action:
    """Convert a Gemma4 tool call into an Action."""
    try:
        action_type = ActionType(name)
    except ValueError:
        # Fallback: treat unknown tool calls as wait
        return Action(ActionType.WAIT, {"seconds": 1.0}, reasoning=f"Unknown action: {name}")
    return Action(action_type, arguments, reasoning=reasoning)


def parse_model_output(output_text: str) -> list[Action]:
    """Parse raw model output text into Actions.

    Handles both structured tool-call output (from apply_chat_template)
    and raw JSON fallback.
    """
    actions = []
    # Try to find JSON tool calls in the output
    # Gemma4 outputs function calls as JSON blocks
    for line in output_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict) and "name" in data:
                actions.append(
                    parse_tool_call(
                        data["name"],
                        data.get("arguments", data.get("parameters", {})),
                        reasoning=data.get("reasoning", ""),
                    )
                )
        except json.JSONDecodeError:
            continue
    return actions
