"""Claude CUA Brain — Anthropic's computer use model via API.

Uses Claude's native computer_use tool for pixel-accurate browser automation.
Claude outputs structured tool calls (click, type, scroll, etc.) with rich
chain-of-thought reasoning — no regex parsing needed.

Backend: Anthropic API (direct HTTP, no SDK dependency)

Key differences from other brains:
- Native computer_use_2025_01_24 tool — Claude understands UIs natively
- Structured tool calls (like Gemma4, unlike EvoCUA's pyautogui text)
- Rich reasoning in thinking blocks — valuable for distillation
- No GPU needed — runs via API, pairs with XdotoolGymEnv on Modal

Usage:
    brain = ClaudeBrain(api_key="sk-ant-...", model="claude-sonnet-4-20250514")
    brain.load()
    result = brain.think(frames=[screenshot], task="Click the login button", ...)

Cost: ~$0.01-0.05 per screenshot step (input tokens dominate)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from io import BytesIO

import requests
from PIL import Image

from .actions import Action, ActionType, parse_tool_call

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a computer use agent. You observe the screen and perform actions to complete tasks.

You receive a sequence of recent screen frames showing how the display has changed over time.
The LAST frame is the current screen state. Earlier frames show what happened before.

Your job:
1. OBSERVE the current screen state carefully
2. REASON step by step about what to do next
3. CALL exactly one tool to perform the next action

# Core rules
- Coordinates are absolute screen pixels. Aim for the CENTER of the target element.
- Look at the LAST frame for current state. Earlier frames show what changed.
- Execute ONE action per turn. After each action, observe the result before acting again.

# Form filling — CRITICAL
- To fill a form: click the input field ONCE to focus it, then call type() with the value.
- Do NOT click an input field multiple times. One click focuses it — then immediately type.
- After typing, move to the next field: click the next input, or press key('Tab').
- To submit a form: press key('Enter') — this is the most reliable method.
- If you already clicked a field and see it is focused, your NEXT action must be type() — not another click.

# Avoiding loops
- NEVER repeat the same action more than twice. If clicking the same spot twice doesn't work, try a different approach.
- If you're stuck: try scrolling, pressing Tab, clicking a different element, or using keyboard shortcuts.

# Completion
- When the task is complete, call done(success=true, summary="...").
- If stuck after multiple attempts, call done(success=false, summary="...").

# Waiting
- If a page is loading or animating, call wait() to observe the result.
- After submitting a form, call wait(seconds=2) before checking the result.\
"""

# Claude computer_use tool + our custom done/wait tools
CLAUDE_TOOLS = [
    {
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": 1280,
        "display_height_px": 720,
        "display_number": 1,
    },
]

# Additional tools that Claude doesn't have natively
EXTRA_TOOLS = [
    {
        "name": "done",
        "description": (
            "Signal that the task is complete. Call this when you have achieved the goal "
            "or determined it cannot be completed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "description": "Whether the task succeeded"},
                "summary": {"type": "string", "description": "Brief summary of what was done or extracted data"},
            },
            "required": ["success", "summary"],
        },
    },
    {
        "name": "wait",
        "description": (
            "Wait and observe the screen. Use this when an action is in progress "
            "(loading, animation) and you need to see the result before acting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "Seconds to wait (default: 1.0)",
                },
            },
        },
    },
]


def _image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@dataclass
class InferenceResult:
    """Result from a single brain inference cycle."""

    action: Action
    raw_output: str
    thinking: str = ""
    tokens_used: int = 0


class ClaudeBrain:
    """Claude CUA brain using Anthropic API for browser automation.

    Args:
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        model: Claude model ID.
        max_tokens: Maximum tokens to generate per call.
        thinking_budget: Token budget for extended thinking (0 to disable).
        screen_size: Display resolution for computer_use tool.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        thinking_budget: int = 2048,
        screen_size: tuple[int, int] = (1280, 720),
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.model_name = f"Claude ({model})"
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        self.screen_size = screen_size

    def load(self) -> None:
        """Verify API key is set and reachable."""
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Provide api_key= or set the env var."
            )
        # Quick validation — list models
        try:
            resp = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2025-01-24",
                },
                timeout=10,
            )
            if resp.status_code == 401:
                raise RuntimeError("Invalid ANTHROPIC_API_KEY")
            logger.info(f"Claude API connected: {self.model}")
        except requests.ConnectionError as e:
            raise RuntimeError(f"Cannot reach Anthropic API: {e}")

    def query(self, prompt: str, response_format: str = "json") -> str:
        """Send a text-only prompt and return the text response.

        Used for plan parsing, classification, extraction (no images).
        """
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=self._headers(),
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            # Extract text from content blocks
            for block in data.get("content", []):
                if block.get("type") == "text":
                    return block["text"]
            return ""
        except Exception as e:
            logger.error(f"query failed: {e}")
            return ""

    def think(
        self,
        frames: list[Image.Image],
        task: str,
        action_history: list[Action] | None = None,
        screen_size: tuple[int, int] = (1920, 1080),
    ) -> InferenceResult:
        """Run perception-reasoning-action via Claude API with computer_use tool."""
        # Update computer_use tool dimensions to match actual screen
        tools = self._build_tools(screen_size)
        messages = self._build_messages(frames, task, action_history, screen_size)

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": messages,
            "tools": tools,
        }

        # Enable extended thinking for richer reasoning (better for distillation)
        if self.thinking_budget > 0:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }

        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=self._headers(),
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Claude API request failed: {e}")
            return InferenceResult(
                action=Action(ActionType.WAIT, {"seconds": 1.0}, reasoning=str(e)),
                raw_output=str(e),
            )

        return self._parse_response(data)

    def _headers(self) -> dict:
        """Build Anthropic API headers."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2025-01-24",
            "content-type": "application/json",
        }

    def _build_tools(self, screen_size: tuple[int, int]) -> list[dict]:
        """Build tool list with correct screen dimensions."""
        computer_tool = {
            "type": "computer_20250124",
            "name": "computer",
            "display_width_px": screen_size[0],
            "display_height_px": screen_size[1],
            "display_number": 1,
        }
        return [computer_tool] + EXTRA_TOOLS

    def _build_messages(
        self,
        frames: list[Image.Image],
        task: str,
        action_history: list[Action] | None,
        screen_size: tuple[int, int],
    ) -> list[dict]:
        """Build Claude API messages with image content."""
        content = []

        # Add frames as images
        n_frames = len(frames)
        for i, frame in enumerate(frames):
            label = "CURRENT" if i == n_frames - 1 else f"t-{n_frames - 1 - i}"
            content.append({"type": "text", "text": f"[Frame {label}]"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _image_to_base64(frame),
                },
            })

        # Task context
        context_parts = [
            f"\nTask: {task}",
            f"Screen size: {screen_size[0]}x{screen_size[1]} pixels",
        ]

        if action_history:
            recent = action_history[-10:]
            history_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(recent))
            context_parts.append(f"Recent actions:\n{history_str}")

        content.append({"type": "text", "text": "\n".join(context_parts)})

        return [{"role": "user", "content": content}]

    def _parse_response(self, data: dict) -> InferenceResult:
        """Parse Claude API response into InferenceResult.

        Claude responses contain content blocks:
        - thinking: extended thinking text (reasoning)
        - text: assistant text
        - tool_use: structured tool calls
        """
        raw_output = json.dumps(data.get("content", []))
        thinking_parts = []
        text_parts = []
        action = None

        for block in data.get("content", []):
            block_type = block.get("type", "")

            if block_type == "thinking":
                thinking_parts.append(block.get("thinking", ""))

            elif block_type == "text":
                text_parts.append(block.get("text", ""))

            elif block_type == "tool_use":
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})

                if tool_name == "computer":
                    action = self._parse_computer_action(tool_input)
                elif tool_name == "done":
                    action = Action(
                        ActionType.DONE,
                        {
                            "success": tool_input.get("success", False),
                            "summary": tool_input.get("summary", ""),
                        },
                    )
                elif tool_name == "wait":
                    action = Action(
                        ActionType.WAIT,
                        {"seconds": tool_input.get("seconds", 1.0)},
                    )

        thinking = "\n".join(thinking_parts)
        text = "\n".join(text_parts)

        # If no tool call found, try to parse from text
        if action is None:
            if text:
                action = self._parse_text_fallback(text)
            else:
                action = Action(ActionType.WAIT, {"seconds": 1.0}, reasoning="No action in response")

        # Combine thinking + text for full reasoning
        full_thinking = thinking
        if text and thinking:
            full_thinking = f"{thinking}\n\n{text}"
        elif text:
            full_thinking = text

        action.reasoning = action.reasoning or full_thinking[:500]

        tokens_used = data.get("usage", {})
        total_tokens = tokens_used.get("input_tokens", 0) + tokens_used.get("output_tokens", 0)

        return InferenceResult(
            action=action,
            raw_output=raw_output,
            thinking=full_thinking,
            tokens_used=total_tokens,
        )

    @staticmethod
    def _parse_computer_action(tool_input: dict) -> Action:
        """Convert Claude's computer_use tool call to our Action format.

        Claude's computer_use actions:
            {"action": "left_click", "coordinate": [x, y]}
            {"action": "type", "text": "hello"}
            {"action": "key", "key": "Return"}
            {"action": "scroll", "coordinate": [x, y], "direction": "down", "amount": 3}
            {"action": "screenshot"}  — request new screenshot
            {"action": "left_click_drag", "start_coordinate": [x1,y1], "coordinate": [x2,y2]}
            {"action": "double_click", "coordinate": [x, y]}
            {"action": "right_click", "coordinate": [x, y]}
        """
        action_type = tool_input.get("action", "")
        coord = tool_input.get("coordinate", [0, 0])

        if action_type in ("left_click", "click"):
            return Action(ActionType.CLICK, {"x": coord[0], "y": coord[1], "button": "left"})

        elif action_type == "right_click":
            return Action(ActionType.CLICK, {"x": coord[0], "y": coord[1], "button": "right"})

        elif action_type == "double_click":
            return Action(ActionType.DOUBLE_CLICK, {"x": coord[0], "y": coord[1]})

        elif action_type == "type":
            return Action(ActionType.TYPE, {"text": tool_input.get("text", "")})

        elif action_type == "key":
            # Claude uses Return/Tab/Escape etc. Normalize.
            key = tool_input.get("key", "")
            key = _normalize_claude_key(key)
            return Action(ActionType.KEY_PRESS, {"keys": key})

        elif action_type == "scroll":
            direction = tool_input.get("direction", "down")
            amount = tool_input.get("amount", 3)
            params = {"direction": direction, "amount": amount}
            if coord and coord != [0, 0]:
                params["x"] = coord[0]
                params["y"] = coord[1]
            return Action(ActionType.SCROLL, params)

        elif action_type == "left_click_drag":
            start = tool_input.get("start_coordinate", [0, 0])
            end = coord
            return Action(ActionType.DRAG, {
                "start_x": start[0], "start_y": start[1],
                "end_x": end[0], "end_y": end[1],
            })

        elif action_type == "screenshot":
            # Claude requesting a new screenshot = wait + observe
            return Action(ActionType.WAIT, {"seconds": 0.5})

        elif action_type == "triple_click":
            # Select all text in field — map to ctrl+a
            return Action(ActionType.KEY_PRESS, {"keys": "ctrl+a"})

        else:
            logger.warning(f"Unknown Claude computer action: {action_type}")
            return Action(ActionType.WAIT, {"seconds": 1.0}, reasoning=f"Unknown: {action_type}")

    @staticmethod
    def _parse_text_fallback(text: str) -> Action:
        """Parse action from text when no tool_use block is present."""
        import re

        text_lower = text.lower()

        if re.search(r'done|complete|finished|task.*(complete|done)', text_lower):
            success = "fail" not in text_lower and "cannot" not in text_lower
            return Action(ActionType.DONE, {"success": success, "summary": text[:200]})

        if re.search(r'wait|loading|progress', text_lower):
            return Action(ActionType.WAIT, {"seconds": 1.0})

        return Action(ActionType.WAIT, {"seconds": 1.0}, reasoning=text[:200])


def _normalize_claude_key(key: str) -> str:
    """Normalize Claude's key names to our format.

    Claude uses: Return, Tab, Escape, BackSpace, space, super, etc.
    We use: enter, tab, escape, backspace, space, super, etc.
    """
    key_map = {
        "Return": "enter",
        "Enter": "enter",
        "BackSpace": "backspace",
        "Escape": "escape",
        "Tab": "tab",
        "Delete": "delete",
        "Home": "home",
        "End": "end",
        "Page_Up": "pageup",
        "Page_Down": "pagedown",
        "Up": "up",
        "Down": "down",
        "Left": "left",
        "Right": "right",
        "F1": "f1", "F2": "f2", "F3": "f3", "F4": "f4",
        "F5": "f5", "F6": "f6", "F7": "f7", "F8": "f8",
        "F9": "f9", "F10": "f10", "F11": "f11", "F12": "f12",
    }

    # Handle key combos: "ctrl+Return" → "ctrl+enter"
    parts = key.split("+")
    normalized = []
    for part in parts:
        part = part.strip()
        normalized.append(key_map.get(part, part.lower()))
    return "+".join(normalized)
