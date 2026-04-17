"""Holo3-35B-A3B CUA Brain — tool-calling via vLLM OpenAI-compatible API.

Holo3 (Hcompany/Holo3-35B-A3B) is a Qwen3.5-based MoE model (35B total,
3B active) scoring 77.8% on OSWorld-Verified. It uses the Qwen3.5 vision
processor, so coordinates follow the same smart-resize system as Qwen2.5-VL
(identical to OpenCUA).

Key design decisions:
- Tool calling (like LlamaCppBrain), NOT pyautogui text (like OpenCUA).
- Qwen smart-resize coordinate conversion (copied from OpenCUA).
- Native <think> block support via enable_thinking toggle.
- Triple fallback parsing: tool_calls -> JSON text -> pyautogui text.

Architecture:
    vLLM server (OpenAI API)  <->  Holo3Brain
         |                          |
    Holo3-35B-A3B weights     Screenshot frames + task
    (BF16, TP=2 on 2xA100)   -> tool calls -> Actions

Usage:
    # Start vLLM with reasoning support:
    python -m vllm.entrypoints.openai.api_server \\
        --model Hcompany/Holo3-35B-A3B \\
        --tensor-parallel-size 2 \\
        --enable-reasoning --reasoning-parser qwen3 \\
        --gpu-memory-utilization 0.90 \\
        --max-model-len 32768 --port 8000

    # Then use this brain:
    brain = Holo3Brain(base_url="http://localhost:8000/v1")
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO

import requests
from PIL import Image

from .actions import TOOLS, Action, ActionType, parse_tool_call

logger = logging.getLogger(__name__)

# ── System prompt (tool-calling style, adapted from LlamaCppBrain) ──────────

SYSTEM_PROMPT = """\
You are a computer use agent. You observe the screen and perform actions to complete tasks.

You receive a sequence of recent screen frames showing how the display has changed over time.
The LAST frame is the current screen state. Earlier frames show what happened before.

Your job:
1. OBSERVE the current screen state carefully
2. REASON about what to do next given the task and what you see
3. CALL exactly one tool to perform an action

Important guidelines:
- Coordinates are in absolute screen pixels
- Look at the last frame for current state; earlier frames for context
- If something is loading or animating, call wait() to observe the result
- If you cannot find an element, try scrolling to reveal it
- When the task is complete, call done(success=true, summary="...")
- If you're stuck after multiple attempts, call done(success=false, summary="...")
- Be precise with click coordinates -- aim for the center of the target element

Form filling tips:
- Click the center of an input field ONCE to focus it, then use type_text()
- Press tab to move between form fields, enter to submit
- After typing, verify the field shows your text before moving on

Browser navigation:
- key_press("alt+left") to go back
- key_press("ctrl+w") to close tab
- key_press("ctrl+tab") to switch tabs
- scroll(direction="down") to see more content

Data extraction:
- Read ALL text visually from the screenshot
- Phone numbers: (555) 555-5555, 555-555-5555, or 10+ consecutive digits
- Read prices, years, makes, models from page titles and content
- Read the current URL from the browser address bar
- When reporting results, include EVERY piece of extracted data

Loop avoidance:
- NEVER repeat the same action more than twice in a row
- If an action fails twice, try a completely different approach
- If you cannot make progress after 5 actions, call done(success=false)\
"""

# ── OpenAI function-calling tool format ─────────────────────────────────────

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": tool["function"],
    }
    for tool in TOOLS
]


# ── Qwen smart-resize coordinate system (same as OpenCUA) ──────────────────

def _smart_resize(height: int, width: int, factor: int = 28,
                  min_pixels: int = 3136, max_pixels: int = 12845056) -> tuple[int, int]:
    """Qwen2.5-VL / Qwen3.5-VL smart resize -- matches OpenCUA's coordinate system.

    The model sees images at this effective resolution. Coordinates in model
    output are relative to these dimensions, so we must map them back to
    actual screen pixels.
    """
    if height < factor or width < factor:
        raise ValueError(f"Image too small: {width}x{height}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        scale = (max_pixels / (height * width)) ** 0.5
        h_bar = max(factor, round(height * scale / factor) * factor)
        w_bar = max(factor, round(width * scale / factor) * factor)
    if h_bar * w_bar < min_pixels:
        scale = (min_pixels / (height * width)) ** 0.5
        h_bar = max(factor, round(height * scale / factor) * factor)
        w_bar = max(factor, round(width * scale / factor) * factor)
    return h_bar, w_bar


def _model_coords_to_screen(model_x: int, model_y: int,
                             screen_width: int, screen_height: int) -> tuple[int, int]:
    """Convert Holo3's model coordinates to actual screen coordinates.

    Holo3 uses the Qwen3.5 vision processor which applies the same
    smart-resize as Qwen2.5-VL. Model outputs coordinates in the
    resized space; we convert them to absolute screen pixels.
    """
    resized_h, resized_w = _smart_resize(screen_height, screen_width)
    rel_x = model_x / resized_w
    rel_y = model_y / resized_h
    abs_x = int(rel_x * screen_width)
    abs_y = int(rel_y * screen_height)
    return abs_x, abs_y


# ── Helpers ─────────────────────────────────────────────────────────────────

def _image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 data URL."""
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


# ── Brain ───────────────────────────────────────────────────────────────────

class Holo3Brain:
    """Holo3-35B-A3B brain via vLLM OpenAI-compatible API with tool calling.

    Args:
        base_url: URL of the vLLM server.
        model: Model name as served by vLLM.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        screen_size: Default screen size for coordinate conversion.
        enable_thinking: Whether to enable native <think> blocks.
    """

    def __init__(
        self,
        base_url: str = "https://api.hcompany.ai/v1",
        model: str = "holo3-35b-a3b",
        api_key: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        screen_size: tuple[int, int] = (1280, 720),
        enable_thinking: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("HAI_API_KEY", "")
        self.model_name = "Holo3-35B-A3B"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.default_screen_size = screen_size
        self.enable_thinking = enable_thinking

    @property
    def _headers(self) -> dict:
        """Auth headers for API requests."""
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def load(self) -> None:
        """Verify the API is reachable."""
        for attempt in range(12):  # Up to 2 min
            try:
                resp = requests.get(f"{self.base_url}/models",
                                    headers=self._headers, timeout=30)
                resp.raise_for_status()
                models = resp.json()
                logger.info(f"Holo3 API connected: {models}")
                return
            except Exception as e:
                if attempt < 11:
                    logger.info(f"Holo3 not ready (attempt {attempt + 1}/12), waiting 10s...")
                    time.sleep(10)
                else:
                    raise RuntimeError(
                        f"Cannot connect to Holo3 API at {self.base_url}\n"
                        f"Error: {e}"
                    )

    def query(self, prompt: str, response_format: str = "json") -> str:
        """Send a text-only prompt and return text response. No images, no actions."""
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload, headers=self._headers,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"].get("content", "")
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
        """Run perception-reasoning-action via Holo3 with tool calling."""
        messages = self._build_messages(frames, task, action_history, screen_size)

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "tools": OPENAI_TOOLS,
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        # Holo3 native thinking toggle (H Company API format)
        if not self.enable_thinking:
            payload["thinking"] = False

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload, headers=self._headers,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Holo3 API request failed: {e}")
            return InferenceResult(
                action=Action(ActionType.WAIT, {"seconds": 1.0}, reasoning=str(e)),
                raw_output=str(e),
            )

        return self._parse_response(data, screen_size)

    def _build_messages(
        self,
        frames: list[Image.Image],
        task: str,
        action_history: list[Action] | None,
        screen_size: tuple[int, int],
    ) -> list[dict]:
        """Build OpenAI-format messages with image content."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        content: list[dict] = []
        n_frames = len(frames)
        for i, frame in enumerate(frames):
            label = "CURRENT" if i == n_frames - 1 else f"t-{n_frames - 1 - i}"
            content.append({"type": "text", "text": f"[Frame {label}]"})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_image_to_base64(frame)}",
                },
            })

        context_parts = [
            f"\nTask: {task}",
            f"Screen size: {screen_size[0]}x{screen_size[1]} pixels",
        ]
        if action_history:
            recent = action_history[-10:]
            history_str = "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(recent))
            context_parts.append(f"Recent actions:\n{history_str}")

        content.append({"type": "text", "text": "\n".join(context_parts)})
        messages.append({"role": "user", "content": content})

        return messages

    # ── Response parsing (triple fallback) ──────────────────────────────────

    def _parse_response(self, data: dict, screen_size: tuple[int, int]) -> InferenceResult:
        """Parse Holo3 response with triple fallback:

        1. Standard OpenAI tool_calls (primary path)
        2. JSON action object in content text
        3. pyautogui-style text in content (last resort)
        """
        choice = data["choices"][0]
        message = choice["message"]
        raw_output = json.dumps(message)

        # Extract thinking from reasoning_content (vLLM with --enable-reasoning)
        # or from <think> blocks in content
        thinking = message.get("reasoning_content", "") or ""
        content_text = message.get("content", "") or ""

        if not thinking and "<think>" in content_text:
            think_match = re.search(r"<think>(.*?)</think>", content_text, re.DOTALL)
            if think_match:
                thinking = think_match.group(1).strip()
                # Remove think block from content for further parsing
                content_text = content_text[:think_match.start()] + content_text[think_match.end():]
                content_text = content_text.strip()

        # Strategy 1: OpenAI tool_calls
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            func = tc.get("function", {})
            name = func.get("name", "wait")
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}

            # Convert model coordinates to screen coordinates for spatial actions
            args = self._convert_coords(name, args, screen_size)

            if not thinking:
                thinking = content_text
            action = parse_tool_call(name, args, reasoning=thinking)
            return InferenceResult(
                action=action,
                raw_output=raw_output,
                thinking=thinking,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
            )

        # Strategy 2: JSON action in content text
        json_action = self._parse_json_action(content_text, screen_size)
        if json_action is not None:
            json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', content_text)
            if json_match and not thinking:
                thinking = content_text[:json_match.start()].strip()
            return InferenceResult(
                action=json_action,
                raw_output=raw_output,
                thinking=thinking,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
            )

        # Strategy 3: pyautogui-style text
        pyautogui_action = self._parse_pyautogui(content_text, screen_size)
        if pyautogui_action is not None:
            pyautogui_match = re.search(r'pyautogui\.\w+\(.*?\)', content_text, re.DOTALL)
            if pyautogui_match and not thinking:
                thinking = content_text[:pyautogui_match.start()].strip()
            return InferenceResult(
                action=pyautogui_action,
                raw_output=raw_output,
                thinking=thinking,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
            )

        # Strategy 4: terminate / DONE / FAIL keywords
        term_match = re.search(r"terminate\(['\"](\w+)['\"]\)", content_text)
        if term_match:
            success = term_match.group(1).lower() == "success"
            if not thinking:
                thinking = content_text[:term_match.start()].strip()
            action = Action(ActionType.DONE, {"success": success, "summary": thinking[:200]})
            return InferenceResult(
                action=action,
                raw_output=raw_output,
                thinking=thinking,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
            )

        if "DONE" in content_text.upper():
            action = Action(ActionType.DONE, {"success": True, "summary": content_text[:200]})
        elif "FAIL" in content_text.upper():
            action = Action(ActionType.DONE, {"success": False, "summary": content_text[:200]})
        else:
            logger.warning(f"No tool call or parseable action: {content_text[:200]}")
            action = Action(
                ActionType.WAIT, {"seconds": 1.0},
                reasoning=f"Could not parse: {content_text[:100]}",
            )

        return InferenceResult(
            action=action,
            raw_output=raw_output,
            thinking=thinking,
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
        )

    def _convert_coords(self, action_name: str, args: dict, screen_size: tuple[int, int]) -> dict:
        """Convert model coordinates to screen coordinates for spatial actions."""
        spatial_actions = {"click", "double_click", "scroll"}
        drag_action = "drag"

        if action_name in spatial_actions:
            if "x" in args and "y" in args:
                sx, sy = _model_coords_to_screen(
                    int(args["x"]), int(args["y"]),
                    screen_size[0], screen_size[1],
                )
                args["x"] = sx
                args["y"] = sy

        if action_name == drag_action:
            for prefix in ("start_", "end_"):
                xk, yk = f"{prefix}x", f"{prefix}y"
                if xk in args and yk in args:
                    sx, sy = _model_coords_to_screen(
                        int(args[xk]), int(args[yk]),
                        screen_size[0], screen_size[1],
                    )
                    args[xk] = sx
                    args[yk] = sy

        return args

    def _parse_json_action(self, text: str, screen_size: tuple[int, int]) -> Action | None:
        """Parse JSON-formatted actions: {"action": "click", "x": N, "y": N}.

        Reused from OpenCUA's fallback strategy for EvoCUA-style output.
        """
        # Strip markdown code fences
        cleaned = re.sub(r'```(?:json|python|)\s*\n?', '', text)
        cleaned = re.sub(r'\n?```', '', cleaned)

        match = re.search(r'\{[^{}]*"action"\s*:\s*"[^"]*"[^{}]*\}', cleaned)
        if not match:
            return None

        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

        action_type = obj.get("action", "").lower()

        if action_type == "click":
            x, y = int(obj.get("x", 0)), int(obj.get("y", 0))
            sx, sy = _model_coords_to_screen(x, y, screen_size[0], screen_size[1])
            button = obj.get("button", "left")
            return Action(ActionType.CLICK, {"x": sx, "y": sy, "button": button})

        if action_type in ("double_click", "doubleclick", "doubleClick"):
            x, y = int(obj.get("x", 0)), int(obj.get("y", 0))
            sx, sy = _model_coords_to_screen(x, y, screen_size[0], screen_size[1])
            return Action(ActionType.DOUBLE_CLICK, {"x": sx, "y": sy})

        if action_type in ("type", "type_text", "typewrite", "write"):
            return Action(ActionType.TYPE, {"text": obj.get("text", "")})

        if action_type in ("key", "key_press", "press", "hotkey"):
            keys = obj.get("keys", obj.get("key", ""))
            if isinstance(keys, list):
                keys = "+".join(keys)
            return Action(ActionType.KEY_PRESS, {"keys": keys})

        if action_type == "scroll":
            direction = obj.get("direction", "down")
            amount = int(obj.get("amount", 3))
            return Action(ActionType.SCROLL, {"direction": direction, "amount": amount})

        if action_type in ("done", "terminate"):
            success = obj.get("success", obj.get("status") == "success")
            summary = obj.get("summary", obj.get("message", ""))
            return Action(ActionType.DONE, {"success": bool(success), "summary": str(summary)[:200]})

        if action_type == "wait":
            seconds = float(obj.get("seconds", 1.0))
            return Action(ActionType.WAIT, {"seconds": seconds})

        return None

    def _parse_pyautogui(self, text: str, screen_size: tuple[int, int]) -> Action | None:
        """Parse pyautogui-style commands as a last-resort fallback.

        Reused from OpenCUA's parsing strategy.
        """
        # click(x=N, y=N) or click(N, N)
        click_match = re.search(r'click\((?:x=)?(\d+),\s*(?:y=)?(\d+)\)', text)
        if click_match:
            mx, my = int(click_match.group(1)), int(click_match.group(2))
            sx, sy = _model_coords_to_screen(mx, my, screen_size[0], screen_size[1])
            return Action(ActionType.CLICK, {"x": sx, "y": sy})

        # doubleClick
        dbl_match = re.search(r'doubleClick\((?:x=)?(\d+),\s*(?:y=)?(\d+)\)', text)
        if dbl_match:
            mx, my = int(dbl_match.group(1)), int(dbl_match.group(2))
            sx, sy = _model_coords_to_screen(mx, my, screen_size[0], screen_size[1])
            return Action(ActionType.DOUBLE_CLICK, {"x": sx, "y": sy})

        # typewrite('text') or write('text')
        type_match = re.search(r"(?:typewrite|write)\(['\"](.+?)['\"]\)", text)
        if type_match:
            return Action(ActionType.TYPE, {"text": type_match.group(1)})

        # hotkey('key1', 'key2')
        hotkey_match = re.search(r"hotkey\((.+?)\)", text)
        if hotkey_match:
            raw = hotkey_match.group(1).strip("[]")
            keys = [k.strip().strip("'\"") for k in raw.split(",")]
            key_map = {"cmd": "ctrl", "command": "ctrl", "return": "enter"}
            keys = [key_map.get(k.lower(), k) for k in keys]
            return Action(ActionType.KEY_PRESS, {"keys": "+".join(keys)})

        # press('key')
        press_match = re.search(r"press\(['\"](.+?)['\"]\)", text)
        if press_match:
            key = press_match.group(1)
            key_map = {"return": "enter", "cmd": "ctrl"}
            key = key_map.get(key.lower(), key)
            return Action(ActionType.KEY_PRESS, {"keys": key})

        # scroll(amount)
        scroll_match = re.search(r'scroll\((-?\d+)\)', text)
        if scroll_match:
            amount = int(scroll_match.group(1))
            direction = "up" if amount > 0 else "down"
            return Action(ActionType.SCROLL, {"direction": direction, "amount": abs(amount)})

        return None
