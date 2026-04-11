"""CUA Brain — specialized vision models for browser automation.

Supports OpenCUA and EvoCUA families — fine-tuned Qwen VL models
trained specifically for computer use. Both output pyautogui actions
and understand GUI elements natively.

OpenCUA (xlang-ai): 7B/32B/72B, based on Qwen2.5-VL
EvoCUA (Meituan):   8B/32B, based on Qwen3-VL (SOTA: 56.7% OSWorld)

Backend: vLLM with OpenAI-compatible API

Key differences from LlamaCppBrain (Gemma4):
- No tool_choice="required" — model outputs actions as text, not tool calls
- Coordinate system uses smart-resize (model coords → screen coords)
- Long chain-of-thought reasoning built into the model
- Multi-image context (up to 3 screenshots for history)

References:
  https://huggingface.co/xlangai/OpenCUA-32B
  https://huggingface.co/meituan/EvoCUA-32B-20260105
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from io import BytesIO

import requests
from PIL import Image

from .actions import Action, ActionType, parse_tool_call

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a computer use agent. You observe screenshots and perform actions to complete tasks.

You receive one or more screenshots. The LAST image is the current screen state.

Your job:
1. OBSERVE the current screen carefully
2. REASON step by step about what to do next
3. Output exactly ONE action as a pyautogui command

Available actions:
- pyautogui.click(x=<int>, y=<int>) — click at coordinates
- pyautogui.doubleClick(x=<int>, y=<int>) — double click
- pyautogui.typewrite('<text>') — type text
- pyautogui.hotkey('<key1>', '<key2>') — press key combo
- pyautogui.press('<key>') — press single key
- pyautogui.scroll(<amount>) — scroll (negative = down)
- DONE — task is complete
- FAIL — task cannot be completed

Rules:
- Coordinates are absolute pixels on the screenshot
- After clicking an input field, use typewrite() to enter text
- Do NOT click the same element repeatedly — try a different approach
- Use hotkey('ctrl', 'a') to select all, then typewrite() to replace text
- After filling a form, use press('enter') or click the submit button\
"""


def _image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _smart_resize(height: int, width: int, factor: int = 28,
                   min_pixels: int = 3136, max_pixels: int = 12845056) -> tuple[int, int]:
    """Qwen2.5-VL smart resize — matches OpenCUA's coordinate system."""
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
    """Convert OpenCUA's model coordinates to actual screen coordinates."""
    resized_h, resized_w = _smart_resize(screen_height, screen_width)
    rel_x = model_x / resized_w
    rel_y = model_y / resized_h
    abs_x = int(rel_x * screen_width)
    abs_y = int(rel_y * screen_height)
    return abs_x, abs_y


@dataclass
class InferenceResult:
    """Result from a single brain inference cycle."""
    action: Action
    raw_output: str
    thinking: str = ""
    tokens_used: int = 0


class OpenCUABrain:
    """OpenCUA brain via vLLM OpenAI-compatible API.

    Args:
        base_url: URL of the vLLM server.
        model: Model name (as served by vLLM).
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        screen_size: Default screen size for coordinate conversion.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "opencua-7b",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        screen_size: tuple[int, int] = (1280, 720),
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.default_screen_size = screen_size
        self.enable_thinking = True

    def load(self) -> None:
        """Verify the vLLM server is running."""
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=10)
            resp.raise_for_status()
            models = resp.json()
            logger.info(f"vLLM server connected: {models}")
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to vLLM at {self.base_url}. "
                f"Start with: vllm serve xlangai/OpenCUA-7B --trust-remote-code\n"
                f"Error: {e}"
            )

    def think(
        self,
        frames: list[Image.Image],
        task: str,
        action_history: list[Action] | None = None,
        screen_size: tuple[int, int] = (1920, 1080),
    ) -> InferenceResult:
        """Run perception-reasoning-action via OpenCUA."""
        messages = self._build_messages(frames, task, action_history, screen_size)

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"vLLM request failed: {e}")
            return InferenceResult(
                action=Action(ActionType.WAIT, {"seconds": 1.0}),
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
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        content = []
        n_frames = len(frames)
        for i, frame in enumerate(frames):
            label = "CURRENT" if i == n_frames - 1 else f"t-{n_frames - 1 - i}"
            content.append({"type": "text", "text": f"[Frame {label}]"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_image_to_base64(frame)}"},
            })

        context_parts = [
            f"\nTask: {task}",
            f"Screen size: {screen_size[0]}x{screen_size[1]} pixels",
        ]
        if action_history:
            recent = action_history[-10:]
            history_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(recent))
            context_parts.append(f"Recent actions:\n{history_str}")

        content.append({"type": "text", "text": "\n".join(context_parts)})
        messages.append({"role": "user", "content": content})

        return messages

    def _parse_response(self, data: dict, screen_size: tuple[int, int]) -> InferenceResult:
        """Parse OpenCUA's text response into an Action."""
        choice = data["choices"][0]
        message = choice["message"]
        text = message.get("content", "")
        raw_output = text

        # Extract thinking (everything before the action)
        thinking = ""
        action_text = text

        # OpenCUA often outputs reasoning then a pyautogui command
        pyautogui_match = re.search(r'pyautogui\.\w+\(.*?\)', text, re.DOTALL)
        if pyautogui_match:
            thinking = text[:pyautogui_match.start()].strip()
            action_text = pyautogui_match.group(0)

        # Parse the action
        action = self._parse_pyautogui(action_text, screen_size)

        # Check for DONE/FAIL
        if "DONE" in text.upper() and not pyautogui_match:
            action = Action(ActionType.DONE, {"success": True, "summary": thinking[:200]})
        elif "FAIL" in text.upper() and not pyautogui_match:
            action = Action(ActionType.DONE, {"success": False, "summary": thinking[:200]})

        return InferenceResult(
            action=action,
            raw_output=raw_output,
            thinking=thinking,
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
        )

    def _parse_pyautogui(self, text: str, screen_size: tuple[int, int]) -> Action:
        """Parse a pyautogui command into a Mantis Action."""

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

        # typewrite('text')
        type_match = re.search(r"typewrite\(['\"](.+?)['\"]\)", text)
        if type_match:
            return Action(ActionType.TYPE, {"text": type_match.group(1)})

        # hotkey('key1', 'key2')
        hotkey_match = re.search(r"hotkey\((.+?)\)", text)
        if hotkey_match:
            keys = [k.strip().strip("'\"") for k in hotkey_match.group(1).split(",")]
            return Action(ActionType.KEY_PRESS, {"keys": "+".join(keys)})

        # press('key')
        press_match = re.search(r"press\(['\"](.+?)['\"]\)", text)
        if press_match:
            return Action(ActionType.KEY_PRESS, {"keys": press_match.group(1)})

        # scroll(amount)
        scroll_match = re.search(r'scroll\((-?\d+)\)', text)
        if scroll_match:
            amount = int(scroll_match.group(1))
            direction = "up" if amount > 0 else "down"
            return Action(ActionType.SCROLL, {"direction": direction, "amount": abs(amount)})

        # Fallback
        return Action(ActionType.WAIT, {"seconds": 1.0}, reasoning=f"Could not parse: {text[:100]}")
