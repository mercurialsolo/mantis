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
You are a computer use agent performing multi-step browser workflows. You observe screenshots and output exactly ONE action per turn.

Available actions:
- pyautogui.click(x=<int>, y=<int>) — click at coordinates (CENTER of target element)
- pyautogui.doubleClick(x=<int>, y=<int>) — double click
- pyautogui.typewrite('<text>') — type text into the currently focused field
- pyautogui.hotkey('<key1>', '<key2>') — press key combo (e.g. ctrl+a, alt+left)
- pyautogui.press('<key>') — press single key (enter, tab, backspace, escape)
- pyautogui.scroll(<amount>) — scroll (negative = down, positive = up)
- terminate('success') — task complete, include ALL results in the message
- terminate('failure') — task cannot be completed

Core rules:
- Click the CENTER of target elements precisely
- After clicking an input field, IMMEDIATELY use typewrite() — do NOT click again
- NEVER repeat the same action more than twice — try a different approach
- Press tab to move between form fields, enter to submit forms

Browser navigation:
- hotkey('alt', 'left') — go back
- hotkey('ctrl', 'w') — close current tab
- hotkey('ctrl', 'tab') — switch tabs
- scroll(-5) to see more content below

Data extraction:
- Read ALL text visually from the screenshot
- Phone numbers: (555) 555-5555, 555-555-5555, or 10+ consecutive digits
- Read prices, years, makes, models from page titles and content
- Read the current URL from the browser address bar
- When reporting results, include EVERY piece of extracted data\
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
        """Verify the vLLM server is running. Waits for cold start."""
        for attempt in range(36):  # Wait up to 6 minutes for cold start (vLLM 32B loading)
            try:
                resp = requests.get(f"{self.base_url}/models", timeout=30)
                resp.raise_for_status()
                models = resp.json()
                logger.info(f"vLLM server connected: {models}")
                return
            except Exception as e:
                if attempt < 35:
                    logger.info(f"Brain not ready (attempt {attempt+1}/36), waiting 10s...")
                    import time
                    time.sleep(10)
                else:
                    raise RuntimeError(
                        f"Cannot connect to vLLM at {self.base_url} after 2 minutes.\n"
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
                json=payload,
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
        """Parse OpenCUA's text response into an Action.

        EvoCUA sometimes outputs JSON actions instead of pyautogui format,
        and may wrap them in markdown fences. We try multiple parse strategies:
        1. pyautogui.foo() format (native OpenCUA)
        2. JSON {"action": "click", "x": N, "y": N} format (EvoCUA variant)
        3. terminate() signals
        4. DONE/FAIL keywords
        """
        choice = data["choices"][0]
        message = choice["message"]
        text = message.get("content", "")
        raw_output = text

        # Strip markdown code fences before parsing
        cleaned = self._strip_code_fences(text)

        # Extract thinking (everything before the action)
        thinking = ""
        action_text = cleaned
        action = None

        # Strategy 1: pyautogui format
        pyautogui_match = re.search(r'pyautogui\.\w+\(.*?\)', cleaned, re.DOTALL)
        if pyautogui_match:
            thinking = cleaned[:pyautogui_match.start()].strip()
            action_text = pyautogui_match.group(0)
            action = self._parse_pyautogui(action_text, screen_size)

        # Strategy 2: JSON action format (EvoCUA variant)
        if action is None or action.action_type == ActionType.WAIT:
            json_action = self._parse_json_action(cleaned, screen_size)
            if json_action is not None:
                # Extract thinking as everything before the JSON block
                json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', cleaned)
                if json_match:
                    thinking = cleaned[:json_match.start()].strip()
                action = json_action

        # Strategy 3: terminate() signals
        if action is None or action.action_type == ActionType.WAIT:
            term_match = re.search(r"terminate\(['\"](\w+)['\"]\)", cleaned)
            if term_match:
                success = term_match.group(1).lower() == "success"
                thinking = cleaned[:term_match.start()].strip()
                action = Action(ActionType.DONE, {"success": success, "summary": thinking[:200]})

        # Strategy 4: DONE/FAIL keywords (last resort)
        if action is None or action.action_type == ActionType.WAIT:
            if not pyautogui_match:
                if "DONE" in text.upper():
                    action = Action(ActionType.DONE, {"success": True, "summary": thinking[:200]})
                elif "FAIL" in text.upper():
                    action = Action(ActionType.DONE, {"success": False, "summary": thinking[:200]})

        # Final fallback
        if action is None:
            action = Action(ActionType.WAIT, {"seconds": 1.0}, reasoning=f"Could not parse: {text[:100]}")

        return InferenceResult(
            action=action,
            raw_output=raw_output,
            thinking=thinking,
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Strip markdown code fences (```json ... ```) from model output."""
        # Remove ```json ... ``` or ``` ... ``` blocks, keeping inner content
        stripped = re.sub(r'```(?:json|python|)\s*\n?', '', text)
        stripped = re.sub(r'\n?```', '', stripped)
        return stripped

    def _parse_json_action(self, text: str, screen_size: tuple[int, int]) -> Action | None:
        """Parse JSON-formatted actions: {"action": "click", "x": N, "y": N}.

        EvoCUA sometimes outputs actions in this format instead of pyautogui.
        """
        # Find JSON object containing "action" key
        match = re.search(r'\{[^{}]*"action"\s*:\s*"[^"]*"[^{}]*\}', text)
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

        if action_type in ("type", "typewrite", "write"):
            return Action(ActionType.TYPE, {"text": obj.get("text", "")})

        if action_type in ("key", "press", "hotkey"):
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

        return None

    def _parse_pyautogui(self, text: str, screen_size: tuple[int, int]) -> Action | None:
        """Parse a pyautogui command into a Mantis Action.

        Returns None instead of WAIT fallback so callers can try other strategies.
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

        # hotkey('key1', 'key2') or hotkey(['key1', 'key2'])
        hotkey_match = re.search(r"hotkey\((.+?)\)", text)
        if hotkey_match:
            raw = hotkey_match.group(1)
            # Strip list brackets if present
            raw = raw.strip("[]")
            keys = [k.strip().strip("'\"") for k in raw.split(",")]
            # Normalize key names
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

        # No pyautogui match — return None so caller can try other strategies
        return None
