"""Gemma4Brain via llama.cpp — fast Metal-accelerated inference on Apple Silicon.

Instead of loading Gemma4 through PyTorch (slow on CPU, MPS allocation limits),
this backend runs llama.cpp as a local OpenAI-compatible server on Metal GPU.

Architecture:
    llama-server (Metal GPU)  ←→  OpenAI API  ←→  Gemma4Brain
         ↑                                            ↑
    GGUF model weights                         Screen frames + task
    (quantized, fast)                          → structured actions

Benefits over transformers backend:
- 10-50x faster inference on Apple Silicon (Metal GPU)
- Lower memory footprint (GGUF quantization)
- OpenAI-compatible API = standard tool calling
- Can run E4B or even 26B-A4B comfortably on 16GB M4

Usage:
    # Start the server:
    llama-server -hf ggml-org/gemma-4-E4B-it-GGUF --port 8080

    # Then use this brain:
    brain = LlamaCppBrain(base_url="http://localhost:8080/v1")
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from io import BytesIO

import requests
from PIL import Image

from .actions import TOOLS, Action, ActionType, parse_tool_call
from .prompts import load_prompt

logger = logging.getLogger(__name__)

# Sourced from mantis_agent.prompts.LLAMACPP_SYSTEM. Override per-tenant via
# MANTIS_PROMPTS_DIR/llamacpp_system.txt.
SYSTEM_PROMPT = load_prompt("llamacpp_system")

# Convert our tool schemas to OpenAI function-calling format
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": tool["function"],
    }
    for tool in TOOLS
]


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


class LlamaCppBrain:
    """Gemma4 brain using llama.cpp server for fast Metal inference.

    Args:
        base_url: URL of the llama-server OpenAI-compatible API.
        model: Model name (passed in API calls, usually ignored by llama-server).
        max_tokens: Maximum tokens to generate per call.
        temperature: Sampling temperature (0 = deterministic).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model: str = "gemma-4",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        use_tool_calling: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = True
        self.use_tool_calling = use_tool_calling

    def load(self) -> None:
        """Verify the llama-server is running."""
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=5)
            resp.raise_for_status()
            models = resp.json()
            logger.info(f"llama-server connected: {models}")
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to llama-server at {self.base_url}. "
                f"Start it with: llama-server -hf ggml-org/gemma-4-E4B-it-GGUF --port 8080\n"
                f"Error: {e}"
            )

    def query(self, prompt: str, response_format: str = "json") -> str:
        """Send a text-only prompt (no images) and return the model's text response.

        Used for structured analysis: plan parsing, classification, extraction.
        Unlike think(), does not send images or expect tool calls.

        Args:
            prompt: Text prompt.
            response_format: "json" for JSON output, "text" for plain text.

        Returns:
            Raw text response from the model.
        """
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        # Don't request tools — we want text output
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
        """Run perception-reasoning-action via llama.cpp OpenAI API."""
        messages = self._build_messages(frames, task, action_history, screen_size)

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.use_tool_calling:
            payload["tools"] = OPENAI_TOOLS
            payload["tool_choice"] = "required"

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"llama-server request failed: {e}")
            return InferenceResult(
                action=Action(ActionType.WAIT, {"seconds": 1.0}, reasoning=str(e)),
                raw_output=str(e),
            )

        return self._parse_response(data)

    def _build_messages(
        self,
        frames: list[Image.Image],
        task: str,
        action_history: list[Action] | None,
        screen_size: tuple[int, int],
    ) -> list[dict]:
        """Build OpenAI-format messages with image content."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # User message with frames as images
        content = []

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
        messages.append({"role": "user", "content": content})

        return messages

    def _parse_response(self, data: dict) -> InferenceResult:
        """Parse OpenAI-format response into InferenceResult."""
        choice = data["choices"][0]
        message = choice["message"]
        raw_output = json.dumps(message)

        # Check for tool calls
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            func = tc.get("function", {})
            name = func.get("name", "wait")
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            thinking = message.get("reasoning_content", "") or message.get("content", "") or ""
            action = parse_tool_call(name, args, reasoning=thinking)
        else:
            # No tool call — try to parse action from text content or thinking
            content = message.get("content", "")
            thinking = message.get("reasoning_content", "") or ""
            # Try content first, then thinking (model may put action in reasoning)
            action_text = content if content.strip() else thinking
            action = self._parse_text_action(action_text)
            if not thinking:
                thinking = content

        return InferenceResult(
            action=action,
            raw_output=raw_output,
            thinking=thinking,
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
        )

    @staticmethod
    def _parse_text_action(text: str) -> Action:
        """Parse action from text output (for models without tool calling).

        Handles formats:
          Action: click({"x": 500, "y": 300})
          click(x=500, y=300)
          pyautogui.click(500, 300)
          type_text({"text": "hello"})
          done({"success": true})
          {"action": "click", "x": 500, "y": 300}       (raw JSON)
          ```json\n{"action": "click", ...}\n```          (fenced JSON)
        """
        import re

        # Strip markdown code fences
        cleaned = re.sub(r'```(?:json|python|)\s*\n?', '', text)
        cleaned = re.sub(r'\n?```', '', cleaned).strip()

        # Try Gemma4 tool-call format: Action: name({"key": value})
        tool_match = re.search(r'(?:Action:\s*)?(\w+)\(\s*(\{.*?\})\s*\)', cleaned, re.DOTALL)
        if tool_match:
            name = tool_match.group(1)
            try:
                args = json.loads(tool_match.group(2))
                return parse_tool_call(name, args)
            except json.JSONDecodeError:
                pass

        # Try raw JSON: {"action": "click", "x": N, "y": N}
        # Also handles nested: {"action": "key_press", "parameters": {"keys": "alt+left"}}
        json_match = re.search(r'\{.*"action"\s*:\s*"(\w+)".*\}', cleaned, re.DOTALL)
        if json_match:
            try:
                obj = json.loads(json_match.group(0))
                action_name = obj.pop("action", "wait")
                # Flatten nested "parameters"/"arguments" into top-level
                for nested_key in ("parameters", "arguments", "params"):
                    if nested_key in obj and isinstance(obj[nested_key], dict):
                        nested = obj.pop(nested_key)
                        obj.update(nested)
                # Map JSON action names to our tool names
                name_map = {
                    "click": "click", "left_click": "click",
                    "type": "type_text", "type_text": "type_text",
                    "scroll": "scroll", "key": "key_press", "key_press": "key_press",
                    "done": "done", "terminate": "done",
                    "double_click": "double_click", "wait": "wait",
                }
                mapped = name_map.get(action_name, action_name)
                # Normalize key variants
                if "key" in obj and "keys" not in obj:
                    obj["keys"] = obj.pop("key")
                return parse_tool_call(mapped, obj)
            except (json.JSONDecodeError, KeyError):
                pass

        # Try pyautogui format: pyautogui.click(x=500, y=300)
        click_match = re.search(r'click\((?:x=)?(\d+),\s*(?:y=)?(\d+)\)', cleaned)
        if click_match:
            return Action(ActionType.CLICK, {"x": int(click_match.group(1)), "y": int(click_match.group(2))})

        type_match = re.search(r'(?:type_text|typewrite|write)\([\'"](.+?)[\'"]\)', cleaned)
        if type_match:
            return Action(ActionType.TYPE, {"text": type_match.group(1)})

        key_match = re.search(r'(?:key_press|press|hotkey)\([\'"](.+?)[\'"]\)', cleaned)
        if key_match:
            return Action(ActionType.KEY_PRESS, {"keys": key_match.group(1)})

        scroll_match = re.search(r'scroll\(.*?[\'"](\w+)[\'"]', cleaned)
        if scroll_match:
            return Action(ActionType.SCROLL, {"direction": scroll_match.group(1), "amount": 3})

        if re.search(r'done|DONE|terminate.*success', cleaned, re.IGNORECASE):
            return Action(ActionType.DONE, {"success": True, "summary": cleaned[:200]})

        if re.search(r'fail|FAIL|terminate.*fail', cleaned, re.IGNORECASE):
            return Action(ActionType.DONE, {"success": False, "summary": cleaned[:200]})

        logger.warning(f"No action parsed from text: {cleaned[:100]}")
        return Action(ActionType.WAIT, {"seconds": 1.0}, reasoning=cleaned[:100])
