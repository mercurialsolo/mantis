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
- To fill a form: click the input field ONCE to focus it, then call type_text() with the value.
- Do NOT click an input field multiple times. One click focuses it — then immediately type.
- After typing, move to the next field: click the next input, or press key_press('Tab').
- To submit a form: press key_press('Enter') — this is the most reliable method. Only click the Submit button if Enter doesn't work.
- If you already clicked a field and see it is focused, your NEXT action must be type_text() — not another click.

# Avoiding loops
- NEVER repeat the same action more than twice. If clicking the same spot twice doesn't work, try a different approach.
- If you're stuck: try scrolling, pressing Tab, clicking a different element, or using keyboard shortcuts.
- Read the task description carefully — it tells you what value to type and where.

# Completion
- When the task is complete, call done(success=true, summary="...").
- If stuck after multiple attempts, call done(success=false, summary="...").

# Waiting
- If a page is loading or animating, call wait() to observe the result.
- After submitting a form, call wait(seconds=2) before checking the result.\
"""

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
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_name = model  # For compatibility with OSWorld adapter
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = True  # For compatibility

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
            "tools": OPENAI_TOOLS,
            "tool_choice": "required",
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
            # llama.cpp puts thinking in reasoning_content, not content
            thinking = message.get("reasoning_content", "") or message.get("content", "") or ""
            action = parse_tool_call(name, args, reasoning=thinking)
        else:
            # No tool call — extract from content
            content = message.get("content", "")
            logger.warning(f"No tool call, treating as wait: {content[:200]}")
            action = Action(
                ActionType.WAIT, {"seconds": 1.0}, reasoning=content
            )
            thinking = content

        return InferenceResult(
            action=action,
            raw_output=raw_output,
            thinking=thinking,
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
        )
