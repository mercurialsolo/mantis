"""Gemma4Brain — unified perception, reasoning, and action in a single model.

This is the core innovation: instead of separate grounding + reasoning models
(like Agent-S's UI-TARS + GPT-5), Gemma4 does all three in one forward pass:
  1. PERCEIVE — understand what's on screen from the frame buffer
  2. REASON — plan the next action given the task and history
  3. ACT — output a structured tool call (click, type, scroll, etc.)

The model receives recent screen frames as a sequence of images, giving it
temporal context to understand transitions, loading states, and the effects
of its previous actions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

from .actions import TOOLS, Action, ActionType, parse_tool_call
from .prompts import load_prompt

logger = logging.getLogger(__name__)

# Model size recommendations for OSWorld:
# - E2B (2.3B): Fast but may miss complex UI elements. Good for simple tasks.
# - E4B (4.5B): Best balance for real-time CUA. Recommended default.
# - 26B-A4B (MoE): Only 4B active — fast with deep reasoning. Best for OSWorld.
# - 31B: Maximum accuracy, but slower. Use if latency isn't critical.
DEFAULT_MODEL = "google/gemma-4-E4B-it"

# Optional override: set MANTIS_GEMMA4_LOCAL_PATH to a directory containing a
# pre-downloaded Gemma4 checkpoint to skip the HuggingFace download. Empty by
# default so the public surface has no developer-machine paths baked in.
_LOCAL_MODEL_PATHS = [
    p for p in (os.environ.get("MANTIS_GEMMA4_LOCAL_PATH", "").strip(),) if p
]

def _resolve_model_path(model_name: str) -> str:
    """Resolve model name to local path if available."""
    # If it's already a local path, use it
    if os.path.isdir(model_name):
        return model_name
    # Check known local paths for default models
    if model_name == DEFAULT_MODEL:
        for path in _LOCAL_MODEL_PATHS:
            if os.path.isdir(path):
                logger.info(f"Using local model at {path}")
                return path
    return model_name

# Sourced from mantis_agent.prompts.GEMMA4_SYSTEM. Override per-tenant via
# MANTIS_PROMPTS_DIR/gemma4_system.txt.
SYSTEM_PROMPT = load_prompt("gemma4_system")


@dataclass
class InferenceResult:
    """Result from a single brain inference cycle."""

    action: Action
    raw_output: str
    thinking: str = ""
    tokens_used: int = 0


class Gemma4Brain:
    """Unified perception-reasoning-action engine powered by Gemma4.

    Args:
        model_name: HuggingFace model ID. See DEFAULT_MODEL.
        device: Device to load model on. "auto" uses device_map.
        enable_thinking: Enable Gemma4's extended thinking mode for complex
                        reasoning. Slower but better for multi-step plans.
        max_new_tokens: Maximum tokens to generate per inference call.
        torch_dtype: Model precision. bfloat16 recommended for speed.
        quantize_4bit: Enable 4-bit quantization to reduce memory usage.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        enable_thinking: bool = True,
        max_new_tokens: int = 1024,
        torch_dtype: torch.dtype = torch.float16,
        quantize_4bit: bool = False,
    ):
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._dtype = torch_dtype
        self._quantize_4bit = quantize_4bit

        # Determine device: CUDA if available, otherwise CPU
        # MPS has allocation limits that prevent loading full models
        if device is not None:
            self._device = device
        elif torch.cuda.is_available():
            self._device = "auto"
        else:
            self._device = None  # CPU — let from_pretrained default

    def load(self) -> None:
        """Load the model and processor. Call once before inference."""
        resolved_path = _resolve_model_path(self.model_name)
        logger.info(f"Loading {resolved_path}...")

        load_kwargs: dict = {
            "dtype": self._dtype,
        }

        if self._quantize_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._dtype,
            )

        if self._device is not None:
            load_kwargs["device_map"] = self._device

        self._model = AutoModelForMultimodalLM.from_pretrained(
            resolved_path, **load_kwargs
        )

        # Load processor from HuggingFace hub if local path is missing chat_template.
        # Local model copies (e.g. from Kaggle) often lack the chat_template
        # required for tool calling. The HF hub version has it.
        try:
            self._processor = AutoProcessor.from_pretrained(resolved_path)
            # Verify chat template is present
            self._processor.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tokenize=False, add_generation_prompt=True,
            )
        except (ValueError, Exception):
            logger.info("Local processor missing chat_template, loading from HuggingFace hub...")
            self._processor = AutoProcessor.from_pretrained(self.model_name)

        logger.info(f"Model loaded on {self._model.device}")

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Call brain.load() before inference")
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            raise RuntimeError("Call brain.load() before inference")
        return self._processor

    def think(
        self,
        frames: list[Image.Image],
        task: str,
        action_history: list[Action] | None = None,
        screen_size: tuple[int, int] = (1920, 1080),
    ) -> InferenceResult:
        """Run a single perception-reasoning-action cycle.

        This is where the magic happens: one model call that sees the screen,
        understands the task, and outputs a structured action.

        Args:
            frames: Recent screen frames (oldest first, last = current state).
            task: The high-level task description.
            action_history: Previous actions taken (for context).
            screen_size: Actual screen dimensions for coordinate context.

        Returns:
            InferenceResult with the chosen action and reasoning.
        """
        messages = self._build_messages(frames, task, action_history, screen_size)

        inputs = self.processor.apply_chat_template(
            messages,
            tools=TOOLS,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking,
            add_generation_prompt=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Deterministic for reproducibility
            )

        # Decode only the generated tokens (strip input)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        raw_output = self.processor.decode(generated_ids, skip_special_tokens=False)

        # Parse the structured response
        parsed = self.processor.parse_response(raw_output)

        return self._build_result(parsed, raw_output)

    def _build_messages(
        self,
        frames: list[Image.Image],
        task: str,
        action_history: list[Action] | None,
        screen_size: tuple[int, int],
    ) -> list[dict]:
        """Build the chat messages with frames as visual context.

        Note: Gemma4's apply_chat_template expects user messages with image
        content as list[dict], but system messages cause issues when the
        processor iterates content looking for images. We fold the system
        prompt into the user message text instead.
        """
        # Build the user message with system prompt + frames + task
        content: list[dict] = []

        # System instructions as leading text (not a separate system message)
        content.append({"type": "text", "text": SYSTEM_PROMPT})

        # Add frames — temporal context for the model
        n_frames = len(frames)
        for i, frame in enumerate(frames):
            label = "CURRENT" if i == n_frames - 1 else f"t-{n_frames - 1 - i}"
            content.append({"type": "text", "text": f"[Frame {label}]"})
            content.append({"type": "image", "image": frame})

        # Task and context
        context_parts = [
            f"\nTask: {task}",
            f"Screen size: {screen_size[0]}x{screen_size[1]} pixels",
        ]

        if action_history:
            recent = action_history[-10:]  # Last 10 actions for context
            history_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(recent))
            context_parts.append(f"Recent actions:\n{history_str}")

        content.append({"type": "text", "text": "\n".join(context_parts)})

        return [{"role": "user", "content": content}]

    def _build_result(self, parsed: dict, raw_output: str) -> InferenceResult:
        """Convert parsed model output into an InferenceResult."""
        thinking = parsed.get("thinking", "")
        content = parsed.get("content", "")

        # Check for tool calls in the parsed output
        # Gemma4 returns: {'tool_calls': [{'type': 'function', 'function': {'name': ..., 'arguments': ...}}]}
        tool_calls = parsed.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]  # Take the first tool call
            # Handle nested 'function' key from Gemma4's response format
            if "function" in tc:
                func = tc["function"]
                name = func.get("name", "wait")
                args = func.get("arguments", {})
            else:
                name = tc.get("name", "wait")
                args = tc.get("arguments", {})
            action = parse_tool_call(name, args, reasoning=thinking or content)
        else:
            # No tool call — model just produced text. Treat as observation/wait.
            logger.warning(f"No tool call in output, treating as wait: {content[:200]}")
            action = Action(
                ActionType.WAIT,
                {"seconds": 1.0},
                reasoning=content or "No action determined",
            )

        return InferenceResult(
            action=action,
            raw_output=raw_output,
            thinking=thinking,
        )
