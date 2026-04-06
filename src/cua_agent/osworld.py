"""OSWorld benchmark adapter.

Plugs our Gemma4-based streaming CUA agent directly into OSWorld's evaluation
harness. Implements the exact interface expected by OSWorld's `lib_run_single.py`:

    agent.reset(runtime_logger=None, vm_ip=None)
    response, actions = agent.predict(instruction, obs)

Where `obs` contains raw PNG screenshot bytes, and `actions` is a list of
pyautogui code strings.

Key advantage over decoupled agents (Agent-S, PromptAgent):
- Single model (Gemma4) does perception + reasoning + action in one pass
- Frame history gives temporal context across steps
- Native bounding box + function calling — no regex parsing

Reference: https://github.com/xlang-ai/OSWorld
"""

from __future__ import annotations

import base64
import logging
import time
from io import BytesIO

from PIL import Image

from .actions import Action, ActionType
from .brain import Gemma4Brain

logger = logging.getLogger("desktopenv.agent")


def _screenshot_bytes_to_pil(screenshot_bytes: bytes) -> Image.Image:
    """Convert raw PNG bytes from OSWorld VM to PIL Image."""
    return Image.open(BytesIO(screenshot_bytes))


def _action_to_pyautogui_code(action: Action) -> str:
    """Convert our Action to pyautogui code string for OSWorld execution."""
    match action.action_type:
        case ActionType.CLICK:
            x, y = action.params["x"], action.params["y"]
            button = action.params.get("button", "left")
            if button == "left":
                return f"pyautogui.click({x}, {y})"
            return f"pyautogui.click({x}, {y}, button='{button}')"

        case ActionType.DOUBLE_CLICK:
            x, y = action.params["x"], action.params["y"]
            return f"pyautogui.doubleClick({x}, {y})"

        case ActionType.TYPE:
            text = action.params["text"].replace("'", "\\'")
            return f"import time; pyautogui.typewrite('{text}', interval=0.02)"

        case ActionType.KEY_PRESS:
            keys_str = action.params["keys"]
            parts = [k.strip() for k in keys_str.split("+")]
            if len(parts) == 1:
                return f"pyautogui.press('{parts[0]}')"
            keys_repr = ", ".join(f"'{k}'" for k in parts)
            return f"pyautogui.hotkey({keys_repr})"

        case ActionType.SCROLL:
            direction = action.params["direction"]
            amount = action.params.get("amount", 3)
            x = action.params.get("x")
            y = action.params.get("y")
            clicks = amount if direction == "up" else -amount
            if x is not None and y is not None:
                return f"pyautogui.scroll({clicks}, x={x}, y={y})"
            return f"pyautogui.scroll({clicks})"

        case ActionType.DRAG:
            sx, sy = action.params["start_x"], action.params["start_y"]
            ex, ey = action.params["end_x"], action.params["end_y"]
            return f"pyautogui.moveTo({sx}, {sy}); pyautogui.drag({ex - sx}, {ey - sy}, duration=0.5)"

        case ActionType.WAIT:
            return "WAIT"

        case ActionType.DONE:
            success = action.params.get("success", False)
            return "DONE" if success else "FAIL"

    return "WAIT"


def _action_to_computer13(action: Action) -> dict:
    """Convert our Action to OSWorld's computer_13 format."""
    match action.action_type:
        case ActionType.CLICK:
            return {
                "action_type": "CLICK",
                "x": action.params["x"],
                "y": action.params["y"],
            }
        case ActionType.DOUBLE_CLICK:
            return {
                "action_type": "DOUBLE_CLICK",
                "x": action.params["x"],
                "y": action.params["y"],
            }
        case ActionType.TYPE:
            return {
                "action_type": "TYPING",
                "text": action.params["text"],
            }
        case ActionType.KEY_PRESS:
            keys_str = action.params["keys"]
            parts = [k.strip() for k in keys_str.split("+")]
            if len(parts) == 1:
                return {"action_type": "PRESS", "key": parts[0]}
            return {"action_type": "HOTKEY", "keys": parts}
        case ActionType.SCROLL:
            direction = action.params["direction"]
            amount = action.params.get("amount", 3)
            dy = amount if direction == "up" else -amount
            dx = amount if direction == "right" else (-amount if direction == "left" else 0)
            if direction in ("up", "down"):
                dx = 0
            else:
                dy = 0
            return {"action_type": "SCROLL", "dx": dx, "dy": dy}
        case ActionType.DRAG:
            return {
                "action_type": "DRAG_TO",
                "x": action.params["end_x"],
                "y": action.params["end_y"],
            }
        case ActionType.WAIT:
            return "WAIT"
        case ActionType.DONE:
            return "DONE" if action.params.get("success", False) else "FAIL"

    return "WAIT"


class Gemma4Agent:
    """OSWorld-compatible agent powered by Gemma4.

    Drop-in replacement for OSWorld's PromptAgent. Plugs directly into
    their evaluation harness via run.py.

    The key difference: instead of separate grounding + reasoning models,
    Gemma4 does all three (perceive, reason, act) in a single forward pass.
    Frame history across steps gives temporal context that screenshot-only
    agents lack.
    """

    def __init__(
        self,
        model: str = "google/gemma-4-E4B-it",
        max_tokens: int = 1024,
        top_p: float = 0.9,
        temperature: float = 0.0,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        max_trajectory_length: int = 5,
        enable_thinking: bool = True,
        quantize_4bit: bool = False,
        platform: str = "ubuntu",
        backend: str = "auto",  # "transformers", "llamacpp", or "auto"
        llamacpp_url: str = "http://localhost:8080/v1",
        **kwargs,
    ):
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.platform = platform

        # Choose backend: llama.cpp for Apple Silicon, transformers for CUDA
        if backend == "auto":
            import platform as _platform
            import torch
            if _platform.system() == "Darwin" or not torch.cuda.is_available():
                backend = "llamacpp"
            else:
                backend = "transformers"

        if backend == "llamacpp":
            from .brain_llamacpp import LlamaCppBrain
            self.brain = LlamaCppBrain(
                base_url=llamacpp_url,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            self.brain = Gemma4Brain(
                model_name=model,
                enable_thinking=enable_thinking,
                max_new_tokens=max_tokens,
                quantize_4bit=quantize_4bit,
            )

        # State across steps within a single task
        self._frame_history: list[Image.Image] = []
        self._action_history: list[Action] = []
        self._step_count = 0

        # OSWorld compatibility
        self.thoughts: list[str] = []
        self.actions: list[str] = []
        self.observations: list[dict] = []

    def load(self) -> None:
        """Load the Gemma4 model. Call once before evaluation."""
        self.brain.load()

    def reset(self, *args, **kwargs) -> None:
        """Reset agent state between OSWorld tasks.

        Accepts arbitrary args for compatibility with OSWorld's harness
        which may pass runtime_logger and vm_ip.
        """
        self._frame_history.clear()
        self._action_history.clear()
        self._step_count = 0
        self.thoughts.clear()
        self.actions.clear()
        self.observations.clear()

    def predict(self, instruction: str, obs: dict) -> tuple[str, list]:
        """Predict the next action(s) based on the current observation.

        This is the exact interface OSWorld's lib_run_single.py calls:
            response, actions = agent.predict(instruction, obs)

        Args:
            instruction: The task description from OSWorld.
            obs: Dict with 'screenshot' (raw PNG bytes), optionally
                 'accessibility_tree' (XML string).

        Returns:
            Tuple of (response_text, list_of_actions).
            Actions are pyautogui code strings or computer_13 dicts.
        """
        self._step_count += 1

        # Convert raw PNG bytes to PIL Image
        screenshot = _screenshot_bytes_to_pil(obs["screenshot"])
        self._frame_history.append(screenshot)

        # Trim frame history to max trajectory length
        if len(self._frame_history) > self.max_trajectory_length:
            self._frame_history = self._frame_history[-self.max_trajectory_length:]

        # Run unified inference: perceive + reason + act in one model call
        t0 = time.time()
        result = self.brain.think(
            frames=self._frame_history,
            task=instruction,
            action_history=self._action_history,
            screen_size=(1920, 1080),
        )
        inference_time = time.time() - t0

        action = result.action
        self._action_history.append(action)

        # Build response text (thinking + action description)
        response = ""
        if result.thinking:
            response += result.thinking + "\n\n"
        response += f"Action: {action}"

        # Convert action to OSWorld format
        if self.action_space == "pyautogui":
            code = _action_to_pyautogui_code(action)
            actions_out = [code]
        else:
            act = _action_to_computer13(action)
            actions_out = [act]

        # Track for OSWorld's trajectory
        self.thoughts.append(response)
        self.actions.append(str(actions_out))
        self.observations.append({
            "screenshot": base64.b64encode(obs["screenshot"]).decode("utf-8"),
            "accessibility_tree": obs.get("accessibility_tree"),
        })

        logger.info(
            f"Step {self._step_count}: {action} ({inference_time:.2f}s)"
        )

        return response, actions_out
