"""Agent-S3 Brain — wraps Simular AI's Agent-S framework as a Mantis brain.

Agent-S3 is a CUA framework with:
- Dedicated grounding model (UI-TARS) for precise element location
- Reflection agent for detecting stuck states
- Best-of-N rollouts (72.6% OSWorld with BoN)
- Support for multiple LLM backends (OpenAI, vLLM, HuggingFace)

This adapter wraps Agent-S3's agent loop behind our Brain protocol,
so it can be used as a drop-in replacement for Gemma4Brain, OpenCUABrain,
or any other brain in the GymRunner.

Architecture:
    GymRunner → AgentSBrain.think(frames, task)
        → Agent-S3 WorkerAgent (reasoning)
        → Agent-S3 OSWorldACI (grounding)
        → Returns Mantis Action

Requires: pip install gui-agents

Reference: https://github.com/simular-ai/agent-s
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from .actions import Action, ActionType

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from Agent-S3 inference."""
    action: Action
    raw_output: str
    thinking: str = ""
    tokens_used: int = 0


def _image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_pyautogui_action(text: str, screen_size: tuple[int, int]) -> Action:
    """Parse Agent-S3's pyautogui output into a Mantis Action.

    Agent-S3 outputs actions as pyautogui commands:
      pyautogui.click(x=500, y=300)
      pyautogui.typewrite('hello')
      pyautogui.hotkey('ctrl', 'a')
    """
    # click(x=N, y=N) or click(N, N)
    click_match = re.search(r'click\((?:x=)?(\d+),\s*(?:y=)?(\d+)\)', text)
    if click_match:
        x, y = int(click_match.group(1)), int(click_match.group(2))
        return Action(ActionType.CLICK, {"x": x, "y": y})

    # doubleClick
    dbl_match = re.search(r'doubleClick\((?:x=)?(\d+),\s*(?:y=)?(\d+)\)', text)
    if dbl_match:
        x, y = int(dbl_match.group(1)), int(dbl_match.group(2))
        return Action(ActionType.DOUBLE_CLICK, {"x": x, "y": y})

    # typewrite('text') or write('text') or type('text')
    type_match = re.search(r"(?:typewrite|write|type)\(['\"](.+?)['\"]\)", text)
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

    # terminate/done/fail
    if re.search(r'terminate|DONE|FAIL', text, re.IGNORECASE):
        success = "success" in text.lower() or "done" in text.lower()
        return Action(ActionType.DONE, {"success": success, "summary": text[:200]})

    return Action(ActionType.WAIT, {"seconds": 1.0}, reasoning=f"Unparsed: {text[:100]}")


class AgentSBrain:
    """Brain adapter that wraps Agent-S3's CUA framework.

    Can run in two modes:
    1. Full Agent-S3: Uses gui-agents package with grounding model
    2. API-only: Uses Agent-S3's prompting strategy via raw LLM API

    Args:
        worker_model: Model for the worker agent (reasoning).
            Can be "gpt-4o", "claude-sonnet-4-20250514", or a vLLM endpoint model name.
        worker_provider: "openai", "anthropic", "vllm", "ollama"
        worker_base_url: Base URL for vLLM/ollama worker endpoint.
        grounding_model: Model for visual grounding (element location).
            Typically "ui-tars-1.5-7b" via vLLM.
        grounding_provider: "huggingface", "vllm"
        grounding_url: URL of the grounding model server.
        screen_size: Screen resolution for coordinate mapping.
        use_reflection: Enable reflection agent for stuck detection.
    """

    def __init__(
        self,
        worker_model: str = "opencua",
        worker_provider: str = "vllm",
        worker_base_url: str = "http://localhost:8000/v1",
        grounding_model: str = "",
        grounding_provider: str = "",
        grounding_url: str = "",
        screen_size: tuple[int, int] = (1280, 720),
        use_reflection: bool = True,
    ):
        self.worker_model = worker_model
        self.worker_provider = worker_provider
        self.worker_base_url = worker_base_url
        self.grounding_model = grounding_model
        self.grounding_provider = grounding_provider
        self.grounding_url = grounding_url
        self.default_screen_size = screen_size
        self.use_reflection = use_reflection
        self.enable_thinking = True
        self.model_name = f"agent-s3-{worker_model}"

        self._agent = None
        self._step_count = 0

    def load(self) -> None:
        """Initialize the Agent-S3 framework."""
        try:
            from gui_agents.s3.agents.agent_s import AgentS3
            from gui_agents.s3.agents.grounding import OSWorldACI
            logger.info("Agent-S3 framework loaded (gui-agents)")

            # Build engine params for the worker
            worker_engine_params = {
                "engine_type": self.worker_provider,
                "model": self.worker_model,
            }
            if self.worker_base_url:
                worker_engine_params["base_url"] = self.worker_base_url

            # Build grounding agent (OSWorldACI) if grounding model specified
            grounding_agent = None
            if self.grounding_model and self.grounding_url:
                grounding_engine_params = {
                    "engine_type": self.grounding_provider or "vllm",
                    "model": self.grounding_model,
                    "base_url": self.grounding_url,
                    "grounding_width": self.default_screen_size[0],
                    "grounding_height": self.default_screen_size[1],
                }
                grounding_agent = OSWorldACI(
                    env=None,
                    platform="linux",
                    engine_params_for_generation=worker_engine_params,
                    engine_params_for_grounding=grounding_engine_params,
                    width=self.default_screen_size[0],
                    height=self.default_screen_size[1],
                )

            self._agent = AgentS3(
                worker_engine_params=worker_engine_params,
                grounding_agent=grounding_agent,
                platform="linux",
                max_trajectory_length=8,
                enable_reflection=self.use_reflection,
            )
            logger.info(f"Agent-S3 initialized: worker={self.worker_model}, grounding={self.grounding_model or 'none'}")

        except ImportError:
            logger.warning("gui-agents not installed — using API-only mode")
            self._agent = None

            import requests
            try:
                resp = requests.get(f"{self.worker_base_url}/models", timeout=10)
                resp.raise_for_status()
                logger.info(f"API-only mode: worker at {self.worker_base_url}")
            except Exception as e:
                raise RuntimeError(f"Cannot connect to worker at {self.worker_base_url}: {e}")

    def think(
        self,
        frames: list[Image.Image],
        task: str,
        action_history: list[Action] | None = None,
        screen_size: tuple[int, int] = (1920, 1080),
    ) -> InferenceResult:
        """Run Agent-S3 inference to get the next action."""
        self._step_count += 1

        if self._agent is not None:
            return self._think_with_agent(frames, task, action_history, screen_size)
        else:
            return self._think_api_only(frames, task, action_history, screen_size)

    def _think_with_agent(
        self, frames, task, action_history, screen_size
    ) -> InferenceResult:
        """Use the full Agent-S3 framework."""
        try:
            current_frame = frames[-1] if frames else Image.new("RGB", screen_size, "white")

            buf = BytesIO()
            current_frame.save(buf, format="PNG")
            screenshot_bytes = buf.getvalue()

            obs = {"screenshot": screenshot_bytes}

            # predict() returns (info_dict, actions_list)
            info, actions = self._agent.predict(
                instruction=task,
                observation=obs,
            )

            # info has: plan, plan_code, execution_code, reflection
            thinking = info.get("plan", "") if isinstance(info, dict) else ""
            reflection = info.get("reflection", "") if isinstance(info, dict) else ""
            if reflection:
                thinking = f"{thinking}\nReflection: {reflection}"

            # actions is a list of pyautogui command strings
            action_text = actions[0] if actions else ""

            if not action_text or action_text.strip() in ("DONE", "FAIL"):
                success = "DONE" in (action_text or "")
                action = Action(ActionType.DONE, {"success": success, "summary": thinking[:200]})
            else:
                action = _parse_pyautogui_action(action_text, screen_size)

            return InferenceResult(
                action=action,
                raw_output=action_text,
                thinking=thinking,
            )

        except Exception as e:
            logger.error(f"Agent-S3 error: {e}")
            return InferenceResult(
                action=Action(ActionType.WAIT, {"seconds": 1.0}),
                raw_output=str(e),
                thinking=f"Error: {e}",
            )

    def _think_api_only(
        self, frames, task, action_history, screen_size
    ) -> InferenceResult:
        """Fallback: use Agent-S3's prompting strategy via raw LLM API.

        When gui-agents isn't installed, we replicate the key prompt
        engineering from Agent-S3 and send it to the worker model directly.
        """
        import requests

        # Agent-S3 style system prompt
        system_prompt = (
            "You are a computer use agent. You see a screenshot and output "
            "exactly ONE pyautogui action to perform.\n\n"
            "Available actions:\n"
            "- pyautogui.click(x=<int>, y=<int>)\n"
            "- pyautogui.typewrite('<text>')\n"
            "- pyautogui.hotkey('<key1>', '<key2>')\n"
            "- pyautogui.press('<key>')\n"
            "- pyautogui.scroll(<amount>)\n"
            "- terminate('success') or terminate('failure')\n\n"
            "Rules:\n"
            "- Click the CENTER of the target element\n"
            "- After clicking an input field, use typewrite() to enter text\n"
            "- Do NOT repeat the same action — try alternatives\n"
            "- Press 'enter' to submit forms\n"
            f"- Screen size: {screen_size[0]}x{screen_size[1]}\n"
        )

        # Build conversation with action history
        messages = [{"role": "system", "content": system_prompt}]

        # Include history frames for context
        content: list[dict] = []
        frames_to_send = frames[-3:] or [Image.new("RGB", screen_size, "white")]
        for i, frame in enumerate(frames_to_send):
            label = "CURRENT" if i == len(frames_to_send) - 1 else "previous"
            content.append({"type": "text", "text": f"[{label}]"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_image_to_base64(frame)}"},
            })

        # Task + action history
        task_text = f"Task: {task}"
        if action_history:
            recent = action_history[-8:]
            task_text += "\n\nPrevious actions:\n" + "\n".join(
                f"  {i+1}. {a}" for i, a in enumerate(recent)
            )
        content.append({"type": "text", "text": task_text})

        messages.append({"role": "user", "content": content})

        try:
            resp = requests.post(
                f"{self.worker_base_url}/chat/completions",
                json={
                    "model": self.worker_model,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.0,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"].get("content", "")

            # Extract action from response
            pyautogui_match = re.search(r'pyautogui\.\w+\(.*?\)', text, re.DOTALL)
            terminate_match = re.search(r"terminate\(['\"](.+?)['\"]\)", text)

            thinking = text[:text.find("pyautogui")] if pyautogui_match else text
            action_text = pyautogui_match.group(0) if pyautogui_match else text

            if terminate_match:
                success = "success" in terminate_match.group(1).lower()
                action = Action(ActionType.DONE, {"success": success, "summary": thinking[:200]})
            else:
                action = _parse_pyautogui_action(action_text, screen_size)

            return InferenceResult(
                action=action,
                raw_output=text,
                thinking=thinking.strip(),
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
            )

        except Exception as e:
            logger.error(f"Agent-S API error: {e}")
            return InferenceResult(
                action=Action(ActionType.WAIT, {"seconds": 1.0}),
                raw_output=str(e),
            )
