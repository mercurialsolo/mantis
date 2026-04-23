"""Lightweight OSWorld agent for Modal deployment.

Self-contained — no imports from cua_agent package. Uses llama-server
via OpenAI-compatible API for Gemma4 inference.
"""

import base64
import json
import logging
import time

import requests

logger = logging.getLogger("desktopenv.agent")

SYSTEM_PROMPT = """\
You are a computer use agent. You observe the screen and perform actions to complete tasks.

Your job:
1. OBSERVE the current screen state carefully
2. REASON about what to do next given the task
3. CALL exactly one tool to perform an action

Coordinates are in absolute screen pixels (1920x1080).
When the task is complete, call done(success=true, summary="...").
If stuck after multiple attempts, call done(success=false, summary="...").\
"""

TOOLS = [
    {"type": "function", "function": {"name": "click", "description": "Click at screen coordinates", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "button": {"type": "string", "enum": ["left", "right"]}}, "required": ["x", "y"]}}},
    {"type": "function", "function": {"name": "double_click", "description": "Double-click", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}}},
    {"type": "function", "function": {"name": "type_text", "description": "Type text", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "key_press", "description": "Press key combo (e.g. 'enter', 'ctrl+a')", "parameters": {"type": "object", "properties": {"keys": {"type": "string"}}, "required": ["keys"]}}},
    {"type": "function", "function": {"name": "scroll", "description": "Scroll", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["up", "down"]}, "amount": {"type": "integer"}}, "required": ["direction"]}}},
    {"type": "function", "function": {"name": "wait", "description": "Wait and observe", "parameters": {"type": "object", "properties": {"seconds": {"type": "number"}}}}},
    {"type": "function", "function": {"name": "done", "description": "Task complete", "parameters": {"type": "object", "properties": {"success": {"type": "boolean"}, "summary": {"type": "string"}}, "required": ["success", "summary"]}}},
]


def _action_to_pyautogui(name: str, args: dict) -> str:
    """Convert tool call to pyautogui code string."""
    if name == "click":
        x, y = args["x"], args["y"]
        btn = args.get("button", "left")
        return f"pyautogui.click({x}, {y})" if btn == "left" else f"pyautogui.click({x}, {y}, button='{btn}')"
    elif name == "double_click":
        return f"pyautogui.doubleClick({args['x']}, {args['y']})"
    elif name == "type_text":
        text = args["text"].replace("'", "\\'")
        return f"pyautogui.typewrite('{text}', interval=0.02)"
    elif name == "key_press":
        parts = [k.strip() for k in args["keys"].split("+")]
        if len(parts) == 1:
            return f"pyautogui.press('{parts[0]}')"
        return f"pyautogui.hotkey({', '.join(repr(k) for k in parts)})"
    elif name == "scroll":
        clicks = args.get("amount", 3)
        if args["direction"] == "down":
            clicks = -clicks
        return f"pyautogui.scroll({clicks})"
    elif name == "wait":
        return "WAIT"
    elif name == "done":
        return "DONE" if args.get("success") else "FAIL"
    return "WAIT"


class ModalGemma4Agent:
    """OSWorld-compatible agent using llama-server on Modal."""

    def __init__(self, llamacpp_url="http://localhost:8080/v1", max_tokens=2048):
        self.url = llamacpp_url
        self.max_tokens = max_tokens
        self.action_space = "pyautogui"
        self._frame_history = []
        self._action_history = []
        self._step = 0
        self.thoughts = []
        self.actions = []
        self.observations = []

    def reset(self, *args, **kwargs):
        self._frame_history.clear()
        self._action_history.clear()
        self._step = 0
        self.thoughts.clear()
        self.actions.clear()
        self.observations.clear()

    def predict(self, instruction: str, obs: dict) -> tuple:
        self._step += 1

        # Convert screenshot to base64
        b64 = base64.b64encode(obs["screenshot"]).decode()
        self._frame_history.append(b64)
        if len(self._frame_history) > 5:
            self._frame_history = self._frame_history[-5:]

        # Build messages with frame history
        content = [{"type": "text", "text": SYSTEM_PROMPT}]

        # Add last 3 frames for temporal context
        frames_to_send = self._frame_history[-3:]
        for i, frame_b64 in enumerate(frames_to_send):
            label = "CURRENT" if i == len(frames_to_send) - 1 else f"t-{len(frames_to_send)-1-i}"
            content.append({"type": "text", "text": f"[Frame {label}]"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame_b64}"}})

        # Task + history
        task_text = f"\nTask: {instruction}\nScreen: 1920x1080"
        if self._action_history:
            recent = self._action_history[-5:]
            task_text += "\nRecent actions:\n" + "\n".join(f"  {i+1}. {a}" for i, a in enumerate(recent))
        content.append({"type": "text", "text": task_text})

        payload = {
            "model": "gemma-4",
            "messages": [{"role": "user", "content": content}],
            "tools": TOOLS,
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }

        try:
            t0 = time.time()
            resp = requests.post(f"{self.url}/chat/completions", json=payload, timeout=600)
            elapsed = time.time() - t0
            data = resp.json()
            msg = data["choices"][0]["message"]
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return str(e), ["WAIT"]

        # Parse tool call
        tool_calls = msg.get("tool_calls", [])
        thinking = msg.get("reasoning_content", "") or msg.get("content", "") or ""

        if tool_calls:
            tc = tool_calls[0]
            func = tc.get("function", {})
            name = func.get("name", "wait")
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            code = _action_to_pyautogui(name, args)
        else:
            code = "WAIT"

        self._action_history.append(code)
        response = f"{thinking}\nAction: {code}" if thinking else f"Action: {code}"

        self.thoughts.append(response)
        self.actions.append(code)
        self.observations.append({"screenshot": b64[:100] + "..."})

        logger.info(f"Step {self._step}: {code} ({elapsed:.1f}s)")
        return response, [code]
