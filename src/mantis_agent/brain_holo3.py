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
from .prompts import load_prompt

logger = logging.getLogger(__name__)


# RL-prep (logprobs capture). vLLM only returns per-token logprobs when
# the request asks for them; without this the modelio record's
# ``response.logprobs`` is always empty, blocking PPO / GRPO (no
# behaviour-policy logprob → no importance ratio / KL). Gated by the
# SAME env flag that opens the Augur DebugSession with
# ``capture_logprobs`` (``observability.augur.should_capture_logprobs``)
# so one switch turns the whole path on. Read directly here — importing
# ``observability.augur`` would pull in ``augur_sdk`` at module import,
# which isn't present in every environment (e.g. local test runs).
_CAPTURE_LOGPROBS_ENV = "MANTIS_CAPTURE_LOGPROBS"
_LOGPROBS_TOP_K_ENV = "MANTIS_LOGPROBS_TOP_K"


def _capture_logprobs_enabled() -> bool:
    return os.environ.get(_CAPTURE_LOGPROBS_ENV, "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _logprobs_top_k() -> int:
    """Number of top alternatives per token (``top_logprobs``). 0 (the
    default) asks only for the sampled token's logprob — all PPO/GRPO
    needs and the smallest payload. Higher k adds alternatives for
    analysis at the cost of response size."""
    try:
        return max(0, int(os.environ.get(_LOGPROBS_TOP_K_ENV, "0") or "0"))
    except ValueError:
        return 0

# ── System prompt (tool-calling style, adapted from LlamaCppBrain) ──────────
# Sourced from mantis_agent.prompts.HOLO3_SYSTEM. Override per-tenant via
# MANTIS_PROMPTS_DIR/holo3_system.txt.
SYSTEM_PROMPT = load_prompt("holo3_system")

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


_PREDICTED_LINE_RE = re.compile(
    r"^\s*Predicted\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_predicted_outcome(text: str) -> str:
    """Pull the first ``Predicted: <delta>`` line out of a Holo3 response.

    The Holo3 system prompt asks for an optional trailing line in this
    format. The brain may omit it (e.g. for done() steps, or when it
    genuinely doesn't know). Returns ``""`` when no line is present so
    the trajectory's ``predicted_outcome`` stays at its default — see #120.

    Trims trailing punctuation/quotes that often appear when the model
    paraphrases the example. Caps at 240 chars to keep trajectories bounded.
    """
    if not text:
        return ""
    match = _PREDICTED_LINE_RE.search(text)
    if not match:
        return ""
    line = match.group(1).strip().strip('"').strip("'").strip()
    return line[:240]


@dataclass
class InferenceResult:
    """Result from a single brain inference cycle."""
    action: Action
    raw_output: str
    thinking: str = ""
    tokens_used: int = 0
    # #120 step 2: brain's prediction of what its action will cause.
    # Empty when the model didn't emit a "Predicted:" line.
    predicted_outcome: str = ""


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
        use_tool_calling: bool = False,  # llama.cpp often breaks with tools — text parsing is more reliable
        extra_headers: dict[str, str] | None = None,
    ):
        """Holo3 client.

        Args:
            extra_headers: Additional headers to send with every request
                (e.g. ``{"X-Mantis-Token": "..."}``). When the caller's
                ``Authorization`` value is also supplied here, it takes
                precedence over the ``Bearer <api_key>`` default — useful
                for gateways that expect ``Api-Key`` or other schemes.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("HAI_API_KEY", "")
        self.use_tool_calling = use_tool_calling
        self.model_name = "Holo3-35B-A3B"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.default_screen_size = screen_size
        self.enable_thinking = enable_thinking
        self.extra_headers = dict(extra_headers or {})

    @property
    def _headers(self) -> dict:
        """Auth headers for API requests.

        Order of precedence (later overrides earlier):
        1. ``Content-Type: application/json``
        2. ``Authorization: Bearer <api_key>`` (when api_key is set)
        3. Anything in ``extra_headers`` — including a custom Authorization
           value, which lets a caller swap ``Bearer`` for ``Api-Key`` when
           the deployment sits behind a gateway that expects it.
        """
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if self.extra_headers:
            h.update(self.extra_headers)
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

    def detect_with_image(self, prompt: str, image: Image.Image, max_tokens: int = 256) -> str:
        """Send a vision prompt (image + question) and return raw text.

        Used by the runner for element-aware decisions: "is this a focused
        input?", "find the submit button". Returns "" on failure so callers
        fall back gracefully. Disables tool calling so the model emits free
        text instead of trying to call click()/type_text().
        """
        image_data = _image_to_base64(image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                {"type": "text", "text": prompt},
            ],
        }]
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if not self.enable_thinking:
            payload["thinking"] = False
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload, headers=self._headers, timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"].get("content", "") or ""
        except Exception as e:
            logger.warning(f"detect_with_image failed: {e}")
            return ""

    def think(
        self,
        frames: list[Image.Image],
        task: str,
        action_history: list[Action] | None = None,
        screen_size: tuple[int, int] = (1920, 1080),
        *,
        retry_attempts: list[dict] | None = None,
        per_step_action_history: list[Action] | None = None,
    ) -> InferenceResult:
        """Run perception-reasoning-action via Holo3 with tool calling.

        ``retry_attempts`` (#435 item 7) and ``per_step_action_history``
        (#435 item 2) — same contract as ``brain_claude.think`` /
        ``brain_fara.think``. Default ``None`` preserves pre-existing
        global-history behaviour for callers that haven't migrated.
        """
        messages = self._build_messages(
            frames, task, action_history, screen_size,
            retry_attempts=retry_attempts,
            per_step_action_history=per_step_action_history,
        )

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self.use_tool_calling:
            payload["tools"] = OPENAI_TOOLS
            payload["tool_choice"] = "auto"

        # Holo3 native thinking toggle (H Company API format)
        if not self.enable_thinking:
            payload["thinking"] = False

        # RL-prep — ask vLLM for per-token logprobs on the planner call
        # when training-data capture is enabled. Off by default (roughly
        # doubles the response payload); the modelio mapper turns the
        # returned block into ``response.logprobs``.
        if _capture_logprobs_enabled():
            payload["logprobs"] = True
            top_k = _logprobs_top_k()
            if top_k > 0:
                payload["top_logprobs"] = top_k

        # Retry with backoff for rate limiting (429)
        data = None
        _modelio_t0 = time.time()
        for attempt in range(4):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload, headers=self._headers,
                    timeout=120,
                )
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 30)  # 5, 10, 20, 30 sec
                    logger.warning(f"Rate limited (429), waiting {wait}s (attempt {attempt+1}/4)")
                    time.sleep(wait)
                    continue
                if resp.status_code == 400:
                    body = resp.text[:500]
                    logger.warning(f"Bad request (400): {body}")
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt < 3:
                    logger.warning(f"Holo3 API attempt {attempt+1} failed: {e}")
                    time.sleep(3)
                else:
                    logger.error(f"Holo3 API request failed after 4 attempts: {e}")
                    return InferenceResult(
                        action=Action(ActionType.WAIT, {"seconds": 2.0}, reasoning=str(e)),
                        raw_output=str(e),
                    )

        if data is None:
            return InferenceResult(
                action=Action(ActionType.WAIT, {"seconds": 5.0}, reasoning="Rate limited"),
                raw_output="429 rate limited after retries",
            )

        # Gap 1 — capture the brain's own decision as a modelio record.
        # The Holo3 client talks to a vLLM OpenAI server, so the
        # Anthropic client's capture hook never sees this call; without
        # this the highest-fidelity prompt->response pair (tagged
        # ``planner`` by the runner wrap) is lost for the production
        # model. No-op when no modelio context is published / Augur is
        # inactive; never raises (Augur spec §4.3).
        self._record_modelio_if_active(
            payload, data, int((time.time() - _modelio_t0) * 1000),
        )

        return self._parse_response(data, screen_size)

    @staticmethod
    def _record_modelio_if_active(
        request_payload: dict, response_json: dict, duration_ms: int,
    ) -> None:
        """Forward a Holo3 chat-completions call to the modelio capture
        layer when a context is active. Mirrors
        ``_anthropic.client._record_modelio_if_active`` for the OpenAI
        response shape; local import keeps observability off the hot
        path until a layer is published."""
        try:
            from .observability.modelio import (
                current_modelio_context,
                record_openai_modelio,
            )
        except Exception:  # noqa: BLE001 — import issues never block calls
            return
        if current_modelio_context() is None:
            return
        try:
            record_openai_modelio(
                request_payload=request_payload,
                response_json=response_json,
                duration_ms=duration_ms,
            )
        except Exception as exc:  # noqa: BLE001 — Augur spec §4.3
            logger.debug("Holo3 modelio capture failed: %s", exc)

    def _build_messages(
        self,
        frames: list[Image.Image],
        task: str,
        action_history: list[Action] | None,
        screen_size: tuple[int, int],
        *,
        retry_attempts: list[dict] | None = None,
        per_step_action_history: list[Action] | None = None,
    ) -> list[dict]:
        """Build OpenAI-format messages with image content."""
        # Context-aware composition: splice in any ContextModule sections
        # whose ``applies_when`` predicate is true for the active dispatch
        # context (pushed via ``push_step_context`` by the calling handler).
        # Modules are the modular replacement for hardcoded per-step
        # sections in holo3_system.txt — adding a new behaviour for a new
        # failure pattern is one module file, not a global prompt edit.
        from .context_modules import compose_system_prompt
        system_prompt = compose_system_prompt(SYSTEM_PROMPT)
        messages = [{"role": "system", "content": system_prompt}]

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
        # #435 item 2: prefer the sub-goal-scoped slice when set.
        recent_actions: list[Action] | None = None
        if per_step_action_history is not None:
            recent_actions = list(per_step_action_history)
        elif action_history:
            # Keep short — Holo3 context is precious.
            recent_actions = action_history[-5:]
        if recent_actions:
            history_str = "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(recent_actions))
            context_parts.append(f"Recent actions:\n{history_str}")

        # #435 item 7: outcome-tagged retry attempts.
        if retry_attempts:
            from .gym import retry_attempts as _retry
            block = _retry.render_attempts_block(retry_attempts)
            if block:
                context_parts.append(block)

        content.append({"type": "text", "text": "\n".join(context_parts)})
        messages.append({"role": "user", "content": content})

        return messages

    # ── Response parsing (triple fallback) ──────────────────────────────────

    def _parse_response(self, data: dict, screen_size: tuple[int, int]) -> InferenceResult:
        """Parse Holo3 response with 5-strategy fallback:

        1. Standard OpenAI tool_calls (primary path)
        2. Holo3 native "Action: name({...})" text format
        3. JSON action {"action": "click", ...} in content
        4. pyautogui-style text (pyautogui.click, etc.)
        5. terminate/DONE/FAIL keywords
        """
        choice = data["choices"][0]
        message = choice["message"]
        raw_output = json.dumps(message)

        # Extract thinking from reasoning_content (vLLM with --enable-reasoning)
        # or from <think> blocks in content
        thinking = message.get("reasoning_content", "") or ""
        content_text = message.get("content", "") or ""

        # #120 step 2: extract the optional "Predicted: <delta>" line the
        # prompt asks for. Empty when the model didn't emit it.
        predicted = _extract_predicted_outcome(content_text)

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
                predicted_outcome=predicted,
            )

        # Strategy 2: Holo3 "Action: name({...})" or "Action: name(key=val)" text format
        holo3_action = self._parse_holo3_action(content_text, screen_size)
        if holo3_action is not None:
            if not thinking:
                # Everything before "Action:" is reasoning
                action_idx = content_text.find("Action:")
                if action_idx > 0:
                    thinking = content_text[:action_idx].strip()
            return InferenceResult(
                action=holo3_action,
                raw_output=raw_output,
                thinking=thinking,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
                predicted_outcome=predicted,
            )

        # Strategy 3: JSON action {"action": "click", ...} in content text
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
                predicted_outcome=predicted,
            )

        # Strategy 4: pyautogui-style text (pyautogui.click, etc.)
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
                predicted_outcome=predicted,
            )

        # Strategy 5: terminate / DONE / FAIL keywords
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
                predicted_outcome=predicted,
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
            predicted_outcome=predicted,
        )

    def _convert_coords(self, action_name: str, args: dict, screen_size: tuple[int, int]) -> dict:
        """Convert model coordinates to screen coordinates for spatial actions."""
        spatial_actions = {"click", "double_click", "scroll"}
        drag_action = "drag"

        if action_name in spatial_actions:
            if "x" in args and "y" in args:
                try:
                    sx, sy = _model_coords_to_screen(
                        int(float(str(args["x"]))), int(float(str(args["y"]))),
                        screen_size[0], screen_size[1],
                    )
                    args["x"] = sx
                    args["y"] = sy
                except (ValueError, TypeError):
                    pass

        if action_name == drag_action:
            for prefix in ("start_", "end_"):
                xk, yk = f"{prefix}x", f"{prefix}y"
                if xk in args and yk in args:
                    try:
                        sx, sy = _model_coords_to_screen(
                            int(float(str(args[xk]))), int(float(str(args[yk]))),
                            screen_size[0], screen_size[1],
                        )
                        args[xk] = sx
                        args[yk] = sy
                    except (ValueError, TypeError):
                        pass

        return args

    def _parse_holo3_action(self, text: str, screen_size: tuple[int, int]) -> Action | None:
        """Parse Holo3's native text format: Action: name({...}) or name(key=val, ...).

        Examples from real Holo3 output:
          Action: scroll({'direction': 'down', 'amount': 5})
          Action: click({'x': 640, 'y': 360})
          Action: type_text({'text': '33101'})
          Action: done({'success': true, 'summary': '...'})
          Action: wait()
        """
        # Match "Action: name(...)" — the Action: prefix is optional
        m = re.search(
            r'(?:Action:\s*)?(\w+)\(\s*(\{.*?\}|\S.*?)\s*\)',
            text, re.DOTALL,
        )
        if not m:
            # Also try bare function call without Action: prefix
            m = re.search(r'\b(click|scroll|type_text|key_press|done|wait|double_click)\(\s*(\{.*?\})\s*\)', text, re.DOTALL)
        if not m:
            # Bare key name like "Escape()" or "Enter()" → key_press
            bare_key = re.search(r'\b(Escape|Enter|Tab|Backspace|Delete)\(\s*\)', text)
            if bare_key:
                return Action(ActionType.KEY_PRESS, {"keys": bare_key.group(1).lower()})
            return None

        func_name = m.group(1).lower()
        raw_args = m.group(2).strip()

        # Try parsing as JSON (single-quoted → double-quoted)
        args = {}
        if raw_args.startswith("{"):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                # Holo3 uses single quotes — convert
                fixed = raw_args.replace("'", '"')
                # Fix Python booleans/None
                fixed = fixed.replace("True", "true").replace("False", "false").replace("None", "null")
                try:
                    args = json.loads(fixed)
                except json.JSONDecodeError:
                    pass

        # If no JSON args parsed, try key=value format: name(key1=val1, key2=val2)
        if not args and "=" in raw_args:
            for kv in re.findall(r"(\w+)\s*=\s*['\"]?([^'\",:}]+)['\"]?", raw_args):
                args[kv[0]] = kv[1].strip()

        def _safe_int(val, default=0) -> int:
            """Extract first integer from possibly malformed value."""
            if isinstance(val, (int, float)):
                return int(val)
            m = re.match(r'-?\d+', str(val).strip())
            return int(m.group(0)) if m else default

        def _has_valid_xy(args: dict) -> bool:
            """#574: a click is only dispatchable when BOTH x and y were
            present in the model's args AND not both zero. Missing keys
            (e.g. ``click()`` or ``click({'button': 'left'})``) and
            explicit ``(0, 0)`` both indicate the model didn't pin a
            target — better to fall through to the WAIT(1s) fallback
            than dispatch to the page origin.
            """
            if "x" not in args or "y" not in args:
                return False
            x = _safe_int(args["x"], default=-1)
            y = _safe_int(args["y"], default=-1)
            if x < 0 or y < 0:
                # Malformed value that didn't parse to a non-negative int.
                return False
            return (x, y) != (0, 0)

        # Map to Action
        if func_name in ("click",):
            # #574: reject click() with missing or (0,0) coords. Returning
            # an Action(CLICK, {x:0, y:0}) silently dispatches to the page
            # origin which always lands off-target (logo / header bg /
            # off-canvas), producing wrong-page nav and DUPLICATE-URL
            # loops on listings plans. ``None`` falls through to the
            # parser chain → eventually Action(WAIT, 1s) at the brain
            # level → clean no-op step.
            if not _has_valid_xy(args):
                return None
            x = _safe_int(args["x"])
            y = _safe_int(args["y"])
            sx, sy = _model_coords_to_screen(x, y, screen_size[0], screen_size[1])
            return Action(ActionType.CLICK, {"x": sx, "y": sy, "button": args.get("button", "left")})

        if func_name in ("double_click", "doubleclick"):
            if not _has_valid_xy(args):
                return None
            x = _safe_int(args["x"])
            y = _safe_int(args["y"])
            sx, sy = _model_coords_to_screen(x, y, screen_size[0], screen_size[1])
            return Action(ActionType.DOUBLE_CLICK, {"x": sx, "y": sy})

        if func_name in ("type_text", "type", "typewrite"):
            return Action(ActionType.TYPE, {"text": str(args.get("text", ""))})

        if func_name in ("key_press", "press", "hotkey"):
            keys = args.get("keys", args.get("key", ""))
            if isinstance(keys, list):
                keys = "+".join(str(k) for k in keys)
            return Action(ActionType.KEY_PRESS, {"keys": str(keys)})

        if func_name == "scroll":
            direction = str(args.get("direction", "down"))
            amount = _safe_int(args.get("amount", 3), default=3)
            return Action(ActionType.SCROLL, {"direction": direction, "amount": amount})

        if func_name in ("done", "terminate"):
            success = args.get("success", True)
            if isinstance(success, str):
                success = success.lower() in ("true", "1", "yes", "success")
            summary = str(args.get("summary", args.get("message", "")))[:200]
            return Action(ActionType.DONE, {"success": bool(success), "summary": summary})

        if func_name == "wait":
            seconds = float(args.get("seconds", 1.0))
            return Action(ActionType.WAIT, {"seconds": seconds})

        return None

    def _parse_json_action(self, text: str, screen_size: tuple[int, int]) -> Action | None:
        """Parse JSON-formatted actions: {"action": "click", "x": N, "y": N}.

        Reused from OpenCUA's fallback strategy for EvoCUA-style output.
        """
        # Strip markdown code fences
        cleaned = re.sub(r'```(?:json|python|)\s*\n?', '', text)
        cleaned = re.sub(r'\n?```', '', cleaned)

        match = re.search(r'\{[^{}]*"(?:action|command|code)"\s*:\s*"[^"]*"[^{}]*\}', cleaned)
        if not match:
            return None

        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

        # Holo3 outputs varying keys: "action", "command", or "code" (which is a function call string)
        action_type = obj.get("action", obj.get("command", "")).lower()
        if not action_type:
            # "code" field contains function call like "wait()" or "click(x=100, y=200)"
            code = obj.get("code", "")
            if code:
                # Delegate to holo3 action parser which handles function call strings
                return self._parse_holo3_action(code, screen_size)

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
