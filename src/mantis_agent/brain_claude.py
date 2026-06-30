"""Claude CUA Brain — Anthropic's computer use model via API.

Uses Claude's native computer_use tool for pixel-accurate browser automation.
Claude outputs structured tool calls (click, type, scroll, etc.) with rich
chain-of-thought reasoning — no regex parsing needed.

Backend: Anthropic API (direct HTTP, no SDK dependency)

Key differences from other brains:
- Native computer_use_2025_01_24 tool — Claude understands UIs natively
- Structured tool calls (like Gemma4, unlike EvoCUA's pyautogui text)
- Rich reasoning in thinking blocks — valuable for distillation
- No GPU needed — runs via API, pairs with XdotoolGymEnv on Modal

Usage:
    brain = ClaudeBrain(api_key="sk-ant-...", model="claude-opus-4-7")
    brain.load()
    result = brain.think(frames=[screenshot], task="Click the login button", ...)

Cost: ~$0.01-0.05 per screenshot step (input tokens dominate)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from io import BytesIO

import requests
from PIL import Image

from .actions import Action, ActionType
from .prompts import load_prompt

logger = logging.getLogger(__name__)

# Sourced from mantis_agent.prompts.CLAUDE_SYSTEM. Override per-tenant via
# MANTIS_PROMPTS_DIR/claude_system.txt.
SYSTEM_PROMPT = load_prompt("claude_system")

# ``anthropic-beta`` header that MUST accompany the computer-use tool type
# (``computer_20251124`` in ``_build_tools``) — the API rejects the tool
# type with HTTP 400 without it. Single source of truth; passed into the
# shared ``AnthropicToolUseClient`` via ``extra_headers`` (the client builds
# its own base headers and would otherwise drop this entirely — the root
# cause of every /v1/cua run brain-looping to a hard-loop halt).
_ANTHROPIC_BETA_HEADER = "computer-use-2025-11-24,context-management-2025-06-27"

# Claude computer_use tool + our custom done/wait tools
CLAUDE_TOOLS = [
    {
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": 1280,
        "display_height_px": 720,
        "display_number": 1,
    },
]

# Additional tools that Claude doesn't have natively
EXTRA_TOOLS = [
    {
        "name": "done",
        "description": (
            "Signal that the task is complete. Call this when you have achieved the goal "
            "or determined it cannot be completed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "description": "Whether the task succeeded"},
                "summary": {"type": "string", "description": "Brief summary of what was done or extracted data"},
            },
            "required": ["success", "summary"],
        },
    },
    {
        "name": "wait",
        "description": (
            "Wait and observe the screen. Use this when an action is in progress "
            "(loading, animation) and you need to see the result before acting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "Seconds to wait (default: 1.0)",
                },
            },
        },
    },
]


def _image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
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
    # #291: brain's structured prediction of post-action signals — either a
    # ``{"expected": [...]}`` JSON block or a back-compat ``Predicted: ...``
    # line. Empty when the model didn't emit one. Parsed downstream by
    # :func:`mantis_agent.gym.predicates.parse_predicates`.
    predicted_outcome: str = ""


class ClaudeBrain:
    """Claude CUA brain using Anthropic API for browser automation.

    Args:
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        model: Claude model ID.
        max_tokens: Maximum tokens to generate per call.
        thinking_budget: Token budget for extended thinking (0 to disable).
        screen_size: Display resolution for computer_use tool.
    """

    # #435 item 5: keep at most this many frames as raw image bytes
    # in the prompt — per cua_notes.md §1 *"Keep only the last 1-3
    # screenshots."* Older frames become text placeholders.
    # Settable per instance via the constructor for callers that
    # want to bypass the prune (e.g. evals comparing pre/post).
    _FRAMES_KEEP_AS_IMAGE: int = 2

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-opus-4-7",
        max_tokens: int = 4096,
        thinking_budget: int = 2048,
        screen_size: tuple[int, int] = (1280, 720),
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.model_name = f"Claude ({model})"
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        self.screen_size = screen_size

    def load(self) -> None:
        """Verify API key is set and reachable."""
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Provide api_key= or set the env var."
            )
        # Quick validation — list models
        try:
            resp = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=10,
            )
            if resp.status_code == 401:
                raise RuntimeError("Invalid ANTHROPIC_API_KEY")
            logger.info(f"Claude API connected: {self.model}")
        except requests.ConnectionError as e:
            raise RuntimeError(f"Cannot reach Anthropic API: {e}")

    def query(self, prompt: str, response_format: str = "json") -> str:
        """Send a text-only prompt and return the text response.

        Used for plan parsing, classification, extraction (no images).
        """
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        # #836: route through the shared retry client so 429/502/503/
        # 504/529 + timeouts don't kill a brain step on a single
        # transient. The shared policy already covers everything we'd
        # implement here.
        try:
            from ._anthropic.client import AnthropicToolUseClient
            client = AnthropicToolUseClient(
                api_key=self.api_key, model=self.model,
                log_prefix="[brain.claude.query]",
                extra_headers={"anthropic-beta": _ANTHROPIC_BETA_HEADER},
            )
            resp = client.post_messages_with_retry(payload, timeout=60)
            if resp is None or resp.status_code != 200:
                logger.warning(
                    "query: Anthropic call failed%s",
                    f" (HTTP {resp.status_code})" if resp is not None else " (network exhaustion)",
                )
                return ""
            data = resp.json()
            for block in data.get("content", []):
                if block.get("type") == "text":
                    return block["text"]
            return ""
        except Exception as e:  # noqa: BLE001 — best-effort; never propagate
            logger.error(f"query failed: {e}")
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
        """Run perception-reasoning-action via Claude API with computer_use tool.

        ``retry_attempts`` (#435 item 7) — outcome-tagged failure records
        from ``MicroPlanRunner._step_failure_history[step_index]``. When
        present, rendered as a ``Recent attempts on this sub-goal:``
        block. Lets Claude refute coordinates / patterns that already
        failed without re-deriving the failure mode from prose.

        ``per_step_action_history`` (#435 item 2) — sub-goal-scoped
        action slice. When provided, overrides the global
        ``action_history`` for the prompt's ``Recent actions:`` block.
        The doc's *"Reset between sub-goals. Bounded length: 1–3
        entries"* guidance — caller (per-step handler) supplies the
        slice already trimmed.
        """
        # Update computer_use tool dimensions to match actual screen
        tools = self._build_tools(screen_size)
        messages = self._build_messages(
            frames, task, action_history, screen_size,
            retry_attempts=retry_attempts,
            per_step_action_history=per_step_action_history,
        )

        # #715 — prompt-cache split. The system prompt + tool array
        # don't change across turns of the same run; mark them so the
        # API caches the prefix and we only pay for the screenshot +
        # task context on subsequent calls. Saves ~30-50% on cost-per-
        # call once the cache is warm (5-minute TTL).
        from ._anthropic.cache import as_cached_system, mark_last_tool_cached

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": as_cached_system(SYSTEM_PROMPT),
            "messages": messages,
            "tools": mark_last_tool_cached(tools),
        }

        # Enable extended thinking for richer reasoning (better for distillation)
        if self.thinking_budget > 0:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }

        # #836: route through the shared retry client. Lower max_attempts
        # (2 instead of 4) because this is the per-step brain inference
        # call — every step pays the latency cost. One retry covers
        # the common transient (529 / 503 / read-timeout) without
        # ballooning step latency on persistent failures.
        try:
            from ._anthropic.client import AnthropicToolUseClient
            client = AnthropicToolUseClient(
                api_key=self.api_key, model=self.model,
                log_prefix="[brain.claude.think]",
                extra_headers={"anthropic-beta": _ANTHROPIC_BETA_HEADER},
            )
            resp = client.post_messages_with_retry(
                payload, timeout=120, max_attempts=2,
            )
            if resp is None:
                logger.error("Claude API network exhaustion (all attempts)")
                return InferenceResult(
                    action=Action(
                        ActionType.WAIT, {"seconds": 1.0},
                        reasoning="Anthropic unreachable",
                    ),
                    raw_output="Anthropic unreachable",
                )
            if resp.status_code != 200:
                logger.error(f"Claude API {resp.status_code}: {resp.text[:500]}")
                resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # noqa: BLE001 — never propagate
            logger.error(f"Claude API request failed: {e}")
            return InferenceResult(
                action=Action(ActionType.WAIT, {"seconds": 1.0}, reasoning=str(e)),
                raw_output=str(e),
            )

        # #715 — cache telemetry. WARNING-level per
        # `feedback_warning_level_for_modal_observability.md`.
        from ._anthropic.cache import extract_cache_telemetry
        tele = extract_cache_telemetry(data)
        if tele.get("cache_read_input_tokens", 0) > 0 or tele.get(
            "cache_creation_input_tokens", 0
        ) > 0:
            logger.warning(
                "  [cache] brain_claude: read=%d created=%d input=%d output=%d",
                tele.get("cache_read_input_tokens", 0),
                tele.get("cache_creation_input_tokens", 0),
                tele.get("input_tokens", 0),
                tele.get("output_tokens", 0),
            )

        # Cost meter (#675 A/B follow-up).
        from .observability.claude_cost_meter import record_from_response
        record_from_response(
            source="brain_claude", model=self.model, response_json=data,
        )

        return self._parse_response(data)

    def _headers(self) -> dict:
        """Build Anthropic API headers.

        #435 item 6: betas bumped per ``docs/cua_notes.md`` §3.

        * ``computer-use-2025-11-24`` — current computer-use beta with
          the matching ``computer_20251124`` tool type (paired with
          ``_build_tools`` below). Was ``computer-use-2025-01-24``,
          which paired with ``computer_20250124``.
        * ``context-management-2025-06-27`` — auto-clears old
          ``tool_result`` blocks when token budget tightens; useful
          for long-running agent loops on Claude Opus 4.7. Off in
          January 2024 beta; on now via this header.

        Multiple beta values are joined with comma per Anthropic
        docs — order doesn't matter.
        """
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": _ANTHROPIC_BETA_HEADER,
            "content-type": "application/json",
        }

    def _build_tools(self, screen_size: tuple[int, int]) -> list[dict]:
        """Build tool list with correct screen dimensions.

        #435 item 6: ``computer_20251124`` tool type pairs with the
        ``computer-use-2025-11-24`` beta header above. The previous
        pairing was ``computer_20250124`` / ``computer-use-2025-01-24``
        — these MUST move together because Anthropic validates that
        the tool type matches the beta version.
        """
        computer_tool = {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": screen_size[0],
            "display_height_px": screen_size[1],
            "display_number": 1,
        }
        return [computer_tool] + EXTRA_TOOLS

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
        """Build Claude API messages with image content.

        #435 item 5: image stand-ins for older frames. Per
        ``docs/cua_notes.md`` §3 *"Image pruning happens in your
        code. Replace older tool_result images with text stand-ins
        like [screenshot omitted] or a small placeholder."*

        Only the most recent ``_FRAMES_KEEP_AS_IMAGE`` frames go in
        as raw image bytes; older frames are emitted as
        ``[screenshot omitted — frame t-N]`` text markers. Keeps
        position markers and turn-ordering intact (which is what
        preserves loop-avoidance signal) without paying the
        ~1-2k tokens/frame image cost on every prior frame.
        """
        content = []

        # Add frames as images, but only keep the most recent N as
        # raw bytes — older ones become text placeholders.
        n_frames = len(frames)
        keep_as_image = self._FRAMES_KEEP_AS_IMAGE
        for i, frame in enumerate(frames):
            label = "CURRENT" if i == n_frames - 1 else f"t-{n_frames - 1 - i}"
            content.append({"type": "text", "text": f"[Frame {label}]"})
            # Distance from the end: 0 = current, 1 = one back, …
            distance_from_end = n_frames - 1 - i
            if distance_from_end < keep_as_image:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _image_to_base64(frame),
                    },
                })
            else:
                content.append({
                    "type": "text",
                    "text": "[screenshot omitted — text-only at this distance]",
                })

        # Task context
        context_parts = [
            f"\nTask: {task}",
            f"Screen size: {screen_size[0]}x{screen_size[1]} pixels",
        ]

        # #435 item 2: prefer the sub-goal-scoped action slice when the
        # caller supplied one. Falls back to the global last-10 slice
        # when not provided — preserves existing behaviour for callers
        # that haven't migrated.
        recent_actions: list[Action] | None = None
        if per_step_action_history is not None:
            recent_actions = list(per_step_action_history)
        elif action_history:
            recent_actions = action_history[-10:]
        if recent_actions:
            history_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(recent_actions))
            context_parts.append(f"Recent actions:\n{history_str}")

        # #435 item 7: outcome-tagged retry attempts from the runner's
        # cross-attempt failure history. Goes AFTER the Recent actions
        # block so a token budget cap truncates the older, less-specific
        # action log first.
        if retry_attempts:
            from .gym import retry_attempts as _retry
            block = _retry.render_attempts_block(retry_attempts)
            if block:
                context_parts.append(block)

        content.append({"type": "text", "text": "\n".join(context_parts)})

        return [{"role": "user", "content": content}]

    def _parse_response(self, data: dict) -> InferenceResult:
        """Parse Claude API response into InferenceResult.

        Claude responses contain content blocks:
        - thinking: extended thinking text (reasoning)
        - text: assistant text
        - tool_use: structured tool calls
        """
        raw_output = json.dumps(data.get("content", []))
        thinking_parts = []
        text_parts = []
        action = None

        for block in data.get("content", []):
            block_type = block.get("type", "")

            if block_type == "thinking":
                thinking_parts.append(block.get("thinking", ""))

            elif block_type == "text":
                text_parts.append(block.get("text", ""))

            elif block_type == "tool_use":
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})

                if tool_name == "computer":
                    action = self._parse_computer_action(tool_input)
                elif tool_name == "done":
                    action = Action(
                        ActionType.DONE,
                        {
                            "success": tool_input.get("success", False),
                            "summary": tool_input.get("summary", ""),
                        },
                    )
                elif tool_name == "wait":
                    action = Action(
                        ActionType.WAIT,
                        {"seconds": tool_input.get("seconds", 1.0)},
                    )

        thinking = "\n".join(thinking_parts)
        text = "\n".join(text_parts)

        # If no tool call found, try to parse from text
        if action is None:
            if text:
                action = self._parse_text_fallback(text)
            else:
                action = Action(ActionType.WAIT, {"seconds": 1.0}, reasoning="No action in response")

        # Combine thinking + text for full reasoning
        full_thinking = thinking
        if text and thinking:
            full_thinking = f"{thinking}\n\n{text}"
        elif text:
            full_thinking = text

        if action.action_type == ActionType.KEY_PRESS and not str(
            action.params.get("keys") or ""
        ).strip():
            inferred_key = _infer_key_from_reasoning(full_thinking)
            if inferred_key:
                logger.warning(
                    "Claude returned an empty key action; inferred %r from reasoning",
                    inferred_key,
                )
                action = Action(
                    ActionType.KEY_PRESS,
                    {"keys": _normalize_claude_key(inferred_key)},
                    reasoning=action.reasoning,
                )
            else:
                logger.warning("Claude returned an empty key action; treating as wait")
                action = Action(
                    ActionType.WAIT,
                    {"seconds": 1.0},
                    reasoning="Claude returned empty key action",
                )

        action.reasoning = action.reasoning or full_thinking[:500]

        tokens_used = data.get("usage", {})
        total_tokens = tokens_used.get("input_tokens", 0) + tokens_used.get("output_tokens", 0)

        # #291: capture the brain's structured prediction (JSON ``expected``
        # block or ``Predicted: ...`` line) for the runner to evaluate against
        # the post-action observation. Search across thinking + assistant text
        # — Claude tends to emit the JSON inside the assistant text block.
        from .gym.predicates import extract_predicted_outcome
        predicted = extract_predicted_outcome(text) or extract_predicted_outcome(thinking)

        return InferenceResult(
            action=action,
            raw_output=raw_output,
            thinking=full_thinking,
            tokens_used=total_tokens,
            predicted_outcome=predicted,
        )

    @staticmethod
    def _parse_computer_action(tool_input: dict) -> Action:
        """Convert Claude's computer_use tool call to our Action format.

        Claude's computer_use actions:
            {"action": "left_click", "coordinate": [x, y]}
            {"action": "type", "text": "hello"}
            {"action": "key", "key": "Return"}
            {"action": "scroll", "coordinate": [x, y], "direction": "down", "amount": 3}
            {"action": "screenshot"}  — request new screenshot
            {"action": "left_click_drag", "start_coordinate": [x1,y1], "coordinate": [x2,y2]}
            {"action": "double_click", "coordinate": [x, y]}
            {"action": "right_click", "coordinate": [x, y]}
        """
        action_type = tool_input.get("action", "")
        coord = tool_input.get("coordinate", [0, 0])

        if action_type in ("left_click", "click"):
            return Action(ActionType.CLICK, {"x": coord[0], "y": coord[1], "button": "left"})

        elif action_type == "right_click":
            return Action(ActionType.CLICK, {"x": coord[0], "y": coord[1], "button": "right"})

        elif action_type == "double_click":
            return Action(ActionType.DOUBLE_CLICK, {"x": coord[0], "y": coord[1]})

        elif action_type == "type":
            return Action(ActionType.TYPE, {"text": tool_input.get("text", "")})

        elif action_type == "key":
            # Claude uses Return/Tab/Escape etc. Normalize.
            key = (
                tool_input.get("key")
                or tool_input.get("text")
                or tool_input.get("keys")
                or ""
            )
            key = _normalize_claude_key(key)
            return Action(ActionType.KEY_PRESS, {"keys": key})

        elif action_type == "scroll":
            direction = tool_input.get("direction", "down")
            amount = tool_input.get("amount", 3)
            params = {"direction": direction, "amount": amount}
            if coord and coord != [0, 0]:
                params["x"] = coord[0]
                params["y"] = coord[1]
            return Action(ActionType.SCROLL, params)

        elif action_type == "left_click_drag":
            start = tool_input.get("start_coordinate", [0, 0])
            end = coord
            return Action(ActionType.DRAG, {
                "start_x": start[0], "start_y": start[1],
                "end_x": end[0], "end_y": end[1],
            })

        elif action_type == "screenshot":
            # Claude requesting a new screenshot = wait + observe
            return Action(ActionType.WAIT, {"seconds": 0.5})

        elif action_type == "wait":
            return Action(ActionType.WAIT, {"seconds": tool_input.get("seconds", 1.0)})

        elif action_type == "triple_click":
            # Select all text in field — map to ctrl+a
            return Action(ActionType.KEY_PRESS, {"keys": "ctrl+a"})

        else:
            logger.warning(f"Unknown Claude computer action: {action_type}")
            return Action(ActionType.WAIT, {"seconds": 1.0}, reasoning=f"Unknown: {action_type}")

    @staticmethod
    def _parse_text_fallback(text: str) -> Action:
        """Parse action from text when no tool_use block is present."""
        import re

        text_lower = text.lower()

        if re.search(r'done|complete|finished|task.*(complete|done)', text_lower):
            success = "fail" not in text_lower and "cannot" not in text_lower
            return Action(ActionType.DONE, {"success": success, "summary": text[:200]})

        if re.search(r'wait|loading|progress', text_lower):
            return Action(ActionType.WAIT, {"seconds": 1.0})

        return Action(ActionType.WAIT, {"seconds": 1.0}, reasoning=text[:200])


def _normalize_claude_key(key: str) -> str:
    """Normalize Claude's key names to our format.

    Claude uses: Return, Tab, Escape, BackSpace, space, super, etc.
    We use: enter, tab, escape, backspace, space, super, etc.
    """
    key_map = {
        "Return": "enter",
        "Enter": "enter",
        "BackSpace": "backspace",
        "Escape": "escape",
        "Tab": "tab",
        "Delete": "delete",
        "Home": "home",
        "End": "end",
        "Page_Up": "pageup",
        "Page_Down": "pagedown",
        "Up": "up",
        "Down": "down",
        "Left": "left",
        "Right": "right",
        "F1": "f1", "F2": "f2", "F3": "f3", "F4": "f4",
        "F5": "f5", "F6": "f6", "F7": "f7", "F8": "f8",
        "F9": "f9", "F10": "f10", "F11": "f11", "F12": "f12",
    }

    # Handle key combos: "ctrl+Return" → "ctrl+enter"
    parts = key.split("+")
    normalized = []
    for part in parts:
        part = part.strip()
        normalized.append(key_map.get(part, part.lower()))
    return "+".join(normalized)


def _infer_key_from_reasoning(text: str) -> str:
    """Recover an intended key when Claude emits ``{"action": "key"}`` empty.

    This has shown up with computer-use responses whose reasoning clearly says
    "press Tab" while the structured key payload is empty. Treat it as a
    narrow repair for common navigation keys rather than guessing arbitrary
    keyboard input.
    """
    if not text:
        return ""

    patterns = [
        ("Tab", r"\b(?:press|use|using|hit|send|tap|try)\s+(?:the\s+)?[`'\"]?tab(?:\s+key)?[`'\"]?\b"),
        ("Return", r"\b(?:press|use|using|hit|send|tap|try)\s+(?:the\s+)?[`'\"]?(?:enter|return)(?:\s+key)?[`'\"]?\b"),
        ("Escape", r"\b(?:press|use|using|hit|send|tap|try)\s+(?:the\s+)?[`'\"]?escape(?:\s+key)?[`'\"]?\b"),
        ("BackSpace", r"\b(?:press|use|using|hit|send|tap|try)\s+(?:the\s+)?[`'\"]?(?:backspace|back space)(?:\s+key)?[`'\"]?\b"),
    ]
    for key, pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return key
    return ""
