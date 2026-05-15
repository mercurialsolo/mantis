"""Fara-7B CUA Brain — tool-calling via vLLM OpenAI-compatible API.

Fara-7B (``microsoft/Fara-7B``) is a Qwen2.5-VL-based 7B agentic model
released by Microsoft (MIT license). It runs natively on vLLM and emits
a single ``computer_use`` tool call per turn.

Differences from Holo3:

* **Native vLLM serve** — Qwen2.5-VL is first-class in vLLM, so no
  llama.cpp / GGUF detour. Start with::

      vllm serve microsoft/Fara-7B --port 5000 --dtype auto

* **Raw screen-pixel coordinates** at a fixed input resolution
  (default ``1428×896``). We resize the screenshot to that size before
  sending and linearly scale coordinates back to the real screen.
  No Qwen smart-resize math.

* **Different action space**. Fara's ``computer_use`` tool emits
  ``left_click``, ``type``, ``key``, ``scroll``, ``visit_url``,
  ``history_back``, ``web_search``, ``wait``, ``terminate``,
  ``mouse_move``, ``pause_and_memorize_fact``. The first six are mapped
  cleanly; ``visit_url``/``history_back``/``web_search`` ride on the
  env's URL auto-navigate path (``TYPE`` with a URL string triggers
  ``ctrl+l`` + paste + Enter in xdotool/playwright); ``mouse_move`` and
  ``pause_and_memorize_fact`` collapse to short ``WAIT`` no-ops because
  no Mantis action mirrors them.

* **Action regressions vs Holo3** — Fara has no ``double_click``,
  ``right_click``, or ``drag``. A planner that needs those should
  prefer Holo3.
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
from urllib.parse import quote_plus

import requests
from PIL import Image

from .actions import Action, ActionType
from .prompts import load_prompt

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = load_prompt("fara_system")

# Microsoft's reference resolution for Fara-7B screenshots. Override
# via constructor or the MANTIS_FARA_INPUT_WH env var ("WxH").
DEFAULT_INPUT_SIZE: tuple[int, int] = (1428, 896)


# ── Fara's computer_use tool (single function with action discriminator) ────

FARA_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": (
                "Perform one computer-use action. Exactly one tool call per "
                "turn. Coordinates are absolute pixels on the screenshot you "
                "were given."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "left_click", "type", "key", "scroll",
                            "wait", "visit_url", "history_back", "web_search",
                            "mouse_move", "pause_and_memorize_fact",
                            "terminate",
                        ],
                    },
                    "coordinate": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "[x, y] in screen pixels.",
                    },
                    "text": {"type": "string"},
                    "url": {"type": "string"},
                    "query": {"type": "string"},
                    "duration": {"type": "number"},
                    "scroll_direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                    },
                    "scroll_amount": {"type": "integer"},
                    "status": {
                        "type": "string",
                        "enum": ["success", "failure"],
                    },
                    "summary": {"type": "string"},
                },
                "required": ["action"],
            },
        },
    },
]


_PREDICTED_LINE_RE = re.compile(
    r"^\s*Predicted\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_predicted_outcome(text: str) -> str:
    """Pull the first ``Predicted: <delta>`` line out of a Fara response.

    Same contract as the Holo3 helper — empty when the model didn't
    emit it. Trims wrapping quotes and caps at 240 chars.
    """
    if not text:
        return ""
    match = _PREDICTED_LINE_RE.search(text)
    if not match:
        return ""
    line = match.group(1).strip().strip('"').strip("'").strip()
    return line[:240]


def _image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _resize_for_model(img: Image.Image, target: tuple[int, int]) -> Image.Image:
    """Resize a screenshot to Fara's input resolution.

    Fara was trained at a fixed resolution; resizing keeps coordinates
    interpretable against the screenshot we hand the model. Image stays
    in RGB; no padding (Fara's vision tower is comfortable with the
    aspect change for typical 16:9 / 16:10 screens).
    """
    if img.size == target:
        return img
    return img.resize(target, Image.LANCZOS)


def _scale_coords(
    x: int, y: int,
    model_size: tuple[int, int],
    screen_size: tuple[int, int],
) -> tuple[int, int]:
    """Map model-space coords back to actual screen pixels.

    ``model_size`` is the resolution we sent the screenshot at (Fara's
    input). ``screen_size`` is the real Chrome viewport.
    """
    mw, mh = model_size
    sw, sh = screen_size
    sx = int(round(x * sw / max(mw, 1)))
    sy = int(round(y * sh / max(mh, 1)))
    return sx, sy


def _resolve_input_size(explicit: tuple[int, int] | None) -> tuple[int, int]:
    if explicit is not None:
        return explicit
    raw = os.environ.get("MANTIS_FARA_INPUT_WH", "").strip()
    if raw:
        m = re.match(r"^(\d+)x(\d+)$", raw, re.IGNORECASE)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return DEFAULT_INPUT_SIZE


@dataclass
class InferenceResult:
    """Result from a single FaraBrain inference cycle.

    Shape mirrors :class:`mantis_agent.brain_holo3.InferenceResult` so the
    task loop can swap brains without code changes.
    """

    action: Action
    raw_output: str
    thinking: str = ""
    tokens_used: int = 0
    predicted_outcome: str = ""


# ── Brain ───────────────────────────────────────────────────────────────────


class FaraBrain:
    """Fara-7B brain via vLLM OpenAI-compatible API.

    Args:
        base_url: vLLM ``/v1`` URL (e.g. ``http://localhost:5000/v1``).
        model: served model name (``microsoft/Fara-7B`` by default).
        api_key: optional bearer token.
        max_tokens: response cap.
        temperature: sampling temperature; default ``0.0``.
        screen_size: actual screen resolution. Coords map to this.
        input_size: resolution Fara was trained at. Screenshots are
            resized to this before send, and output coords are scaled
            back to ``screen_size``.
        use_tool_calling: send the Fara ``computer_use`` tool schema.
            Set to ``False`` to receive plain text and rely on the
            text-format parser only.
        extra_headers: additional HTTP headers (e.g. gateway auth).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5000/v1",
        model: str = "microsoft/Fara-7B",
        api_key: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        screen_size: tuple[int, int] = (1280, 720),
        input_size: tuple[int, int] | None = None,
        use_tool_calling: bool = True,
        extra_headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("FARA_API_KEY", "")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.default_screen_size = screen_size
        self.input_size = _resolve_input_size(input_size)
        self.use_tool_calling = use_tool_calling
        self.extra_headers = dict(extra_headers or {})
        self.model_name = "Fara-7B"

    @property
    def _headers(self) -> dict:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if self.extra_headers:
            h.update(self.extra_headers)
        return h

    def load(self) -> None:
        """Verify the vLLM server is reachable. 2 min budget."""
        for attempt in range(12):
            try:
                resp = requests.get(
                    f"{self.base_url}/models",
                    headers=self._headers, timeout=30,
                )
                resp.raise_for_status()
                logger.info("Fara API connected: %s", resp.json())
                return
            except Exception as e:  # noqa: BLE001 — startup probe
                if attempt < 11:
                    logger.info(
                        "Fara not ready (attempt %d/12), waiting 10s...",
                        attempt + 1,
                    )
                    time.sleep(10)
                else:
                    raise RuntimeError(
                        f"Cannot connect to Fara API at {self.base_url}: {e}"
                    )

    def query(self, prompt: str, response_format: str = "json") -> str:
        """Text-only prompt → text response. Used by helper detection paths."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload, headers=self._headers, timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"].get("content", "") or ""
        except Exception as e:  # noqa: BLE001
            logger.error("Fara query failed: %s", e)
            return ""

    def detect_with_image(
        self, prompt: str, image: Image.Image, max_tokens: int = 256,
    ) -> str:
        """Vision prompt → free text. Tool calling is disabled here."""
        resized = _resize_for_model(image, self.input_size)
        payload: dict = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{_image_to_base64(resized)}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload, headers=self._headers, timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"].get("content", "") or ""
        except Exception as e:  # noqa: BLE001
            logger.warning("Fara detect_with_image failed: %s", e)
            return ""

    def think(
        self,
        frames: list[Image.Image],
        task: str,
        action_history: list[Action] | None = None,
        screen_size: tuple[int, int] = (1920, 1080),
    ) -> InferenceResult:
        """Single perception-reasoning-action cycle."""
        messages = self._build_messages(frames, task, action_history)

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.use_tool_calling:
            payload["tools"] = FARA_TOOLS
            payload["tool_choice"] = "auto"

        data: dict | None = None
        for attempt in range(4):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload, headers=self._headers, timeout=120,
                )
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 30)
                    logger.warning(
                        "Fara 429 — waiting %ds (attempt %d/4)",
                        wait, attempt + 1,
                    )
                    time.sleep(wait)
                    continue
                if resp.status_code == 400:
                    logger.warning("Fara 400: %s", resp.text[:500])
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:  # noqa: BLE001
                if attempt < 3:
                    logger.warning(
                        "Fara attempt %d failed: %s", attempt + 1, e,
                    )
                    time.sleep(3)
                else:
                    logger.error("Fara request failed after 4 attempts: %s", e)
                    return InferenceResult(
                        action=Action(
                            ActionType.WAIT,
                            {"seconds": 2.0},
                            reasoning=str(e),
                        ),
                        raw_output=str(e),
                    )

        if data is None:
            return InferenceResult(
                action=Action(
                    ActionType.WAIT, {"seconds": 5.0},
                    reasoning="Rate limited",
                ),
                raw_output="429 rate limited after retries",
            )

        return self._parse_response(data, screen_size)

    def _build_messages(
        self,
        frames: list[Image.Image],
        task: str,
        action_history: list[Action] | None,
    ) -> list[dict]:
        """OpenAI-format messages with Fara's input-resolution screenshots."""
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        content: list[dict] = []
        n_frames = len(frames)
        for i, frame in enumerate(frames):
            resized = _resize_for_model(frame, self.input_size)
            label = "CURRENT" if i == n_frames - 1 else f"t-{n_frames - 1 - i}"
            content.append({"type": "text", "text": f"[Frame {label}]"})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_image_to_base64(resized)}",
                },
            })

        parts = [
            f"\nTask: {task}",
            (
                f"Screen size: {self.input_size[0]}x{self.input_size[1]} pixels"
                " (coordinates are in this space)"
            ),
        ]
        if action_history:
            recent = action_history[-5:]
            history = "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(recent))
            parts.append(f"Recent actions:\n{history}")
        # #435: re-feed the model's own ``pause_and_memorize_fact``
        # entries from earlier turns. The Fara model card calls this
        # out as an explicit scratchpad primitive; the doc's §4
        # *"plumb it through"* note is satisfied by accumulating
        # non-empty ``Action.memorize_fact`` strings off the action
        # history and rendering them in their own block. Bounded to
        # the last 10 facts so a chatty plan doesn't blow the prompt
        # footprint.
        if action_history:
            facts = [
                str(a.memorize_fact)
                for a in action_history
                if getattr(a, "memorize_fact", "")
            ][-10:]
            if facts:
                fact_block = "\n".join(f"  - {f}" for f in facts)
                parts.append(f"Memorized facts (your earlier notes):\n{fact_block}")
        content.append({"type": "text", "text": "\n".join(parts)})

        messages.append({"role": "user", "content": content})
        return messages

    # ── Response parsing ────────────────────────────────────────────────

    def _parse_response(
        self,
        data: dict,
        screen_size: tuple[int, int],
    ) -> InferenceResult:
        """Parse Fara's response. Triple fallback:

        1. OpenAI ``tool_calls`` with ``computer_use`` (primary).
        2. ``computer_use({...})`` text in content.
        3. terminate/DONE/FAIL keyword.
        """
        choice = data["choices"][0]
        message = choice["message"]
        raw_output = json.dumps(message)

        thinking = message.get("reasoning_content", "") or ""
        content_text = message.get("content", "") or ""

        predicted = _extract_predicted_outcome(content_text)

        if not thinking and "<think>" in content_text:
            m = re.search(r"<think>(.*?)</think>", content_text, re.DOTALL)
            if m:
                thinking = m.group(1).strip()
                content_text = (
                    content_text[:m.start()] + content_text[m.end():]
                ).strip()

        # Strategy 1: OpenAI tool_calls
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            tc = tool_calls[0]
            func = tc.get("function", {})
            name = func.get("name", "")
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            action = self._action_from_fara(name, args, screen_size)
            if not thinking:
                thinking = content_text
            if action is not None:
                return InferenceResult(
                    action=action, raw_output=raw_output,
                    thinking=thinking, predicted_outcome=predicted,
                    tokens_used=data.get("usage", {}).get("total_tokens", 0),
                )

        # Strategy 2: computer_use({...}) text
        text_action = self._parse_text_call(content_text, screen_size)
        if text_action is not None:
            if not thinking:
                idx = content_text.find("computer_use")
                if idx > 0:
                    thinking = content_text[:idx].strip()
            return InferenceResult(
                action=text_action, raw_output=raw_output,
                thinking=thinking, predicted_outcome=predicted,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
            )

        # Strategy 3: terminate / DONE / FAIL
        term = re.search(
            r"terminate\([^)]*['\"](success|failure)['\"]",
            content_text, re.IGNORECASE,
        )
        if term:
            success = term.group(1).lower() == "success"
            if not thinking:
                thinking = content_text[:term.start()].strip()
            return InferenceResult(
                action=Action(
                    ActionType.DONE,
                    {"success": success, "summary": thinking[:200]},
                ),
                raw_output=raw_output, thinking=thinking,
                predicted_outcome=predicted,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
            )

        upper = content_text.upper()
        if "DONE" in upper:
            action = Action(
                ActionType.DONE,
                {"success": True, "summary": content_text[:200]},
            )
        elif "FAIL" in upper:
            action = Action(
                ActionType.DONE,
                {"success": False, "summary": content_text[:200]},
            )
        else:
            logger.warning(
                "Fara: no parseable action in %r", content_text[:200],
            )
            action = Action(
                ActionType.WAIT, {"seconds": 1.0},
                reasoning=f"unparsed: {content_text[:100]}",
            )

        return InferenceResult(
            action=action, raw_output=raw_output,
            thinking=thinking, predicted_outcome=predicted,
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
        )

    def _parse_text_call(
        self, text: str, screen_size: tuple[int, int],
    ) -> Action | None:
        """Recover ``computer_use({...})`` JSON-ish calls from content text.

        Tolerates Python-style single quotes / ``True``/``False`` and
        bare ``{action: ...}`` blocks not wrapped in ``computer_use(...)``.
        """
        m = re.search(
            r"computer_use\s*\(\s*(\{.*?\})\s*\)",
            text, re.DOTALL,
        )
        raw_args: str | None = m.group(1) if m else None
        if raw_args is None:
            m2 = re.search(
                r"\{[^{}]*['\"]action['\"]\s*:\s*['\"]\w+['\"][^{}]*\}",
                text, re.DOTALL,
            )
            if m2:
                raw_args = m2.group(0)
        if raw_args is None:
            return None
        args = _coerce_json(raw_args)
        if not args:
            return None
        return self._action_from_fara(
            "computer_use", args, screen_size,
        )

    # ── Fara action → Mantis Action ─────────────────────────────────────

    def _action_from_fara(
        self,
        tool_name: str,
        args: dict,
        screen_size: tuple[int, int],
    ) -> Action | None:
        """Translate one Fara ``computer_use`` call into a Mantis Action.

        ``tool_name`` is ignored when ``args["action"]`` is present —
        Fara packs every action under one tool. Returns ``None`` if the
        payload is too malformed to recover from; the caller falls
        through to the next strategy.
        """
        action = str(args.get("action", "")).lower().strip()
        if not action and tool_name != "computer_use":
            action = tool_name.lower()

        coord = args.get("coordinate") or args.get("coord")
        x = y = None
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            try:
                x = int(float(coord[0]))
                y = int(float(coord[1]))
            except (TypeError, ValueError):
                x = y = None
        if x is None and "x" in args and "y" in args:
            try:
                x = int(float(args["x"]))
                y = int(float(args["y"]))
            except (TypeError, ValueError):
                x = y = None

        def _to_screen(mx: int, my: int) -> tuple[int, int]:
            return _scale_coords(mx, my, self.input_size, screen_size)

        if action == "left_click":
            if x is None or y is None:
                return None
            sx, sy = _to_screen(x, y)
            return Action(
                ActionType.CLICK,
                {"x": sx, "y": sy, "button": "left"},
            )

        if action in ("type", "type_text"):
            # Fara emits both verbs depending on training-data variance — the
            # documented action is ``type`` but in practice many turns come
            # through as ``type_text`` (see #405). Treat as aliases.
            #
            # Fara also folds two semantics into the same call via flags
            # rather than emitting separate verbs (#405 follow-up):
            #   * ``delete_existing_text: true`` — clear field before typing
            #   * ``press_enter: true``         — Return after typing
            # Both are forwarded to env executors via TYPE params; xdotool
            # and playwright envs honour them. URL-shaped text keeps its
            # own auto-navigate path (which already clears + submits).
            text = str(args.get("text", ""))
            params: dict = {"text": text}
            if args.get("delete_existing_text") or args.get("clear_first"):
                params["clear_first"] = True
            if args.get("press_enter"):
                params["press_enter"] = True
            return Action(ActionType.TYPE, params)

        if action in ("key", "press_key"):
            keys = args.get("text") or args.get("key") or args.get("keys") or ""
            if isinstance(keys, list):
                keys = "+".join(str(k) for k in keys)
            return Action(ActionType.KEY_PRESS, {"keys": str(keys)})

        if action in ("submit", "enter"):
            # Form submit / Enter — Fara variants emit either verb. Always
            # map to a Return key press; works on focused inputs and on
            # selected buttons. If args carry an explicit ``text`` (e.g.
            # ``submit(text="Return")``), honour it.
            keys = str(args.get("text") or args.get("key") or "Return")
            return Action(
                ActionType.KEY_PRESS, {"keys": keys},
                reasoning=f"fara {action}",
            )

        if action == "scroll":
            direction = str(
                args.get("scroll_direction") or args.get("direction") or "down"
            )
            try:
                amount = int(args.get("scroll_amount") or args.get("amount") or 3)
            except (TypeError, ValueError):
                amount = 3
            params: dict = {"direction": direction, "amount": amount}
            if x is not None and y is not None:
                sx, sy = _to_screen(x, y)
                params["x"] = sx
                params["y"] = sy
            return Action(ActionType.SCROLL, params)

        if action == "wait":
            try:
                seconds = float(args.get("duration") or args.get("seconds") or 1.0)
            except (TypeError, ValueError):
                seconds = 1.0
            return Action(ActionType.WAIT, {"seconds": seconds})

        if action == "visit_url":
            url = str(args.get("url") or args.get("text") or "").strip()
            if not url:
                return None
            # xdotool / playwright env both treat TYPE-of-URL as
            # ctrl+l → paste → Return. That's exactly visit_url.
            return Action(
                ActionType.TYPE,
                {"text": url},
                reasoning=f"fara visit_url:{url}",
            )

        if action == "history_back":
            return Action(
                ActionType.KEY_PRESS,
                {"keys": "alt+Left"},
                reasoning="fara history_back",
            )

        if action == "web_search":
            q = str(args.get("query") or args.get("text") or "").strip()
            if not q:
                return None
            url = f"https://www.google.com/search?q={quote_plus(q)}"
            return Action(
                ActionType.TYPE,
                {"text": url},
                reasoning=f"fara web_search:{q[:80]}",
            )

        if action == "mouse_move":
            # No Mantis equivalent; xdotool clicks already mousemove
            # before the press, so this collapses to a short WAIT and
            # the model's follow-up click will land at the right pixel.
            logger.debug("Fara mouse_move → WAIT (no-op)")
            return Action(
                ActionType.WAIT, {"seconds": 0.2},
                reasoning="fara mouse_move noop",
            )

        if action == "pause_and_memorize_fact":
            # #435: surface the fact through the structured
            # ``memorize_fact`` field instead of stuffing it in
            # ``reasoning``. The runner pops it into its
            # ``_memorized_facts`` list and the next turn's prompt
            # re-feeds the list as a ``Memorized facts:`` block, which
            # is what the doc means by *"essentially an explicit
            # scratchpad primitive"* in §4. The action itself is still
            # a no-op WAIT — the side-effect is the memo, not movement.
            note = str(args.get("text") or args.get("fact") or "")[:200]
            return Action(
                ActionType.WAIT, {"seconds": 0.2},
                reasoning=f"fara memo:{note}",
                memorize_fact=note,
            )

        if action == "terminate":
            status = str(args.get("status", "success")).lower()
            success = status == "success"
            summary = str(args.get("summary") or args.get("message") or "")[:200]
            return Action(
                ActionType.DONE,
                {"success": success, "summary": summary},
            )

        # Regressions: Fara doesn't natively emit these, but if a
        # planner-wrapped variant slips through, do the obvious thing.
        if action in ("double_click", "doubleclick"):
            if x is None or y is None:
                return None
            sx, sy = _to_screen(x, y)
            return Action(
                ActionType.DOUBLE_CLICK, {"x": sx, "y": sy},
                reasoning="fara double_click (non-native)",
            )

        logger.warning("Fara: unknown action %r — emitting WAIT", action)
        return Action(
            ActionType.WAIT, {"seconds": 0.5},
            reasoning=f"fara unknown:{action}",
        )


def _coerce_json(raw: str) -> dict:
    """Best-effort JSON loader for model-emitted dict-ish text.

    Tries strict JSON first; on failure, swaps single quotes for double
    and normalises Python ``True``/``False``/``None``.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    fixed = (
        raw.replace("'", '"')
        .replace("True", "true")
        .replace("False", "false")
        .replace("None", "null")
    )
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return {}
