"""Claude director — escalation-on-failure helper for the runner.

Called only when Holo3 is demonstrably stuck (loop detector triggered).
Sends the current screenshot + plan + recent action history to Claude and
asks "what's the next action?" — Claude returns a concrete action that the
runner substitutes for Holo3's stuck output. Used at most once per stuck
event so the bench stays predominantly Holo3-driven.

CUA-pure in spirit: Claude is also a vision LLM looking at the same
screenshot Holo3 sees. No DOM, no CDP — just a stronger model giving
tactical guidance when the primary brain stalls.

Why not always: at ~$0.01-0.05 + ~1-2 s latency per call, calling Claude
on every step would dominate cost and latency for what is meant to be a
Holo3 benchmark. Failure-only escalation keeps the architecture honest
about which model produced each action.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from io import BytesIO
from typing import Any

import requests
from PIL import Image

from ..actions import Action, ActionType

logger = logging.getLogger(__name__)


CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = os.environ.get("MANTIS_CLAUDE_DIRECTOR_MODEL", "claude-sonnet-4-5")


def _image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


# Director is intentionally NOT allowed to emit type_text — the
# runtime's force-fill pipeline owns form filling, and the two paths
# stomped on each other in run 027 (director typed user_id, force-fill
# typed it again later because the director didn't update force-fill
# state). Drop type_text from the action map; if Claude returns it
# anyway, _decision_to_action returns None and we fall through to the
# soft-loop nudge.
_ACTION_TYPE_MAP = {
    "click": ActionType.CLICK,
    "scroll": ActionType.SCROLL,
    "key_press": ActionType.KEY_PRESS,
    "wait": ActionType.WAIT,
}


def suggest_unstuck_action(
    plan: str,
    screenshot: Image.Image | None,
    recent_actions: list[Action],
    api_key: str | None,
    *,
    fill_done: list[str] | None = None,
    fill_pending: list[str] | None = None,
    submitted: bool = False,
    timeout: int = 20,
) -> Action | None:
    """Ask Claude what the agent should do, given it's stuck.

    Returns ``None`` if no API key is available, the screenshot is
    missing, the API call fails, or Claude doesn't return a parseable
    action. Falls back to the runner's existing soft-loop nudge in
    those cases — the director is purely additive scaffolding.

    ``fill_done`` / ``fill_pending`` / ``submitted`` propagate force-fill
    state so the director doesn't re-suggest already-completed actions
    (re-typing the username, clicking submit before password is typed,
    etc — observed in run 022).
    """
    if not api_key:
        return None
    if screenshot is None:
        return None

    history_lines = []
    for i, a in enumerate(recent_actions[-10:]):
        kind = a.action_type.value
        params = a.params or {}
        # Trim secret-like values (force-fill type_text) from the snapshot
        # we send to Claude — the password is not Claude's business.
        if kind == "type_text":
            text = str(params.get("text", ""))
            params = {"text": f"<{len(text)} chars>"}
        history_lines.append(f"  - {kind}({params})")
    history_str = "\n".join(history_lines) if history_lines else "  (no prior actions)"

    state_lines: list[str] = []
    if fill_done:
        state_lines.append(
            "  Form fields ALREADY FILLED (do NOT suggest typing these "
            f"again): {', '.join(fill_done)}"
        )
    if fill_pending:
        state_lines.append(
            "  Form fields STILL PENDING (these must be filled before "
            f"clicking submit): {', '.join(fill_pending)}"
        )
    if submitted:
        state_lines.append(
            "  Form has ALREADY been submitted via Enter; do NOT suggest "
            "clicking a submit button again — focus on what comes after."
        )
    state_str = "\n".join(state_lines) if state_lines else "  (no force-fill state yet)"

    prompt = (
        "You are observing a CUA agent (Holo3) that has gotten stuck on a "
        "browser workflow. The agent has been issuing the same or similar "
        "actions without making visible progress, so the runtime escalated "
        "to you for tactical guidance.\n\n"
        "WORKFLOW PLAN:\n"
        f"{plan[:4000]}\n\n"
        "RECENT ACTIONS the agent took (most recent last):\n"
        f"{history_str}\n\n"
        "RUNTIME STATE — known progress flags:\n"
        f"{state_str}\n\n"
        "Look at the current screenshot and tell the agent the SINGLE next "
        "action that will most likely break it out of the loop. Pick the "
        "smallest, safest action that visibly advances the workflow. "
        "Honor the runtime-state flags above: never re-fill an "
        "already-filled field, never click submit until pending fields "
        "are filled, never re-submit a submitted form.\n\n"
        "IMPORTANT: do NOT suggest type_text. The runtime owns form filling "
        "via a separate force-fill pipeline; the runtime-state flags above "
        "tell you which fields are pending. Your job is unsticking the agent "
        "with click / scroll / key_press / wait actions only.\n\n"
        "Reply STRICT JSON only, no prose:\n"
        '{\n'
        '  "action_type": "click" | "scroll" | "key_press" | "wait",\n'
        '  "x": <int — center pixel, only when action_type=click>,\n'
        '  "y": <int — center pixel, only when action_type=click>,\n'
        '  "keys": "<string e.g. Tab, Return, only when action_type=key_press>",\n'
        '  "direction": "up" | "down" | "left" | "right" (only for scroll),\n'
        '  "amount": <int — scroll amount, only for scroll>,\n'
        '  "seconds": <number, only when action_type=wait>,\n'
        '  "rationale": "<one short sentence — what you see and why this action>"\n'
        "}"
    )

    try:
        body = {
            "model": CLAUDE_MODEL,
            "max_tokens": 400,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _image_to_base64(screenshot),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        resp = requests.post(
            CLAUDE_API_URL,
            json=body,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("claude director request failed: %s", exc)
        return None

    text = ""
    for block in data.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            text += block.get("text", "")
    if not text:
        return None

    m = _JSON_OBJ_RE.search(text)
    if not m:
        logger.info("claude director: no JSON in response: %r", text[:200])
        return None
    try:
        decision = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        logger.info("claude director: JSON decode failed (%s): %r", exc, text[:200])
        return None

    return _decision_to_action(decision)


def _decision_to_action(decision: dict) -> Action | None:
    raw_type = str(decision.get("action_type") or "").strip().lower()
    action_type = _ACTION_TYPE_MAP.get(raw_type)
    if action_type is None:
        return None

    rationale = str(decision.get("rationale") or "").strip()
    reasoning = f"claude-director: {rationale}" if rationale else "claude-director"

    try:
        if action_type == ActionType.CLICK:
            return Action(
                action_type,
                {"x": int(decision["x"]), "y": int(decision["y"]), "button": "left"},
                reasoning=reasoning,
            )
        if action_type == ActionType.KEY_PRESS:
            return Action(action_type, {"keys": str(decision.get("keys", ""))}, reasoning=reasoning)
        if action_type == ActionType.SCROLL:
            params = {
                "direction": str(decision.get("direction") or "down"),
                "amount": int(decision.get("amount") or 5),
            }
            return Action(action_type, params, reasoning=reasoning)
        if action_type == ActionType.WAIT:
            return Action(action_type, {"seconds": float(decision.get("seconds") or 2.0)}, reasoning=reasoning)
    except (KeyError, TypeError, ValueError) as exc:
        logger.info("claude director: malformed decision (%s): %r", exc, decision)
        return None
    return None
