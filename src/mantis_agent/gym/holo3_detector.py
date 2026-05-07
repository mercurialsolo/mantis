"""Holo3-as-detector — runtime helpers that ask Holo3 vision questions.

The runner uses these to make CUA-pure element-aware decisions without
touching the DOM:

- :func:`extract_form_values` — one-shot text query parsing the plan into
  an ordered list of values to type into form fields.
- :func:`detect_focused_field` — given a screenshot, asks Holo3 whether an
  editable input is currently focused.
- :func:`find_submit_button` — given a screenshot, asks Holo3 to locate
  the visible submit/login/update button.

All helpers fail soft: they return ``None`` (or an empty list) on any
parse / API error so the calling code falls back to its prior heuristic
or simply skips the substitution.

The same Holo3 model that emits agent actions answers these queries —
keeping detection in the CUA loop. They use ``Holo3Brain.detect_with_image``
(no tool calls, free-text output), separate from the agent's ``think`` path.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)
_JSON_ARR_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _first_json(text: str, pattern: re.Pattern[str]) -> Any | None:
    """Pull the first JSON literal matching ``pattern`` out of ``text``."""
    if not text:
        return None
    m = pattern.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _log_extracted(values: list[dict[str, str]]) -> None:
    """Log each extracted value (label only — never log secret values)."""
    if not values:
        return
    pairs = ", ".join(f"{v.get('label')!r}=<{len(v.get('value',''))} chars>" for v in values)
    logger.info("extract_form_values: %d values: %s", len(values), pairs)


def extract_form_values(brain: Any, task: str) -> list[dict[str, str]]:
    """Ask Holo3 to extract every {label, value} the plan instructs to type.

    Returns a list of ``{"label": str, "value": str}`` dicts in the order
    the plan suggests they will be typed. ``label`` describes the field
    in 1-3 words ("user id", "password", "industry vertical", "zip code")
    and is used by force-fill to match the focused field's visible label
    so the right value lands in the right field — not just FIFO order.

    Returns ``[]`` on failure or when the plan has no form values.
    """
    if not isinstance(task, str) or not task.strip():
        return []
    if not hasattr(brain, "query"):
        return []
    prompt = (
        "You will read a workflow plan and extract every value the plan "
        "instructs the agent to TYPE into a form field. For each value, "
        "output a {label, value} pair where label describes the field in "
        "1-3 lowercase words (e.g. \"user id\", \"password\", \"industry "
        "vertical\", \"zip code\", \"search radius\", \"company name\").\n\n"
        "Include: usernames, passwords, emails, search terms, ZIP codes, "
        "free-text answers, dropdown selections that the agent must type.\n"
        "Exclude: URLs the agent should navigate to, names of buttons to "
        "click, headings, step numbers.\n\n"
        "Output STRICT JSON only — a single array of objects, no prose. "
        "Empty array [] if the plan has no form-fill values.\n"
        "Schema: [{\"label\": \"<short field label>\", \"value\": \"<exact text to type>\"}, ...]\n\n"
        f"PLAN:\n{task[:6000]}"
    )
    try:
        raw = brain.query(prompt)
    except Exception as exc:
        logger.warning("extract_form_values brain.query failed: %s", exc)
        return []
    parsed = _first_json(raw, _JSON_ARR_RE)
    if not isinstance(parsed, list):
        return []
    out: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().lower()
        value = str(item.get("value") or "").strip()
        if not value:
            continue
        out.append({"label": label, "value": value})
    _log_extracted(out)
    return out


def detect_focused_field(
    brain: Any,
    screenshot: Image.Image,
    click_coords: tuple[int, int] | None = None,
) -> dict | None:
    """Ask Holo3 whether the screenshot has an editable input currently focused.

    Returns a dict ``{focused, label, type}`` on success, ``None`` on
    failure or when the model says no field is focused. ``label`` is the
    field's visible label/placeholder; ``type`` is the input semantic
    (``text`` / ``password`` / ``email`` / ``other``).

    Logs the detector's raw decision for diagnostics — without this we
    can't tell whether substitution mismatches come from bad detection or
    bad matching.
    """
    if not hasattr(brain, "detect_with_image"):
        return None
    where = (
        f" The agent just clicked at ({click_coords[0]}, {click_coords[1]})."
        if click_coords else ""
    )
    prompt = (
        "Look at this screenshot. Is the user currently focused on an "
        "editable text input or textarea (cursor visible inside it, "
        "border highlighted, or otherwise selected as the active "
        f"input)?{where}\n\n"
        "Output STRICT JSON only:\n"
        '{"focused": true|false, '
        '"label": "<visible label or placeholder, empty string if none>", '
        '"type": "text"|"password"|"email"|"number"|"search"|"other"}'
    )
    raw = brain.detect_with_image(prompt, screenshot)
    parsed = _first_json(raw, _JSON_OBJ_RE)
    if not isinstance(parsed, dict):
        logger.info("detect_focused_field: raw=%r → no JSON", (raw or "")[:200])
        return None
    logger.info(
        "detect_focused_field: click=%s → focused=%s label=%r type=%r",
        click_coords, parsed.get("focused"),
        parsed.get("label"), parsed.get("type"),
    )
    if not bool(parsed.get("focused", False)):
        return None
    return {
        "focused": True,
        "label": str(parsed.get("label") or ""),
        "type": str(parsed.get("type") or "other"),
    }


def verify_done(
    brain: Any,
    screenshot: Image.Image | None,
    plan: str,
    summary: str,
) -> dict | None:
    """Ask Holo3 whether the screenshot supports an alleged ``done(success=True)``.

    Run 023 surfaced false-success done: Holo3 emitted
    ``done(success=True, summary='Updated lead industry to Space Exploration')``
    after only completing login. The summary parroted the plan's expected
    end-state without the agent actually doing the work. This verifier
    asks Holo3 to compare the current screenshot against the claim.

    Returns ``{"valid": bool, "reason": str}`` or ``None`` if the call
    fails. The runner gates ``done(success=True)`` on ``valid=True``.
    """
    if not hasattr(brain, "detect_with_image"):
        return None
    if screenshot is None:
        return None
    prompt = (
        "You are observing a CUA agent that just declared its workflow "
        "complete. Verify the claim against what is on screen.\n\n"
        "WORKFLOW PLAN:\n"
        f"{plan[:3000]}\n\n"
        "AGENT'S DONE SUMMARY:\n"
        f"\"{summary}\"\n\n"
        "Look at the current screenshot. Does the screen show concrete "
        "evidence that the workflow described in the plan was actually "
        "completed? Look for: a final confirmation message or success "
        "indicator on screen; the expected end-state UI (post-update "
        "view, completion page, etc.); a URL change consistent with the "
        "workflow concluding. Be strict — if the summary claims actions "
        "that are not visibly evidenced on screen, mark valid=false.\n\n"
        "Output STRICT JSON only:\n"
        '{"valid": true|false, '
        '"reason": "<one short sentence: what you actually see vs what was claimed>"}'
    )
    raw = brain.detect_with_image(prompt, screenshot)
    parsed = _first_json(raw, _JSON_OBJ_RE)
    if not isinstance(parsed, dict):
        logger.info("verify_done: raw=%r → no JSON", (raw or "")[:200])
        return None
    out = {
        "valid": bool(parsed.get("valid", False)),
        "reason": str(parsed.get("reason") or ""),
    }
    logger.info("verify_done: valid=%s reason=%r", out["valid"], out["reason"][:200])
    return out


def find_submit_button(
    brain: Any,
    screenshot: Image.Image,
    plan_intent: str | None = None,
) -> dict[str, int | str] | None:
    """Ask Holo3 to locate a submit / login / update button on screen.

    Returns ``{x, y, label}`` (absolute pixel coords at the button center)
    or ``None`` if no button is visible. The runner uses this to inject
    a final click after the last force-fill value is typed — Holo3's
    well-documented "doesn't submit after typing" failure mode.
    """
    if not hasattr(brain, "detect_with_image"):
        return None
    extra = ""
    if plan_intent:
        # Trim down — too much plan text drowns the question.
        snippet = plan_intent.replace("\n", " ").strip()[:400]
        extra = f"\n\nThe agent just finished typing form values for this plan: \"{snippet}\""
    prompt = (
        "Look at this screenshot. Find a visible button that submits "
        "the form on screen (text like 'Sign in', 'Log in', 'Login', "
        "'Submit', 'Continue', 'Next', 'Update', 'Save', 'Update Lead'). "
        "Pick the button most likely to advance the workflow."
        f"{extra}\n\n"
        "Output STRICT JSON only:\n"
        '{"found": true|false, "x": <int>, "y": <int>, '
        '"label": "<button text>"}\n'
        "Coordinates are absolute screen pixels at the center of the "
        "button. Set found=false if no submit-style button is visible."
    )
    raw = brain.detect_with_image(prompt, screenshot)
    parsed = _first_json(raw, _JSON_OBJ_RE)
    if not isinstance(parsed, dict):
        logger.info("find_submit_button: raw=%r → no JSON", (raw or "")[:200])
        return None
    logger.info(
        "find_submit_button: found=%s label=%r at (%s,%s)",
        parsed.get("found"), parsed.get("label"),
        parsed.get("x"), parsed.get("y"),
    )
    if not bool(parsed.get("found", False)):
        return None
    try:
        return {
            "x": int(parsed["x"]),
            "y": int(parsed["y"]),
            "label": str(parsed.get("label") or "submit"),
        }
    except (KeyError, TypeError, ValueError):
        return None
