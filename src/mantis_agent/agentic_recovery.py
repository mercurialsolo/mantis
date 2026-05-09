"""Agentic failure-recovery loop — analyse + adapt instead of halting.

When a required step fails terminally (after retries + handler
escalation), the runner's existing path halts with a HALT reason.
This module adds the missing layer: ask Claude to analyse the
failure (step intent + last screenshot + failure data) and decide
how to recover.

Four recovery modes, in order of cheapness:

1. ``add_hint`` — the step is structurally correct but the search
   needs a clarifying instruction. Example: ``"the button is
   labeled 'Save', not 'Update Lead'."`` The hint is appended to
   the next retry's prompt; the step itself is unchanged.

2. ``edit_step`` — the step has wrong type / params / labels and
   needs a localised replacement. Example: change a ``submit`` step
   with ``label='Update Lead'`` to ``submit`` with
   ``label='Save', aliases=['Submit', 'Update']``. Only the failed
   step is modified; surrounding plan structure is preserved.

3. ``insert_steps`` — the step's precondition isn't met and a
   helper sub-flow is needed first (scroll, dismiss modal, navigate
   back, wait for render). Helper steps splice in BEFORE the failed
   step; existing loop_target references are renumbered.

4. ``halt`` — the page state genuinely doesn't permit the step
   (target doesn't exist on the site, anti-bot block, account
   suspension). No plan tweak would help; surface the failure.

Bounded by per-step + per-run budgets so the recovery loop can't
run away. The legacy halt path is the fallback when budgets exhaust
or no Claude key is available.

Public API:

- :class:`RecoveryDecision` — typed dataclass for the four modes
- :func:`analyse_failure_and_recover` — the Claude tool_use call;
  returns a ``RecoveryDecision`` on success or ``None`` on any
  fallback path (no key, API error, schema violation)

The hook point lives in
:meth:`mantis_agent.gym.step_recovery.StepRecoveryPolicy.handle_failure`
— this module is decoupled and importable on its own.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


# Bumping this string invalidates any caller that keys recovery
# decisions by ``(plan_signature, step_index, prompt_version)``. Not
# used today but reserved so a future caching layer (issue #224
# Phase 2 pattern) can integrate cleanly.
RECOVERY_PROMPT_VERSION = "v1"


# Per-step + per-run recovery budgets. The policy enforces these
# before invoking the LLM call so a pathological page / step can't
# burn the run on recovery alone.
DEFAULT_MAX_RECOVERIES_PER_STEP = 2
DEFAULT_MAX_RECOVERIES_PER_RUN = 5


@dataclass
class RecoveryDecision:
    """The structured response from :func:`analyse_failure_and_recover`.

    ``mode`` is the discriminator. Other fields are populated based
    on the mode:

    - ``add_hint``  → ``hint`` (str)
    - ``edit_step`` → ``edited_step`` (dict with ``intent`` / ``type``
      / ``params``; missing keys preserve the original step's value)
    - ``insert_steps`` → ``inserted_steps`` (list of step-shaped
      dicts; spliced BEFORE the failed step)
    - ``halt`` → no extra fields
    """

    mode: str  # "add_hint" | "edit_step" | "insert_steps" | "halt"
    reasoning: str = ""
    hint: str = ""
    edited_step: dict[str, Any] = field(default_factory=dict)
    inserted_steps: list[dict[str, Any]] = field(default_factory=list)


_ANALYSIS_TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["add_hint", "edit_step", "insert_steps", "halt"],
            "description": (
                "How to recover. Pick the cheapest option that would "
                "plausibly succeed: add_hint < edit_step < insert_steps "
                "< halt. Only choose halt when no plan tweak would help "
                "(target genuinely missing, anti-bot block)."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": (
                "One-sentence explanation of what you saw on the "
                "screenshot and why this mode fits."
            ),
        },
        "hint": {
            "type": "string",
            "description": (
                "When mode=add_hint: a short clarifying instruction "
                "appended to the next retry's search prompt. Example: "
                "'the button is labeled Save not Update Lead.' "
                "Empty string for other modes."
            ),
        },
        "edited_step": {
            "type": "object",
            "description": (
                "When mode=edit_step: replacement field values for the "
                "failed step. Provide ONLY the fields that need to "
                "change. Empty object for other modes."
            ),
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "Replacement intent prose, or empty.",
                },
                "type": {
                    "type": "string",
                    "description": (
                        "Replacement step type — e.g. submit, click, "
                        "select_option, fill_field. Empty to keep "
                        "original."
                    ),
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Replacement params dict. Common keys: label, "
                        "aliases (list), value, dropdown_label, "
                        "option_label."
                    ),
                },
            },
        },
        "inserted_steps": {
            "type": "array",
            "description": (
                "When mode=insert_steps: helper sub-flow to splice "
                "BEFORE the failed step. Each entry is a step shape "
                "with at least 'intent' and 'type'. Empty for other "
                "modes."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "type": {"type": "string"},
                    "params": {"type": "object"},
                },
                "required": ["intent", "type"],
            },
        },
    },
    "required": ["mode", "reasoning"],
}


_ANALYSIS_PROMPT_TEMPLATE = """\
You are analysing a terminal failure in a CUA (Computer Use Agent) plan.

A required step has failed after retries AND after the runner's
default handler escalation. Your job: look at the screenshot of the
current page, the failed step's intent / type / params, and the
failure data, then choose the cheapest recovery that would plausibly
succeed.

FAILED STEP
===========
intent: {intent}
type:   {step_type}
params: {params}

FAILURE DATA: {failure_data}
RETRY ATTEMPTS: {attempts}

PLAN CONTEXT (steps already completed successfully):
{plan_context}

RECOVERY MODES (pick ONE via the record_recovery tool):

1. add_hint — the step is fundamentally correct but the search
   needs a clarifying instruction. The hint is appended to the
   prompt the next retry will pass to its grounder. Use this when
   you can see what the right element is and just need to tell
   the searcher to pick it. Example: the failed step said
   "Click 'Update Lead'" but the actual button reads "Save" — so
   hint="The save button is labeled 'Save', not 'Update Lead'."

2. edit_step — the step's structure is wrong. Use when:
   - the step type is wrong (submit when it should be select_option,
     click when it should be fill_field, etc.)
   - the labels in params are wrong and the right ones are visible
     on screen
   - aliases are needed because the button label varies
   Provide ONLY the fields to change in ``edited_step``. Existing
   fields are preserved.

3. insert_steps — a precondition is missing. Use when the step
   itself is correct but the page isn't ready: a modal is blocking,
   the form section isn't expanded, scroll is needed, or a previous
   page-state didn't take. Provide a small sub-flow (1-3 steps) to
   put the page in the right state. The runner splices these BEFORE
   the failed step then re-attempts the original step.

4. halt — recovery is impossible. Use when:
   - the target genuinely doesn't exist on the site
   - the page shows a hard error (anti-bot, 403, server error)
   - the step's premise contradicts visible page state
   No plan tweak would help; surface the failure to the operator.

Be conservative: only halt when truly stuck. add_hint is preferred
when you can see what the right action is. edit_step is preferred
when the structure is just slightly off. insert_steps when a
precondition is missing.
"""


def analyse_failure_and_recover(
    *,
    step: Any,
    failure_data: str,
    screenshot: Image.Image | None,
    plan_context: list[str],
    attempts: int,
    api_key: str = "",
    model: str = "claude-haiku-4-5-20251001",
) -> RecoveryDecision | None:
    """Ask Claude to analyse a step failure and pick a recovery mode.

    Issue #224 follow-up: replaces the current "retry until budget
    exhausts then HALT" terminal path with a Claude-mediated decision.
    On success returns a :class:`RecoveryDecision` the caller applies.
    On any fallback path (no API key, API error, schema violation,
    network error) returns ``None`` — the caller is expected to fall
    back to the legacy halt behaviour.

    Args:
        step: The :class:`MicroIntent` that failed. ``intent``,
            ``type``, ``params`` are surfaced into the prompt.
        failure_data: The ``StepResult.data`` from the last attempt
            (e.g. ``"form_target_not_found"`` /
            ``"submit:Login@(540,220):no_state_change"``).
        screenshot: Last screenshot before halting. Optional —
            without it Claude reasons from the failure data alone.
        plan_context: Intent strings of the previously-successful
            steps in the plan. Helps Claude understand "where in
            the workflow the failure happened."
        attempts: How many times this step has been attempted
            (drives Claude's confidence calibration — many attempts
            = stronger evidence the structure is wrong).
        api_key: Anthropic API key. Falls back to the
            ``ANTHROPIC_API_KEY`` env var.
        model: Claude model. Defaults to Haiku-tier (the analysis
            task is structured-JSON lift, not deep reasoning).

    Returns:
        :class:`RecoveryDecision` on a successful tool_use call,
        ``None`` on any fallback path. The caller is responsible
        for budget enforcement (per-step + per-run); this function
        does NOT track budgets — it just performs the analysis when
        invoked.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning(
            "agentic_recovery: no ANTHROPIC_API_KEY — falling back "
            "to legacy halt path"
        )
        return None

    parsed = _call_recovery_tool(
        step=step,
        failure_data=failure_data,
        screenshot=screenshot,
        plan_context=plan_context,
        attempts=attempts,
        api_key=api_key,
        model=model,
    )
    if parsed is None:
        return None

    return _build_decision(parsed)


def _call_recovery_tool(
    *,
    step: Any,
    failure_data: str,
    screenshot: Image.Image | None,
    plan_context: list[str],
    attempts: int,
    api_key: str,
    model: str,
) -> dict | None:
    """Tool_use call. Returns the validated input dict or ``None``."""
    import requests

    prompt = _ANALYSIS_PROMPT_TEMPLATE.format(
        intent=getattr(step, "intent", "")[:300],
        step_type=getattr(step, "type", ""),
        params=str(getattr(step, "params", {}) or {})[:300],
        failure_data=failure_data[:300],
        attempts=attempts,
        plan_context=_format_plan_context(plan_context),
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if screenshot is not None:
        try:
            buf = BytesIO()
            screenshot.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.insert(
                0,
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("recovery: screenshot encode failed: %s", exc)

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 1500,
                "tools": [{
                    "name": "record_recovery",
                    "description": (
                        "Record the recovery decision for the failed "
                        "step."
                    ),
                    "input_schema": _ANALYSIS_TOOL_INPUT_SCHEMA,
                }],
                "tool_choice": {"type": "tool", "name": "record_recovery"},
                "messages": [{"role": "user", "content": content}],
            },
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning(
                "agentic_recovery API error %s: %s",
                resp.status_code, resp.text[:300],
            )
            return None
        for block in resp.json().get("content", []):
            if (
                block.get("type") == "tool_use"
                and block.get("name") == "record_recovery"
            ):
                tool_input = block.get("input")
                if isinstance(tool_input, dict):
                    return tool_input
        return None
    except Exception as exc:  # noqa: BLE001 — log + fall back
        logger.warning("agentic_recovery call failed: %s", exc)
        return None


def _build_decision(data: dict[str, Any]) -> RecoveryDecision:
    """Coerce a validated tool_use input dict into RecoveryDecision."""
    mode = str(data.get("mode") or "halt")
    if mode not in {"add_hint", "edit_step", "insert_steps", "halt"}:
        # Defensive — schema enforces enum but the LLM occasionally
        # emits a typo. Treat as halt rather than crash the runner.
        mode = "halt"

    inserted: list[dict[str, Any]] = []
    for raw in data.get("inserted_steps") or []:
        if not isinstance(raw, dict):
            continue
        intent = str(raw.get("intent") or "").strip()
        step_type = str(raw.get("type") or "").strip()
        if not intent or not step_type:
            continue
        inserted.append({
            "intent": intent,
            "type": step_type,
            "params": dict(raw.get("params") or {}),
        })

    return RecoveryDecision(
        mode=mode,
        reasoning=str(data.get("reasoning") or "")[:500],
        hint=str(data.get("hint") or "")[:500],
        edited_step=dict(data.get("edited_step") or {}),
        inserted_steps=inserted,
    )


def _format_plan_context(steps: list[str]) -> str:
    if not steps:
        return "  (this is the first step that failed)"
    return "\n".join(f"  [{i}] {s[:100]}" for i, s in enumerate(steps))


# ── Plan-splice helper (exposed for the runner to call on insert_steps) ─


def splice_inserted_steps(
    plan_steps: list[Any],
    insertion_index: int,
    inserted: list[Any],
) -> list[Any]:
    """Splice ``inserted`` before ``plan_steps[insertion_index]``.

    Loop steps in the plan use absolute step indices via
    ``loop_target``. After splicing N steps in at index K, every
    existing step's loop_target X with X >= K must shift to X + N
    so the loop still points at the correct target.

    Returns a NEW list; ``plan_steps`` is not mutated.
    """
    if not inserted:
        return list(plan_steps)

    n = len(inserted)
    new_steps: list[Any] = []
    for i, s in enumerate(plan_steps):
        if i == insertion_index:
            new_steps.extend(inserted)
        # Shift loop targets pointing AT or BEYOND the insertion
        # index — those steps now live N slots later.
        loop_target = getattr(s, "loop_target", -1)
        if loop_target is not None and loop_target >= insertion_index:
            try:
                s.loop_target = loop_target + n
            except AttributeError:
                pass
        new_steps.append(s)

    # Edge case: insertion_index == len(plan_steps) — append at end.
    if insertion_index >= len(plan_steps):
        new_steps.extend(inserted)
    return new_steps
