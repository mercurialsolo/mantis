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
import time
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
#
# #567: a submission can override these via the
# ``max_recoveries_per_step`` / ``max_recoveries_per_run`` runtime
# fields (threaded through ``MicroPlanRunner.max_recoveries_per_step``
# and ``.max_recoveries_per_run``). The DEFAULT_* values remain the
# fallback when no override is supplied — preserves legacy behavior
# for callers that don't set the runtime fields.
DEFAULT_MAX_RECOVERIES_PER_STEP = 2
DEFAULT_MAX_RECOVERIES_PER_RUN = 5


def effective_max_recoveries(runner: Any) -> tuple[int, int]:
    """Resolve the per-step + per-run recovery caps for this run.

    Reads ``runner.max_recoveries_per_step`` / ``.max_recoveries_per_run``
    when set (positive int), else falls back to the module DEFAULTS.
    Defensive: handles ``None``, missing attr, non-positive values
    by treating them as "use default" — never returns 0 (would brick
    the recovery layer entirely).
    """
    per_step = DEFAULT_MAX_RECOVERIES_PER_STEP
    per_run = DEFAULT_MAX_RECOVERIES_PER_RUN
    try:
        v = getattr(runner, "max_recoveries_per_step", None)
        if isinstance(v, int) and v > 0:
            per_step = v
        v = getattr(runner, "max_recoveries_per_run", None)
        if isinstance(v, int) and v > 0:
            per_run = v
    except Exception:  # noqa: BLE001 — never break recovery
        pass
    return per_step, per_run


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


# Recovery analysis prompt body lives at
# ``src/mantis_agent/prompts/files/recovery_analysis.txt``. Operators
# can override via ``MANTIS_PROMPTS_DIR=/path/to/prompts`` without
# editing this module.


def analyse_failure_and_recover(
    *,
    step: Any,
    failure_data: str,
    screenshot: Image.Image | None,
    plan_context: list[str],
    attempts: int,
    api_key: str = "",
    model: str = "claude-haiku-4-5-20251001",
    prior_hints: list[str] | None = None,
    page_context: dict | None = None,
    env: Any = None,
    plan_hash: str = "",
    workflow_id: str = "",
    step_index: int = -1,
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
        page_context: Optional dict surfacing runtime page state
            the LLM can compare against the failure. Recognised keys:

            * ``anchor_url`` (str) — the URL the most recent
              successful ``navigate`` step landed on
              (``runner._last_known_url``). When this differs from
              the current URL, the page has drifted and the LLM
              should prefer ``insert_steps=[{type: navigate,
              params: {url: anchor_url}}]`` over more brain retries.
            * ``current_url`` (str) — what ``env.current_url``
              reports right now. Pairs with ``anchor_url``.
            * ``scroll_y`` (int|float) — current ``window.scrollY``.
              When ``scroll_y >= viewport_h`` the viewport is past
              the fold and the LLM should prefer
              ``insert_steps=[{type: scroll, params: {direction:
              up}}]`` to bring the extraction target back into view.
            * ``viewport_h`` (int|float) — current ``window.innerHeight``.
              Pairs with ``scroll_y``.

            All values are best-effort; missing keys are simply
            omitted from the prompt. The decision still belongs to
            the LLM — these hints just give it the values it would
            otherwise have to infer from the screenshot pixels.

    Returns:
        :class:`RecoveryDecision` on a successful tool_use call,
        ``None`` on any fallback path. The caller is responsible
        for budget enforcement (per-step + per-run); this function
        does NOT track budgets — it just performs the analysis when
        invoked.
    """
    # Plan-evolution Phase 1 (#705): a `bad_url:<subclass>` failure means
    # the navigate step landed somewhere unusable (DNS error, 404,
    # wrong domain, soft-404). Run the URL-rewrite sources in cheapness
    # order (pattern_transform, then page_links if env supports CDP).
    # If a candidate emerges, dispatch as an `edit_step` decision
    # rewriting the navigate intent + params.url; the existing
    # step_recovery dispatcher merges the params and the navigate
    # handler picks up the new URL on retry. If no candidate emerges,
    # halt cleanly with the structured reason so result.json + Augur
    # show what was tried.
    #
    # `blocked` subclass is filtered upstream (handled by
    # external_pause / CF challenge paths) and never reaches this point.
    fd_lc = (failure_data or "").lower()
    if fd_lc.startswith("bad_url:") or fd_lc.startswith("bad_url="):
        from .gym.url_recovery import propose_url_rewrites
        subclass = (failure_data or "").split(":", 1)[-1].split("=", 1)[-1].strip()
        failed_url = _extract_failed_url_from_step(step, page_context or {})
        report = propose_url_rewrites(
            failed_url=failed_url,
            failure_subclass=subclass,
            intent_text=getattr(step, "intent", "") or "",
            env=env,
        )
        if report.proposals:
            best = report.proposals[0]
            logger.warning(
                "agentic_recovery: bad_url subclass=%s → rewrite_url "
                "source=%s confidence=%.2f new_url=%s (notes=%s)",
                subclass, best.source, best.confidence,
                best.new_url[:120], best.notes,
            )
            edited_step = {
                "intent": _rewrite_intent_url(
                    getattr(step, "intent", "") or "",
                    failed_url,
                    best.new_url,
                ),
                "params": {"url": best.new_url},
            }

            # Plan-evolution Phase 2 (#706): persist the candidate so
            # subsequent runs of the same plan can apply it via overlay
            # once it promotes (3 consecutive successes). Failures here
            # are non-fatal — the rewrite still applies in-memory for
            # THIS run.
            if plan_hash and workflow_id and step_index >= 0:
                try:
                    from .recipes.plan_evolution_store import (
                        record_rewrite_candidate,
                    )
                    record_rewrite_candidate(
                        plan_hash=plan_hash,
                        workflow_id=workflow_id,
                        step_index=step_index,
                        original_step={
                            "intent": getattr(step, "intent", "") or "",
                            "type": getattr(step, "type", "") or "",
                            "params": dict(getattr(step, "params", {}) or {}),
                        },
                        rewritten_step=edited_step,
                        source=best.source,
                        confidence=best.confidence,
                    )
                except Exception as exc:  # noqa: BLE001 — never block recovery
                    logger.debug(
                        "plan_evolution: record_rewrite_candidate raised: %s", exc,
                    )

            return RecoveryDecision(
                mode="edit_step",
                edited_step=edited_step,
                reasoning=(
                    f"rewrite_url:{best.source} {failed_url} → {best.new_url} "
                    f"(confidence={best.confidence:.2f}; {best.notes})"
                ),
            )

        logger.warning(
            "agentic_recovery: bad_url subclass=%s — no rewrite_url candidates "
            "(sources_tried=%s sources_skipped=%s); halting cleanly",
            subclass, report.sources_tried, report.sources_skipped,
        )
        return RecoveryDecision(
            mode="halt",
            reasoning=(
                f"bad_url:{subclass} — rewrite_url found no candidates "
                f"(tried {','.join(report.sources_tried)})"
            ),
        )

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
        prior_hints=prior_hints or [],
        page_context=page_context or {},
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
    prior_hints: list[str] | None = None,
    page_context: dict | None = None,
) -> dict | None:
    """Tool_use call. Returns the validated input dict or ``None``."""
    import requests

    from .prompts import load_prompt

    hints = [h for h in (prior_hints or []) if str(h).strip()]
    if hints:
        hints_block = "PRIOR HINTS TEXT:\n" + "\n".join(
            f"  - {str(h)[:200]}" for h in hints[-3:]
        )
    else:
        hints_block = ""

    prompt = load_prompt(
        "recovery_analysis",
        intent=getattr(step, "intent", "")[:300],
        step_type=getattr(step, "type", ""),
        params=str(getattr(step, "params", {}) or {})[:300],
        failure_data=failure_data[:300],
        attempts=attempts,
        plan_context=_format_plan_context(plan_context),
        prior_hint_count=len(hints),
        prior_hints_block=hints_block,
        page_state=_format_page_context(page_context or {}),
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
        # #715 — prompt-cache split. The tool input_schema is the
        # largest stable block in this call (~1.5KB of JSON Schema).
        # Mark it for caching so the 5-minute TTL covers a typical
        # 30-min boattrader run's recovery calls.
        from ._anthropic.cache import mark_last_tool_cached
        recovery_tool = {
            "name": "record_recovery",
            "description": (
                "Record the recovery decision for the failed "
                "step."
            ),
            "input_schema": _ANALYSIS_TOOL_INPUT_SCHEMA,
        }
        request_body = {
            "model": model,
            "max_tokens": 1500,
            "tools": mark_last_tool_cached([recovery_tool]),
            "tool_choice": {"type": "tool", "name": "record_recovery"},
            "messages": [{"role": "user", "content": content}],
        }
        t0 = time.monotonic()
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=request_body,
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning(
                "agentic_recovery API error %s: %s",
                resp.status_code, resp.text[:300],
            )
            return None
        # #715 — cache telemetry. WARNING-level visible in Modal.
        from ._anthropic.cache import extract_cache_telemetry as _cache_tele
        _t = _cache_tele(resp.json() if resp.content else {})
        if _t.get("cache_read_input_tokens", 0) > 0 or _t.get(
            "cache_creation_input_tokens", 0
        ) > 0:
            logger.warning(
                "  [cache] agentic_recovery: read=%d created=%d",
                _t.get("cache_read_input_tokens", 0),
                _t.get("cache_creation_input_tokens", 0),
            )
        # Cost meter (#675 A/B follow-up).
        from .observability.claude_cost_meter import record_from_response
        record_from_response(
            source="recovery", model=model,
            response_json=resp.json() if resp.content else None,
        )
        # #523 PR B-5 — capture this call as a ``step_recovery`` modelio
        # record when an upstream caller (step_recovery._try_agentic_
        # recovery via publish_modelio_context) has set the context.
        # Silent no-op otherwise. Best-effort — telemetry never breaks
        # recovery analysis.
        try:
            from .observability.modelio import (
                current_modelio_context,
                record_anthropic_modelio,
            )
            ctx = current_modelio_context()
            if ctx is not None:
                record_anthropic_modelio(
                    request_payload=request_body,
                    response_json=resp.json(),
                    duration_ms=int((time.monotonic() - t0) * 1000),
                    ctx=ctx,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("agentic_recovery modelio capture failed: %s", exc)
        for block in resp.json().get("content", []):
            if (
                block.get("type") == "tool_use"
                and block.get("name") == "record_recovery"
            ):
                tool_input = block.get("input")
                if isinstance(tool_input, dict):
                    return tool_input
        # #431: surface the malformed-response path so a silent skip is
        # debuggable. Anthropic occasionally emits a tool_use block with
        # the wrong name, or a text-only block when the model decides
        # not to call the forced tool — both look like "no recovery"
        # from the caller's perspective without a log line.
        logger.warning(
            "agentic_recovery_skipped: 200 OK but no record_recovery "
            "tool_use block in response (text only or wrong tool)"
        )
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


# ── plan-evolution Phase 1 helpers ───────────────────────────────────

import re as _re  # noqa: E402 — keep heavy imports up top, plain re here


_URL_RE = _re.compile(r'https?://[^\s"]+')


def _extract_failed_url_from_step(step: Any, page_context: dict) -> str:
    """Find the URL the navigate step tried to load.

    Order: step.params['url'] > URL match in step.intent > page_context
    ['current_url']. Returns "" only if every source is empty (shouldn't
    happen for a bad_url failure; defensive).
    """
    params = getattr(step, "params", None) or {}
    params_url = str(params.get("url", "") or "").strip()
    if params_url:
        m = _URL_RE.search(params_url)
        if m:
            return m.group(0)
        if params_url.startswith("http"):
            return params_url
    intent = getattr(step, "intent", "") or ""
    m = _URL_RE.search(intent)
    if m:
        return m.group(0)
    return str(page_context.get("current_url", "") or "").strip()


def _rewrite_intent_url(intent: str, old_url: str, new_url: str) -> str:
    """Swap old_url → new_url inside the intent prose, preserving the rest.

    When the intent doesn't literally contain old_url (the decomposer
    paraphrased), returns the intent unchanged — params.url is the
    authoritative source for the navigate handler anyway. The intent
    rewrite is for log readability + future audit.
    """
    if not intent or not old_url or old_url == new_url:
        return intent
    if old_url in intent:
        return intent.replace(old_url, new_url)
    return intent


def _format_page_context(page: dict) -> str:
    """Render runtime page state for the recovery prompt.

    Emits only the keys that are populated — missing values omit
    their line entirely so the prompt stays terse and the LLM
    doesn't reason from placeholders. Returns the literal string
    ``(unavailable)`` when nothing is known, so the prompt section
    still renders coherently.
    """
    lines: list[str] = []
    anchor = str(page.get("anchor_url") or "").strip()
    current = str(page.get("current_url") or "").strip()
    if anchor:
        lines.append(f"  anchor_url (last navigate): {anchor[:200]}")
    if current:
        lines.append(f"  current_url:                {current[:200]}")
    if anchor and current and anchor != current:
        lines.append("  (URL drifted — the page is no longer the one the plan anchored to)")

    scroll_y = page.get("scroll_y")
    viewport_h = page.get("viewport_h")
    try:
        if scroll_y is not None and viewport_h is not None:
            sy = float(scroll_y)
            vh = float(viewport_h)
            lines.append(
                f"  scroll: scrollY={sy:.0f}px viewport_h={vh:.0f}px "
                f"(ratio={sy / vh:.2f} viewports below top)"
            )
            if vh > 0 and sy >= vh:
                lines.append(
                    "  (page is past the fold — extraction targets above "
                    "the current viewport are not visible)"
                )
    except (TypeError, ValueError):
        pass

    if not lines:
        return "  (unavailable)"
    return "\n".join(lines)


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
