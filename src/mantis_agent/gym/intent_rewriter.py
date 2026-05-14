"""Self-healing primitive — Phase B of epic #377.

When a step fails with a ``failure_class`` that suggests the **intent
itself** is the problem (goal-shaped, ambiguous target, no observable
effect), ask Claude to propose a more mechanical / more specific
alternative intent. The next retry uses the rewritten intent.

Bounded by design:

* Only the classes in :data:`REWRITE_TRIGGERING_CLASSES` trigger a
  rewrite call. Other failures (selector_miss, cf_challenge,
  extractor_error, …) follow their existing recovery paths.
* One rewrite per step per run (configurable via the caller's budget
  tracker). A repeatedly-failing rewrite is a signal the plan needs
  human attention — burning more Claude calls just compounds the
  problem.
* Returns ``None`` on API / parse failure, ``KEEP`` response, or empty
  output. Caller treats ``None`` as "no rewrite available; let the
  recovery policy decide".

Generic by design:

* Takes ``(intent, list[FailureContext])`` — no plan, no domain
  knowledge, no URL patterns. The prompt instructs Claude on the
  pattern transformations (goal → literal, ambiguous → specific) and
  Claude does the work over any intent.
* The rewriter is a pure function plus an HTTP shim. The runner glue
  in :mod:`~.run_executor` decides when to call it; this module just
  decides what to say.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


REWRITE_TRIGGERING_CLASSES: frozenset[str] = frozenset({
    "brain_loop_exhausted",  # goal-shaped intent, brain looped
    "wrong_target",          # click landed on the wrong destination
    "no_state_change",       # action reported success without observable effect
})


_REWRITE_PROMPT = """\
A browser-automation plan step has failed and needs to be retried with a different intent.

Original intent: {intent}

Per-attempt failure history (newest first):
{failure_summary}

Rewrite the intent to be more MECHANICAL and SPECIFIC. Apply whichever of these patterns fits the failure:

- If the original is goal-shaped (multi-clause, "reveal X / Y / Z"), rewrite to a literal action like "Scroll down by one viewport".
- If the target is ambiguous ("the first card", "the link"), rewrite to disambiguate ("the first card under the X section", "the link labelled Y").
- If the action had no observable effect, rewrite to specify a different element or a verify-first approach.
- Keep the same verb (click stays click, scroll stays scroll). Do NOT change the step type.

Constraints:
- Output ONLY the rewritten intent string (or KEEP).
- No JSON, no markdown, no prose preamble.
- If you cannot meaningfully improve the intent, output exactly: KEEP
"""


@dataclass(frozen=True)
class FailureContext:
    """One attempt's worth of failure info — fed to the rewriter so
    Claude can see what was tried and why it failed.

    Fields are exactly the columns a result.json reader sees on each
    failed step. ``screenshot_png`` is optional — when supplied, the
    rewriter sends it alongside the prompt for grounding.
    """

    failure_class: str
    data: str
    page_title: str = ""
    final_url: str = ""
    screenshot_png: bytes | None = None


def should_attempt_rewrite(
    failure_class: str,
    *,
    attempts_used: int,
    max_attempts: int = 1,
) -> bool:
    """Pure predicate: would this step+failure trigger a rewrite right
    now? Lets the caller decide before paying the API cost.

    The caller maintains the attempts counter (typically a
    ``runner._step_rewrite_attempts`` dict keyed by step_index).
    """
    if not failure_class or failure_class not in REWRITE_TRIGGERING_CLASSES:
        return False
    return attempts_used < max_attempts


def propose_rewrite(
    intent: str,
    failures: list[FailureContext],
    *,
    api_key: str = "",
    model: str = "claude-opus-4-7",
    timeout: float = 20.0,
) -> str | None:
    """Ask Claude to propose a more mechanical / specific intent.

    Returns the rewritten intent string, or ``None`` if no useful
    rewrite is available (Claude returned ``KEEP``, the API call
    failed, the response was empty, or no api_key was provided).

    The most recent failure's screenshot (when present) is attached
    as an image block so Claude can see what the page looked like
    when the original intent failed. Older failures contribute their
    prose context only.
    """
    if not intent or not failures:
        return None
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        logger.debug("intent_rewriter: no ANTHROPIC_API_KEY; skipping rewrite")
        return None

    try:
        import requests
    except ImportError:
        logger.debug("intent_rewriter: requests not installed; skipping rewrite")
        return None

    failure_summary = _format_failures(failures)
    prompt = _REWRITE_PROMPT.format(intent=intent, failure_summary=failure_summary)

    content: list[dict[str, Any]] = []
    # Attach the newest failure's screenshot when present, so Claude
    # sees what the page looked like when the original intent failed.
    latest = failures[0] if failures else None
    if latest and latest.screenshot_png:
        try:
            b64 = base64.b64encode(latest.screenshot_png).decode("ascii")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64,
                },
            })
        except Exception as exc:  # noqa: BLE001
            logger.debug("intent_rewriter: screenshot encode failed: %s", exc)
    content.append({"type": "text", "text": prompt})

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 200,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 — never break runs
        logger.debug("intent_rewriter: API call raised: %s", exc)
        return None

    if resp.status_code != 200:
        logger.debug(
            "intent_rewriter: API error %s: %s",
            resp.status_code, resp.text[:200],
        )
        return None

    try:
        body = resp.json()
    except (ValueError, json.JSONDecodeError):
        return None

    for block in body.get("content", []):
        if block.get("type") == "text":
            text = (block.get("text") or "").strip()
            if not text:
                return None
            if text.upper() == "KEEP":
                logger.info("intent_rewriter: Claude returned KEEP — no rewrite")
                return None
            # Strip surrounding quotes if Claude wrapped the response.
            if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
                text = text[1:-1].strip()
            if text and text.upper() != "KEEP":
                logger.info(
                    "intent_rewriter: rewrote intent (was %d chars, now %d chars)",
                    len(intent), len(text),
                )
                return text
            return None
    return None


def _format_failures(failures: list[FailureContext]) -> str:
    lines: list[str] = []
    for i, f in enumerate(failures, 1):
        bits = [f"#{i} failure_class={f.failure_class!r}"]
        if f.data:
            bits.append(f"data={f.data[:120]!r}")
        if f.page_title:
            bits.append(f"page_title={f.page_title[:80]!r}")
        if f.final_url:
            bits.append(f"final_url={f.final_url[:80]!r}")
        lines.append("- " + ", ".join(bits))
    return "\n".join(lines) if lines else "- (no failures recorded)"


__all__ = [
    "REWRITE_TRIGGERING_CLASSES",
    "FailureContext",
    "propose_rewrite",
    "should_attempt_rewrite",
]
