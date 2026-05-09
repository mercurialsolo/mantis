"""Derive-first ObjectiveSpec producer (issue #224 Phase 2).

The Phase 1 overlay primitives (``ExtractionSchema.overlay`` /
``SiteConfig.overlay`` / ``recipes.load_site_config``) gave callers
the merge half of the derive-first config story. This module ships
the producer half: :func:`derive_from_plan` lifts the structured
:class:`ObjectiveSpec` shape directly from the plan text the user
authored, so host integrations can do::

    from mantis_agent import objective, recipes
    from mantis_agent.extraction import ExtractionSchema

    spec   = objective.derive_from_plan(plan_text)
    schema = ExtractionSchema.from_objective(spec).overlay(
        recipes.load_schema(name) if name else None
    )

without picking a recipe per template plan. Most plans don't name a
recipe; the ones that do treat it as a *hardening hint* — empirical
spam tokens accumulated from production runs — not as the primary
config selector.

The lift is a single Anthropic ``tool_use`` call with a strict input
schema, so the response is server-side-validated as JSON of the
right shape (no fragile prose parsing). Calls are cached
in-process by ``hash(plan_text + DERIVE_PROMPT_VERSION)`` — repeated
runs in the same process are free, and bumping the prompt version
forces a uniform cache miss.

Top-level re-exports of :class:`ObjectiveSpec`,
:class:`OutputField`, :class:`CompletionCondition` keep
``from mantis_agent.objective import ...`` ergonomic for callers
that don't want to know which submodule the dataclasses physically
live in.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

from .graph.objective import (
    CompletionCondition,
    ObjectiveSpec,
    OutputField,
)

__all__ = [
    "CompletionCondition",
    "DERIVE_PROMPT_VERSION",
    "ObjectiveSpec",
    "OutputField",
    "derive_from_plan",
]

logger = logging.getLogger(__name__)


# Bumping this string invalidates every cached entry — use when the
# prompt or tool schema below changes in a way that makes prior
# parses incorrect (renamed fields, tightened required-flag rules,
# new mandatory output keys, etc.).
DERIVE_PROMPT_VERSION = "v1"


# In-process cache keyed by sha256(DERIVE_PROMPT_VERSION + plan_text).
# Bounded loosely — a single process rarely processes more than a
# handful of unique plans, and ``ObjectiveSpec`` instances are small
# so the memory cost is trivial. Cleared by tests via
# :func:`_reset_cache_for_tests`.
_DERIVE_CACHE: dict[str, ObjectiveSpec] = {}


_DERIVE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_entity": {
            "type": "string",
            "description": (
                "What kind of item the plan extracts, in the user's "
                "vocabulary. Examples: 'boat listing', 'job posting', "
                "'real estate listing', 'product', 'user profile'. "
                "Empty string only if truly indeterminate."
            ),
        },
        "domains": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Target website domains mentioned in the plan, e.g. "
                "['boattrader.com']. Strip ``www.`` prefixes."
            ),
        },
        "output_fields": {
            "type": "array",
            "description": (
                "Every field the plan extracts. Use ``required=true`` ONLY "
                "when the plan calls the field out as REQUIRED, VIABLE-"
                "blocking, or 'must have'. Otherwise ``required=false``. "
                "Conservative is correct — over-marking required fields "
                "produces false rejections downstream."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["str", "int", "bool", "float"],
                    },
                    "required": {"type": "boolean"},
                    "example": {
                        "type": "string",
                        "description": (
                            "A representative value from the plan or a "
                            "sensible synthesised one. Empty string OK."
                        ),
                    },
                },
                "required": ["name"],
            },
        },
        "forbidden_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Every UI label the plan tells the agent NOT to click "
                "('Do NOT click X', 'STRICTLY PROHIBITED'). Empty if "
                "the plan doesn't enumerate any."
            ),
        },
        "allowed_reveal_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Every UI label the plan tells the agent IT MAY click "
                "to surface hidden info ('Show more', 'Show phone', "
                "'View details'). Empty if not enumerated."
            ),
        },
        "viability_definition": {
            "type": "string",
            "description": (
                "The plan's viability rule, paraphrased to one "
                "sentence. Example: 'A listing is VIABLE only if the "
                "seller's phone number appears in the description.' "
                "Empty if the plan does not define viability."
            ),
        },
        "spam_text_indicators": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Plan-enumerated spam / dealer text tokens. Most plans "
                "don't enumerate these — leave empty when the plan is "
                "silent. Do NOT invent dealer names like 'marinemax' "
                "unless the plan explicitly lists them."
            ),
        },
        "spam_seller_indicators": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Plan-enumerated seller-name spam indicators (e.g. "
                "'llc', 'inc', 'brokerage'). Same rule: empty unless "
                "the plan enumerates them."
            ),
        },
        "spam_label": {
            "type": "string",
            "description": (
                "What the plan calls non-organic listings, e.g. "
                "'dealer', 'recruiter', 'broker spam'. Empty if "
                "unspecified."
            ),
        },
    },
    "required": ["target_entity", "output_fields"],
}


_DERIVE_PROMPT = """\
Read the CUA (Computer Use Agent) plan below and lift its semantic
shape into a structured ObjectiveSpec via the ``record_objective``
tool.

PLAN TEXT
=========
{plan_text}
=========

Be precise:

- ``output_fields``: include every field the plan extracts. Use
  ``required: true`` ONLY when the plan explicitly calls a field out
  as REQUIRED / VIABLE / "must have". Default new fields to
  ``required: false``.
- ``forbidden_actions`` / ``allowed_reveal_actions``: only include
  UI labels the plan literally enumerates. Don't paraphrase.
- ``viability_definition``: paraphrase the plan's viability rule
  if any; empty string otherwise.
- ``spam_text_indicators`` / ``spam_seller_indicators`` / ``spam_label``:
  leave empty unless the plan explicitly enumerates dealer / spam
  tokens. The recipe overlay layer (``recipes.load_schema``) is
  where empirical-from-production tokens live; the plan-derived
  spec must not invent them.
"""


def derive_from_plan(
    plan_text: str,
    *,
    api_key: str = "",
    model: str = "claude-haiku-4-5-20251001",
) -> ObjectiveSpec:
    """Lift an :class:`ObjectiveSpec` from a free-text CUA plan.

    Issue #224 Phase 2 producer: a single Claude tool_use call lifts
    the structured fields downstream consumers
    (:meth:`ExtractionSchema.from_objective`,
    :class:`PlanDecomposer`) need. Result is cached in-process
    keyed by ``sha256(prompt_version + plan_text)`` so repeated
    invocations for the same plan are free.

    Args:
        plan_text: Free-text plan body. Multi-paragraph; arbitrary
            shape. The plans/ directory is the canonical source.
        api_key: Anthropic API key. Falls back to the
            ``ANTHROPIC_API_KEY`` env var. When neither is set, a
            heuristic fallback produces a minimal-but-valid spec
            so the call doesn't hard-fail in offline contexts.
        model: Anthropic model identifier. Defaults to a
            Haiku-tier model — the lift task is small and the cost
            difference vs Sonnet/Opus matters for plans that get
            decomposed on every CI run.

    Returns:
        :class:`ObjectiveSpec` populated with the derive-time
        fields (``target_entity``, ``output_schema``,
        ``forbidden_actions``, ``allowed_reveal_actions``,
        ``viability_definition``, ``spam_*``). Empty list / string
        for fields the plan didn't address — the overlay layer
        is where the recipe fills them in.

    Cache lifecycle: per-process, never expires within a process.
    Bump :data:`DERIVE_PROMPT_VERSION` when the schema or prompt
    semantics change to invalidate every entry uniformly.
    """
    cache_key = _cache_key(plan_text)
    cached = _DERIVE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning(
            "derive_from_plan: no ANTHROPIC_API_KEY — falling back to "
            "heuristic ObjectiveSpec.parse. Pass api_key= or set the "
            "env var for the tool_use lift."
        )
        spec = ObjectiveSpec.parse(plan_text, api_key="")
        _DERIVE_CACHE[cache_key] = spec
        return spec

    parsed = _call_derive_tool(plan_text, api_key=api_key, model=model)
    if parsed is None:
        logger.warning(
            "derive_from_plan: tool_use lift returned no validated "
            "input — falling back to heuristic ObjectiveSpec.parse"
        )
        spec = ObjectiveSpec.parse(plan_text, api_key="")
    else:
        spec = _build_spec_from_tool_input(plan_text, parsed)

    _DERIVE_CACHE[cache_key] = spec
    return spec


def _cache_key(plan_text: str) -> str:
    return hashlib.sha256(
        (DERIVE_PROMPT_VERSION + "\n" + plan_text).encode("utf-8")
    ).hexdigest()


def _call_derive_tool(
    plan_text: str,
    *,
    api_key: str,
    model: str,
) -> dict | None:
    """Single Anthropic tool_use call. Returns the validated input
    dict, or ``None`` on any API / network / shape mismatch."""
    import requests

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
                    "name": "record_objective",
                    "description": (
                        "Record the structured ObjectiveSpec lifted "
                        "from the plan."
                    ),
                    "input_schema": _DERIVE_TOOL_SCHEMA,
                }],
                "tool_choice": {"type": "tool", "name": "record_objective"},
                "messages": [{
                    "role": "user",
                    "content": _DERIVE_PROMPT.format(plan_text=plan_text),
                }],
            },
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning(
                "derive_from_plan API error %s: %s",
                resp.status_code,
                resp.text[:500],
            )
            return None
        for block in resp.json().get("content", []):
            if (
                block.get("type") == "tool_use"
                and block.get("name") == "record_objective"
            ):
                tool_input = block.get("input")
                if isinstance(tool_input, dict):
                    return tool_input
        return None
    except Exception as exc:  # noqa: BLE001 — log + fall back
        logger.warning("derive_from_plan call failed: %s", exc)
        return None


def _build_spec_from_tool_input(
    raw_text: str, data: dict[str, Any],
) -> ObjectiveSpec:
    """Coerce a validated tool_use input dict into ObjectiveSpec.

    The tool schema accepts arrays of partially-typed objects, but the
    dataclass wants strict :class:`OutputField` instances. We also
    drop fields with empty ``name`` defensively.
    """
    output_fields: list[OutputField] = []
    for f in data.get("output_fields") or []:
        if not isinstance(f, dict):
            continue
        name = (f.get("name") or "").strip()
        if not name:
            continue
        output_fields.append(
            OutputField(
                name=name,
                type=str(f.get("type") or "str"),
                required=bool(f.get("required", False)),
                example=str(f.get("example") or ""),
            )
        )

    return ObjectiveSpec(
        raw_text=raw_text,
        domains=list(data.get("domains") or []),
        target_entity=str(data.get("target_entity") or ""),
        forbidden_actions=list(data.get("forbidden_actions") or []),
        allowed_reveal_actions=list(data.get("allowed_reveal_actions") or []),
        output_schema=output_fields,
        viability_definition=str(data.get("viability_definition") or ""),
        spam_text_indicators=list(data.get("spam_text_indicators") or []),
        spam_seller_indicators=list(data.get("spam_seller_indicators") or []),
        spam_label=str(data.get("spam_label") or ""),
    )


def _reset_cache_for_tests() -> None:
    """Test-only hook: clear the in-process cache so each test starts
    from a clean slate. Production callers must not call this — the
    cache is per-process by design."""
    _DERIVE_CACHE.clear()
