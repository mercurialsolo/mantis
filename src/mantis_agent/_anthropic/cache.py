"""Prompt-cache helpers for Anthropic API calls (#715, brain-context Phase 0).

Anthropic's prompt cache lets us mark a content block with
``cache_control: {type: "ephemeral"}`` to tell the API "cache everything
up to and including this block for the next 5 minutes". Subsequent
requests that match the cached prefix only pay for the suffix.

Mantis's brain code (brain_claude, ClaudeGrounding, ClaudeExtractor,
agentic_recovery) all share the same call shape: a stable prefix
(system prompt + tool definitions + task template) followed by a
mutable suffix (screenshot + per-step state). This module is the single
source of truth for partitioning + marking the boundary.

See `docs/reference/brain-context-optimization.md` § Phase 0 for the
phased rollout.
"""

from __future__ import annotations

from typing import Any

# Magic marker shape — single source of truth so we don't hand-write
# ``{"type": "ephemeral"}`` in every call site.
_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}


def as_cached_system(text: str) -> list[dict[str, Any]]:
    """Convert a string ``system`` prompt into a single-block list with
    ``cache_control: ephemeral`` on it.

    Anthropic accepts ``system`` as either a string or a list of text
    blocks. Only the list form supports ``cache_control``. The block is
    marked so the API caches everything up to and including the system
    prompt — typically the largest stable section in any Mantis call.
    """
    if not text:
        return []
    return [{
        "type": "text",
        "text": text,
        "cache_control": _EPHEMERAL,
    }]


def mark_last_tool_cached(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add ``cache_control: ephemeral`` to the last tool definition.

    Caches the entire ``tools`` array (Anthropic caches up to and
    including any block marked with ``cache_control``). Returns a new
    list — never mutates the caller's input.
    """
    if not tools:
        return tools
    out = [dict(t) for t in tools]
    out[-1] = {**out[-1], "cache_control": _EPHEMERAL}
    return out


def cached_text_block(text: str) -> dict[str, Any]:
    """Convenience: a single ``text`` content block with cache_control.

    Use inside a ``user`` message ``content`` list to mark the stable
    portion of a prompt — typically the instruction template that
    doesn't vary per call. The screenshot / dynamic fields go after.
    """
    return {
        "type": "text",
        "text": text,
        "cache_control": _EPHEMERAL,
    }


def extract_cache_telemetry(response_json: dict[str, Any]) -> dict[str, int]:
    """Pull cache-creation + cache-read token counts from a response.

    Returns ``{}`` when the response lacks usage fields (older API
    versions, error responses). Surfaced in Augur so operators can
    audit cache hit rates per run.

    Field reference: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    """
    if not isinstance(response_json, dict):
        return {}
    usage = response_json.get("usage") or {}
    out: dict[str, int] = {}
    for src_key, out_key in (
        ("cache_creation_input_tokens", "cache_creation_input_tokens"),
        ("cache_read_input_tokens", "cache_read_input_tokens"),
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
    ):
        v = usage.get(src_key)
        if isinstance(v, int):
            out[out_key] = v
    return out
