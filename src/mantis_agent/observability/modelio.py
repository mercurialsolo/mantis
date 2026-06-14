"""Per-LLM-call modelio capture wiring (#523, augur-sdk 0.1.6+).

This module provides the plumbing that lets the shared
``AnthropicToolUseClient`` emit one canonical training-data record
(``modelio/<step:04d>-<layer>-<seq>.json``) per Anthropic Messages
call without each call site needing to assemble the record itself.

Two-piece design:

1. A contextvar pair (:func:`publish_modelio_context` /
   :func:`current_modelio_context`) so callers higher up the stack
   declare the *layer* the next LLM call belongs to (planner,
   grounding, verifier, step_recovery, judge). The client reads the
   contextvar at request time. No layer published → capture is a no-op.

2. :func:`record_anthropic_modelio`, the mapper that converts the
   Anthropic request payload + raw response JSON to the schema-pinned
   modelio shape and forwards to :meth:`AugurAdapter.record_modelio`.

**Vendor field-name divergence (memorize this):**

The canonical ``modelio.schema.json`` ``response.usage`` block uses
**OpenAI** field names — ``prompt_tokens`` / ``completion_tokens``
— with ``additionalProperties: false``. Anthropic responses ship
``input_tokens`` / ``output_tokens`` plus
``cache_creation_input_tokens`` / ``cache_read_input_tokens``. The
mapper translates at the boundary; downstream SFT/DPO tooling reads
the OpenAI shape uniformly.

PR B-1 lands this module + the client-side capture hook. No call
sites publish a context yet — those follow in PR B-2..B-5 (one per
layer) so blast radius stays narrow.
"""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

logger = logging.getLogger(__name__)


_ALLOWED_LAYERS: frozenset[str] = frozenset({
    "planner", "grounding", "model", "verifier", "step_recovery", "judge",
})


@dataclass(frozen=True)
class ModelIOContext:
    """The triple a caller pushes onto the contextvar before invoking
    the LLM client. ``augur`` is the open adapter; ``layer`` picks the
    modelio.schema.json layer enum; ``step_index`` is the 0-based
    Mantis step index (the adapter bumps to 1-based on the way into
    the SDK)."""

    augur: Any  # AugurAdapter, but typed Any to avoid circular import.
    layer: str
    step_index: int | None = None


_current: contextvars.ContextVar[ModelIOContext | None] = contextvars.ContextVar(
    "mantis_current_modelio_context",
    default=None,
)


def current_modelio_context() -> ModelIOContext | None:
    """Return the active context if any call up the stack published one."""
    return _current.get()


@contextmanager
def publish_modelio_context(
    augur: Any,
    layer: str,
    step_index: int | None = None,
) -> Iterator[None]:
    """Declare that any LLM call made inside the ``with`` block should
    be captured as ``layer`` (with optional ``step_index``).

    Validates layer up-front so a typo at a call site surfaces during
    development rather than silently dropping captures. Falls back to
    a debug log + no-op when ``augur`` is None or inactive — telemetry
    never breaks the run.
    """
    if layer not in _ALLOWED_LAYERS:
        logger.warning(
            "publish_modelio_context: unknown layer %r (allowed=%s); skipping",
            layer, sorted(_ALLOWED_LAYERS),
        )
        yield
        return
    if augur is None or not getattr(augur, "active", False):
        yield
        return
    token = _current.set(ModelIOContext(augur=augur, layer=layer, step_index=step_index))
    try:
        yield
    finally:
        _current.reset(token)


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp matching ``modelio.schema.json``'s
    ``ts`` field (``date-time`` format)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _map_anthropic_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    """Anthropic ``response.usage`` → OpenAI-shape ``usage`` block.

    Anthropic ships ``input_tokens`` / ``output_tokens`` plus optional
    ``cache_creation_input_tokens`` / ``cache_read_input_tokens``;
    modelio.schema.json's ``usage`` slot requires ``prompt_tokens`` /
    ``completion_tokens`` with ``cache_hit_tokens`` /
    ``cache_creation_tokens`` for the cache fields.
    ``additionalProperties: false`` — so we MUST drop anything else
    (Anthropic occasionally adds new usage keys; we ignore unknown).
    """
    if not isinstance(usage, dict):
        return {}
    out: dict[str, int] = {}
    if (v := usage.get("input_tokens")) is not None:
        out["prompt_tokens"] = int(v)
    if (v := usage.get("output_tokens")) is not None:
        out["completion_tokens"] = int(v)
    if (v := usage.get("cache_read_input_tokens")) is not None:
        out["cache_hit_tokens"] = int(v)
    if (v := usage.get("cache_creation_input_tokens")) is not None:
        out["cache_creation_tokens"] = int(v)
    return out


def _extract_response_text_and_tools(
    payload_json: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Pull a flattened ``text`` + ``tool_calls`` view off an Anthropic
    response. modelio.schema.json's ``response`` slot accepts both —
    text is a single concatenated string (null when only tool blocks),
    tool_calls is the verbatim ``tool_use`` block list (provider-shaped,
    additionalProperties: true)."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in payload_json.get("content", []) or []:
        btype = block.get("type")
        if btype == "text":
            t = block.get("text")
            if isinstance(t, str) and t:
                text_parts.append(t)
        elif btype == "tool_use":
            tool_calls.append(block)
    text = "\n".join(text_parts) if text_parts else None
    return text, tool_calls


def record_anthropic_modelio(
    *,
    request_payload: dict[str, Any],
    response_json: dict[str, Any],
    duration_ms: int,
    ctx: ModelIOContext | None = None,
) -> None:
    """Build a modelio record from an Anthropic Messages call and stage
    it via :meth:`AugurAdapter.record_modelio`. No-op when no context
    has been published or the adapter is inactive.

    Failures are swallowed (per Augur spec §4.3 — telemetry never
    breaks the run). Validation failures are demoted to WARN so a
    schema-side surprise surfaces in deploy verification but doesn't
    interrupt the user's run.
    """
    ctx = ctx if ctx is not None else current_modelio_context()
    if ctx is None or ctx.augur is None or not getattr(ctx.augur, "active", False):
        return

    text, tool_calls = _extract_response_text_and_tools(response_json)
    usage = _map_anthropic_usage(response_json.get("usage"))

    request_block: dict[str, Any] = {
        "model": str(request_payload.get("model", "")),
    }
    # The schema allows messages / tools / params; keep the ones we
    # actually sent so SFT pipelines see the exact prompt shape.
    if "messages" in request_payload:
        request_block["messages"] = request_payload["messages"]
    if "tools" in request_payload:
        request_block["tools"] = request_payload["tools"]
    if "system" in request_payload:
        request_block["system"] = request_payload["system"]
    # Sampling / generation params (provider-shaped). Whitelist the
    # ones we set to avoid leaking the entire payload twice.
    params = {
        k: v for k, v in request_payload.items()
        if k in {"max_tokens", "temperature", "top_p", "top_k", "stop_sequences", "tool_choice"}
    }
    if params:
        request_block["params"] = params

    response_block: dict[str, Any] = {}
    if text is not None:
        response_block["text"] = text
    if tool_calls:
        response_block["tool_calls"] = tool_calls
    stop_reason = response_json.get("stop_reason")
    if isinstance(stop_reason, str) and stop_reason:
        response_block["stop_reason"] = stop_reason
    if usage:
        response_block["usage"] = usage
    # #687 (augur-sdk 0.5.0+): when the vendor returns logprobs on the
    # response, map them through the SDK's canonical extractor and
    # stamp on ``response.logprobs``. Returns ``None`` when the
    # response didn't carry logprobs (Anthropic Messages API today
    # doesn't surface them on most call paths). The session-level
    # ``capture_logprobs=True`` opt-in still gates the empty-vs-absent
    # semantics: when the flag is on AND the vendor returned nothing,
    # the SDK writes ``response.logprobs = []`` so downstream
    # consumers can distinguish "requested but vendor returned
    # nothing" from "not requested" (absent / null).
    try:
        from augur_sdk import ModelApiAdapterBase
        logprobs = ModelApiAdapterBase.extract_logprobs_from_response(response_json)
        if logprobs is not None:
            response_block["logprobs"] = logprobs
    except Exception:  # noqa: BLE001 — observability never fatal
        # Pre-0.5.0 SDK, import failed, or extractor raised.
        # Leave logprobs unset; the session-level opt-in still
        # controls whether the SDK writes ``[]`` on close.
        pass

    record: dict[str, Any] = {
        "schema_version": "0.1",
        "layer": ctx.layer,
        "ts": _utc_now_iso(),
        "request": request_block,
        "response": response_block,
    }
    if ctx.step_index is not None:
        # modelio.schema.json's step_index is 0-based (minimum: 0) —
        # the file-path bump to 1-based is the SDK's concern.
        record["step_index"] = int(ctx.step_index)
    if duration_ms > 0:
        record["duration_ms"] = int(duration_ms)

    try:
        ctx.augur.record_modelio(
            record,
            step_index=ctx.step_index,
            layer=ctx.layer,
        )
    except Exception as exc:  # noqa: BLE001 — Augur spec §4.3
        logger.warning(
            "record_anthropic_modelio: capture failed layer=%s step=%s: %s",
            ctx.layer, ctx.step_index, exc,
        )


def _map_openai_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    """OpenAI / vLLM ``response.usage`` → schema ``usage`` block.

    The chat-completions usage block is already the schema's native
    shape (``prompt_tokens`` / ``completion_tokens``), so this is mostly
    a whitelist — ``additionalProperties: false`` means we MUST drop
    anything else (vLLM adds ``total_tokens`` and a
    ``prompt_tokens_details`` sub-object we don't carry). The cached-
    prompt count, when present, maps to ``cache_hit_tokens``.
    """
    if not isinstance(usage, dict):
        return {}
    out: dict[str, int] = {}
    if (v := usage.get("prompt_tokens")) is not None:
        out["prompt_tokens"] = int(v)
    if (v := usage.get("completion_tokens")) is not None:
        out["completion_tokens"] = int(v)
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and (v := details.get("cached_tokens")) is not None:
        out["cache_hit_tokens"] = int(v)
    return out


def _extract_openai_response(
    response_json: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]], str | None]:
    """Pull ``text`` + ``tool_calls`` + ``finish_reason`` off the first
    choice of an OpenAI / vLLM chat-completions response.

    Mirrors :func:`_extract_response_text_and_tools` for the Anthropic
    shape: text is the message content (null when only tool calls were
    returned), tool_calls is the verbatim provider-shaped list."""
    choices = response_json.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None, [], None
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    text = content if isinstance(content, str) and content else None
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        tool_calls = []
    finish = choice.get("finish_reason")
    finish_reason = finish if isinstance(finish, str) and finish else None
    return text, tool_calls, finish_reason


def _extract_openai_logprobs(
    response_json: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Map an OpenAI / vLLM ``choices[0].logprobs.content`` block onto
    modelio.schema.json's ``response.logprobs`` shape.

    Needed for RL pipelines (PPO / GRPO importance weighting, DPO) that
    consume the behaviour-policy token logprobs. We map explicitly
    rather than passing the vendor block through because the schema's
    item is ``{token?, token_id?, logprob, top_alternatives?}`` with
    ``additionalProperties: false`` — the vendor ships ``bytes`` and
    ``top_logprobs`` (not ``top_alternatives``), which would fail
    validation verbatim.

    Returns ``None`` when the response carries no logprobs (the vendor
    wasn't asked, or returned a tool-only turn with no token logprobs)
    so the caller leaves the field unset.
    """
    choices = response_json.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    block = choices[0].get("logprobs") or {}
    content = block.get("content")
    if not isinstance(content, list):
        return None

    def _alt(a: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(a, dict) or a.get("logprob") is None:
            return None
        out: dict[str, Any] = {"logprob": float(a["logprob"])}
        if isinstance(a.get("token"), str):
            out["token"] = a["token"]
        if a.get("token_id") is not None:
            out["token_id"] = int(a["token_id"])
        return out

    entries: list[dict[str, Any]] = []
    for tok in content:
        if not isinstance(tok, dict) or tok.get("logprob") is None:
            continue
        entry: dict[str, Any] = {"logprob": float(tok["logprob"])}
        if isinstance(tok.get("token"), str):
            entry["token"] = tok["token"]
        if tok.get("token_id") is not None:
            entry["token_id"] = int(tok["token_id"])
        alts = [m for a in (tok.get("top_logprobs") or []) if (m := _alt(a))]
        if alts:
            entry["top_alternatives"] = alts
        entries.append(entry)
    return entries or None


def record_openai_modelio(
    *,
    request_payload: dict[str, Any],
    response_json: dict[str, Any],
    duration_ms: int,
    ctx: ModelIOContext | None = None,
) -> None:
    """Build a modelio record from an OpenAI / vLLM chat-completions call
    (the Holo3 brain client) and stage it via
    :meth:`AugurAdapter.record_modelio`.

    Sibling of :func:`record_anthropic_modelio` for the
    ``/chat/completions`` shape — needed because the Holo3 brain talks
    to a vLLM OpenAI server, not the instrumented Anthropic client, so
    the brain's own ``planner`` decision is otherwise never captured.
    No-op when no context is published or the adapter is inactive;
    failures are swallowed (Augur spec §4.3).
    """
    ctx = ctx if ctx is not None else current_modelio_context()
    if ctx is None or ctx.augur is None or not getattr(ctx.augur, "active", False):
        return

    text, tool_calls, finish_reason = _extract_openai_response(response_json)
    usage = _map_openai_usage(response_json.get("usage"))

    request_block: dict[str, Any] = {
        "model": str(request_payload.get("model", "")),
    }
    if "messages" in request_payload:
        request_block["messages"] = request_payload["messages"]
    if "tools" in request_payload:
        request_block["tools"] = request_payload["tools"]
    params = {
        k: v for k, v in request_payload.items()
        if k in {"max_tokens", "temperature", "top_p", "top_k", "stop", "tool_choice"}
    }
    if params:
        request_block["params"] = params

    response_block: dict[str, Any] = {}
    if text is not None:
        response_block["text"] = text
    if tool_calls:
        response_block["tool_calls"] = tool_calls
    if finish_reason is not None:
        response_block["stop_reason"] = finish_reason
    if usage:
        response_block["usage"] = usage
    # The OpenAI path knows its own response shape — map logprobs
    # directly to the schema (verified against modelio.schema.json)
    # rather than via the vendor-agnostic SDK extractor, which would
    # have to guess the provider. ``None`` when the vendor wasn't asked
    # for logprobs (the default) → field stays unset.
    logprobs = _extract_openai_logprobs(response_json)
    if logprobs is not None:
        response_block["logprobs"] = logprobs

    record: dict[str, Any] = {
        "schema_version": "0.1",
        "layer": ctx.layer,
        "ts": _utc_now_iso(),
        "request": request_block,
        "response": response_block,
    }
    if ctx.step_index is not None:
        record["step_index"] = int(ctx.step_index)
    if duration_ms > 0:
        record["duration_ms"] = int(duration_ms)

    try:
        ctx.augur.record_modelio(
            record,
            step_index=ctx.step_index,
            layer=ctx.layer,
        )
    except Exception as exc:  # noqa: BLE001 — Augur spec §4.3
        logger.warning(
            "record_openai_modelio: capture failed layer=%s step=%s: %s",
            ctx.layer, ctx.step_index, exc,
        )


__all__ = [
    "ModelIOContext",
    "current_modelio_context",
    "publish_modelio_context",
    "record_anthropic_modelio",
    "record_openai_modelio",
]
