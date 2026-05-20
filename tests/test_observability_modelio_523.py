"""Tests for the #523 modelio capture wiring (PR B-1).

PR B-1 only lands the plumbing — the client-side hook + the
mapper + the contextvar. No call sites push a layer yet, so on
disk no modelio records get written from a real run. PR B-2..B-5
wire each layer one at a time.

These tests exercise the plumbing in isolation:

* mapper round-trips (Anthropic input_tokens / output_tokens /
  cache_read_input_tokens / cache_creation_input_tokens →
  modelio.schema.json prompt_tokens / completion_tokens /
  cache_hit_tokens / cache_creation_tokens)
* contextvar is plumbed and respected by the client
* validates against the SDK's modelio.schema.json (this is the
  fix-the-30-minute-debug-cycle test the SDK author flagged)
* no-op behavior when no context is published (the default)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("augur_sdk")

from mantis_agent.observability.augur import AugurAdapter
from mantis_agent.observability.modelio import (
    ModelIOContext,
    _map_anthropic_usage,
    current_modelio_context,
    publish_modelio_context,
    record_anthropic_modelio,
)


# ── usage mapper (the SDK-author-flagged compatibility test) ─────────────


def test_map_anthropic_usage_renames_to_openai_shape():
    """Anthropic's response.usage uses input_tokens / output_tokens;
    modelio.schema.json's usage block uses prompt_tokens /
    completion_tokens (additionalProperties: false). Wrong shape
    raises a Draft 2020-12 validation error in record_modelio — caught
    once during early integration so we test the mapping locks in."""
    anthropic_usage = {
        "input_tokens": 12_345,
        "output_tokens": 678,
        "cache_read_input_tokens": 9_000,
        "cache_creation_input_tokens": 2_500,
    }
    mapped = _map_anthropic_usage(anthropic_usage)
    assert mapped == {
        "prompt_tokens": 12_345,
        "completion_tokens": 678,
        "cache_hit_tokens": 9_000,
        "cache_creation_tokens": 2_500,
    }
    # All-Anthropic-only fields stripped
    assert "input_tokens" not in mapped
    assert "output_tokens" not in mapped


def test_map_anthropic_usage_handles_missing_fields():
    """Anthropic always returns input/output; cache fields are only on
    requests with cache breakpoints. Missing → omitted from the output
    (the SDK schema allows partial usage blocks)."""
    assert _map_anthropic_usage({"input_tokens": 100, "output_tokens": 50}) == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
    }
    assert _map_anthropic_usage({}) == {}
    assert _map_anthropic_usage(None) == {}


def test_map_anthropic_usage_drops_unknown_keys():
    """additionalProperties: false on the schema — when Anthropic adds
    a new usage field (they have, repeatedly), we must drop it rather
    than passing through and breaking validation."""
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "new_unsupported_field": 999,  # would fail validation
    }
    mapped = _map_anthropic_usage(usage)
    assert "new_unsupported_field" not in mapped
    assert mapped == {"prompt_tokens": 100, "completion_tokens": 50}


# ── contextvar plumbing ──────────────────────────────────────────────────


def test_current_modelio_context_returns_none_by_default():
    """No publish → None. The client's _record_modelio_if_active gates
    on this — the default path through the hot LLM call site must be
    a no-op until call sites opt in."""
    assert current_modelio_context() is None


def test_publish_modelio_context_sets_and_resets(tmp_path: Path, monkeypatch):
    """publish_modelio_context is a contextmanager — leaving the with
    block must restore the prior value (None or otherwise)."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="ctxvar_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    assert current_modelio_context() is None
    with publish_modelio_context(a, layer="planner", step_index=0):
        ctx = current_modelio_context()
        assert ctx is not None
        assert ctx.layer == "planner"
        assert ctx.step_index == 0
        assert ctx.augur is a
    # Reset to None on exit.
    assert current_modelio_context() is None
    a.close(status="completed")


def test_publish_modelio_context_rejects_unknown_layer(tmp_path: Path, monkeypatch):
    """Layer typos must surface as a warning, not silently drop captures
    — yielding through the block as if no context was published."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="ctxvar_v2", tenant_id="t", session_name="s", out_dir=tmp_path)
    with publish_modelio_context(a, layer="bogus_typo", step_index=0):
        # Unknown layer → no context published.
        assert current_modelio_context() is None
    a.close(status="completed")


def test_publish_modelio_context_noop_when_adapter_inactive():
    """When augur is None or inactive, publish should still cleanly
    yield through — telemetry never breaks the run."""
    with publish_modelio_context(None, layer="planner", step_index=0):
        assert current_modelio_context() is None


# ── end-to-end: record a real modelio bundle file ────────────────────────


_ANTHROPIC_RESPONSE_FIXTURE: dict[str, Any] = {
    "id": "msg_abc",
    "model": "claude-sonnet-4-7-20260120",
    "role": "assistant",
    "stop_reason": "tool_use",
    "type": "message",
    "content": [
        {
            "type": "tool_use", "id": "tu_001",
            "name": "extract_lead",
            "input": {"name": "Alice", "email": "alice@example.com"},
        },
    ],
    "usage": {
        "input_tokens": 1_500,
        "output_tokens": 80,
        "cache_read_input_tokens": 400,
    },
}

_REQUEST_PAYLOAD: dict[str, Any] = {
    "model": "claude-sonnet-4-7-20260120",
    "max_tokens": 500,
    "tools": [{"name": "extract_lead", "description": "...", "input_schema": {}}],
    "tool_choice": {"type": "tool", "name": "extract_lead"},
    "messages": [{"role": "user", "content": [{"type": "text", "text": "extract this"}]}],
}


def test_record_anthropic_modelio_writes_validated_bundle_file(
    tmp_path: Path, monkeypatch,
):
    """End-to-end: with a real AugurAdapter open + a context
    published, record_anthropic_modelio stages a file under modelio/
    that satisfies modelio.schema.json (validate=True is the default;
    the SDK raises on shape mismatches)."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="modelio_e2e", tenant_id="t", session_name="s", out_dir=tmp_path)
    with publish_modelio_context(a, layer="planner", step_index=0):
        record_anthropic_modelio(
            request_payload=_REQUEST_PAYLOAD,
            response_json=_ANTHROPIC_RESPONSE_FIXTURE,
            duration_ms=842,
        )
    a.close(status="completed")
    # Bundle path convention: modelio/<step:04d>-<layer>-<seq>.json
    modelio_dir = tmp_path / "modelio"
    assert modelio_dir.exists(), "record_modelio should have created the modelio/ dir"
    files = sorted(modelio_dir.glob("*.json"))
    assert files, "no modelio file written"
    record = json.loads(files[0].read_text())
    assert record["layer"] == "planner"
    assert record["step_index"] == 0  # 0-based on the record (path is 1-based)
    assert record["request"]["model"] == "claude-sonnet-4-7-20260120"
    # OpenAI-shape usage — NOT input_tokens / output_tokens
    assert record["response"]["usage"]["prompt_tokens"] == 1_500
    assert record["response"]["usage"]["completion_tokens"] == 80
    assert record["response"]["usage"]["cache_hit_tokens"] == 400
    assert record["duration_ms"] == 842
    assert record["response"]["stop_reason"] == "tool_use"
    # The tool_use block is captured verbatim under tool_calls.
    assert record["response"]["tool_calls"][0]["name"] == "extract_lead"


def test_record_anthropic_modelio_noop_without_context(tmp_path: Path, monkeypatch):
    """Without a published context, the mapper is a clean no-op —
    no exceptions, no files written."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="modelio_noop", tenant_id="t", session_name="s", out_dir=tmp_path)
    record_anthropic_modelio(
        request_payload=_REQUEST_PAYLOAD,
        response_json=_ANTHROPIC_RESPONSE_FIXTURE,
        duration_ms=100,
    )  # no context published
    a.close(status="completed")
    assert not (tmp_path / "modelio").exists() or not list((tmp_path / "modelio").glob("*.json"))


def test_record_anthropic_modelio_step_index_none_means_run_scoped(
    tmp_path: Path, monkeypatch,
):
    """An initial plan call that precedes any step is run-scoped —
    step_index=None should land as null on the record (schema allows
    that) and the SDK stages under modelio/0000-*-<seq>.json."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="modelio_run_scoped", tenant_id="t", session_name="s", out_dir=tmp_path)
    with publish_modelio_context(a, layer="planner", step_index=None):
        record_anthropic_modelio(
            request_payload=_REQUEST_PAYLOAD,
            response_json=_ANTHROPIC_RESPONSE_FIXTURE,
            duration_ms=200,
        )
    a.close(status="completed")
    files = sorted((tmp_path / "modelio").glob("*.json"))
    assert files
    record = json.loads(files[0].read_text())
    assert record.get("step_index") in (None, 0)  # null OR absent both pass schema


# ── client integration: capture fires only when context is published ─────


def test_anthropic_client_capture_fires_when_context_published(
    tmp_path: Path, monkeypatch,
):
    """When publish_modelio_context() is active during a successful
    call_with_tool_schema, the client should invoke
    record_anthropic_modelio via _record_modelio_if_active. Test by
    asserting a modelio file lands on disk after one captured call."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    a = AugurAdapter(run_id="client_capture", tenant_id="t", session_name="s", out_dir=tmp_path)

    # Build a fake response object the client thinks came from requests.post.
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = _ANTHROPIC_RESPONSE_FIXTURE

    from mantis_agent._anthropic.client import AnthropicToolUseClient
    client = AnthropicToolUseClient(api_key="test", model="claude-sonnet-4-7-20260120")
    monkeypatch.setattr(client, "post_messages_with_retry", lambda *a, **k: fake_resp)

    from PIL import Image
    img = Image.new("RGB", (100, 100), color="white")

    with publish_modelio_context(a, layer="planner", step_index=0):
        out = client.call_with_tool_schema(
            img, "extract this",
            tool_name="extract_lead",
            tool_description="...",
            input_schema={"type": "object"},
        )
    a.close(status="completed")
    assert out == {"name": "Alice", "email": "alice@example.com"}
    files = sorted((tmp_path / "modelio").glob("*.json"))
    assert files, "client did not invoke record_modelio with a context active"


def test_anthropic_client_capture_skips_when_no_context(
    tmp_path: Path, monkeypatch,
):
    """The default path — no caller publishes a layer — must be a
    no-op. Critical: PR B-1 should NOT change the on-disk shape of
    any existing user's run until PR B-2..B-5 wire individual layers."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    a = AugurAdapter(run_id="client_noop", tenant_id="t", session_name="s", out_dir=tmp_path)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = _ANTHROPIC_RESPONSE_FIXTURE

    from mantis_agent._anthropic.client import AnthropicToolUseClient
    client = AnthropicToolUseClient(api_key="test", model="claude-sonnet-4-7-20260120")
    monkeypatch.setattr(client, "post_messages_with_retry", lambda *a, **k: fake_resp)

    from PIL import Image
    img = Image.new("RGB", (100, 100), color="white")

    # NO publish_modelio_context wrapper.
    out = client.call_with_tool_schema(
        img, "extract this",
        tool_name="extract_lead",
        tool_description="...",
        input_schema={"type": "object"},
    )
    a.close(status="completed")
    assert out == {"name": "Alice", "email": "alice@example.com"}
    # No modelio/ dir, OR an empty one — depends on SDK init.
    modelio_dir = tmp_path / "modelio"
    if modelio_dir.exists():
        assert not list(modelio_dir.glob("*.json"))


def test_modelio_context_is_dataclass_with_expected_fields():
    """The contextvar payload is :class:`ModelIOContext` — guard
    against accidental rename / drop of fields PR B-2..B-5 will rely
    on."""
    ctx = ModelIOContext(augur=object(), layer="planner", step_index=3)
    assert ctx.layer == "planner"
    assert ctx.step_index == 3
    # Frozen — mutation should raise.
    with pytest.raises(Exception):  # FrozenInstanceError
        ctx.layer = "grounding"  # type: ignore[misc]
