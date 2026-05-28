"""Tests for ClaudeExtractor prompt caching — #720 follow-up to #715.

Locks the contract: extractor calls via the shared tool-use client
default to ``cache_tools=True``, and the underlying client marks the
tool definition with ``cache_control: ephemeral`` regardless of
whether the single- or multi-screenshot path is used.
"""

from __future__ import annotations

from unittest.mock import patch

from PIL import Image


def _make_payload_capture():
    captured: dict = {}

    class _FakeResp:
        status_code = 200

        def json(self) -> dict:
            return {
                "content": [{
                    "type": "tool_use",
                    "name": "extract",
                    "input": {"field": "value"},
                }],
                "usage": {
                    "input_tokens": 50, "output_tokens": 5,
                    "cache_read_input_tokens": 1200,
                    "cache_creation_input_tokens": 0,
                },
            }

    def _post(self, payload, timeout=None):  # noqa: ARG001
        captured["payload"] = payload
        return _FakeResp()

    return captured, _post


# ── shared client: single-screenshot path ─────────────────────────────


def test_client_call_with_tool_schema_marks_cache_when_opted_in() -> None:
    from mantis_agent._anthropic.client import AnthropicToolUseClient

    captured, fake_post = _make_payload_capture()

    client = AnthropicToolUseClient(api_key="fake", model="claude-opus-4-7")
    with patch.object(
        AnthropicToolUseClient, "post_messages_with_retry", fake_post
    ):
        client.call_with_tool_schema(
            screenshot=Image.new("RGB", (16, 16), "white"),
            prompt="test",
            tool_name="extract",
            tool_description="extract data",
            input_schema={"type": "object"},
            cache_tools=True,
        )

    tools = captured["payload"]["tools"]
    assert tools[0]["cache_control"] == {"type": "ephemeral"}


def test_client_call_with_tool_schema_no_cache_when_opted_out() -> None:
    """Default: no cache marker. Preserves back-compat for callers
    that don't opt in (e.g. one-off verifier calls)."""
    from mantis_agent._anthropic.client import AnthropicToolUseClient

    captured, fake_post = _make_payload_capture()

    client = AnthropicToolUseClient(api_key="fake", model="claude-opus-4-7")
    with patch.object(
        AnthropicToolUseClient, "post_messages_with_retry", fake_post
    ):
        client.call_with_tool_schema(
            screenshot=Image.new("RGB", (16, 16), "white"),
            prompt="test",
            tool_name="extract",
            tool_description="extract data",
            input_schema={"type": "object"},
            # cache_tools defaults False
        )

    tools = captured["payload"]["tools"]
    assert "cache_control" not in tools[0]


# ── shared client: multi-screenshot path ──────────────────────────────


def test_client_call_with_tool_schema_multi_supports_cache_tools() -> None:
    """The multi-screenshot helper newly grew the cache_tools kwarg —
    used by the extractor's _call_with_tool_schema_multi shim."""
    from mantis_agent._anthropic.client import AnthropicToolUseClient

    captured, fake_post = _make_payload_capture()

    client = AnthropicToolUseClient(api_key="fake", model="claude-opus-4-7")
    with patch.object(
        AnthropicToolUseClient, "post_messages_with_retry", fake_post
    ):
        client.call_with_tool_schema_multi(
            screenshots=[
                Image.new("RGB", (16, 16), "white"),
                Image.new("RGB", (16, 16), "black"),
            ],
            prompt="test",
            tool_name="extract",
            tool_description="extract data",
            input_schema={"type": "object"},
            cache_tools=True,
        )

    tools = captured["payload"]["tools"]
    assert tools[0]["cache_control"] == {"type": "ephemeral"}


def test_client_call_with_tool_schema_multi_default_no_cache() -> None:
    from mantis_agent._anthropic.client import AnthropicToolUseClient

    captured, fake_post = _make_payload_capture()

    client = AnthropicToolUseClient(api_key="fake", model="claude-opus-4-7")
    with patch.object(
        AnthropicToolUseClient, "post_messages_with_retry", fake_post
    ):
        client.call_with_tool_schema_multi(
            screenshots=[Image.new("RGB", (16, 16), "white")],
            prompt="test",
            tool_name="extract",
            tool_description="extract data",
            input_schema={"type": "object"},
        )

    tools = captured["payload"]["tools"]
    assert "cache_control" not in tools[0]


# ── extractor shims default cache_tools=True ─────────────────────────


def test_extractor_call_with_tool_schema_passes_cache_tools_true() -> None:
    """The extractor's back-compat shim must default cache_tools=True
    so production extract calls hit the cache."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    extractor = ClaudeExtractor(api_key="fake")
    captured: dict = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return {"field": "value"}

    with patch.object(extractor._client, "call_with_tool_schema", _capture):
        extractor._call_with_tool_schema(
            screenshot=Image.new("RGB", (16, 16), "white"),
            prompt="extract this",
            tool_name="extract",
            tool_description="extract data",
            input_schema={"type": "object"},
        )

    assert captured.get("cache_tools") is True


def test_extractor_call_with_tool_schema_multi_passes_cache_tools_true() -> None:
    from mantis_agent.extraction.extractor import ClaudeExtractor

    extractor = ClaudeExtractor(api_key="fake")
    captured: dict = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return {"field": "value"}

    with patch.object(extractor._client, "call_with_tool_schema_multi", _capture):
        extractor._call_with_tool_schema_multi(
            screenshots=[Image.new("RGB", (16, 16), "white")],
            prompt="extract this",
            tool_name="extract",
            tool_description="extract data",
            input_schema={"type": "object"},
        )

    assert captured.get("cache_tools") is True


# ── telemetry log fires on cache hits ────────────────────────────────


def test_client_emits_cache_telemetry_when_hits(caplog) -> None:
    """A successful response carrying cache_read_input_tokens > 0
    must produce a WARNING-level `[cache] extractor:` log line."""
    import logging

    from mantis_agent._anthropic.client import AnthropicToolUseClient

    _, fake_post = _make_payload_capture()
    client = AnthropicToolUseClient(api_key="fake", model="claude-opus-4-7")

    with patch.object(
        AnthropicToolUseClient, "post_messages_with_retry", fake_post
    ):
        with caplog.at_level(logging.WARNING):
            client.call_with_tool_schema(
                screenshot=Image.new("RGB", (16, 16), "white"),
                prompt="test",
                tool_name="extract",
                tool_description="extract data",
                input_schema={"type": "object"},
                cache_tools=True,
            )

    cache_lines = [r for r in caplog.records if "[cache] extractor:" in r.message]
    assert cache_lines, "expected at least one [cache] extractor: warning"
    msg = cache_lines[0].message
    assert "read=1200" in msg


def test_client_no_telemetry_when_cache_tools_false(caplog) -> None:
    """When the caller opted out, don't emit the telemetry line even
    if the response happens to carry cache fields (would clutter the
    log without delivering any cache benefit)."""
    import logging

    from mantis_agent._anthropic.client import AnthropicToolUseClient

    _, fake_post = _make_payload_capture()
    client = AnthropicToolUseClient(api_key="fake", model="claude-opus-4-7")

    with patch.object(
        AnthropicToolUseClient, "post_messages_with_retry", fake_post
    ):
        with caplog.at_level(logging.WARNING):
            client.call_with_tool_schema(
                screenshot=Image.new("RGB", (16, 16), "white"),
                prompt="test",
                tool_name="extract",
                tool_description="extract data",
                input_schema={"type": "object"},
                # cache_tools defaults False
            )

    cache_lines = [r for r in caplog.records if "[cache] extractor:" in r.message]
    assert not cache_lines


def test_client_multi_emits_extractor_multi_telemetry(caplog) -> None:
    import logging

    from mantis_agent._anthropic.client import AnthropicToolUseClient

    _, fake_post = _make_payload_capture()
    client = AnthropicToolUseClient(api_key="fake", model="claude-opus-4-7")

    with patch.object(
        AnthropicToolUseClient, "post_messages_with_retry", fake_post
    ):
        with caplog.at_level(logging.WARNING):
            client.call_with_tool_schema_multi(
                screenshots=[Image.new("RGB", (16, 16), "white")],
                prompt="test",
                tool_name="extract",
                tool_description="extract data",
                input_schema={"type": "object"},
                cache_tools=True,
            )

    cache_lines = [r for r in caplog.records if "[cache] extractor_multi:" in r.message]
    assert cache_lines, "expected at least one [cache] extractor_multi: warning"
