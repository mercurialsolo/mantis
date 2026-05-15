"""Direct coverage for :mod:`mantis_agent._anthropic.client` (#406).

The retry mechanics are already pinned via ``tests/test_extractor_retry.py``
through the legacy ``ClaudeExtractor._post_anthropic_with_retry`` shim,
so we don't re-walk the full retry decision tree here. This file instead
pins what the legacy tests couldn't reach:

- ``call_with_tool_schema`` / ``call_with_tool_schema_multi`` continue
  to be the entry points other modules can rely on (so a future caller
  outside ``ClaudeExtractor`` — e.g. ``ClaudeFormTargetProvider`` —
  gets the same contract).
- The ``log_prefix`` flows into warning messages so per-caller log
  filtering still works (Modal log queries that watch
  ``ClaudeExtractor`` vs ``ClaudeFormTarget`` won't be broken by the
  factoring).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mantis_agent._anthropic.client import (
    _TRANSIENT_STATUS_CODES,
    AnthropicToolUseClient,
    _retry_delay,
)


def _fake_response(status_code: int, headers: dict | None = None, *, body: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = f"<body status={status_code}>"
    resp.json = MagicMock(return_value=body or {})
    return resp


def _img() -> Image.Image:
    return Image.new("RGB", (10, 10), color=(255, 255, 255))


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch) -> None:
    monkeypatch.setattr("mantis_agent._anthropic.client.time.sleep", lambda *_: None)


# ── module-level constants are still importable ─────────────────────


def test_transient_status_codes_includes_529() -> None:
    """529 (Anthropic overload) is the canonical retry case — moving
    the constant must not lose it."""
    assert 529 in _TRANSIENT_STATUS_CODES
    assert 429 in _TRANSIENT_STATUS_CODES


def test_retry_delay_honours_numeric_retry_after() -> None:
    assert _retry_delay(0, "3") == 3.0


# ── post_messages_with_retry ────────────────────────────────────────


def test_post_messages_recovers_from_529() -> None:
    """Direct call to the client (not via the extractor shim) survives
    the same 529-then-200 sequence."""
    client = AnthropicToolUseClient(api_key="k", model="m")
    seq = [_fake_response(529), _fake_response(200)]
    with patch("requests.post", side_effect=seq) as post:
        out = client.post_messages_with_retry({}, timeout=30)
    assert out.status_code == 200
    assert post.call_count == 2


def test_post_messages_honours_log_prefix(caplog) -> None:
    """When the client is constructed with a custom ``log_prefix``,
    transient-error log lines carry that prefix — keeps the existing
    'ClaudeExtractor transient HTTP 529 …' wording stable AND lets a
    future ClaudeFormTarget log under its own banner."""
    import logging
    caplog.set_level(logging.INFO, logger="mantis_agent._anthropic.client")
    client = AnthropicToolUseClient(api_key="k", model="m", log_prefix="ClaudeFormTarget")
    seq = [_fake_response(529), _fake_response(200)]
    with patch("requests.post", side_effect=seq):
        client.post_messages_with_retry({}, timeout=30)
    transient_log = next(
        (r.message for r in caplog.records if "transient HTTP" in r.message),
        "",
    )
    assert "ClaudeFormTarget transient HTTP 529" in transient_log


# ── call_with_tool_schema ───────────────────────────────────────────


def test_call_with_tool_schema_returns_validated_input() -> None:
    """Happy path: 200 + a tool_use block of the right name → its
    ``input`` dict flows back unchanged."""
    client = AnthropicToolUseClient(api_key="k", model="m")
    body = {"content": [{"type": "tool_use", "name": "find_x", "input": {"x": 10, "y": 20}}]}
    with patch("requests.post", return_value=_fake_response(200, body=body)):
        out = client.call_with_tool_schema(
            _img(), "find the email field",
            tool_name="find_x",
            tool_description="locate",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        )
    assert out == {"x": 10, "y": 20}


def test_call_with_tool_schema_returns_none_without_api_key() -> None:
    """No API key → no call attempted, return None (matches legacy)."""
    client = AnthropicToolUseClient(api_key="", model="m")
    with patch("requests.post") as post:
        out = client.call_with_tool_schema(
            _img(), "x",
            tool_name="t",
            tool_description="d",
            input_schema={"type": "object"},
        )
    assert out is None
    post.assert_not_called()


def test_call_with_tool_schema_returns_none_when_no_matching_tool_block() -> None:
    """200 + content blocks that don't match the requested tool name
    → None (caller treats as not-found)."""
    client = AnthropicToolUseClient(api_key="k", model="m")
    body = {"content": [{"type": "tool_use", "name": "different_tool", "input": {"x": 1}}]}
    with patch("requests.post", return_value=_fake_response(200, body=body)):
        out = client.call_with_tool_schema(
            _img(), "x",
            tool_name="find_email",
            tool_description="d",
            input_schema={"type": "object"},
        )
    assert out is None


# ── call_with_tool_schema_multi ─────────────────────────────────────


def test_call_with_tool_schema_multi_returns_validated_input() -> None:
    """Multi-screenshot path also returns the validated dict."""
    client = AnthropicToolUseClient(api_key="k", model="m")
    body = {"content": [{"type": "tool_use", "name": "before_after", "input": {"changed": True}}]}
    with patch("requests.post", return_value=_fake_response(200, body=body)):
        out = client.call_with_tool_schema_multi(
            [_img(), _img()], "diff",
            tool_name="before_after",
            tool_description="d",
            input_schema={"type": "object"},
            labels=["BEFORE", "AFTER"],
        )
    assert out == {"changed": True}
