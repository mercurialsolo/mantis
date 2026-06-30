"""Regression: the CUA Claude brain must send the computer-use ``anthropic-beta``
header through the shared retry client.

Root cause (cua-issues follow-up): when brain_claude was routed through the
shared ``AnthropicToolUseClient`` for retry handling, the
``anthropic-beta: computer-use-2025-11-24`` header was dropped — the client
built its own base headers and had no seam to carry it. Without the beta,
the API rejects the ``computer_20251124`` tool type with HTTP 400 on EVERY
call, so the brain never produces an action and every /v1/cua run loops to a
hard-loop halt. These tests pin the ``extra_headers`` seam and the brain's
use of it.
"""

from __future__ import annotations

from typing import Any

import pytest

from mantis_agent._anthropic.client import AnthropicToolUseClient
from mantis_agent import brain_claude


class _FakeResp:
    status_code = 200

    def json(self) -> dict[str, Any]:
        return {"content": []}


def _capture_post(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Patch requests.post and return the list that records each call's headers."""
    captured: list[dict] = []

    def _fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        captured.append(dict(headers or {}))
        return _FakeResp()

    import requests
    monkeypatch.setattr(requests, "post", _fake_post)
    return captured


def test_extra_headers_are_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_post(monkeypatch)
    client = AnthropicToolUseClient(
        api_key="sk-test", model="claude-sonnet-4-6",
        extra_headers={"anthropic-beta": "computer-use-2025-11-24"},
    )
    client.post_messages_with_retry({"messages": []}, timeout=5)

    assert captured, "requests.post was never called"
    hdr = captured[-1]
    assert hdr.get("anthropic-beta") == "computer-use-2025-11-24"
    # base headers still present
    assert hdr.get("x-api-key") == "sk-test"
    assert hdr.get("anthropic-version") == "2023-06-01"


def test_no_extra_headers_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Text/grounding/extraction callers (no extra_headers) stay beta-free."""
    captured = _capture_post(monkeypatch)
    client = AnthropicToolUseClient(api_key="sk-test", model="claude-sonnet-4-6")
    client.post_messages_with_retry({"messages": []}, timeout=5)

    assert "anthropic-beta" not in captured[-1]


def test_beta_header_constant_pairs_with_tool_version() -> None:
    """The beta string must name the 2025-11-24 computer-use beta that the
    ``computer_20251124`` tool type requires (they must move together)."""
    assert "computer-use-2025-11-24" in brain_claude._ANTHROPIC_BETA_HEADER
    tool = brain_claude.ClaudeBrain._build_tools.__get__(
        object.__new__(brain_claude.ClaudeBrain)
    )((1280, 720))[0]
    assert tool["type"] == "computer_20251124"


def test_brain_think_passes_beta_to_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """ClaudeBrain.think constructs the shared client WITH the beta header."""
    seen: list[dict] = []

    real_init = AnthropicToolUseClient.__init__

    def _spy_init(self, *args, **kwargs):
        seen.append(dict(kwargs.get("extra_headers") or {}))
        # Short-circuit the network: make post return a no-op response.
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(AnthropicToolUseClient, "__init__", _spy_init)
    _capture_post(monkeypatch)

    brain = brain_claude.ClaudeBrain(api_key="sk-test", model="claude-sonnet-4-6")
    from PIL import Image
    brain.think([Image.new("RGB", (64, 64))], task="read the page", screen_size=(1280, 720))

    assert any(h.get("anthropic-beta") == brain_claude._ANTHROPIC_BETA_HEADER for h in seen), (
        f"think() did not pass the computer-use beta header; saw {seen!r}"
    )
