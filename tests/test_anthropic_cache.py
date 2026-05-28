"""Tests for prompt-cache split — brain-context Phase 0 (#715).

Locks the cache-boundary contract:

- `as_cached_system(text)` wraps a string in a single text block with
  `cache_control: ephemeral`.
- `mark_last_tool_cached(tools)` adds `cache_control: ephemeral` to the
  last tool definition without mutating the input.
- `extract_cache_telemetry(response_json)` pulls the four cache /
  token counters from the Anthropic response shape.
- brain_claude, agentic_recovery, ClaudeGrounding all emit the cache
  marker on the right block when constructing requests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mantis_agent._anthropic.cache import (
    as_cached_system,
    cached_text_block,
    extract_cache_telemetry,
    mark_last_tool_cached,
)


# ── helpers ───────────────────────────────────────────────────────────


def test_as_cached_system_wraps_string_in_block_list() -> None:
    out = as_cached_system("you are a helpful assistant")
    assert out == [{
        "type": "text",
        "text": "you are a helpful assistant",
        "cache_control": {"type": "ephemeral"},
    }]


def test_as_cached_system_empty_returns_empty_list() -> None:
    assert as_cached_system("") == []


def test_mark_last_tool_cached_marks_only_last() -> None:
    tools = [
        {"name": "tool_a", "description": "first"},
        {"name": "tool_b", "description": "second"},
        {"name": "tool_c", "description": "third"},
    ]
    out = mark_last_tool_cached(tools)
    # First two unmarked
    assert "cache_control" not in out[0]
    assert "cache_control" not in out[1]
    # Last marked
    assert out[2]["cache_control"] == {"type": "ephemeral"}
    # Other fields preserved on the marked tool
    assert out[2]["name"] == "tool_c"
    assert out[2]["description"] == "third"


def test_mark_last_tool_cached_does_not_mutate_input() -> None:
    tools = [{"name": "tool_a"}, {"name": "tool_b"}]
    original = [dict(t) for t in tools]
    mark_last_tool_cached(tools)
    assert tools == original


def test_mark_last_tool_cached_empty_returns_empty() -> None:
    assert mark_last_tool_cached([]) == []


def test_cached_text_block_shape() -> None:
    block = cached_text_block("static instructions")
    assert block == {
        "type": "text",
        "text": "static instructions",
        "cache_control": {"type": "ephemeral"},
    }


# ── telemetry ─────────────────────────────────────────────────────────


def test_extract_cache_telemetry_full_shape() -> None:
    resp = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 800,
            "cache_read_input_tokens": 1200,
        },
    }
    out = extract_cache_telemetry(resp)
    assert out == {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 800,
        "cache_read_input_tokens": 1200,
    }


def test_extract_cache_telemetry_omits_missing_fields() -> None:
    resp = {"usage": {"input_tokens": 10}}
    out = extract_cache_telemetry(resp)
    assert out == {"input_tokens": 10}
    assert "cache_read_input_tokens" not in out


def test_extract_cache_telemetry_no_usage_returns_empty() -> None:
    assert extract_cache_telemetry({}) == {}
    assert extract_cache_telemetry({"foo": "bar"}) == {}


def test_extract_cache_telemetry_handles_non_dict() -> None:
    assert extract_cache_telemetry(None) == {}  # type: ignore[arg-type]
    assert extract_cache_telemetry("error") == {}  # type: ignore[arg-type]


# ── brain_claude integration ──────────────────────────────────────────


def test_brain_claude_sends_cached_system_and_tools() -> None:
    """brain_claude.think() must build a request with system as a
    cached block list AND the last tool marked for caching."""
    from PIL import Image

    from mantis_agent.brain_claude import ClaudeBrain

    captured: dict = {}

    class _FakeResp:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            # Minimal valid Claude response — tool_use to satisfy parsing
            return {
                "content": [
                    {"type": "tool_use", "name": "computer",
                     "input": {"action": "wait"}},
                ],
                "usage": {
                    "input_tokens": 10, "output_tokens": 5,
                    "cache_read_input_tokens": 1500,
                    "cache_creation_input_tokens": 0,
                },
            }

    def _capture_post(url, headers=None, json=None, timeout=None):  # noqa: ARG001
        captured["payload"] = json
        return _FakeResp()

    brain = ClaudeBrain(api_key="fake-key")
    with patch("mantis_agent.brain_claude.requests.post", side_effect=_capture_post):
        brain.think(
            frames=[Image.new("RGB", (10, 10), "white")],
            task="test", screen_size=(1280, 720),
        )

    payload = captured["payload"]
    # System is a list of cached blocks (not a string)
    assert isinstance(payload["system"], list)
    assert payload["system"][-1]["cache_control"] == {"type": "ephemeral"}
    # Last tool carries cache_control
    assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    # Non-last tools don't carry it (unless there's only one tool)
    if len(payload["tools"]) > 1:
        assert "cache_control" not in payload["tools"][0]


# ── grounding integration ─────────────────────────────────────────────


def test_grounding_sends_cached_system() -> None:
    """ClaudeGrounding splits the static instructions into a cached
    system block; per-call data goes in the user message."""
    from PIL import Image

    from mantis_agent.grounding import ClaudeGrounding

    captured: dict = {}

    class _FakeResp:
        status_code = 200

        def json(self) -> dict:
            return {
                "content": [{"type": "text", "text": "100 200"}],
                "usage": {
                    "input_tokens": 50, "output_tokens": 5,
                    "cache_read_input_tokens": 100,
                    "cache_creation_input_tokens": 0,
                },
            }

    def _capture_post(url, headers=None, json=None, timeout=None):  # noqa: ARG001
        captured["payload"] = json
        return _FakeResp()

    g = ClaudeGrounding(api_key="fake-key")
    with patch("requests.post", side_effect=_capture_post), \
         patch("mantis_agent.grounding.encode_screenshot_for_claude",
               return_value=("b64stub", "image/png")):
        g.ground(
            screenshot=Image.new("RGB", (1280, 720), "white"),
            description="click the Save button",
            initial_x=100, initial_y=200,
        )

    payload = captured["payload"]
    # System is a cached list
    assert isinstance(payload["system"], list)
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    # Static instructions live in system; per-call data lives in user
    sys_text = payload["system"][0]["text"]
    assert "RULES" in sys_text
    assert "Output ONLY two numbers" in sys_text
    # Description (mutable) is NOT in the system block
    assert "click the Save button" not in sys_text


# ── agentic_recovery integration ──────────────────────────────────────


def test_agentic_recovery_marks_tool_cached() -> None:
    """The recovery tool input_schema is the largest stable block in
    that call — must be marked for caching."""
    from mantis_agent.agentic_recovery import _call_recovery_tool

    captured: dict = {}

    class _FakeResp:
        status_code = 200
        content = b"{}"

        def json(self) -> dict:
            return {
                "content": [{"type": "tool_use", "name": "record_recovery",
                             "input": {"mode": "halt", "reasoning": "test"}}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

    def _capture_post(url, headers=None, json=None, timeout=None):  # noqa: ARG001
        captured["payload"] = json
        return _FakeResp()

    step = MagicMock(intent="click X", type="click", params={})
    with patch("requests.post", side_effect=_capture_post):
        _call_recovery_tool(
            step=step, failure_data="x", screenshot=None,
            plan_context=[], attempts=1, api_key="fake", model="claude-haiku",
            prior_hints=[], page_context={},
        )

    payload = captured["payload"]
    tools = payload["tools"]
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}
    assert tools[-1]["name"] == "record_recovery"
