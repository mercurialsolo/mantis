"""Gap 1 (Holo3 completion) — capture the brain's own decision as modelio.

The runner wraps ``brain.think()`` in a ``planner`` context (Gap 1),
but Holo3 talks to a vLLM OpenAI server, not the instrumented
Anthropic client — so the brain's decision produced *no* modelio
record on the production model. Live verification on Modal confirmed
this: a Holo3 run captured only ``grounding``-layer records (the
Anthropic-backed form-targeting calls), never the brain itself.

This pins the fix: an OpenAI-shape mapper + a capture hook on the
Holo3 client so the planner prompt->response pair is staged via
``record_modelio`` under whatever layer the runner published.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mantis_agent.observability.modelio import (
    _extract_openai_response,
    _map_openai_usage,
    publish_modelio_context,
    record_openai_modelio,
)


@pytest.fixture
def fake_augur():
    augur = MagicMock()
    augur.active = True
    augur.record_modelio = MagicMock()
    return augur


def _vllm_response(*, content="I'll click login", with_tool=True, with_usage=True):
    message = {"content": content}
    if with_tool:
        message["tool_calls"] = [{
            "id": "call_1", "type": "function",
            "function": {"name": "click", "arguments": '{"x": 100, "y": 200}'},
        }]
    resp = {"choices": [{"message": message, "finish_reason": "tool_calls"}]}
    if with_usage:
        resp["usage"] = {
            "prompt_tokens": 1500, "completion_tokens": 42, "total_tokens": 1542,
            "prompt_tokens_details": {"cached_tokens": 1200},
        }
    return resp


# ── mappers ─────────────────────────────────────────────────────────


def test_map_openai_usage_whitelists_schema_keys():
    out = _map_openai_usage({
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 8},
    })
    # total_tokens dropped (additionalProperties:false); cached → cache_hit.
    assert out == {"prompt_tokens": 10, "completion_tokens": 5, "cache_hit_tokens": 8}


def test_map_openai_usage_handles_missing():
    assert _map_openai_usage(None) == {}
    assert _map_openai_usage({}) == {}


def test_extract_openai_response_pulls_text_tools_finish():
    text, tools, finish = _extract_openai_response(_vllm_response())
    assert text == "I'll click login"
    assert tools and tools[0]["function"]["name"] == "click"
    assert finish == "tool_calls"


def test_extract_openai_response_tool_only_has_null_text():
    text, tools, finish = _extract_openai_response(
        _vllm_response(content="", with_tool=True)
    )
    assert text is None
    assert tools


def test_extract_openai_response_handles_empty_choices():
    assert _extract_openai_response({"choices": []}) == (None, [], None)
    assert _extract_openai_response({}) == (None, [], None)


# ── record_openai_modelio ───────────────────────────────────────────


def test_record_stages_modelio_under_published_layer(fake_augur):
    payload = {
        "model": "Holo3-35B-A3B",
        "messages": [{"role": "user", "content": "go"}],
        "tools": [{"type": "function"}],
        "max_tokens": 512, "temperature": 0.0,
    }
    with publish_modelio_context(fake_augur, layer="planner", step_index=3):
        record_openai_modelio(
            request_payload=payload, response_json=_vllm_response(), duration_ms=87,
        )
    fake_augur.record_modelio.assert_called_once()
    rec = fake_augur.record_modelio.call_args.args[0]
    assert rec["layer"] == "planner"
    assert rec["step_index"] == 3
    assert rec["duration_ms"] == 87
    assert rec["request"]["model"] == "Holo3-35B-A3B"
    assert rec["request"]["params"]["max_tokens"] == 512
    assert rec["response"]["text"] == "I'll click login"
    assert rec["response"]["tool_calls"]
    assert rec["response"]["stop_reason"] == "tool_calls"
    assert rec["response"]["usage"]["prompt_tokens"] == 1500
    # layer forwarded to record_modelio kwarg too
    assert fake_augur.record_modelio.call_args.kwargs["layer"] == "planner"


def test_record_is_noop_without_context(fake_augur):
    # No publish_modelio_context active.
    record_openai_modelio(
        request_payload={"model": "x"}, response_json=_vllm_response(), duration_ms=1,
    )
    fake_augur.record_modelio.assert_not_called()


def test_record_is_noop_when_inactive():
    inactive = MagicMock()
    inactive.active = False
    inactive.record_modelio = MagicMock()
    with publish_modelio_context(inactive, layer="planner", step_index=0):
        # publish_modelio_context itself no-ops on inactive, so context
        # is None inside; record must not stage.
        record_openai_modelio(
            request_payload={"model": "x"}, response_json=_vllm_response(), duration_ms=1,
        )
    inactive.record_modelio.assert_not_called()


def test_record_never_raises_on_bad_response(fake_augur):
    with publish_modelio_context(fake_augur, layer="planner", step_index=0):
        # Malformed response — must not raise.
        record_openai_modelio(
            request_payload={"model": "x"}, response_json={"weird": True}, duration_ms=1,
        )
    # A record with empty response block still stages (text/tools just absent).
    fake_augur.record_modelio.assert_called_once()


# ── Holo3 client hook ───────────────────────────────────────────────


def test_holo3_hook_stages_under_planner_when_context_active(fake_augur):
    from mantis_agent.brain_holo3 import Holo3Brain

    payload = {"model": "Holo3-35B-A3B", "messages": [], "max_tokens": 256}
    with publish_modelio_context(fake_augur, layer="planner", step_index=2):
        Holo3Brain._record_modelio_if_active(payload, _vllm_response(), 50)
    fake_augur.record_modelio.assert_called_once()
    assert fake_augur.record_modelio.call_args.args[0]["layer"] == "planner"


def test_holo3_hook_noop_without_context(fake_augur):
    from mantis_agent.brain_holo3 import Holo3Brain

    Holo3Brain._record_modelio_if_active({"model": "x"}, _vllm_response(), 50)
    fake_augur.record_modelio.assert_not_called()
