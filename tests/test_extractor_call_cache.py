"""Tests for ClaudeExtractor `_call` + `_call_many` prompt caching.

#720 follow-up: PR #721 cached the tool-use shims (`_call_with_tool_schema*`)
but the dominant extract_multi path uses `_call_many` (raw text completion),
not the tool-use shims. This module tests the cache_control on the actual
hot paths.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

from PIL import Image


def _fake_resp(input_tokens: int = 50, cache_read: int = 1200):
    class _FakeResp:
        status_code = 200

        def json(self):
            return {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 5,
                    "cache_read_input_tokens": cache_read,
                    "cache_creation_input_tokens": 0,
                },
            }

    return _FakeResp()


# ── _call (single screenshot) ────────────────────────────────────────


def test_call_marks_prompt_with_cache_control_by_default() -> None:
    from mantis_agent.extraction.extractor import ClaudeExtractor

    captured: dict = {}

    def _capture_post(url, headers=None, json=None, timeout=None):  # noqa: ARG001
        captured["payload"] = json
        return _fake_resp()

    extractor = ClaudeExtractor(api_key="fake")
    with patch("requests.post", side_effect=_capture_post):
        extractor._call(
            screenshot=Image.new("RGB", (16, 16), "white"),
            prompt="Extract the fields",
        )

    content = captured["payload"]["messages"][0]["content"]
    # First block is the prompt with cache_control
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Extract the fields"
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    # Image comes second, no cache_control on it
    assert content[1]["type"] == "image"
    assert "cache_control" not in content[1]


def test_call_skips_cache_control_when_opted_out() -> None:
    from mantis_agent.extraction.extractor import ClaudeExtractor

    captured: dict = {}

    def _capture_post(url, headers=None, json=None, timeout=None):  # noqa: ARG001
        captured["payload"] = json
        return _fake_resp(cache_read=0)

    extractor = ClaudeExtractor(api_key="fake")
    with patch("requests.post", side_effect=_capture_post):
        extractor._call(
            screenshot=Image.new("RGB", (16, 16), "white"),
            prompt="one-off extract",
            cache_prompt=False,
        )

    content = captured["payload"]["messages"][0]["content"]
    assert "cache_control" not in content[0]


def test_call_emits_cache_telemetry_on_hit(caplog) -> None:
    from mantis_agent.extraction.extractor import ClaudeExtractor

    def _capture_post(url, headers=None, json=None, timeout=None):  # noqa: ARG001
        return _fake_resp(cache_read=2400)

    extractor = ClaudeExtractor(api_key="fake")
    with patch("requests.post", side_effect=_capture_post):
        with caplog.at_level(logging.WARNING):
            extractor._call(
                screenshot=Image.new("RGB", (16, 16), "white"),
                prompt="extract",
            )

    msgs = [r.message for r in caplog.records if "[cache] extract:" in r.message]
    assert msgs
    assert "read=2400" in msgs[0]


def test_call_no_telemetry_when_cache_prompt_false(caplog) -> None:
    from mantis_agent.extraction.extractor import ClaudeExtractor

    def _capture_post(url, headers=None, json=None, timeout=None):  # noqa: ARG001
        return _fake_resp(cache_read=2400)  # response includes cache fields

    extractor = ClaudeExtractor(api_key="fake")
    with patch("requests.post", side_effect=_capture_post):
        with caplog.at_level(logging.WARNING):
            extractor._call(
                screenshot=Image.new("RGB", (16, 16), "white"),
                prompt="extract",
                cache_prompt=False,
            )

    # No telemetry log when caller didn't opt in
    msgs = [r.message for r in caplog.records if "[cache] extract:" in r.message]
    assert not msgs


# ── _call_many (multi screenshot) ────────────────────────────────────


def test_call_many_marks_prompt_with_cache_control_by_default() -> None:
    from mantis_agent.extraction.extractor import ClaudeExtractor

    captured: dict = {}

    def _capture_post(url, headers=None, json=None, timeout=None):  # noqa: ARG001
        captured["payload"] = json
        return _fake_resp()

    extractor = ClaudeExtractor(api_key="fake")
    with patch("requests.post", side_effect=_capture_post):
        extractor._call_many(
            screenshots=[
                Image.new("RGB", (16, 16), "white"),
                Image.new("RGB", (16, 16), "black"),
            ],
            prompt="Extract leads from these screenshots",
        )

    content = captured["payload"]["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Extract leads from these screenshots"
    assert content[0]["cache_control"] == {"type": "ephemeral"}


def test_call_many_skips_cache_when_opted_out() -> None:
    from mantis_agent.extraction.extractor import ClaudeExtractor

    captured: dict = {}

    def _capture_post(url, headers=None, json=None, timeout=None):  # noqa: ARG001
        captured["payload"] = json
        return _fake_resp(cache_read=0)

    extractor = ClaudeExtractor(api_key="fake")
    with patch("requests.post", side_effect=_capture_post):
        extractor._call_many(
            screenshots=[Image.new("RGB", (16, 16), "white")],
            prompt="extract",
            cache_prompt=False,
        )

    content = captured["payload"]["messages"][0]["content"]
    assert "cache_control" not in content[0]


def test_call_many_emits_extract_many_telemetry(caplog) -> None:
    from mantis_agent.extraction.extractor import ClaudeExtractor

    def _capture_post(url, headers=None, json=None, timeout=None):  # noqa: ARG001
        return _fake_resp(cache_read=3000)

    extractor = ClaudeExtractor(api_key="fake")
    with patch("requests.post", side_effect=_capture_post):
        with caplog.at_level(logging.WARNING):
            extractor._call_many(
                screenshots=[Image.new("RGB", (16, 16), "white")],
                prompt="extract leads",
            )

    msgs = [r.message for r in caplog.records if "[cache] extract_many:" in r.message]
    assert msgs
    assert "read=3000" in msgs[0]


def test_call_many_no_telemetry_when_opted_out(caplog) -> None:
    from mantis_agent.extraction.extractor import ClaudeExtractor

    def _capture_post(url, headers=None, json=None, timeout=None):  # noqa: ARG001
        return _fake_resp(cache_read=2400)

    extractor = ClaudeExtractor(api_key="fake")
    with patch("requests.post", side_effect=_capture_post):
        with caplog.at_level(logging.WARNING):
            extractor._call_many(
                screenshots=[Image.new("RGB", (16, 16), "white")],
                prompt="extract",
                cache_prompt=False,
            )

    msgs = [r.message for r in caplog.records if "[cache] extract_many:" in r.message]
    assert not msgs
