"""Tests for the extractor's per-call max_tokens budget.

The boattrader smoke from PR #218's CLI surfaced a real failure
mode: ``find_listing_content_control`` calls ``self._call(...)`` with
the default ``max_tokens=200``. Claude's vision prompts often produce
a prose preamble before the JSON answer ("I need to look for
controls that expand content..."). With only 200 tokens of budget,
the prose consumes the entire output and the trailing JSON is
truncated mid-string — the parser fails and the step halts.

Confirmed in /tmp/mantis_debug:
- Response 1: prose + ``{"x": 302, "y": 204, "action": "expand",
  "label": "Price", "reason": "Filter section with``  ← truncated.
- Response 2: prose + valid JSON (lucky, fit in budget).
- Response 3: pure JSON (lucky, no prose).
- Response 4: ``{"action": "none"}`` (Claude said nothing visible).

Generic fix: raise the default to 500. Anthropic only bills generated
tokens, so concise responses still cost ~the same; verbose ones are
no longer truncated.

Tests pin the new default and confirm callers that override (e.g.
``extract`` at 1500 for multi-field JSON) are still honoured.
"""

from __future__ import annotations

import inspect

from mantis_agent.extraction.extractor import ClaudeExtractor


def test_default_max_tokens_is_500() -> None:
    """The default max_tokens of ``_call`` must be at least large
    enough to fit the prose-then-JSON pattern Claude empirically
    produces. 200 was not enough; 500 is conservative headroom."""
    sig = inspect.signature(ClaudeExtractor._call)
    default = sig.parameters["max_tokens"].default
    assert default >= 500, (
        f"_call default max_tokens={default}; must be >= 500 to avoid "
        "truncating prose-then-JSON responses (boattrader smoke #218)"
    )


def test_default_was_raised_from_legacy_200() -> None:
    """Forward-only guard: regressing back to 200 reintroduces the
    truncation cascade. Spelled out so a future bisect doesn't
    silently revert."""
    sig = inspect.signature(ClaudeExtractor._call)
    default = sig.parameters["max_tokens"].default
    assert default != 200, (
        "_call default reverted to 200 — see boattrader smoke from "
        "PR #218 for the truncation failure mode this guards against"
    )


def test_explicit_max_tokens_override_is_honoured() -> None:
    """Higher-budget call sites (e.g. ``extract`` at 1500 for multi-
    field JSON; ``find_filter_target`` at 350; ``listing_url`` at 450)
    pass max_tokens explicitly. The default bump must not collapse
    those overrides — the parameter is still a kwarg, not a hardcode."""
    src = inspect.getsource(ClaudeExtractor._call)
    # Function signature still accepts max_tokens as a parameter.
    assert "max_tokens" in src
    # And forwards it into the API request body (no constant override).
    assert '"max_tokens": max_tokens' in src


# ── Content-control prompt — concrete example + anti-malformed guard ────


def test_content_control_prompt_has_concrete_worked_example() -> None:
    """Claude follows concrete examples better than placeholder ``N``
    templates. The worked example must use literal integer values
    (the smoke that triggered this fix saw Claude emit
    ``{\"x\": 302, 43, \"y\": 43, ...}`` — extra positional value —
    when the prompt only showed ``\"x\": N``)."""
    from mantis_agent.extraction.extractor import ClaudeExtractor
    from mantis_agent.extraction.schema import ExtractionSchema

    schema = ExtractionSchema(
        entity_name="boat",
        allowed_controls=("expand_description", "show_phone"),
        forbidden_controls=("contact seller", "request info"),
        fields=[],
    )
    extractor = ClaudeExtractor(api_key="dummy", schema=schema)
    prompt = extractor._get_content_control_prompt()
    # Concrete example with real integers, not ``N``.
    assert '"x": 740' in prompt
    assert '"y": 320' in prompt
    # Sentinel "no visible control" example uses zero coordinates.
    assert '"x": 0, "y": 0' in prompt


def test_content_control_prompt_warns_against_tuple_style_xy() -> None:
    """The smoke saw Claude emit ``{\"x\": 302, 43, \"y\": 43, ...}``
    — an extra unlabeled positional value before ``y``. The prompt
    must explicitly call this out so future runs don't repeat."""
    from mantis_agent.extraction.extractor import ClaudeExtractor
    from mantis_agent.extraction.schema import ExtractionSchema

    schema = ExtractionSchema(
        entity_name="boat",
        allowed_controls=("expand_description",),
        forbidden_controls=("contact seller",),
        fields=[],
    )
    extractor = ClaudeExtractor(api_key="dummy", schema=schema)
    prompt = extractor._get_content_control_prompt()
    text_lower = prompt.lower()
    # Either word that captures the warning intent.
    assert "tuple" in text_lower or "positional" in text_lower
    # And the explicit anti-pattern shape from the smoke.
    assert "302, 43" in prompt or "unlabeled" in text_lower


def test_content_control_prompt_demands_no_prose_preamble() -> None:
    """The prompt must explicitly forbid the prose-preamble pattern
    that triggered max_tokens truncation in the previous smoke."""
    from mantis_agent.extraction.extractor import ClaudeExtractor
    from mantis_agent.extraction.schema import ExtractionSchema

    schema = ExtractionSchema(
        entity_name="boat",
        allowed_controls=("expand_description",),
        forbidden_controls=("contact seller",),
        fields=[],
    )
    extractor = ClaudeExtractor(api_key="dummy", schema=schema)
    prompt = extractor._get_content_control_prompt()
    text_lower = prompt.lower()
    assert "no prose" in text_lower or "no commentary" in text_lower


# ── tool_use schema enforcement (Anthropic structured output) ──────────


def test_call_with_tool_schema_helper_exists() -> None:
    """The new helper is the canonical seam for structured-output
    extractor calls. Replaces the prompt-only \"output ONLY valid
    JSON\" pattern that the boattrader smoke proved unreliable."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    assert hasattr(ClaudeExtractor, "_call_with_tool_schema")
    sig = inspect.signature(ClaudeExtractor._call_with_tool_schema)
    # Required keyword-only args.
    for arg in ("tool_name", "tool_description", "input_schema"):
        assert arg in sig.parameters
        assert sig.parameters[arg].kind == inspect.Parameter.KEYWORD_ONLY


def test_call_with_tool_schema_forces_tool_choice(monkeypatch) -> None:
    """The Anthropic request must set ``tool_choice`` to the named tool
    so Claude is FORCED to emit the schema-validated tool_use block.
    Without forcing, Claude can fall back to text content (the failure
    mode this whole helper is supposed to prevent)."""
    from PIL import Image

    from mantis_agent.extraction.extractor import ClaudeExtractor

    captured: dict = {}

    class _FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "test_tool",
                        "input": {"x": 100, "y": 200, "ok": True},
                    },
                ],
            }

    def fake_post(url, headers, json, timeout):  # noqa: A002 — match requests sig
        captured["body"] = json
        return _FakeResponse()

    import requests as real_requests
    monkeypatch.setattr(real_requests, "post", fake_post)

    extractor = ClaudeExtractor(api_key="dummy")
    img = Image.new("RGB", (10, 10), color=(255, 255, 255))
    out = extractor._call_with_tool_schema(
        img,
        "find the thing",
        tool_name="test_tool",
        tool_description="report the thing",
        input_schema={
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "ok": {"type": "boolean"},
            },
            "required": ["x", "y", "ok"],
        },
    )

    # Forced tool-choice with the named tool.
    assert captured["body"]["tool_choice"] == {"type": "tool", "name": "test_tool"}
    # Tool definition is in the request.
    assert captured["body"]["tools"][0]["name"] == "test_tool"
    assert "input_schema" in captured["body"]["tools"][0]
    # Validated tool_use input flows back as the helper's return.
    assert out == {"x": 100, "y": 200, "ok": True}


def test_call_with_tool_schema_returns_none_on_no_tool_use_block(monkeypatch) -> None:
    """Defensive: if Anthropic returns a response without the expected
    tool_use block (e.g. API regression, misconfigured request), the
    helper returns None so the caller can fall back to its existing
    not_found handling. No crash."""
    from PIL import Image

    from mantis_agent.extraction.extractor import ClaudeExtractor

    class _FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {"content": [{"type": "text", "text": "I refuse to use the tool."}]}

    import requests as real_requests
    monkeypatch.setattr(real_requests, "post", lambda *a, **k: _FakeResponse())

    extractor = ClaudeExtractor(api_key="dummy")
    img = Image.new("RGB", (10, 10), color=(255, 255, 255))
    out = extractor._call_with_tool_schema(
        img, "x", tool_name="t", tool_description="d",
        input_schema={"type": "object"},
    )
    assert out is None


def test_call_with_tool_schema_returns_none_on_api_error(monkeypatch) -> None:
    """Non-200 responses must NOT crash — they propagate as None and
    the caller's not_found handling kicks in."""
    from PIL import Image

    from mantis_agent.extraction.extractor import ClaudeExtractor

    class _Err:
        status_code = 500
        text = "internal error"

    import requests as real_requests
    monkeypatch.setattr(real_requests, "post", lambda *a, **k: _Err())

    extractor = ClaudeExtractor(api_key="dummy")
    img = Image.new("RGB", (10, 10), color=(255, 255, 255))
    out = extractor._call_with_tool_schema(
        img, "x", tool_name="t", tool_description="d",
        input_schema={"type": "object"},
    )
    assert out is None


def test_find_listing_content_control_uses_tool_schema(monkeypatch) -> None:
    """Integration: the content-control finder must route through the
    schema-enforced helper, not the legacy ``_call`` + ``_parse_json``
    pair. Pin the wiring so a future refactor doesn't silently revert."""
    from unittest.mock import MagicMock

    from PIL import Image

    from mantis_agent.extraction.extractor import ClaudeExtractor
    from mantis_agent.extraction.schema import ExtractionSchema

    schema = ExtractionSchema(
        entity_name="boat",
        allowed_controls=("expand_description", "show_phone"),
        forbidden_controls=("contact seller",),
        fields=[],
    )
    extractor = ClaudeExtractor(api_key="dummy", schema=schema)
    extractor._call_with_tool_schema = MagicMock(  # type: ignore[method-assign]
        return_value={
            "x": 740, "y": 320, "action": "expand_description",
            "label": "Show more", "reason": "see-more chevron",
        },
    )
    extractor._call = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError("legacy _call must not be used"),
    )

    img = Image.new("RGB", (1280, 720), color=(255, 255, 255))
    result = extractor.find_listing_content_control(img)

    extractor._call_with_tool_schema.assert_called_once()
    extractor._call.assert_not_called()
    assert result == {
        "x": 740, "y": 320, "action": "expand_description",
        "label": "Show more", "reason": "see-more chevron",
    }
    # The schema's allowed_controls + "none" sentinel must be the
    # enum the tool's input_schema accepts — that's how the structured
    # output stays plan-driven (per #209's analysis-stage discipline).
    call_kwargs = extractor._call_with_tool_schema.call_args.kwargs
    enum = call_kwargs["input_schema"]["properties"]["action"]["enum"]
    assert "expand_description" in enum
    assert "show_phone" in enum
    assert "none" in enum


def test_find_listing_content_control_returns_none_when_tool_returns_none_action(monkeypatch) -> None:
    """If Claude reports ``action=none`` via the tool_use block, the
    finder returns None — same contract the legacy path had."""
    from unittest.mock import MagicMock

    from PIL import Image

    from mantis_agent.extraction.extractor import ClaudeExtractor
    from mantis_agent.extraction.schema import ExtractionSchema

    schema = ExtractionSchema(
        entity_name="boat",
        allowed_controls=("expand_description",),
        forbidden_controls=("contact seller",),
        fields=[],
    )
    extractor = ClaudeExtractor(api_key="dummy", schema=schema)
    extractor._call_with_tool_schema = MagicMock(  # type: ignore[method-assign]
        return_value={
            "x": 0, "y": 0, "action": "none",
            "label": "", "reason": "none visible",
        },
    )

    img = Image.new("RGB", (10, 10), color=(255, 255, 255))
    assert extractor.find_listing_content_control(img) is None
