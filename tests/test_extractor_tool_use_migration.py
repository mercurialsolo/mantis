"""Tests for the tool_use schema rollout across the extractor surface.

#219 introduced the ``_call_with_tool_schema`` seam and migrated
``find_listing_content_control`` to it. This rollout extends the same
treatment to:

  • ``verify_gate``       — boolean+string gate result
  • ``find_filter_target`` — five-field click/type/select target
  • ``find_form_target``   — five-field, same shape as filter
  • ``extract``            — dynamic, schema-driven multi-field

Each migration replaces the prompt-only "Output ONLY valid JSON"
pattern (which produced prose-only / truncated / malformed responses
in production) with Anthropic's server-validated tool_use shape. The
tests pin: the helper is invoked with the right tool name + schema,
the legacy ``_call`` + ``_parse_json`` path is NOT used, and the
returned value flows through unchanged.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from mantis_agent.extraction.extractor import ClaudeExtractor
from mantis_agent.extraction.schema import ExtractionSchema


def _img() -> Image.Image:
    return Image.new("RGB", (10, 10), color=(255, 255, 255))


def _wired(extractor: ClaudeExtractor) -> tuple[MagicMock, MagicMock]:
    """Replace the tool_use seam + legacy _call. Returns (tool_mock, call_mock).

    Tests assert the tool seam is called and ``_call`` is NOT — that's
    how we lock in the migration so a future refactor doesn't silently
    revert to the prompt-only path.

    #406 moved the form-target methods (find_form_target /
    find_target_by_affordance / verify_dropdown_value) out of
    ClaudeExtractor into a dedicated provider that holds its own
    AnthropicToolUseClient. The same mock is wired onto both seams so
    tests written before that split keep working without case analysis.
    """
    tool = MagicMock()
    extractor._call_with_tool_schema = tool  # type: ignore[method-assign]
    extractor._form_target_provider._client.call_with_tool_schema = tool  # type: ignore[method-assign]
    call = MagicMock(side_effect=AssertionError("legacy _call must not be used"))
    extractor._call = call  # type: ignore[method-assign]
    return tool, call


# ── verify_gate ─────────────────────────────────────────────────────────


def _wire_verify(
    extractor: ClaudeExtractor,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Replace the Haiku + Opus verify clients with mocks (#421).

    Returns (haiku_mock, opus_mock, call_mock). The legacy ``_call``
    seam is also blocked so a regression that drops the tool_use
    routing surfaces as a test failure rather than silent prose
    parsing.
    """
    haiku = MagicMock()
    opus = MagicMock()
    extractor._verify_client.call_with_tool_schema = haiku  # type: ignore[method-assign]
    extractor._verify_escalation_client.call_with_tool_schema = opus  # type: ignore[method-assign]
    call = MagicMock(side_effect=AssertionError("legacy _call must not be used"))
    extractor._call = call  # type: ignore[method-assign]
    return haiku, opus, call


def test_verify_gate_routes_through_haiku_first() -> None:
    extractor = ClaudeExtractor(api_key="dummy")
    haiku, opus, call = _wire_verify(extractor)
    haiku.return_value = {"passed": True, "reason": "Filters applied"}

    passed, reason = extractor.verify_gate(_img(), "Filters applied")

    haiku.assert_called_once()
    opus.assert_not_called()  # no escalation when Haiku says PASS
    call.assert_not_called()
    assert passed is True
    assert reason == "Filters applied"


def test_verify_gate_schema_shape() -> None:
    """The tool's input_schema must lock the boolean / string fields."""
    extractor = ClaudeExtractor(api_key="dummy")
    haiku, _, _ = _wire_verify(extractor)
    haiku.return_value = {"passed": True, "reason": "ok"}

    extractor.verify_gate(_img(), "Filters applied")

    schema = haiku.call_args.kwargs["input_schema"]
    assert schema["properties"]["passed"]["type"] == "boolean"
    assert schema["properties"]["reason"]["type"] == "string"
    assert set(schema["required"]) == {"passed", "reason"}


def test_verify_gate_returns_false_on_no_tool_use() -> None:
    """Server-side validation can't fail on shape, but the tool_use
    block can be missing entirely (API regression). The runner halts
    on a False gate, which is the right safety default. No escalation
    fires when the Haiku call itself failed to return a usable result —
    a transient API blip shouldn't be papered over by burning an Opus
    call too."""
    extractor = ClaudeExtractor(api_key="dummy")
    haiku, opus, _ = _wire_verify(extractor)
    haiku.return_value = None

    passed, reason = extractor.verify_gate(_img(), "anything")
    assert passed is False
    assert "tool_use" in reason
    opus.assert_not_called()


def test_verify_gate_escalates_to_opus_on_haiku_fail() -> None:
    """Haiku FAIL must trigger one Opus re-ask (#421 §3). Trust Opus
    when it disagrees — recovery loops on a Haiku false-negative cost
    ~$0.50, the escalation costs ~$0.003."""
    extractor = ClaudeExtractor(api_key="dummy")
    haiku, opus, _ = _wire_verify(extractor)
    haiku.return_value = {"passed": False, "reason": "haiku-uncertain"}
    opus.return_value = {"passed": True, "reason": "opus-saw-filter-pill"}

    passed, reason = extractor.verify_gate(_img(), "filters applied")

    haiku.assert_called_once()
    opus.assert_called_once()
    assert passed is True
    assert reason == "opus-saw-filter-pill"


def test_verify_gate_keeps_fail_when_opus_also_fails() -> None:
    """Both verdicts FAIL → final FAIL with Opus's reason (more
    authoritative than Haiku's)."""
    extractor = ClaudeExtractor(api_key="dummy")
    haiku, opus, _ = _wire_verify(extractor)
    haiku.return_value = {"passed": False, "reason": "haiku-no-filter"}
    opus.return_value = {"passed": False, "reason": "opus-confirmed-no-filter"}

    passed, reason = extractor.verify_gate(_img(), "filters applied")

    assert passed is False
    assert reason == "opus-confirmed-no-filter"


def test_verify_gate_keeps_haiku_verdict_when_escalation_errors() -> None:
    """Opus call errors out (returns None) — fall back to Haiku's
    FAIL verdict rather than hide it behind a None."""
    extractor = ClaudeExtractor(api_key="dummy")
    haiku, opus, _ = _wire_verify(extractor)
    haiku.return_value = {"passed": False, "reason": "haiku-no-filter"}
    opus.return_value = None

    passed, reason = extractor.verify_gate(_img(), "filters applied")

    assert passed is False
    assert reason == "haiku-no-filter"


def test_verify_gate_opts_into_cache_tools() -> None:
    """#421 §4: ``verify_gate`` must mark the tool spec as
    cache-eligible so the schema/description amortise across all
    gates in a run."""
    extractor = ClaudeExtractor(api_key="dummy")
    haiku, _, _ = _wire_verify(extractor)
    haiku.return_value = {"passed": True, "reason": "ok"}

    extractor.verify_gate(_img(), "anything")

    assert haiku.call_args.kwargs.get("cache_tools") is True


def test_verify_gate_uses_haiku_default_model() -> None:
    """#421 §2: the verify client defaults to Haiku 4.5; the
    escalation client defaults to Opus 4.7. Operators can pin
    either via env, but the constructor defaults must be stable."""
    extractor = ClaudeExtractor(api_key="dummy")
    assert extractor._verify_client.model == "claude-haiku-4-5-20251001"
    assert extractor._verify_escalation_client.model == "claude-opus-4-7"


# ── find_filter_target ──────────────────────────────────────────────────


def test_find_filter_target_routes_through_tool_schema() -> None:
    extractor = ClaudeExtractor(api_key="dummy")
    tool, call = _wired(extractor)
    tool.return_value = {
        "x": 220, "y": 110, "action": "click",
        "value": "", "label": "Private Seller pill",
    }

    out = extractor.find_filter_target(_img(), "Click Private Seller filter")

    tool.assert_called_once()
    call.assert_not_called()
    assert out == {
        "x": 220, "y": 110, "action": "click",
        "value": "", "label": "Private Seller pill",
    }


def test_find_filter_target_schema_includes_full_action_enum() -> None:
    extractor = ClaudeExtractor(api_key="dummy")
    tool, _ = _wired(extractor)
    tool.return_value = {
        "x": 0, "y": 0, "action": "not_found", "value": "", "label": "x",
    }

    extractor.find_filter_target(_img(), "x")

    schema = tool.call_args.kwargs["input_schema"]
    enum = schema["properties"]["action"]["enum"]
    # ``right_click`` (#373) joined the enum so Claude can adaptively
    # suggest a context-menu open when a filter chip is right-click-
    # only (rare but observed on a few SaaS filter pickers).
    assert set(enum) == {"click", "right_click", "type", "select", "not_found"}
    assert set(schema["required"]) == {"x", "y", "action", "value", "label"}


def test_find_filter_target_returns_none_on_not_found() -> None:
    extractor = ClaudeExtractor(api_key="dummy")
    tool, _ = _wired(extractor)
    tool.return_value = {
        "x": 0, "y": 0, "action": "not_found",
        "value": "", "label": "no matching filter visible",
    }
    assert extractor.find_filter_target(_img(), "Pick X") is None


def test_find_filter_target_returns_none_on_zero_coordinates() -> None:
    """Defensive: even if Claude returns action='click' but emits
    (0, 0), treat as not-located rather than clicking origin."""
    extractor = ClaudeExtractor(api_key="dummy")
    tool, _ = _wired(extractor)
    tool.return_value = {
        "x": 0, "y": 0, "action": "click",
        "value": "", "label": "uncertain",
    }
    assert extractor.find_filter_target(_img(), "Pick X") is None


def test_find_filter_target_returns_none_on_no_tool_use() -> None:
    extractor = ClaudeExtractor(api_key="dummy")
    tool, _ = _wired(extractor)
    tool.return_value = None
    assert extractor.find_filter_target(_img(), "Pick X") is None


# ── find_form_target ────────────────────────────────────────────────────


def test_find_form_target_routes_through_tool_schema() -> None:
    extractor = ClaudeExtractor(api_key="dummy")
    tool, call = _wired(extractor)
    tool.return_value = {
        "x": 691, "y": 284, "action": "type",
        "value": "alice", "label": "User ID input field",
    }

    out = extractor.find_form_target(
        _img(),
        "Click the user ID input field and type alice",
        target_label="user ID",
        target_value="alice",
    )

    tool.assert_called_once()
    call.assert_not_called()
    assert out == {
        "x": 691, "y": 284, "action": "type",
        "value": "alice", "label": "User ID input field",
    }


def test_find_form_target_schema_matches_filter_target_shape() -> None:
    """``form`` and ``filter`` targets share the same shape — both are
    'find one labelled element and tell me how to interact'. Pin that
    parity so a future schema drift on one method gets noticed."""
    extractor = ClaudeExtractor(api_key="dummy")

    # Capture form schema.
    tool_form, _ = _wired(extractor)
    tool_form.return_value = {
        "x": 0, "y": 0, "action": "not_found",
        "value": "", "label": "",
    }
    extractor.find_form_target(_img(), "x")
    form_schema = tool_form.call_args.kwargs["input_schema"]

    # Capture filter schema (rewire fresh mocks).
    tool_filter, _ = _wired(extractor)
    tool_filter.return_value = {
        "x": 0, "y": 0, "action": "not_found",
        "value": "", "label": "",
    }
    extractor.find_filter_target(_img(), "x")
    filter_schema = tool_filter.call_args.kwargs["input_schema"]

    assert form_schema == filter_schema


def test_find_form_target_returns_none_on_not_found() -> None:
    extractor = ClaudeExtractor(api_key="dummy")
    tool, _ = _wired(extractor)
    tool.return_value = {
        "x": 0, "y": 0, "action": "not_found",
        "value": "", "label": "Cloudflare challenge",
    }
    assert extractor.find_form_target(_img(), "Click X") is None


def test_find_form_target_returns_none_on_no_tool_use() -> None:
    extractor = ClaudeExtractor(api_key="dummy")
    tool, _ = _wired(extractor)
    tool.return_value = None
    assert extractor.find_form_target(_img(), "Click X") is None


# ── extract (dynamic schema) ───────────────────────────────────────────


def test_extract_routes_through_tool_schema_with_legacy_shape() -> None:
    """Without an ExtractionSchema, ``extract`` falls back to the
    canonical marketplace-listing input_schema."""
    extractor = ClaudeExtractor(api_key="dummy")
    tool, call = _wired(extractor)
    tool.return_value = {
        "year": "2024", "make": "Sea Ray", "model": "Sundancer",
        "price": "$189000", "phone": "786-555-1234",
        "url": "https://x/boat/1",
        "seller": "Bob", "is_dealer": False,
    }

    result = extractor.extract(_img())

    tool.assert_called_once()
    call.assert_not_called()
    assert result.year == "2024"
    assert result.make == "Sea Ray"
    assert result.is_dealer is False
    assert result.confidence == 0.9


def test_extract_dynamic_schema_built_from_extraction_schema_fields() -> None:
    """When an ExtractionSchema is set, the tool_use input_schema
    must include every schema field as a property + the universal
    ``is_spam`` boolean. Schema field types map to JSON-Schema types
    (str → string, bool → boolean, int → integer)."""
    schema = ExtractionSchema(
        entity_name="job",
        spam_label="recruiter",
        fields=[
            {"name": "title", "type": "str", "required": True},
            {"name": "company", "type": "str"},
            {"name": "salary_min", "type": "int"},
            {"name": "remote", "type": "bool"},
        ],
    )
    extractor = ClaudeExtractor(api_key="dummy", schema=schema)
    tool, _ = _wired(extractor)
    tool.return_value = {
        "title": "Senior SRE", "company": "Acme",
        "salary_min": 200000, "remote": True, "is_spam": False,
    }

    extractor.extract(_img())

    input_schema = tool.call_args.kwargs["input_schema"]
    props = input_schema["properties"]
    assert props["title"] == {"type": "string"}
    assert props["company"] == {"type": "string"}
    assert props["salary_min"] == {"type": "integer"}
    assert props["remote"] == {"type": "boolean"}
    assert props["is_spam"] == {"type": "boolean"}
    # Every property is required — the structure-validating effect is
    # what makes tool_use stronger than prompt-only "Output JSON".
    assert set(input_schema["required"]) == set(props.keys())


def test_extract_returns_low_confidence_when_tool_returns_none() -> None:
    extractor = ClaudeExtractor(api_key="dummy")
    tool, _ = _wired(extractor)
    tool.return_value = None

    result = extractor.extract(_img())
    assert result.confidence == 0.1
    assert "no tool_use" in result.raw_response


def test_extract_passes_higher_max_tokens() -> None:
    """Multi-field extracts can exceed the default 500-token budget;
    the extract path should request at least 1500 tokens."""
    extractor = ClaudeExtractor(api_key="dummy")
    tool, _ = _wired(extractor)
    tool.return_value = {
        "year": "", "make": "", "model": "", "price": "",
        "phone": "", "url": "", "seller": "", "is_dealer": False,
    }

    extractor.extract(_img())
    assert tool.call_args.kwargs["max_tokens"] >= 1500


# ── Schema → JSON-Schema type mapping ──────────────────────────────────


@pytest.mark.parametrize(
    "schema_type,json_type",
    [
        ("str", "string"),
        ("string", "string"),
        ("int", "integer"),
        ("integer", "integer"),
        ("float", "number"),
        ("number", "number"),
        ("bool", "boolean"),
        ("boolean", "boolean"),
        ("UnknownType", "string"),  # default fallback
    ],
)
def test_extract_field_type_map_is_complete(
    schema_type: str, json_type: str,
) -> None:
    """Every schema field type ExtractionSchema documents must map to
    a JSON-Schema type the Anthropic tool_use API accepts."""
    schema = ExtractionSchema(
        entity_name="thing",
        fields=[{"name": "value", "type": schema_type}],
    )
    extractor = ClaudeExtractor(api_key="dummy", schema=schema)
    input_schema = extractor._build_extract_input_schema()
    assert input_schema["properties"]["value"]["type"] == json_type
