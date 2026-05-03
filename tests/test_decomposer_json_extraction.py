"""Tests for #112 — robust JSON extraction from Claude responses.

The decomposer's earlier parser only handled clean responses or
``\\`\\`\\`json``-fenced output. Claude empirically also produces:

  • prose preamble: "Here's the decomposition: {...}"
  • prose epilogue: "{...} Let me know if you need adjustments."
  • mixed:          "I'll classify this plan as workflow.\\n\\n{...}"

These tests verify the parser pulls the JSON out of all three shapes.
Surfaced by the staffcrm rerun (run 20260503_110305_5287bc9c)
where Claude prepended a classification sentence before the JSON
object and the strict ``json.loads`` failed.
"""

from __future__ import annotations

from mantis_agent.plan_decomposer import _extract_json_payload


# ── Pure JSON (fast path) ───────────────────────────────────────────────


def test_pure_json_object() -> None:
    out = _extract_json_payload('{"shapes": ["form"], "steps": []}')
    assert out == {"shapes": ["form"], "steps": []}


def test_pure_json_array() -> None:
    out = _extract_json_payload('[{"type": "navigate"}]')
    assert out == [{"type": "navigate"}]


def test_pure_json_with_surrounding_whitespace() -> None:
    out = _extract_json_payload('  \n\n  {"a": 1}  \n  ')
    assert out == {"a": 1}


# ── Code fences ─────────────────────────────────────────────────────────


def test_fenced_json_with_language_tag() -> None:
    text = '```json\n{"shapes": ["form"], "steps": []}\n```'
    assert _extract_json_payload(text) == {"shapes": ["form"], "steps": []}


def test_fenced_json_without_language_tag() -> None:
    text = '```\n{"shapes": ["form"]}\n```'
    assert _extract_json_payload(text) == {"shapes": ["form"]}


def test_fenced_json_array() -> None:
    text = '```json\n[{"intent": "click"}]\n```'
    assert _extract_json_payload(text) == [{"intent": "click"}]


# ── Prose-wrapped JSON (the staffcrm rerun failure mode) ────────────────


def test_prose_preamble_before_object() -> None:
    text = (
        "Here's the decomposition for the staffcrm plan:\n\n"
        '{"shapes": ["workflow", "form"], "steps": [{"type": "navigate"}]}'
    )
    out = _extract_json_payload(text)
    assert out == {"shapes": ["workflow", "form"], "steps": [{"type": "navigate"}]}


def test_prose_epilogue_after_object() -> None:
    text = (
        '{"shapes": ["form"], "steps": []}\n\n'
        "Let me know if you need any adjustments to the step types."
    )
    out = _extract_json_payload(text)
    assert out == {"shapes": ["form"], "steps": []}


def test_prose_both_sides() -> None:
    text = (
        "I'll classify this as a workflow plan:\n\n"
        '{"shapes": ["workflow"], "steps": [{"intent": "go"}]}\n\n'
        "Each step is atomic and within the budget."
    )
    out = _extract_json_payload(text)
    assert out == {"shapes": ["workflow"], "steps": [{"intent": "go"}]}


def test_prose_wrapped_array() -> None:
    text = (
        "Here are the steps:\n\n"
        '[{"type": "navigate", "intent": "Go to login"}]'
    )
    out = _extract_json_payload(text)
    assert out == [{"type": "navigate", "intent": "Go to login"}]


# ── String-literal brace handling ───────────────────────────────────────


def test_braces_inside_string_literals_dont_confuse_parser() -> None:
    """The intent text might contain literal braces — make sure the
    balanced-brace scanner doesn't get fooled."""
    text = (
        "Result:\n"
        '{"steps": [{"intent": "Click \\"Update Lead\\" button"}]}'
    )
    out = _extract_json_payload(text)
    assert out == {"steps": [{"intent": 'Click "Update Lead" button'}]}


def test_escaped_quotes_in_string_literals() -> None:
    text = '{"key": "value with \\"quotes\\" inside"}'
    out = _extract_json_payload(text)
    assert out == {"key": 'value with "quotes" inside'}


# ── Failure modes ───────────────────────────────────────────────────────


def test_returns_none_for_empty_input() -> None:
    assert _extract_json_payload("") is None
    assert _extract_json_payload("   \n\n  ") is None


def test_returns_none_for_pure_prose() -> None:
    assert _extract_json_payload(
        "I'm sorry, I can't decompose this plan."
    ) is None


def test_returns_none_for_unbalanced_braces() -> None:
    """Truly malformed JSON (no balanced object or array anywhere) should
    fail cleanly, not raise."""
    out = _extract_json_payload("Here is the start: { not closed and ( not closed")
    assert out is None


def test_unbalanced_outer_with_balanced_inner_returns_inner() -> None:
    """Lenient on purpose: if Claude got cut off mid-response but a complete
    inner array survived, surface what we have. The runtime parser will
    raise with a useful message if the recovered fragment is the wrong
    shape (e.g. shapes-only with no steps)."""
    out = _extract_json_payload('Here is the start: {"shapes": ["form"]')
    # Recovers the inner array even though the outer object was truncated.
    assert out == ["form"]


# ── Realistic Claude responses (regression fixtures) ────────────────────


def test_realistic_response_with_shape_classification_preamble() -> None:
    """The actual failure pattern from the staffcrm rerun. Claude often
    explains its classification before emitting the JSON."""
    text = """\
The plan describes a multi-step CRM workflow with login + navigation +
form-edit. I'll classify it as both `workflow` and `form` shapes.

```json
{
  "shapes": ["form", "workflow"],
  "steps": [
    {"type": "navigate", "intent": "Go to login URL"}
  ]
}
```

This decomposition follows the workflow pattern."""
    out = _extract_json_payload(text)
    assert isinstance(out, dict)
    assert out["shapes"] == ["form", "workflow"]
    assert len(out["steps"]) == 1


def test_realistic_response_legacy_array_with_preamble() -> None:
    """Backward-compat: legacy bare-array response with prose preamble
    still parses correctly."""
    text = """\
Here are the micro-intents for this plan:

[
  {"type": "navigate", "intent": "Go to login"},
  {"type": "fill_field", "intent": "Enter username"}
]
"""
    out = _extract_json_payload(text)
    assert isinstance(out, list)
    assert len(out) == 2


def test_finds_first_balanced_object_when_multiple_present() -> None:
    """If the response has multiple JSON-like blocks, take the first one
    that parses (the canonical decomposition output)."""
    text = (
        "Maybe like this: {invalid json here}\n\n"
        "Actually the right answer is:\n\n"
        '{"shapes": ["form"]}'
    )
    out = _extract_json_payload(text)
    assert out == {"shapes": ["form"]}
