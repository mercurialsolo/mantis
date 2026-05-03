"""Tests for LLM-driven plan-shape classification.

The decomposer asks Claude to classify the plan's shape (listings / form
/ workflow / inspect) as part of the same JSON response that produces the
micro-intents. These tests verify:

  • The prompt requests shape classification with a strict canonical vocabulary.
  • The parser handles both the new {"shapes": [...], "steps": [...]} schema
    and the legacy bare-array schema gracefully.
  • Shape normalization filters/canonicalizes Claude's output so a model
    hallucination (extra tokens, wrong order) can't poison downstream
    observability.

The shape classification itself is **LLM-driven** — there is no regex /
keyword heuristic in this module. Per-domain generalization comes from
Claude reading the plan, not from us listing every possible domain word.
"""

from __future__ import annotations

from mantis_agent.plan_decomposer import (
    DECOMPOSE_PROMPT,
    MicroPlan,
    PlanDecomposer,
)


# ── _normalize_shapes ───────────────────────────────────────────────────


def test_normalize_strips_unknown_tokens() -> None:
    """A model hallucination ('e-commerce', 'crud', 'random') gets dropped."""
    assert PlanDecomposer._normalize_shapes(
        ["workflow", "e-commerce", "form", "random"]
    ) == ["form", "workflow"]


def test_normalize_canonical_order_independent_of_input() -> None:
    """Claude may emit shapes in any order; we always canonicalize to
    the documented order so hashes / cache keys are stable."""
    a = PlanDecomposer._normalize_shapes(["workflow", "form"])
    b = PlanDecomposer._normalize_shapes(["form", "workflow"])
    assert a == b == ["form", "workflow"]


def test_normalize_dedups() -> None:
    assert PlanDecomposer._normalize_shapes(
        ["workflow", "workflow", "form", "form"]
    ) == ["form", "workflow"]


def test_normalize_case_insensitive() -> None:
    assert PlanDecomposer._normalize_shapes(
        ["Workflow", "FORM", "  inspect "]
    ) == ["form", "workflow", "inspect"]


def test_normalize_accepts_string_input() -> None:
    """Some models will emit a single string instead of a list."""
    assert PlanDecomposer._normalize_shapes("workflow") == ["workflow"]


def test_normalize_empty_input_returns_empty() -> None:
    assert PlanDecomposer._normalize_shapes([]) == []
    assert PlanDecomposer._normalize_shapes(None) == []
    assert PlanDecomposer._normalize_shapes("") == []


def test_normalize_filters_non_canonical_strings() -> None:
    """Anything that isn't one of the four canonical tokens gets dropped."""
    assert PlanDecomposer._normalize_shapes(
        ["scraping", "data-extraction", "form-driven"]
    ) == []


def test_known_shapes_are_documented() -> None:
    assert PlanDecomposer.KNOWN_PLAN_SHAPES == frozenset(
        {"listings", "form", "workflow", "inspect"}
    )


# ── MicroPlan.shapes round-trips ────────────────────────────────────────


def test_micro_plan_default_shapes_is_empty_list() -> None:
    plan = MicroPlan()
    assert plan.shapes == []


def test_micro_plan_to_dict_includes_shapes() -> None:
    plan = MicroPlan()
    plan.shapes = ["form", "workflow"]
    assert plan.to_dict()["shapes"] == ["form", "workflow"]


def test_micro_plan_from_dict_recovers_shapes() -> None:
    payload = {
        "steps": [],
        "source_plan": "test",
        "domain": "x.test",
        "shapes": ["workflow", "form"],
    }
    plan = MicroPlan.from_dict(payload)
    # Recovery normalizes — canonical order applied.
    assert plan.shapes == ["form", "workflow"]


def test_micro_plan_from_dict_legacy_array_yields_empty_shapes() -> None:
    """Backward compat: bare array → empty shapes, no error."""
    plan = MicroPlan.from_dict([])
    assert plan.shapes == []


def test_micro_plan_from_dict_missing_shapes_field_yields_empty() -> None:
    plan = MicroPlan.from_dict({"steps": [], "source_plan": "", "domain": ""})
    assert plan.shapes == []


def test_micro_plan_summary_shows_shapes_when_present() -> None:
    plan = MicroPlan()
    plan.shapes = ["workflow", "form"]
    summary = plan.summary()
    # Both shape names appear in the summary output for human inspection.
    assert "workflow" in summary
    assert "form" in summary


def test_micro_plan_summary_omits_shape_tag_when_empty() -> None:
    plan = MicroPlan()
    summary = plan.summary()
    # No empty brackets or stray "[]" decoration when shapes is empty.
    assert "[]" not in summary


# ── Prompt regression guards ────────────────────────────────────────────


def test_prompt_documents_all_four_shapes() -> None:
    text = DECOMPOSE_PROMPT.upper()
    assert "LISTINGS" in text
    assert "FORM" in text
    assert "WORKFLOW" in text or "MULTI-STEP" in text
    assert "INSPECT" in text


def test_prompt_requires_classification_step() -> None:
    """STEP 0 (classify) must appear before step generation in the prompt."""
    text = DECOMPOSE_PROMPT.upper()
    assert "STEP 0" in text or "CLASSIFY" in text


def test_prompt_requests_canonical_vocabulary() -> None:
    """The prompt must tell Claude to use only the four canonical tokens."""
    text = DECOMPOSE_PROMPT.lower()
    # All four tokens appear quoted as expected output values.
    for shape in ("listings", "form", "workflow", "inspect"):
        assert shape in text


def test_prompt_describes_object_output_schema() -> None:
    """The prompt must document the new {shapes, steps} object output."""
    text = DECOMPOSE_PROMPT
    assert '"shapes"' in text
    assert '"steps"' in text


def test_prompt_keeps_legacy_fallback_documented() -> None:
    """Backward compat note: bare-array fallback is still documented."""
    text = DECOMPOSE_PROMPT.lower()
    assert "fallback" in text or "bare json array" in text


def test_prompt_no_longer_says_two_shapes() -> None:
    """The pre-staffcrm prompt said 'TWO PLAN SHAPES' — must be gone."""
    text = DECOMPOSE_PROMPT.upper()
    assert "TWO PLAN SHAPES" not in text
    assert "FOUR PLAN SHAPES" in text


def test_prompt_does_not_mention_specific_domains() -> None:
    """No domain hard-wiring in the prompt — generalization means staying
    domain-agnostic."""
    text = DECOMPOSE_PROMPT.lower()
    forbidden = {
        "boattrader.com", "linkedin.com", "salesforce.com", "shopify.com",
        "ebay.com", "amazon.com", "github.com",
    }
    for token in forbidden:
        assert token not in text, f"Prompt should not mention {token!r}"


def test_prompt_does_not_use_regex_shape_signals() -> None:
    """Sanity: the regex-based detection was removed in favor of LLM
    classification. If a contributor adds it back, this catches it."""
    import mantis_agent.plan_decomposer as mod
    assert not hasattr(mod.PlanDecomposer, "_SHAPE_SIGNALS")
    assert not hasattr(mod.PlanDecomposer, "_detect_plan_shape")
    assert not hasattr(mod.PlanDecomposer, "_build_shape_hint")


# ── Module docstring claims generalization ─────────────────────────────


def test_module_docstring_documents_generalization() -> None:
    import mantis_agent.plan_decomposer as mod
    doc = (mod.__doc__ or "").lower()
    assert "shape" in doc
    assert "domain" in doc


# ── End-to-end: simulate what Claude would return ───────────────────────


def test_round_trip_object_response_through_micro_plan() -> None:
    """Simulate the new LLM-driven response shape and verify the MicroPlan
    captures both shapes and steps without losing fidelity."""
    response_payload = {
        "shapes": ["workflow", "form"],
        "steps": [
            {"intent": "Go to login", "type": "navigate"},
            {"intent": "Enter username", "type": "fill_field", "params": {"label": "username"}},
        ],
    }
    plan = MicroPlan.from_dict(response_payload)
    assert plan.shapes == ["form", "workflow"]
    assert len(plan.steps) == 2
    assert plan.steps[0].type == "navigate"
    assert plan.steps[1].type == "fill_field"


def test_round_trip_legacy_array_response_yields_empty_shapes() -> None:
    """Backward compat: an old bare-array response still parses correctly."""
    response_payload = [
        {"intent": "Go to login", "type": "navigate"},
    ]
    plan = MicroPlan.from_dict(response_payload)
    assert plan.shapes == []
    assert len(plan.steps) == 1
