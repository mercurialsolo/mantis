"""Decomposer emits inline `extract` blocks from intent prose (#785 follow-up).

The Claude-side prompt change (live trip to the API) is not unit-tested
here — those land in `tests/test_plan_decomposer_live.py` if/when they
land. This module pins the *parser* contract: when Claude returns an
extract block in the step dict, the parser carries it through to
MicroIntent, defensively normalized.
"""

from __future__ import annotations


from mantis_agent.plan_decomposer import PlanDecomposer


# ── _coerce_extract_block ────────────────────────────────────────────


def test_coerce_passes_through_canonical_shape():
    raw = {
        "schema_name": "hn_top5",
        "entity_name": "hn_story",
        "fields": [
            {"name": "rank", "type": "int", "required": True},
            {"name": "title", "type": "str", "required": True},
            {"name": "points", "type": "int", "required": False},
        ],
        "max_items": 5,
    }
    out = PlanDecomposer._coerce_extract_block(raw)
    assert out["schema_name"] == "hn_top5"
    assert out["entity_name"] == "hn_story"
    assert out["max_items"] == 5
    assert len(out["fields"]) == 3
    assert out["fields"][0] == {"name": "rank", "type": "int", "required": True}


def test_coerce_returns_empty_for_non_dict():
    assert PlanDecomposer._coerce_extract_block(None) == {}
    assert PlanDecomposer._coerce_extract_block("title,points") == {}
    assert PlanDecomposer._coerce_extract_block(42) == {}


def test_coerce_returns_empty_when_fields_missing_or_not_list():
    assert PlanDecomposer._coerce_extract_block({"schema_name": "x"}) == {}
    assert PlanDecomposer._coerce_extract_block({"fields": "title,price"}) == {}
    assert PlanDecomposer._coerce_extract_block({"fields": []}) == {}


def test_coerce_drops_fields_missing_name():
    raw = {
        "fields": [
            {"name": "title", "type": "str", "required": True},
            {"type": "int"},  # no name — dropped
            {"name": "", "type": "str"},  # empty name — dropped
            {"name": "points", "type": "int"},
        ],
    }
    out = PlanDecomposer._coerce_extract_block(raw)
    names = [f["name"] for f in out["fields"]]
    assert names == ["title", "points"]


def test_coerce_defaults_required_true_when_omitted():
    """ExtractionSchema.from_dict treats missing `required` as True;
    the coercion mirrors that so the runner contract is consistent."""
    raw = {"fields": [{"name": "title", "type": "str"}]}
    out = PlanDecomposer._coerce_extract_block(raw)
    assert out["fields"][0]["required"] is True


def test_coerce_defaults_type_to_str_when_omitted():
    raw = {"fields": [{"name": "title"}]}
    out = PlanDecomposer._coerce_extract_block(raw)
    assert out["fields"][0]["type"] == "str"


def test_coerce_drops_invalid_max_items():
    raw = {
        "fields": [{"name": "title", "type": "str"}],
        "max_items": 0,  # invalid (must be > 0)
    }
    out = PlanDecomposer._coerce_extract_block(raw)
    assert "max_items" not in out


def test_coerce_drops_unknown_top_level_keys():
    raw = {
        "schema_name": "x",
        "fields": [{"name": "title", "type": "str"}],
        "side_dish": "pasta",  # not in the canonical shape
    }
    out = PlanDecomposer._coerce_extract_block(raw)
    assert "side_dish" not in out


def test_coerce_returns_empty_when_all_fields_get_dropped():
    raw = {"fields": [{"type": "str"}, {"name": "", "type": "int"}]}
    assert PlanDecomposer._coerce_extract_block(raw) == {}


# ── parser end-to-end ────────────────────────────────────────────────


def test_build_intent_carries_extract_block_to_micro_intent():
    """Simulates what Claude's decomposer response looks like for an
    extraction step that came from prose like "Extract title, points
    of each story". The parser should build a MicroIntent whose
    `extract` field carries the schema."""
    raw_step = {
        "intent": "Extract title and points for each story",
        "type": "extract_data",
        "claude_only": True,
        "extract": {
            "schema_name": "stories",
            "entity_name": "story",
            "fields": [
                {"name": "title", "type": "str", "required": True},
                {"name": "points", "type": "int", "required": False},
            ],
            "max_items": 5,
        },
    }
    step = PlanDecomposer._build_intent(raw_step)
    assert step.extract["schema_name"] == "stories"
    assert len(step.extract["fields"]) == 2
    assert step.extract["fields"][0]["required"] is True


def test_build_intent_legacy_step_has_empty_extract():
    """Steps without an `extract` block round-trip through the parser
    with the default empty dict — preserves recipe-bound behavior for
    boattrader and friends."""
    raw_step = {
        "intent": "Extract listing details",
        "type": "extract_data",
        "claude_only": True,
        # no `extract` block
    }
    step = PlanDecomposer._build_intent(raw_step)
    assert step.extract == {}


def test_build_intent_malformed_extract_falls_back_to_empty():
    """If Claude emits a partially-shaped extract block (e.g. fields
    not a list), the parser drops it. Better than carrying a broken
    schema downstream — the validator will surface no_schema_configured
    with the runtime warning instead."""
    raw_step = {
        "intent": "Extract things",
        "type": "extract_data",
        "claude_only": True,
        "extract": {"fields": "title,points"},  # malformed — not a list
    }
    step = PlanDecomposer._build_intent(raw_step)
    assert step.extract == {}


def test_build_intent_extract_with_only_one_required_field():
    """Common case: prose mentions just one identifier ('the title of
    each item'). The decomposer should emit a single required field
    and the parser preserves it."""
    raw_step = {
        "intent": "Extract just the title of each item",
        "type": "extract_data",
        "claude_only": True,
        "extract": {
            "fields": [{"name": "title", "type": "str", "required": True}],
        },
    }
    step = PlanDecomposer._build_intent(raw_step)
    assert step.extract["fields"] == [{"name": "title", "type": "str", "required": True}]


def test_build_intent_does_not_add_extract_to_navigate_step():
    """Sanity: an `extract` block on a navigate step is a no-op
    (handler only consumes it on extract_data / extract_url). The
    parser still carries it (one place for the field), but the
    runtime behavior is unchanged."""
    raw_step = {
        "intent": "Navigate to HN",
        "type": "navigate",
        "params": {"url": "https://news.ycombinator.com/"},
        "extract": {
            "fields": [{"name": "title", "type": "str", "required": True}],
        },
    }
    step = PlanDecomposer._build_intent(raw_step)
    # Field is carried (the parser is permissive), but the click_step
    # handler is the only consumer — navigate ignores it.
    assert step.extract["fields"][0]["name"] == "title"


# ── prompt version invalidation ─────────────────────────────────────


def test_prompt_version_bump_invalidates_cached_extract_decomposition():
    """When the prompt grows new behavior (here: emitting `extract`
    blocks), the cached decompositions must be re-derived. The
    prompt_version literal in decompose_text gates this — bumping
    it changes the cache key.

    This test pins that the bump happened. If you reset the version
    back to v35 without removing the extract-block instructions, the
    cache wouldn't invalidate and old plans would silently miss
    extract blocks for one run.
    """
    import inspect

    src = inspect.getsource(PlanDecomposer.decompose_text)
    # The literal lives in decompose_text. v36+ means the change landed.
    assert 'prompt_version = "v36' in src or 'prompt_version = "v37' in src or 'prompt_version = "v38' in src
