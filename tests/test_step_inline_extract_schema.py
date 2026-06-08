"""Plan-author inline extraction schema (#785 follow-up).

Verifies that a `step.extract` block on a `claude_only` step (declared
inline in the plan) is consumed at runtime: the validator enforces the
plan's `required_fields` instead of falling through to the recipe
schema or to `no_schema_configured`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from mantis_agent.extraction import ExtractionResult, ExtractionSchema
from mantis_agent.plan_decomposer import MicroIntent


# ── Wire-level: MicroIntent carries `extract` ────────────────────


def test_micro_intent_default_extract_is_empty_dict():
    step = MicroIntent(intent="extract", type="extract_data")
    assert step.extract == {}


def test_micro_intent_can_carry_extract_block():
    extract = {
        "schema_name": "hn_top5",
        "fields": [
            {"name": "rank", "type": "int", "required": True},
            {"name": "title", "type": "str", "required": True},
        ],
        "max_items": 5,
    }
    step = MicroIntent(
        intent="extract top 5 stories",
        type="extract_data",
        claude_only=True,
        extract=extract,
    )
    assert step.extract == extract


def test_micro_plan_steps_to_dicts_carries_extract():
    from mantis_agent.server_utils import micro_plan_steps_to_dicts

    step = MicroIntent(
        intent="x",
        type="extract_data",
        claude_only=True,
        extract={
            "fields": [{"name": "title", "type": "str", "required": True}],
        },
    )
    dicts = micro_plan_steps_to_dicts([step])
    assert len(dicts) == 1
    assert dicts[0]["extract"] == {
        "fields": [{"name": "title", "type": "str", "required": True}],
    }


def test_micro_plan_steps_to_dicts_omits_unset_extract_as_empty():
    """Legacy steps without an extract block round-trip as {} — empty
    dict means "fall back to recipe schema," not None."""
    from mantis_agent.server_utils import micro_plan_steps_to_dicts

    step = MicroIntent(intent="x", type="navigate")
    dicts = micro_plan_steps_to_dicts([step])
    assert dicts[0]["extract"] == {}


# ── ExtractionSchema.from_dict round-trip ───────────────────────


def test_extraction_schema_from_step_extract_dict():
    """The runtime path: claude_step builds a transient schema via
    ExtractionSchema.from_dict(step.extract). Verify the schema's
    required_fields default from each field's `required` flag.
    """
    schema = ExtractionSchema.from_dict(
        {
            "schema_name": "hn_top5",
            "fields": [
                {"name": "rank", "type": "int", "required": True},
                {"name": "title", "type": "str", "required": True},
                {"name": "story_url", "type": "str", "required": False},
                {"name": "points", "type": "int", "required": False},
            ],
            "max_items": 5,
        }
    )
    assert sorted(schema.required_fields) == ["rank", "title"]
    assert len(schema.fields) == 4


def test_result_with_inline_schema_enforces_plan_required_fields():
    """End-to-end: result with the plan's inline schema rejects when
    the plan's required fields are missing — not boattrader's year/make.
    """
    schema = ExtractionSchema.from_dict(
        {
            "fields": [
                {"name": "title", "type": "str", "required": True},
                {"name": "story_url", "type": "str", "required": False},
            ],
        }
    )
    # No `title` extracted → schema's required_field is missing.
    result = ExtractionResult(
        _schema=schema,
        extracted_fields={"story_url": "https://example.com/a"},
    )
    reason = result.missing_required_reason()
    assert "title" in reason
    assert "year" not in reason  # the legacy boattrader fallback is gone
    assert "make" not in reason


def test_result_with_inline_schema_viable_when_required_fields_present():
    schema = ExtractionSchema.from_dict(
        {
            "fields": [{"name": "title", "type": "str", "required": True}],
        }
    )
    result = ExtractionResult(
        _schema=schema,
        extracted_fields={"title": "Show HN: Hacker News scraper that works"},
    )
    # Note: is_viable() also calls is_private_seller(), which is True
    # by default for results without a `seller` field. Plans that need
    # other viability semantics can extend the schema.
    assert result.missing_required_reason() == ""


# ── claude_step schema swap (handler-level) ────────────────────


def _fake_step_context(extractor: Any) -> Any:
    """Build a minimal StepContext stand-in for handler unit tests."""
    ctx = MagicMock()
    ctx.extractor = extractor
    ctx.env = MagicMock()
    ctx.dynamic_verifier = MagicMock()
    ctx.extraction_cache = MagicMock()
    ctx.state = {"index": 0}
    return ctx


def test_handler_swaps_schema_when_step_has_extract():
    """ClaudeStepHandler.execute() should override extractor.schema
    with the inline schema for the duration of the step, then restore.
    """
    from mantis_agent.gym.step_handlers.claude_step import ClaudeStepHandler

    original_schema = ExtractionSchema(
        entity_name="boat", fields=[], required_fields=["year", "make"]
    )
    extractor = MagicMock()
    extractor.schema = original_schema

    swapped_schemas: list[Any] = []

    def _capture_execute(*_a, **_kw):
        # During the wrapped body, the extractor should carry the
        # plan-declared schema.
        swapped_schemas.append(extractor.schema)
        from mantis_agent.gym.checkpoint import StepResult

        return StepResult(step_index=0, intent="x", success=True)

    runner = MagicMock()
    runner._last_extracted = {}
    handler = ClaudeStepHandler(runner)
    handler._execute = _capture_execute  # type: ignore[assignment]

    step = MicroIntent(
        intent="extract HN top 5",
        type="extract_data",
        claude_only=True,
        extract={
            "schema_name": "hn",
            "fields": [{"name": "title", "type": "str", "required": True}],
        },
    )
    ctx = _fake_step_context(extractor)
    handler.execute(step, ctx)

    # During the call: schema was swapped.
    assert len(swapped_schemas) == 1
    swapped = swapped_schemas[0]
    assert swapped is not original_schema
    assert swapped.required_fields == ["title"]

    # After the call: schema is restored.
    assert extractor.schema is original_schema


def test_handler_does_not_swap_when_step_has_no_extract():
    from mantis_agent.gym.step_handlers.claude_step import ClaudeStepHandler

    original_schema = ExtractionSchema(
        entity_name="boat", fields=[], required_fields=["year", "make"]
    )
    extractor = MagicMock()
    extractor.schema = original_schema

    saw_schemas: list[Any] = []

    def _capture_execute(*_a, **_kw):
        saw_schemas.append(extractor.schema)
        from mantis_agent.gym.checkpoint import StepResult

        return StepResult(step_index=0, intent="x", success=True)

    runner = MagicMock()
    handler = ClaudeStepHandler(runner)
    handler._execute = _capture_execute  # type: ignore[assignment]

    step = MicroIntent(intent="extract", type="extract_data")  # no .extract
    handler.execute(step, _fake_step_context(extractor))

    # Schema was NOT touched.
    assert saw_schemas[0] is original_schema
    assert extractor.schema is original_schema


def test_handler_restores_schema_when_execute_raises():
    """If the wrapped extract path raises, the schema swap is still
    restored — try/finally semantics.
    """
    from mantis_agent.gym.step_handlers.claude_step import ClaudeStepHandler

    original_schema = ExtractionSchema(
        entity_name="boat", fields=[], required_fields=["year", "make"]
    )
    extractor = MagicMock()
    extractor.schema = original_schema

    def _boom(*_a, **_kw):
        raise RuntimeError("kaboom")

    runner = MagicMock()
    handler = ClaudeStepHandler(runner)
    handler._execute = _boom  # type: ignore[assignment]

    step = MicroIntent(
        intent="x",
        type="extract_data",
        claude_only=True,
        extract={"fields": [{"name": "title", "type": "str", "required": True}]},
    )
    try:
        handler.execute(step, _fake_step_context(extractor))
    except RuntimeError:
        pass

    # Schema is restored despite the exception.
    assert extractor.schema is original_schema


def test_handler_falls_back_to_recipe_on_malformed_extract():
    """Malformed step.extract (missing fields) should log a warning
    and fall back to the recipe schema — not break the run.
    """
    from mantis_agent.gym.step_handlers.claude_step import ClaudeStepHandler

    original_schema = ExtractionSchema(
        entity_name="boat", fields=[], required_fields=["year", "make"]
    )
    extractor = MagicMock()
    extractor.schema = original_schema

    saw: list[Any] = []

    def _capture_execute(*_a, **_kw):
        saw.append(extractor.schema)
        from mantis_agent.gym.checkpoint import StepResult

        return StepResult(step_index=0, intent="x", success=True)

    runner = MagicMock()
    handler = ClaudeStepHandler(runner)
    handler._execute = _capture_execute  # type: ignore[assignment]

    step = MicroIntent(
        intent="x",
        type="extract_data",
        claude_only=True,
        # Missing the `fields` key — ExtractionSchema.from_dict raises.
        # Handler should swallow and proceed with recipe schema.
        extract={"schema_name": "broken"},
    )
    handler.execute(step, _fake_step_context(extractor))
    # Recipe schema preserved (no swap happened).
    assert saw[0] is original_schema
