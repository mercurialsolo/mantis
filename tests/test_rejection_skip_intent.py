"""Tests for issue #246 — recipe-rejection skip-semantic envelope.

Ahead of this fix, ``ClaudeStepHandler`` returned a generic
``StepResult(success=False, data="REJECTED_DEALER|...")`` whenever
the extractor flagged a row as a dealer / spam / non-private
listing. Hosts that orchestrate the agent over a tool surface
(staffai's hybrid backend) saw an opaque step failure and treated
it as *"extraction is broken, try a different angle"* — the agent
then re-navigated, scrolled more, and asked Claude to read other
sections, none of which can succeed because the page genuinely is
a dealer listing and the recipe correctly rejects it. Net effect:
the agent stuck looping on a single dealer-flagged URL, never
moving on to the next listing.

Recipes know which rejections are *terminal for this row* (dealer,
spam — host should advance) versus *retryable / informational*
(incomplete required fields — host could try the detail page).
This module pins the contract that propagates that knowledge:

1. ``ExtractionSchema.rejection_intents`` — recipe-author dict that
   maps a rejection-reason key to a host-facing intent string.
2. ``StepResult.skip`` / ``StepResult.skip_reason`` — runner-side
   fields populated when the recipe says ``intent == 'skip'``.
3. ``ClaudeStepHandler`` extract_data branch consults the schema
   on every rejection and sets ``skip`` accordingly.
4. ``marketplace_listings`` recipe carries ``{'dealer': 'skip', ...}``
   so its existing dealer rejections become host-actionable.
"""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock

from mantis_agent.extraction import ExtractionSchema
from mantis_agent.extraction.result import ExtractionResult
from mantis_agent.gym.checkpoint import StepResult


# ── StepResult: new fields, defaults, persistence ────────────────


def test_step_result_has_skip_and_skip_reason_fields() -> None:
    names = {f.name for f in fields(StepResult)}
    assert "skip" in names
    assert "skip_reason" in names


def test_step_result_skip_defaults_to_false() -> None:
    """Default StepResult must have ``skip=False`` and
    ``skip_reason=None`` so legacy callers (every existing test +
    every existing handler) see no change."""
    r = StepResult(step_index=0, intent="x", success=True)
    assert r.skip is False
    assert r.skip_reason is None


def test_step_result_skip_round_trips_via_to_dict_from_dict() -> None:
    """The skip envelope persists through the checkpoint JSON the
    same way ``data`` / ``steps_used`` do — otherwise resume from
    a checkpoint loses the host-actionable signal."""
    r = StepResult(
        step_index=3, intent="extract row", success=False,
        data="REJECTED_DEALER|extractor marked as dealer|...",
        skip=True, skip_reason="dealer",
    )
    payload = r.to_dict()
    assert payload["skip"] is True
    assert payload["skip_reason"] == "dealer"
    rt = StepResult.from_dict(payload)
    assert rt.skip is True
    assert rt.skip_reason == "dealer"


def test_step_result_persisted_tuple_includes_skip_fields() -> None:
    """``_PERSISTED`` is the explicit allowlist of checkpoint-
    serialised fields. New fields must be added here or they
    silently drop on save."""
    persisted = set(StepResult._PERSISTED)
    assert "skip" in persisted
    assert "skip_reason" in persisted


# ── ExtractionSchema: rejection_intents ──────────────────────────


def test_extraction_schema_has_rejection_intents_field() -> None:
    s = ExtractionSchema()
    assert hasattr(s, "rejection_intents")
    assert s.rejection_intents == {}


def test_extraction_schema_rejection_intents_is_per_instance() -> None:
    """Default factory: each instance gets its own dict, mutating
    one must not leak into a sibling. Default-factory regression
    guard for the dataclass."""
    a = ExtractionSchema()
    b = ExtractionSchema()
    a.rejection_intents["dealer"] = "skip"
    assert "dealer" not in b.rejection_intents


def test_extraction_schema_overlay_merges_rejection_intents() -> None:
    """``overlay()`` is the derive-first composition path
    (issue #224). When a derived schema (built from the plan
    text via ``from_objective``) is overlaid with a recipe, the
    recipe's empirical knowledge of which rejections mean *skip*
    must reach the runner. Recipe extends derived; derived keys
    win on conflict (operator-authored override)."""
    derived = ExtractionSchema(rejection_intents={"dealer": "extract_more"})
    recipe = ExtractionSchema(rejection_intents={
        "dealer": "skip",          # would lose to derived
        "spam": "skip",            # added by recipe
        "parse_error": "retry",    # added by recipe
    })
    merged = derived.overlay(recipe)
    assert merged.rejection_intents["dealer"] == "extract_more"  # derived wins
    assert merged.rejection_intents["spam"] == "skip"
    assert merged.rejection_intents["parse_error"] == "retry"


def test_extraction_schema_overlay_with_empty_recipe_intents_keeps_derived() -> None:
    derived = ExtractionSchema(rejection_intents={"dealer": "skip"})
    recipe = ExtractionSchema()  # no rejection_intents
    merged = derived.overlay(recipe)
    assert merged.rejection_intents == {"dealer": "skip"}


# ── marketplace_listings recipe: explicit dealer intent ──────────


def test_marketplace_listings_recipe_marks_dealer_as_skip() -> None:
    """Issue #246's reproducer: BoatTrader detail pages flag as
    dealer, the runner returned generic step failure, the
    orchestrator looped. The recipe must annotate dealer rejections
    as host-actionable skips."""
    from mantis_agent.recipes.marketplace_listings.schema import SCHEMA
    assert SCHEMA.rejection_intents.get("dealer") == "skip"


def test_marketplace_listings_recipe_distinguishes_incomplete_from_dealer() -> None:
    """Incomplete-required rejections are different — the row may
    enrich on the detail page. The host should NOT skip the row;
    it should drive a follow-up read. Recipe authors annotate this
    explicitly."""
    from mantis_agent.recipes.marketplace_listings.schema import SCHEMA
    assert SCHEMA.rejection_intents.get("incomplete_required") != "skip"


# ── ClaudeStepHandler integration: skip propagation ──────────────


def _build_step_handler_ctx(*, recipe_schema: ExtractionSchema, dealer: bool, missing: bool):
    """Construct the minimal MicroPlanRunner / StepContext / mock
    extractor needed for the extract_data branch. Returns
    ``(handler, step, ctx)``."""
    from mantis_agent.gym.step_context import StepContext
    from mantis_agent.gym.step_handlers.claude_step import ClaudeStepHandler
    from mantis_agent.plan_decomposer import MicroIntent

    runner = MagicMock()
    runner.costs = {"claude_extract": 0, "gpu_steps": 0, "gpu_seconds": 0}
    runner._last_extracted = {}
    runner._current_page = 1
    runner.scanner.is_duplicate.return_value = False

    extractor = MagicMock()
    extractor.schema = recipe_schema

    # ExtractionResult stub with the relevant predicate methods.
    data = MagicMock(spec=ExtractionResult)
    data.url = "https://www.example.com/boat/1234"
    data.dealer_reason.return_value = "extractor marked as dealer" if dealer else ""
    data.missing_required_reason.return_value = (
        "missing required field: year" if missing else ""
    )
    data.is_viable.return_value = not (dealer or missing)
    data.to_summary.return_value = "summary"
    data.raw_response = "raw"
    data.extracted_fields = {}
    data.url = "https://www.example.com/boat/1234"

    handler = ClaudeStepHandler(runner)

    # Patch the deep-extract helper so we don't need a real
    # screenshot pipeline.
    handler._extract_listing_data_deep = lambda shot, ctx: (data, 1)

    env = MagicMock()
    env.screenshot.return_value = MagicMock()

    extraction_cache = MagicMock()
    extraction_cache.read_enabled = False
    extraction_cache.get.return_value = None

    dynamic_verifier = MagicMock()

    ctx = StepContext(
        env=env, brain=None, extractor=extractor, grounding=None,
        cost_meter=None, dynamic_verifier=dynamic_verifier,
        scanner=runner.scanner, site_config=None,
        tool_channel=None, extraction_cache=extraction_cache,
        state={"index": 5},
    )

    step = MicroIntent(
        intent="Read structured fields off this boat detail page",
        type="extract_data",
        claude_only=True,
    )
    return handler, step, ctx


def test_dealer_rejection_with_skip_intent_sets_step_result_skip(monkeypatch) -> None:
    """End-to-end: recipe says ``dealer → skip``, extractor flags a
    row as dealer, handler returns a StepResult with ``skip=True``
    and ``skip_reason='dealer'``."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.claude_step.time.sleep",
        lambda *_: None,
    )
    schema = ExtractionSchema(
        spam_label="dealer",
        rejection_intents={"dealer": "skip"},
    )
    handler, step, ctx = _build_step_handler_ctx(
        recipe_schema=schema, dealer=True, missing=False,
    )
    result = handler.execute(step, ctx)

    assert result.success is False
    assert result.skip is True
    assert result.skip_reason == "dealer"
    assert result.data.startswith("REJECTED_DEALER|")


def test_dealer_rejection_without_recipe_intent_does_not_skip(monkeypatch) -> None:
    """Default empty ``rejection_intents`` preserves today's
    behavior: a dealer rejection looks like a step failure with
    ``skip=False`` so legacy callers see no change. The host opts
    in by setting ``rejection_intents`` on the recipe."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.claude_step.time.sleep",
        lambda *_: None,
    )
    schema = ExtractionSchema(spam_label="dealer")  # no rejection_intents
    handler, step, ctx = _build_step_handler_ctx(
        recipe_schema=schema, dealer=True, missing=False,
    )
    result = handler.execute(step, ctx)

    assert result.success is False
    assert result.skip is False
    assert result.skip_reason is None


def test_incomplete_rejection_with_extract_more_intent_does_not_skip(monkeypatch) -> None:
    """``incomplete_required`` mapped to ``extract_more`` is not a
    skip — the host should follow up with a deeper read, not move
    on. ``StepResult.skip`` stays False."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.claude_step.time.sleep",
        lambda *_: None,
    )
    schema = ExtractionSchema(
        required_fields=["year", "make"],
        rejection_intents={
            "dealer": "skip",
            "incomplete_required": "extract_more",
        },
    )
    handler, step, ctx = _build_step_handler_ctx(
        recipe_schema=schema, dealer=False, missing=True,
    )
    result = handler.execute(step, ctx)

    assert result.success is False
    assert result.skip is False
    assert result.skip_reason is None
    assert result.data.startswith("REJECTED_INCOMPLETE|")


def test_incomplete_rejection_with_skip_intent_does_skip(monkeypatch) -> None:
    """Recipes can choose to treat incomplete-required as
    terminal-skip for verticals where missing required fields mean
    the row is fundamentally unsuitable (no future read can
    enrich it). ``skip`` follows the recipe's annotation."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.claude_step.time.sleep",
        lambda *_: None,
    )
    schema = ExtractionSchema(
        required_fields=["year", "make"],
        rejection_intents={"incomplete_required": "skip"},
    )
    handler, step, ctx = _build_step_handler_ctx(
        recipe_schema=schema, dealer=False, missing=True,
    )
    result = handler.execute(step, ctx)

    assert result.skip is True
    assert result.skip_reason == "incomplete_required"
