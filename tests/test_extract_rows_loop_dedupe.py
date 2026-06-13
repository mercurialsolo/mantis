"""Loop-extract-dedupe for multi-row extraction.

Pre-fix, the multi-row branch did a single extract pass — got back
whatever Claude saw on the initial viewport, then returned. For
listings pages where the first viewport only shows N/M cards (YC
grid, infinite-scroll feeds), the run silently capped at N. The
loop bridges the gap by alternating extract + CDP-scroll + dedupe.

Coverage:
- happy path: 2 passes fill ``max_items`` (10) from two viewport
  screenshots that overlap on some cards
- dedup by the first ``required=True`` field (primary key)
- stop after 2 consecutive empty passes (page exhausted)
- stop when CDP scroll is unavailable (env can't scroll)
- max_items reached before scroll cap exhausts
- no rows ever returned → ``no_visible_rows`` failure
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from mantis_agent.extraction import ExtractionSchema
from mantis_agent.gym.step_handlers.claude_step import (
    _INNER_SCROLLER_JS,
    ClaudeStepHandler,
    _cdp_scroll_once,
    _filter_new_rows,
    _pick_dedup_key,
)
from mantis_agent.plan_decomposer import MicroIntent


def _ctx(extractor: Any, env_with_cdp: bool = True) -> Any:
    ctx = MagicMock()
    ctx.extractor = extractor
    env = MagicMock()
    env.screenshot.return_value = MagicMock()
    if env_with_cdp:
        env.cdp_evaluate = MagicMock(return_value=None)
    else:
        # Simulate an env where cdp_evaluate isn't available
        # (so the loop terminates cleanly on the first scroll attempt).
        delattr(env, "cdp_evaluate")
    ctx.env = env
    ctx.state = {"index": 0}
    return ctx


def _runner_with_costs() -> Any:
    runner = MagicMock()
    runner.costs = {}
    return runner


# ── helpers ────────────────────────────────────────────────────────────


def test_pick_dedup_key_prefers_first_required_field() -> None:
    schema = ExtractionSchema(
        entity_name="yc_company", fields=[],
        required_fields=["name", "rank"],
    )
    assert _pick_dedup_key(schema) == "name"


def test_pick_dedup_key_falls_back_to_first_field_when_no_required() -> None:
    schema = ExtractionSchema(
        entity_name="x",
        fields=[{"name": "foo"}, {"name": "bar"}],
        required_fields=[],
    )
    assert _pick_dedup_key(schema) == "foo"


def test_filter_new_rows_dedupes_case_insensitively() -> None:
    seen: set = set()
    first = _filter_new_rows(
        [{"name": "Cardinal"}, {"name": "Doomersion"}],
        seen, "name",
    )
    assert len(first) == 2
    second = _filter_new_rows(
        [{"name": "cardinal"}, {"name": "Sequence Markets"}],
        seen, "name",
    )
    # "cardinal" already seen (case-insensitive); Sequence Markets is new.
    assert len(second) == 1
    assert second[0]["name"] == "Sequence Markets"


def test_filter_new_rows_drops_empty_keys() -> None:
    seen: set = set()
    fresh = _filter_new_rows(
        [{"name": ""}, {"name": "Cardinal"}, {"name": "  "}],
        seen, "name",
    )
    assert [r["name"] for r in fresh] == ["Cardinal"]


# ── loop end-to-end ────────────────────────────────────────────────────


def test_loop_two_passes_dedup_and_combine() -> None:
    """First pass yields 5 cards, scroll, second pass yields 5 cards
    (3 fresh, 2 dupes). Final row count = 8, dedup honored."""
    schema = ExtractionSchema(
        entity_name="yc_company",
        fields=[{"name": "name", "type": "str", "required": True}],
        required_fields=["name"],
        max_items=10,
    )
    extractor = MagicMock()
    extractor.schema = schema
    pass_returns = [
        [{"rank": "1", "name": "Cardinal"},
         {"rank": "2", "name": "Doomersion"},
         {"rank": "3", "name": "Sequence"},
         {"rank": "4", "name": "Ditto"},
         {"rank": "5", "name": "Servo7"}],
        [{"rank": "4", "name": "Ditto"},  # duplicate
         {"rank": "5", "name": "Servo7"},  # duplicate
         {"rank": "6", "name": "Voxel"},
         {"rank": "7", "name": "Aurorin"},
         {"rank": "8", "name": "Fixture"}],
        [{"rank": "8", "name": "Fixture"},  # duplicate
         {"rank": "9", "name": "Unifold"},
         {"rank": "10", "name": "Carrot Labs"}],
    ]
    extractor.extract_rows.side_effect = pass_returns
    runner = _runner_with_costs()
    handler = ClaudeStepHandler(runner)

    step = MicroIntent(
        intent="x", type="extract_data", claude_only=True,
        extract={"fields": [{"name": "name", "type": "str", "required": True}], "max_items": 10},
    )
    ctx = _ctx(extractor, env_with_cdp=True)

    result = handler._execute_rows(step, ctx, schema)
    assert result.success is True
    names = [r["name"] for r in result.extracted_rows]
    assert names == [
        "Cardinal", "Doomersion", "Sequence", "Ditto", "Servo7",
        "Voxel", "Aurorin", "Fixture", "Unifold", "Carrot Labs",
    ]
    # 3 extract calls — max_items (10) hit on the 3rd pass.
    assert runner.costs["claude_extract"] == 3


def test_loop_stops_after_two_consecutive_empty_passes() -> None:
    schema = ExtractionSchema(
        entity_name="yc", fields=[], required_fields=["name"],
        max_items=10,
    )
    extractor = MagicMock()
    extractor.schema = schema
    pass_returns = [
        [{"name": "A"}, {"name": "B"}, {"name": "C"}],
        [{"name": "A"}, {"name": "B"}, {"name": "C"}],  # all dupes
        [{"name": "A"}, {"name": "B"}, {"name": "C"}],  # all dupes again
        [{"name": "Z"}],  # would be fresh, but loop stops before getting here
    ]
    extractor.extract_rows.side_effect = pass_returns
    runner = _runner_with_costs()
    handler = ClaudeStepHandler(runner)
    step = MicroIntent(intent="x", type="extract_data", claude_only=True)
    ctx = _ctx(extractor, env_with_cdp=True)
    result = handler._execute_rows(step, ctx, schema)
    assert result.success is True
    # 3 passes ran (initial + 2 empty), then loop bailed.
    assert runner.costs["claude_extract"] == 3
    assert len(result.extracted_rows) == 3


def test_loop_stops_when_cdp_scroll_unavailable() -> None:
    schema = ExtractionSchema(
        entity_name="yc", fields=[], required_fields=["name"],
        max_items=10,
    )
    extractor = MagicMock()
    extractor.schema = schema
    extractor.extract_rows.return_value = [{"name": "Hero"}]
    runner = _runner_with_costs()
    handler = ClaudeStepHandler(runner)
    step = MicroIntent(intent="x", type="extract_data", claude_only=True)
    ctx = _ctx(extractor, env_with_cdp=False)
    result = handler._execute_rows(step, ctx, schema)
    assert result.success is True
    assert [r["name"] for r in result.extracted_rows] == ["Hero"]
    # Only one extract call — the CDP scroll bailed, loop ended.
    assert runner.costs["claude_extract"] == 1


def test_loop_stops_when_max_items_reached() -> None:
    schema = ExtractionSchema(
        entity_name="yc", fields=[], required_fields=["name"],
        max_items=3,
    )
    extractor = MagicMock()
    extractor.schema = schema
    extractor.extract_rows.return_value = [
        {"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"},
    ]
    runner = _runner_with_costs()
    handler = ClaudeStepHandler(runner)
    step = MicroIntent(intent="x", type="extract_data", claude_only=True)
    ctx = _ctx(extractor, env_with_cdp=True)
    result = handler._execute_rows(step, ctx, schema)
    assert result.success is True
    # Capped at 3 even though Claude returned 4.
    assert [r["name"] for r in result.extracted_rows] == ["A", "B", "C"]
    # Single pass — max_items hit after the first extract.
    assert runner.costs["claude_extract"] == 1


def test_loop_zero_rows_returns_no_visible_rows_failure() -> None:
    schema = ExtractionSchema(
        entity_name="yc", fields=[], required_fields=["name"],
        max_items=10,
    )
    extractor = MagicMock()
    extractor.schema = schema
    extractor.extract_rows.return_value = []
    runner = _runner_with_costs()
    handler = ClaudeStepHandler(runner)
    step = MicroIntent(intent="x", type="extract_data", claude_only=True)
    ctx = _ctx(extractor, env_with_cdp=False)
    result = handler._execute_rows(step, ctx, schema)
    assert result.success is False
    assert "no_visible_rows" in result.data


# ── #880: inner-overflow-container scroll ──────────────────────────────


def test_cdp_scroll_js_drives_inner_overflow_container() -> None:
    """The scroll payload must fall back to an inner scroll container
    (YC virtualized directory) when the window scroller is pinned —
    the pre-#880 ``window.scrollBy``-only payload was a no-op there."""
    # window scroller
    assert "window.scrollBy" in _INNER_SCROLLER_JS
    assert "document.scrollingElement" in _INNER_SCROLLER_JS
    # inner overflow scroller fallback
    assert "overflowY" in _INNER_SCROLLER_JS
    assert "scrollTop" in _INNER_SCROLLER_JS
    # virtualized lists re-render off a scroll event
    assert "new Event('scroll'" in _INNER_SCROLLER_JS


def test_cdp_scroll_once_returns_false_on_explicit_no_movement() -> None:
    """A real env returns the JS bool: False ⇒ nothing scrolled (page
    at the bottom / nothing scrollable) ⇒ stop the loop now."""
    env = MagicMock()
    env.cdp_evaluate = MagicMock(return_value=False)
    assert _cdp_scroll_once(env) is False
    env.cdp_evaluate.assert_called_once_with(_INNER_SCROLLER_JS)


def test_cdp_scroll_once_returns_true_on_movement() -> None:
    env = MagicMock()
    env.cdp_evaluate = MagicMock(return_value=True)
    assert _cdp_scroll_once(env) is True


def test_cdp_scroll_once_back_compat_none_keeps_looping() -> None:
    """Test/replay envs (and older CDP shims) return None — preserve the
    legacy 'scroll issued' contract so they keep looping to the empty-
    pass guard rather than stopping after one pass."""
    env = MagicMock()
    env.cdp_evaluate = MagicMock(return_value=None)
    assert _cdp_scroll_once(env) is True


def test_cdp_scroll_once_false_when_no_cdp() -> None:
    env = MagicMock()
    delattr(env, "cdp_evaluate")
    assert _cdp_scroll_once(env) is False


def test_loop_stops_when_scroll_reports_no_movement() -> None:
    """New #880 path: when the scroller is genuinely exhausted (JS
    returns False), the loop stops after the current pass instead of
    burning two empty extract passes."""
    schema = ExtractionSchema(
        entity_name="yc", fields=[], required_fields=["name"],
        max_items=10,
    )
    extractor = MagicMock()
    extractor.schema = schema
    extractor.extract_rows.return_value = [{"name": "A"}, {"name": "B"}]
    runner = _runner_with_costs()
    handler = ClaudeStepHandler(runner)
    step = MicroIntent(intent="x", type="extract_data", claude_only=True)
    ctx = _ctx(extractor, env_with_cdp=True)
    ctx.env.cdp_evaluate = MagicMock(return_value=False)  # nothing scrolled
    result = handler._execute_rows(step, ctx, schema)
    assert result.success is True
    # One extract pass, then the no-movement scroll stopped the loop.
    assert runner.costs["claude_extract"] == 1
    assert [r["name"] for r in result.extracted_rows] == ["A", "B"]
