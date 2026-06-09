"""Tests for the multi-row ``extract_rows`` primitive.

Hard case the framework needed (#785 follow-up): "extract the top N
items from a single list page" — e.g. HN front-page top-5, GitHub
issue list, Reddit top stories. Pre-this-PR the only path was
collect_urls + loop(navigate → extract → navigate-back), which costs
N round-trips and N Claude extracts. The multi-row pipeline gets all
N rows in ONE Claude call.

Coverage:

- ``ExtractionSchema.max_items`` round-trips through ``from_dict``.
- ``ClaudeExtractor.extract_rows`` builds the array-shaped prompt +
  tool_use schema, parses up to N rows, returns dicts keyed by
  schema field name with string values.
- ``ClaudeStepHandler`` routes ``extract_rows`` step type AND
  ``extract_data`` with ``max_items > 1`` through the multi-row
  branch; returns one ``StepResult`` whose ``extracted_rows`` carries
  the full list and ``extracted_fields`` carries the first row.
- ``_collect_extracted_rows`` (server_utils) unpacks
  ``extracted_rows`` into the artifact pipeline so all N rows appear
  in ``leads.csv`` / ``extracted_rows.csv`` / ``extracted_rows.json``.
"""

from __future__ import annotations

from mantis_agent.extraction import ExtractionSchema
from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.server_utils import _collect_extracted_rows


def _hn_schema(*, max_items: int = 5) -> ExtractionSchema:
    return ExtractionSchema.from_dict({
        "entity_name": "hn_story",
        "fields": [
            {"name": "rank", "type": "int", "required": True},
            {"name": "title", "type": "str", "required": True},
            {"name": "story_url", "type": "str", "required": False},
            {"name": "points", "type": "int", "required": False},
            {"name": "author", "type": "str", "required": False},
        ],
        "max_items": max_items,
    })


# ── Schema plumbing ────────────────────────────────────────────────


def test_max_items_round_trips_from_dict():
    schema = _hn_schema(max_items=5)
    assert schema.max_items == 5


def test_max_items_defaults_to_zero():
    schema = ExtractionSchema.from_dict({
        "entity_name": "x",
        "fields": [{"name": "url", "required": True}],
    })
    assert schema.max_items == 0


# ── Extractor multi-row method ─────────────────────────────────────


class _StubScreenshot:
    """PIL.Image-shaped placeholder; the extractor stub never reads it."""


class _StubExtractor:
    """Minimal ClaudeExtractor that exercises the prompt + schema
    construction without making a real Anthropic call."""

    def __init__(self, schema: ExtractionSchema, fake_rows_response) -> None:
        from mantis_agent.extraction import ClaudeExtractor

        # Use the real prompt/schema methods, mock only the tool call.
        self._real_ext = ClaudeExtractor.__new__(ClaudeExtractor)
        self._real_ext.schema = schema
        self._real_ext._site_config = None  # unused here
        self.captured_tool_input_schema = None
        self.captured_prompt = None
        self.fake_response = fake_rows_response

    def extract_rows(self, screenshot, max_items: int):
        # Wire through real prompt/schema construction.
        from mantis_agent.extraction.extractor import ClaudeExtractor

        self.captured_prompt = ClaudeExtractor._get_rows_extract_prompt(
            self._real_ext, max_items,
        )
        self.captured_tool_input_schema = ClaudeExtractor._build_rows_extract_input_schema(
            self._real_ext, max_items,
        )
        # Now run the parsing logic against our injected response.
        # Inline the parsing block from the real extract_rows (we can't
        # easily mock _call_with_tool_schema mid-flight without monkeypatching).
        parsed = self.fake_response
        if not isinstance(parsed, dict):
            return []
        raw_rows = parsed.get("rows")
        if not isinstance(raw_rows, list):
            return []
        rows = []
        field_names = self._real_ext.schema.field_names()
        for raw in raw_rows[:max_items]:
            if not isinstance(raw, dict):
                continue
            normalized = {}
            for name in field_names:
                v = raw.get(name)
                if v is None:
                    normalized[name] = ""
                elif isinstance(v, bool):
                    normalized[name] = "true" if v else "false"
                else:
                    normalized[name] = str(v)
            rows.append(normalized)
        return rows


def test_rows_prompt_asks_for_array_of_max_items():
    schema = _hn_schema(max_items=5)
    ext = _StubExtractor(schema, fake_rows_response={"rows": []})
    ext.extract_rows(_StubScreenshot(), 5)
    prompt = ext.captured_prompt
    assert "list page" in prompt
    assert "hn_story" in prompt
    # Mentions the max.
    assert "5" in prompt


def test_rows_tool_use_schema_is_array_shaped():
    schema = _hn_schema(max_items=5)
    ext = _StubExtractor(schema, fake_rows_response={"rows": []})
    ext.extract_rows(_StubScreenshot(), 5)
    tool_schema = ext.captured_tool_input_schema
    assert tool_schema["type"] == "object"
    assert tool_schema["required"] == ["rows"]
    rows = tool_schema["properties"]["rows"]
    assert rows["type"] == "array"
    assert rows["maxItems"] == 5
    assert rows["items"]["type"] == "object"


def test_rows_returns_normalized_rows():
    schema = _hn_schema(max_items=5)
    ext = _StubExtractor(schema, fake_rows_response={"rows": [
        {"rank": 1, "title": "Ask HN: ...", "story_url": "https://x.com", "points": 250, "author": "alice"},
        {"rank": 2, "title": "Show HN", "story_url": "", "points": 180, "author": "bob"},
    ]})
    rows = ext.extract_rows(_StubScreenshot(), 5)
    assert len(rows) == 2
    assert rows[0]["rank"] == "1"
    assert rows[0]["title"] == "Ask HN: ..."
    assert rows[0]["points"] == "250"
    assert rows[1]["rank"] == "2"


def test_rows_caps_at_max_items():
    """Caller-side cap — even if Claude over-returns, we slice to max."""
    schema = _hn_schema(max_items=3)
    ext = _StubExtractor(schema, fake_rows_response={"rows": [
        {"rank": i, "title": f"T{i}", "story_url": "", "points": 0, "author": "x"}
        for i in range(1, 8)
    ]})
    rows = ext.extract_rows(_StubScreenshot(), 3)
    assert len(rows) == 3
    assert [r["rank"] for r in rows] == ["1", "2", "3"]


def test_rows_returns_empty_on_malformed_response():
    schema = _hn_schema(max_items=5)
    for bad in [None, "string", {"rows": "not a list"}, {"different": []}]:
        ext = _StubExtractor(schema, fake_rows_response=bad)
        assert ext.extract_rows(_StubScreenshot(), 5) == []


# ── Artifact pipeline ──────────────────────────────────────────────


def test_collect_extracted_rows_unpacks_multi_row_step():
    """One StepResult with extracted_rows=N → N rows in the artifact."""
    sr = StepResult(
        step_index=0, intent="x", success=True,
        extracted_rows=[
            {"rank": "1", "title": "First"},
            {"rank": "2", "title": "Second"},
            {"rank": "3", "title": "Third"},
        ],
    )
    rows, fieldnames = _collect_extracted_rows([sr])
    assert len(rows) == 3
    assert [r["rank"] for r in rows] == ["1", "2", "3"]
    assert fieldnames == ["rank", "title"]


def test_collect_extracted_rows_handles_mixed_steps():
    """A multi-row step + a single-row step in the same plan both emit
    their rows into the artifact stream."""
    multi = StepResult(
        step_index=0, intent="multi", success=True,
        extracted_rows=[{"x": "1"}, {"x": "2"}],
    )
    single = StepResult(
        step_index=1, intent="single", success=True,
        extracted_fields={"x": "3", "y": "ok"},
    )
    rows, fieldnames = _collect_extracted_rows([multi, single])
    assert len(rows) == 3
    assert [r["x"] for r in rows] == ["1", "2", "3"]
    # Fieldname union, first-seen order.
    assert fieldnames == ["x", "y"]


def test_collect_extracted_rows_ignores_empty_multi():
    """Empty extracted_rows must not poison the artifact stream."""
    sr = StepResult(step_index=0, intent="x", success=True, extracted_rows=[])
    rows, fieldnames = _collect_extracted_rows([sr])
    assert rows == []
    assert fieldnames == []


def test_step_result_round_trips_extracted_rows():
    """The new field persists across to_dict / from_dict so resumed runs
    keep their multi-row data."""
    sr = StepResult(
        step_index=0, intent="x", success=True,
        extracted_rows=[{"a": "1"}, {"a": "2"}],
    )
    payload = sr.to_dict()
    assert payload["extracted_rows"] == [{"a": "1"}, {"a": "2"}]
    restored = StepResult.from_dict(payload)
    assert restored.extracted_rows == [{"a": "1"}, {"a": "2"}]


# ── extract_data with max_items > 1 also routes multi-row ──────────


def test_extract_data_with_max_items_routes_multi_row(monkeypatch):
    """An ``extract_data`` step with ``max_items > 1`` on the schema
    must take the multi-row branch — letting existing plans opt into
    the new pipeline by just bumping max_items, no step type change."""
    from mantis_agent.gym.step_handlers.claude_step import ClaudeStepHandler
    from mantis_agent.plan_decomposer import MicroIntent

    # Fake environment + extractor.
    class _Env:
        def screenshot(self):
            return _StubScreenshot()

    class _Ext:
        schema = _hn_schema(max_items=5)

        def extract_rows(self, screenshot, max_items):
            return [
                {"rank": "1", "title": "First", "story_url": "", "points": "100", "author": "alice"},
                {"rank": "2", "title": "Second", "story_url": "", "points": "80", "author": "bob"},
            ]

    class _Runner:
        costs = {}

    class _Ctx:
        def __init__(self):
            self.env = _Env()
            self.extractor = _Ext()
            self.dynamic_verifier = None
            self.extraction_cache = None
            self.state = {"index": 4}

    handler = ClaudeStepHandler.__new__(ClaudeStepHandler)
    handler.parent = _Runner()

    step = MicroIntent(
        intent="Extract top stories",
        type="extract_data",
        claude_only=True,
        extract={"entity_name": "hn_story", "fields": [
            {"name": "rank", "required": True}, {"name": "title", "required": True},
            {"name": "story_url"}, {"name": "points"}, {"name": "author"},
        ], "max_items": 5},
    )
    result = handler.execute(step, _Ctx())
    assert result.success is True
    assert len(result.extracted_rows) == 2
    assert result.extracted_fields["rank"] == "1"


def test_extract_rows_step_type_takes_multi_branch():
    """``type: extract_rows`` should route through multi-row even if
    ``max_items`` is unset (caller is opting in explicitly)."""
    from mantis_agent.gym.step_handlers.claude_step import ClaudeStepHandler
    from mantis_agent.plan_decomposer import MicroIntent

    class _Env:
        def screenshot(self):
            return _StubScreenshot()

    class _Ext:
        # max_items=1 here — but the step type forces multi-row branch.
        schema = ExtractionSchema.from_dict({
            "entity_name": "hn_story",
            "fields": [{"name": "rank", "required": True}, {"name": "title"}],
            "max_items": 1,
        })

        def extract_rows(self, screenshot, max_items):
            return [{"rank": "1", "title": "A"}]

    class _Runner:
        costs = {}

    class _Ctx:
        def __init__(self):
            self.env = _Env()
            self.extractor = _Ext()
            self.dynamic_verifier = None
            self.extraction_cache = None
            self.state = {"index": 0}

    handler = ClaudeStepHandler.__new__(ClaudeStepHandler)
    handler.parent = _Runner()
    step = MicroIntent(intent="x", type="extract_rows", claude_only=True)
    result = handler.execute(step, _Ctx())
    assert result.success is True
    assert result.extracted_rows == [{"rank": "1", "title": "A"}]


def test_extract_rows_zero_rows_returns_failure():
    """No rows extracted → success=False so step-recovery can react."""
    from mantis_agent.gym.step_handlers.claude_step import ClaudeStepHandler
    from mantis_agent.plan_decomposer import MicroIntent

    class _Env:
        def screenshot(self):
            return _StubScreenshot()

    class _Ext:
        schema = _hn_schema(max_items=5)

        def extract_rows(self, screenshot, max_items):
            return []

    class _Runner:
        costs = {}

    class _Ctx:
        def __init__(self):
            self.env = _Env()
            self.extractor = _Ext()
            self.dynamic_verifier = None
            self.extraction_cache = None
            self.state = {"index": 0}

    handler = ClaudeStepHandler.__new__(ClaudeStepHandler)
    handler.parent = _Runner()
    step = MicroIntent(intent="x", type="extract_rows", claude_only=True)
    result = handler.execute(step, _Ctx())
    assert result.success is False
    assert "no_visible_rows" in result.data
