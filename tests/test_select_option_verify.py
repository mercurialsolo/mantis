"""Tests for ``select_option`` post-click value verification.

The two-phase select_option flow (open dropdown → click option) can
land on an adjacent menu item — visually similar options stacked
vertically share x-bands, and a y-coordinate inferred from the
language-model's spatial reasoning can fall on the wrong row by a
handful of pixels. Without a read-back step, the runner reports
``select:Priority=High`` when the dropdown actually committed
``Critical``; the verify gate later sees the wrong value and the
recovery loop wastes a budget chasing a fix the layer below already
knew about.

This module locks in two contracts:

1. ``ClaudeExtractor.verify_dropdown_value`` — schema, return shape,
   and graceful None on API failure.
2. ``ClaudeGuidedFormHandler``'s select_option branch — calls the
   verifier after the option click, returns ``select_mismatch:...``
   on a confirmed wrong value, and ignores transient verify failures
   (None) rather than over-triggering the recovery loop.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.actions import ActionType
from mantis_agent.gym.step_context import StepContext
from mantis_agent.gym.step_handlers.form import ClaudeGuidedFormHandler
from mantis_agent.plan_decomposer import MicroIntent


# ── Form-handler integration tests ─────────────────────────────────


class _FakeRunner:
    def __init__(self) -> None:
        self.costs: dict[str, float] = {
            "claude_extract": 0,
            "gpu_steps": 0,
            "gpu_seconds": 0,
        }
        self._url_history: list[str] = []
        self._last_known_url: str = ""

    def _best_effort_current_url(self) -> str:
        return self._url_history.pop(0) if self._url_history else ""

    def _adaptive_submit_settle(self, *, url_before: str) -> float:
        return 0.0

    def _safe_screenshot(self):
        return MagicMock()

    def _dump_debug_screenshot(self, name_stem: str, screenshot) -> None:
        pass


def _ctx(runner: _FakeRunner, *, env=None, extractor=None) -> StepContext:
    return StepContext(
        env=env or MagicMock(),
        brain=None,
        extractor=extractor or MagicMock(),
        grounding=None,
        cost_meter=None,
        dynamic_verifier=None,
        scanner=None,
        site_config=None,
        tool_channel=None,
        extraction_cache=None,
        state={"index": 9},
    )


def _patched_form(monkeypatch) -> None:
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform",
        lambda a, b: a,
    )


def test_select_option_verify_match_returns_success(monkeypatch):
    """Verifier reports ``matches=True`` → success path unchanged."""
    _patched_form(monkeypatch)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.side_effect = [
        {"x": 100, "y": 200},  # dropdown
        {"x": 110, "y": 240},  # option click
    ]
    extractor.verify_dropdown_value.return_value = {
        "matches": True, "observed": "High",
    }
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Set priority", type="select_option",
        params={"dropdown_label": "Priority", "option_label": "High"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is True
    assert result.data == "select:Priority=High"
    extractor.verify_dropdown_value.assert_called_once()
    # 2 find_form_target calls + 1 verify call = 3 Claude calls
    assert runner.costs["claude_extract"] == 3


def test_select_option_blurs_dropdown_after_pick(monkeypatch):
    """A Tab keystroke must follow the option click, before the
    verifier screenshots. React-style controlled selects only fire
    onChange on blur — without this, the form serializes the
    previous default on submit even when the dropdown visually
    displays the new option."""
    _patched_form(monkeypatch)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.side_effect = [
        {"x": 100, "y": 200},  # dropdown
        {"x": 110, "y": 240},  # option click
    ]
    extractor.verify_dropdown_value.return_value = {
        "matches": True, "observed": "High",
    }
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Set priority", type="select_option",
        params={"dropdown_label": "Priority", "option_label": "High"},
    )
    ClaudeGuidedFormHandler(runner).execute(step, ctx)

    # Sequence inspection: dropdown click → option click → Tab blur
    # → screenshot for verify. The Tab keystroke must come AFTER
    # the option click, BEFORE the verify screenshot.
    actions = env.step.call_args_list
    types = [c.args[0].action_type for c in actions]
    assert types[0] == ActionType.CLICK   # dropdown
    assert types[1] == ActionType.CLICK   # option
    # Index 2 must be a KEY_PRESS Tab (the blur-and-commit step).
    third = actions[2].args[0]
    assert third.action_type == ActionType.KEY_PRESS
    assert third.params == {"keys": "Tab"}


def test_select_option_verify_mismatch_returns_failure_with_observed(monkeypatch):
    """Verifier reports ``matches=False`` → step fails with
    ``select_mismatch:got=<observed>_wanted=<expected>``."""
    _patched_form(monkeypatch)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.side_effect = [
        {"x": 100, "y": 200},  # dropdown
        {"x": 110, "y": 252},  # option click — landed on Critical row
    ]
    extractor.verify_dropdown_value.return_value = {
        "matches": False, "observed": "Critical",
    }
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Set priority", type="select_option",
        params={"dropdown_label": "Priority", "option_label": "High"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is False
    assert result.data == "select_mismatch:got=Critical_wanted=High"
    # An Escape keypress is fired to close any stray menu.
    keypress_calls = [
        c for c in env.step.call_args_list
        if c.args[0].action_type == ActionType.KEY_PRESS
        and c.args[0].params == {"keys": "Escape"}
    ]
    assert len(keypress_calls) == 1


def test_select_option_verify_returns_none_keeps_success(monkeypatch):
    """Verifier returns None on API failure → don't fail the step.

    Forcing a recovery cycle on every Claude API blip would spam the
    recovery budget. Treat None as 'could not verify; trust the click
    happened' — the downstream verify gate is the safety net."""
    _patched_form(monkeypatch)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.side_effect = [
        {"x": 100, "y": 200},  # dropdown
        {"x": 110, "y": 240},  # option
    ]
    extractor.verify_dropdown_value.return_value = None
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Set priority", type="select_option",
        params={"dropdown_label": "Priority", "option_label": "High"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is True
    assert result.data == "select:Priority=High"


def test_select_option_verify_swallows_exception_keeps_success(monkeypatch):
    """If verify_dropdown_value raises, the runner shouldn't crash —
    log and continue, same as the None-response branch."""
    _patched_form(monkeypatch)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.side_effect = [
        {"x": 100, "y": 200},
        {"x": 110, "y": 240},
    ]
    extractor.verify_dropdown_value.side_effect = RuntimeError("boom")
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Set priority", type="select_option",
        params={"dropdown_label": "Priority", "option_label": "High"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is True
    assert result.data == "select:Priority=High"


def test_select_option_verify_skipped_when_dropdown_or_option_blank(monkeypatch):
    """Verifier needs both a label *and* an expected value to
    construct a meaningful prompt. If either is empty (legacy plans
    that pass only ``intent``), fall back to the un-verified
    success path rather than calling verify with empty strings."""
    _patched_form(monkeypatch)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.side_effect = [
        {"x": 100, "y": 200},
        {"x": 110, "y": 240},
    ]
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Pick something", type="select_option",
        params={},  # no dropdown_label / option_label
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    # Without labels, success is reported (legacy compatibility) but
    # verify is never called — no expected_value to test against.
    assert result.success is True
    extractor.verify_dropdown_value.assert_not_called()


# ── Extractor-level tests ───────────────────────────────────────────


def test_verify_dropdown_value_returns_match_dict_on_success(monkeypatch):
    """Happy path: ``_call_with_tool_schema`` returns ``observed``
    only (no ``matches`` field — that's the debiased contract). The
    public method computes the match locally and reports both."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    extractor = ClaudeExtractor(api_key="dummy")
    extractor._form_target_provider._client.call_with_tool_schema = (
        lambda *a, **kw: {"observed": "High"}
    )

    out = extractor.verify_dropdown_value(
        screenshot=MagicMock(width=1280, height=720),
        dropdown_label="Priority",
        expected_value="High",
    )
    assert out == {"matches": True, "observed": "High"}


def test_verify_dropdown_value_returns_mismatch_dict(monkeypatch):
    """LLM reports ``observed=Critical`` neutrally — local matcher
    decides matches=False against expected=High."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    extractor = ClaudeExtractor(api_key="dummy")
    extractor._form_target_provider._client.call_with_tool_schema = (
        lambda *a, **kw: {"observed": "Critical"}
    )

    out = extractor.verify_dropdown_value(
        screenshot=MagicMock(width=1280, height=720),
        dropdown_label="Priority",
        expected_value="High",
    )
    assert out == {"matches": False, "observed": "Critical"}


def test_verify_dropdown_value_substring_match_on_either_side(monkeypatch):
    """Common UI shapes: dropdowns render ``"High Priority"`` for
    expected ``"High"``, or render ``"Contacted"`` when option list
    label was ``"Contacted (4)"``. Both should match — the local
    matcher does case-insensitive substring on either side."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    extractor = ClaudeExtractor(api_key="dummy")
    extractor._form_target_provider._client.call_with_tool_schema = (
        lambda *a, **kw: {"observed": "High Priority"}
    )

    out = extractor.verify_dropdown_value(
        screenshot=MagicMock(width=1280, height=720),
        dropdown_label="Priority",
        expected_value="High",
    )
    assert out["matches"] is True

    extractor._form_target_provider._client.call_with_tool_schema = (
        lambda *a, **kw: {"observed": "Contacted"}
    )
    out = extractor.verify_dropdown_value(
        screenshot=MagicMock(width=1280, height=720),
        dropdown_label="Status",
        expected_value="Contacted (4)",
    )
    assert out["matches"] is True


def test_verify_dropdown_value_empty_observed_does_not_match(monkeypatch):
    """An empty ``observed`` (LLM couldn't read the dropdown) must
    not silently pass as a match — that would defeat the validator."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    extractor = ClaudeExtractor(api_key="dummy")
    extractor._form_target_provider._client.call_with_tool_schema = (
        lambda *a, **kw: {"observed": ""}
    )

    out = extractor.verify_dropdown_value(
        screenshot=MagicMock(width=1280, height=720),
        dropdown_label="Priority",
        expected_value="High",
    )
    assert out == {"matches": False, "observed": ""}


def test_verify_dropdown_value_prompt_does_not_leak_expected_value(monkeypatch):
    """The expected value must not appear in the prompt — that's the
    debias guarantee. If we ever leak it back into the prompt the
    LLM will go back to confidently confirming whatever we asked
    for."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    captured: dict = {}

    def fake_call(screenshot, prompt, *, tool_name, tool_description, input_schema, max_tokens):
        captured["prompt"] = prompt
        return {"observed": "High"}

    extractor = ClaudeExtractor(api_key="dummy")
    extractor._form_target_provider._client.call_with_tool_schema = fake_call

    extractor.verify_dropdown_value(
        screenshot=MagicMock(width=1280, height=720),
        dropdown_label="Priority",
        expected_value="High",
    )
    # Dropdown label is fine — orient the LLM to the right control.
    assert "Priority" in captured["prompt"]
    # Expected value must not be in the prompt — that's the bias we
    # specifically removed.
    assert "High" not in captured["prompt"]


def test_verify_dropdown_value_returns_none_on_api_failure(monkeypatch):
    """``_call_with_tool_schema`` returns None on API failure —
    surface it so the form handler can treat it as 'unverified, trust
    the click'."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    extractor = ClaudeExtractor(api_key="dummy")
    extractor._form_target_provider._client.call_with_tool_schema = (
        lambda *a, **kw: None
    )

    out = extractor.verify_dropdown_value(
        screenshot=MagicMock(width=1280, height=720),
        dropdown_label="Priority",
        expected_value="High",
    )
    assert out is None


def test_verify_dropdown_value_passes_correct_tool_schema(monkeypatch):
    """The tool schema requires only ``observed`` — match is
    computed locally to keep the LLM unbiased."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    captured: dict = {}

    def fake_call(screenshot, prompt, *, tool_name, tool_description, input_schema, max_tokens):
        captured["tool_name"] = tool_name
        captured["input_schema"] = input_schema
        return {"observed": "High"}

    extractor = ClaudeExtractor(api_key="dummy")
    extractor._form_target_provider._client.call_with_tool_schema = fake_call

    extractor.verify_dropdown_value(
        screenshot=MagicMock(width=1920, height=1080),
        dropdown_label="Priority",
        expected_value="High",
    )

    assert captured["tool_name"] == "report_dropdown_value"
    schema = captured["input_schema"]
    assert "observed" in schema["properties"]
    assert "matches" not in schema["properties"]  # debiased: no LLM-side decision
    assert set(schema["required"]) == {"observed"}
    assert schema["properties"]["observed"]["type"] == "string"


# ── Local semantic matcher ─────────────────────────────────────────


def test_semantic_dropdown_match_exact():
    from mantis_agent.extraction.extractor import ClaudeExtractor
    assert ClaudeExtractor._semantic_dropdown_match("High", "High") is True


def test_semantic_dropdown_match_case_and_whitespace():
    from mantis_agent.extraction.extractor import ClaudeExtractor
    assert ClaudeExtractor._semantic_dropdown_match("  high  ", "HIGH") is True


def test_semantic_dropdown_match_substring_either_direction():
    from mantis_agent.extraction.extractor import ClaudeExtractor
    # observed shorter
    assert ClaudeExtractor._semantic_dropdown_match("High", "High Priority") is True
    # expected shorter
    assert ClaudeExtractor._semantic_dropdown_match("High Priority", "High") is True


def test_semantic_dropdown_match_distinct_values():
    from mantis_agent.extraction.extractor import ClaudeExtractor
    # Critical and High share no substring relationship — must NOT match.
    assert ClaudeExtractor._semantic_dropdown_match("Critical", "High") is False


def test_semantic_dropdown_match_empty_strings():
    from mantis_agent.extraction.extractor import ClaudeExtractor
    # Empty observed → not a match (avoids silent pass when LLM
    # blanks the read).
    assert ClaudeExtractor._semantic_dropdown_match("", "High") is False
    assert ClaudeExtractor._semantic_dropdown_match("High", "") is False
    assert ClaudeExtractor._semantic_dropdown_match("", "") is False
