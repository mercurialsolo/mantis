"""ClaudeGuidedFilterHandler unit tests — Phase 2 of EPIC #161.

Sidebar filter dispatch with three action types: ``click``, ``type``,
``select``. Tests pin each branch.

- click action: simple click + 2s settle
- type action: click + triple-click + ctrl+a + Delete + TYPE + Return
- select action: open dropdown + Claude finds option + click
- select with option-not-found: Escape to close + failure
- target not found after 8 viewport scrolls → failure
- step_type property
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.actions import ActionType
from mantis_agent.gym.step_context import StepContext
from mantis_agent.gym.step_handlers.filter import ClaudeGuidedFilterHandler
from mantis_agent.plan_decomposer import MicroIntent


class _FakeRunner:
    def __init__(self) -> None:
        self.costs: dict[str, float] = {"claude_extract": 0, "claude_grounding": 0, "gpu_steps": 0}
        self._last_known_url = ""
        self._results_base_url = "https://example.com/cars"
        self.scroll_state_calls: list[dict] = []

    def _set_scroll_state(self, **kwargs) -> None:
        self.scroll_state_calls.append(kwargs)

    def _current_results_page_url(self) -> str:
        return self._results_base_url


def _ctx(runner, *, env=None, extractor=None) -> StepContext:
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
        state={"index": 6},
    )


def _step(intent: str = "Set seller type to private") -> MicroIntent:
    return MicroIntent(intent=intent, type="filter", section="setup")


def test_click_action_succeeds(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.filter.time.sleep", lambda *_: None)
    monkeypatch.setattr("mantis_agent.gym.step_handlers.filter.random.uniform", lambda a, b: a)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_filter_target.return_value = {
        "x": 100, "y": 200, "action": "click", "value": "", "label": "Private Seller",
    }
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    result = ClaudeGuidedFilterHandler(runner).execute(_step(), ctx)

    assert result.success is True
    assert runner.costs["gpu_steps"] == 1
    assert runner.costs["claude_extract"] == 1
    # Final scroll state set after filter applies
    assert runner.scroll_state_calls
    assert runner.scroll_state_calls[-1]["context"] == "results_after_filter"


def test_type_action_clears_then_types_then_enters(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.filter.time.sleep", lambda *_: None)
    monkeypatch.setattr("mantis_agent.gym.step_handlers.filter.random.uniform", lambda a, b: a)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_filter_target.return_value = {
        "x": 200, "y": 300, "action": "type", "value": "33101", "label": "Zip Code",
    }
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    result = ClaudeGuidedFilterHandler(runner).execute(_step(intent="Enter zip 33101"), ctx)

    assert result.success is True
    # Examine the env.step calls — filter type does:
    #   sidebar reset (10x SCROLL up) + initial CLICK + triple-click (3x CLICK) +
    #   ctrl+a + Delete + TYPE + Return
    types = [c.args[0].action_type for c in env.step.call_args_list]
    assert ActionType.TYPE in types
    type_call = [c for c in env.step.call_args_list if c.args[0].action_type == ActionType.TYPE][0]
    assert type_call.args[0].params == {"text": "33101"}
    # Final keypress Return
    keypress_calls = [c for c in env.step.call_args_list if c.args[0].action_type == ActionType.KEY_PRESS]
    return_calls = [c for c in keypress_calls if c.args[0].params.get("keys") == "Return"]
    assert len(return_calls) >= 1


def test_select_action_opens_dropdown_picks_option(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.filter.time.sleep", lambda *_: None)
    monkeypatch.setattr("mantis_agent.gym.step_handlers.filter.random.uniform", lambda a, b: a)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_filter_target.side_effect = [
        # First call: find dropdown
        {"x": 150, "y": 250, "action": "select", "value": "Florida", "label": "State"},
        # Second call: find option in opened dropdown
        {"x": 160, "y": 290, "action": "click", "value": "", "label": "Florida"},
    ]
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    result = ClaudeGuidedFilterHandler(runner).execute(_step(intent="Pick state Florida"), ctx)

    assert result.success is True
    assert runner.costs["claude_extract"] == 2  # dropdown + option
    # Two clicks for dropdown + option
    click_calls = [c for c in env.step.call_args_list if c.args[0].action_type == ActionType.CLICK]
    assert len(click_calls) >= 2


def test_select_with_option_not_found_presses_escape(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.filter.time.sleep", lambda *_: None)
    monkeypatch.setattr("mantis_agent.gym.step_handlers.filter.random.uniform", lambda a, b: a)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_filter_target.side_effect = [
        {"x": 150, "y": 250, "action": "select", "value": "Mars", "label": "State"},
        None,  # option not found
    ]
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    result = ClaudeGuidedFilterHandler(runner).execute(_step(intent="Pick state Mars"), ctx)

    assert result.success is False
    # Escape was sent to close the dropdown
    keypress_calls = [c for c in env.step.call_args_list if c.args[0].action_type == ActionType.KEY_PRESS]
    escape_calls = [c for c in keypress_calls if c.args[0].params.get("keys") == "Escape"]
    assert len(escape_calls) >= 1


def test_target_not_found_after_8_viewport_scrolls(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.filter.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_filter_target.return_value = None  # never found
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    result = ClaudeGuidedFilterHandler(runner).execute(_step(), ctx)

    assert result.success is False
    # 8 viewport scans billed to claude_extract
    assert runner.costs["claude_extract"] == 8


def test_step_type_property():
    handler = ClaudeGuidedFilterHandler(_FakeRunner())
    assert handler.step_type == "filter"
