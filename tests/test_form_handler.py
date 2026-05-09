"""ClaudeGuidedFormHandler unit tests — Phase 2 of EPIC #161.

Three step types share one handler (fill_field / submit /
select_option). Tests pin the high-value paths:

- fill_field success: target found, click → triple-click → ctrl+a +
  Delete → type
- fill_field target not found: returns ``form_target_not_found``
- submit success: target found, adaptive settle, returns success
- submit target not found after scroll-and-rescan: returns
  ``form_target_not_found`` (and resets scroll to Home)
- submit click-no-navigation triggers Enter-key fallback
- select_option success: open dropdown → pick option
- select_option dropdown not found: returns ``form_target_not_found``
- select_option option not found in open menu: returns
  ``option_not_found`` and presses Escape to close

No Xvfb, no GymRunner, no real ClaudeExtractor.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from mantis_agent.actions import ActionType
from mantis_agent.gym.step_context import StepContext
from mantis_agent.gym.step_handlers.form import ClaudeGuidedFormHandler
from mantis_agent.plan_decomposer import MicroIntent


class _FakeRunner:
    """Minimal back-reference for the form handler's parent calls."""

    def __init__(self) -> None:
        self.costs: dict[str, float] = {
            "claude_extract": 0,
            "gpu_steps": 0,
            "gpu_seconds": 0,
        }
        self._url_history: list[str] = ["", ""]  # before, after
        self.dump_calls: list[tuple[str, Any]] = []
        self._last_known_url: str = ""

    def _best_effort_current_url(self) -> str:
        return self._url_history.pop(0) if self._url_history else ""

    def _adaptive_submit_settle(self, *, url_before: str) -> float:
        return 0.5  # cheap stand-in for the bounded URL-poll

    def _safe_screenshot(self) -> Any:
        return MagicMock()

    def _dump_debug_screenshot(self, name_stem: str, screenshot: Any) -> None:
        self.dump_calls.append((name_stem, screenshot))


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
        state={"index": 5},
    )


# ── fill_field ──────────────────────────────────────────────────────


def test_fill_field_target_not_found_returns_form_target_not_found(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.return_value = None
    # Vision-affordance fallback also misses → form_target_not_found.
    extractor.find_target_by_affordance.return_value = None
    ctx = _ctx(runner, extractor=extractor)

    step = MicroIntent(
        intent="Fill in email", type="fill_field",
        params={"label": "Email", "value": "x@y.com"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is False
    assert result.data == "form_target_not_found"
    # 1 label-match + 1 visual-affordance fallback = 2 Claude calls.
    assert runner.costs["claude_extract"] == 2


def test_fill_field_success_clicks_triple_clicks_clears_and_types(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform",
        lambda a, b: a,
    )

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.return_value = {"x": 100, "y": 200}
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Fill email", type="fill_field",
        params={"label": "Email", "value": "user@example.com"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is True
    assert result.data == "fill:Email"
    assert runner.costs["gpu_steps"] == 1
    # Click + triple-click (3 total) + ctrl+a + Delete + TYPE = 6 env.step calls
    types = [call.args[0].action_type for call in env.step.call_args_list]
    assert types[0] == ActionType.CLICK
    assert types[1] == ActionType.CLICK
    assert types[2] == ActionType.CLICK
    assert types[3] == ActionType.KEY_PRESS  # ctrl+a
    assert types[4] == ActionType.KEY_PRESS  # Delete
    assert types[5] == ActionType.TYPE


# ── submit ──────────────────────────────────────────────────────────


def test_submit_target_not_found_after_scroll(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.return_value = None  # never found
    extractor.find_target_by_affordance.return_value = None  # vision miss too
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Submit form", type="submit",
        params={"label": "Sign in"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is False
    assert result.data == "form_target_not_found"
    # 1 initial + End probe + Home probe + 6 Page_Down sweeps + 1
    # visual-affordance fallback = 10 calls. The affordance pass
    # is the language-agnostic / icon-aware fallback that fires
    # after label-match scroll-probe exhausts.
    assert runner.costs["claude_extract"] == 10
    # Final Home keypress to reset scroll for the next step.
    final_keypress = env.step.call_args_list[-1].args[0]
    assert final_keypress.params == {"keys": "Home"}


def test_submit_click_navigated_propagates_url_to_last_known_url(monkeypatch):
    """Successful click that changes the URL must update runner._last_known_url
    so step_snapshot.diff sees the change and run_executor doesn't demote
    the success to no_state_change."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform",
        lambda a, b: a,
    )

    runner = _FakeRunner()
    runner._last_known_url = "https://app.example/dashboard"
    runner._url_history = [
        "https://app.example/dashboard",  # url_before
        "https://app.example/leads",      # url_after_click — navigated
    ]
    extractor = MagicMock()
    extractor.find_form_target.return_value = {"x": 100, "y": 142}
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Open Leads", type="submit",
        params={"label": "Leads"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is True
    assert runner._last_known_url == "https://app.example/leads"


def test_submit_url_unchanged_does_not_update_last_known_url(monkeypatch):
    """When neither click nor Enter fallback navigates, _last_known_url
    must stay at its prior value — recovery uses it as the rollback URL."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform",
        lambda a, b: a,
    )

    runner = _FakeRunner()
    runner._last_known_url = "https://app.example/results"
    runner._url_history = [
        "https://app.example/results",  # url_before
        "https://app.example/results",  # url_after_click — unchanged
        "https://app.example/results",  # url_after_enter — still unchanged
    ]
    extractor = MagicMock()
    extractor.find_form_target.return_value = {"x": 100, "y": 142}
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Submit", type="submit",
        params={"label": "Submit"},
    )
    ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert runner._last_known_url == "https://app.example/results"


def test_submit_url_changed_after_enter_fallback_updates_last_known_url(monkeypatch):
    """Click doesn't navigate but Enter fallback does — the post-Enter URL
    is what should land in _last_known_url."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform",
        lambda a, b: a,
    )

    runner = _FakeRunner()
    runner._last_known_url = "https://app.example/login"
    runner._url_history = [
        "https://app.example/login",     # url_before
        "https://app.example/login",     # url_after_click — unchanged → trigger Enter
        "https://app.example/dashboard", # url_after_enter — navigated
    ]
    extractor = MagicMock()
    extractor.find_form_target.return_value = {"x": 100, "y": 142}
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Sign in", type="submit",
        params={"label": "Sign in"},
    )
    ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert runner._last_known_url == "https://app.example/dashboard"


def test_submit_click_then_enter_fallback_when_url_does_not_change(monkeypatch):
    """Click that doesn't navigate triggers an Enter-key fallback."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform",
        lambda a, b: a,
    )

    runner = _FakeRunner()
    # URL doesn't change after click → fallback fires
    runner._url_history = [
        "https://app.example/login",  # url_before
        "https://app.example/login",  # url_after_click (unchanged)
    ]
    extractor = MagicMock()
    extractor.find_form_target.return_value = {"x": 200, "y": 400}
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Submit", type="submit",
        params={"label": "Submit"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is True
    # Two adaptive settles billed (one for click, one for Enter fallback)
    assert runner.costs["gpu_seconds"] == 1.0
    # gpu_steps incremented for both click AND Enter
    assert runner.costs["gpu_steps"] == 2
    # Last env.step is the Return keypress
    last_call = env.step.call_args_list[-1]
    assert last_call.args[0].action_type == ActionType.KEY_PRESS
    assert last_call.args[0].params == {"keys": "Return"}


# ── select_option ───────────────────────────────────────────────────


def test_select_option_dropdown_not_found(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.return_value = None
    ctx = _ctx(runner, extractor=extractor)

    step = MicroIntent(
        intent="Pick industry", type="select_option",
        params={"dropdown_label": "Industry", "option_label": "Tech"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is False
    assert result.data == "form_target_not_found"


def test_select_option_open_succeeds_but_option_not_found_presses_escape(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform",
        lambda a, b: a,
    )

    runner = _FakeRunner()
    extractor = MagicMock()
    # First call: dropdown found. Second call: option not found in open menu.
    extractor.find_form_target.side_effect = [
        {"x": 100, "y": 200},  # dropdown
        None,                  # option lookup miss
    ]
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Select industry", type="select_option",
        params={"dropdown_label": "Industry", "option_label": "Tech"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is False
    assert result.data == "option_not_found"
    # First step: click the dropdown. Last step: Escape to close.
    last_call = env.step.call_args_list[-1]
    assert last_call.args[0].action_type == ActionType.KEY_PRESS
    assert last_call.args[0].params == {"keys": "Escape"}


def test_select_option_full_success(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform",
        lambda a, b: a,
    )

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.side_effect = [
        {"x": 100, "y": 200},  # dropdown
        {"x": 110, "y": 240},  # option
    ]
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Select Tech", type="select_option",
        params={"dropdown_label": "Industry", "option_label": "Tech"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is True
    assert result.data == "select:Industry=Tech"
    # 2 click steps (dropdown + option)
    click_calls = [c for c in env.step.call_args_list if c.args[0].action_type == ActionType.CLICK]
    assert len(click_calls) == 2
    assert runner.costs["gpu_steps"] == 2


def test_unknown_step_type_returns_failure(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    runner = _FakeRunner()
    ctx = _ctx(runner)

    step = MicroIntent(intent="Mystery", type="unknown_form_type")
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)
    assert result.success is False
    assert result.data == ""


def test_step_type_property():
    handler = ClaudeGuidedFormHandler(_FakeRunner())
    assert handler.step_type == "submit"
