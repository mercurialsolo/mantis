"""Form-shaped intent vocabulary (issue #80).

Pins the contract that:
- MicroIntent supports `params` for structured field/value payloads.
- The decomposer's _build_intent applies sensible defaults for the new types
  (required=True, grounding=True, section="setup", form types in FORM_STEP_TYPES).
- MicroPlan.to_dict / from_dict round-trip preserves params.
- The runner has a dispatch entry for the new types that uses
  ClaudeExtractor.find_form_target instead of find_all_listings.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from mantis_agent import MicroIntent, MicroPlan
from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
from mantis_agent.gym.micro_runner import MicroPlanRunner, StepResult
from mantis_agent.plan_decomposer import PlanDecomposer


# ── _build_intent: defaults for new form types ─────────────────────────


@pytest.mark.parametrize("step_type", ["fill_field", "submit", "select_option"])
def test_form_step_defaults_required_grounded_setup(step_type: str):
    intent = PlanDecomposer._build_intent({"intent": "x", "type": step_type})
    assert intent.type == step_type
    assert intent.required is True, "form steps must be required by default"
    assert intent.grounding is True, "form steps must be claude-grounded"
    assert intent.section == "setup", "form steps default to setup section"
    # Form steps go through the form dispatch, not Holo3-direct, not Claude-only.
    assert intent.claude_only is False


def test_form_step_constants_listed():
    """Regression guard: FORM_STEP_TYPES must contain exactly the three issue-80 types."""
    assert set(PlanDecomposer.FORM_STEP_TYPES) == {"fill_field", "submit", "select_option"}


def test_build_intent_params_default_empty():
    intent = PlanDecomposer._build_intent({"intent": "x", "type": "click"})
    assert intent.params == {}


def test_build_intent_params_round_trip():
    intent = PlanDecomposer._build_intent({
        "intent": "Enter the user ID",
        "type": "fill_field",
        "params": {"label": "User ID", "value": "sarah.connor"},
    })
    assert intent.params == {"label": "User ID", "value": "sarah.connor"}


def test_build_intent_params_rejects_non_dict():
    """A malformed `params` (e.g. a string from a misbehaving decomposer) is dropped."""
    intent = PlanDecomposer._build_intent({
        "intent": "x", "type": "fill_field", "params": "not-a-dict",
    })
    assert intent.params == {}


# ── MicroPlan.to_dict / from_dict round-trip preserves params ──────────


def test_microplan_round_trip_preserves_form_params():
    plan = MicroPlan(
        steps=[
            MicroIntent(
                intent="Navigate to login",
                type="navigate",
                budget=3,
                section="setup",
                required=True,
            ),
            MicroIntent(
                intent="Enter the user ID",
                type="fill_field",
                budget=4,
                section="setup",
                required=True,
                grounding=True,
                params={"label": "User ID", "value": "sarah.connor"},
            ),
            MicroIntent(
                intent="Submit the form",
                type="submit",
                budget=4,
                section="setup",
                required=True,
                grounding=True,
                params={"label": "Login"},
            ),
        ],
        domain="staffai-test-crm.exe.xyz",
    )
    encoded = plan.to_dict()
    restored = MicroPlan.from_dict(encoded)

    assert len(restored.steps) == 3
    assert restored.steps[1].type == "fill_field"
    assert restored.steps[1].params == {"label": "User ID", "value": "sarah.connor"}
    assert restored.steps[2].type == "submit"
    assert restored.steps[2].params == {"label": "Login"}


# ── Runner dispatch routes form types to find_form_target ──────────────


class _FakeEnv(GymEnvironment):
    def __init__(self):
        self.actions: list[Action] = []
        self._img = Image.new("RGB", (320, 200), "white")

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        return GymObservation(screenshot=self._img)

    def step(self, action: Action) -> GymResult:
        self.actions.append(action)
        return GymResult(GymObservation(screenshot=self._img), 0.0, False, {})

    def screenshot(self) -> Image.Image:
        return self._img

    def close(self) -> None:
        pass

    @property
    def screen_size(self):
        return (320, 200)


def _runner_with_extractor(env: _FakeEnv, extractor: MagicMock) -> MicroPlanRunner:
    return MicroPlanRunner(
        brain=None, env=env, grounding=None, extractor=extractor,
        checkpoint_path="/tmp/_form_runner_test.json",
        run_key="test", session_name="test",
        max_cost=999.0, max_time_minutes=999,
    )


def test_fill_field_calls_find_form_target_not_find_all_listings():
    """The whole point of #80: form steps must NOT call find_all_listings."""
    env = _FakeEnv()
    extractor = MagicMock()
    extractor.find_form_target.return_value = {
        "x": 100, "y": 50, "action": "click", "value": "", "label": "User ID",
    }
    runner = _runner_with_extractor(env, extractor)

    intent = MicroIntent(
        intent="Enter the user ID",
        type="fill_field",
        budget=4,
        section="setup",
        required=True,
        grounding=True,
        params={"label": "User ID", "value": "sarah.connor"},
    )
    result = runner._execute_claude_guided_form(intent, index=0)

    assert result.success is True
    assert result.data.startswith("fill:")
    extractor.find_form_target.assert_called_once()
    # Critical regression guard — find_all_listings is the listings extractor
    # whose "0 cards, 0 new" was the issue-80 failure mode.
    extractor.find_all_listings.assert_not_called()
    # The runner should have clicked the field, cleared, and typed the value.
    types = [a.action_type for a in env.actions]
    assert ActionType.CLICK in types
    assert ActionType.TYPE in types


def test_submit_clicks_labelled_button():
    env = _FakeEnv()
    extractor = MagicMock()
    extractor.find_form_target.return_value = {
        "x": 200, "y": 75, "action": "click", "value": "", "label": "Login",
    }
    runner = _runner_with_extractor(env, extractor)

    intent = MicroIntent(
        intent="Click Login",
        type="submit",
        budget=4,
        section="setup",
        required=True,
        params={"label": "Login"},
    )
    result = runner._execute_claude_guided_form(intent, index=0)

    assert result.success is True
    assert result.data.startswith("submit:")
    extractor.find_all_listings.assert_not_called()
    # Submit performs exactly one click — no clear/type rigmarole.
    click_actions = [a for a in env.actions if a.action_type == ActionType.CLICK]
    assert len(click_actions) == 1
    assert click_actions[0].params["x"] == 200


def test_submit_target_not_found_returns_failure_no_crash():
    env = _FakeEnv()
    extractor = MagicMock()
    extractor.find_form_target.return_value = None
    runner = _runner_with_extractor(env, extractor)

    intent = MicroIntent(
        intent="Click Login",
        type="submit",
        params={"label": "Login"},
    )
    result = runner._execute_claude_guided_form(intent, index=0)

    assert result.success is False
    assert result.data == "form_target_not_found"


def test_select_option_two_phase_open_then_pick():
    env = _FakeEnv()
    extractor = MagicMock()
    # First call: locate the dropdown control. Second: locate the option.
    extractor.find_form_target.side_effect = [
        {"x": 300, "y": 100, "action": "click", "value": "", "label": "Industry Vertical"},
        {"x": 320, "y": 200, "action": "click", "value": "", "label": "Space Exploration"},
    ]
    runner = _runner_with_extractor(env, extractor)

    intent = MicroIntent(
        intent="Pick Space Exploration",
        type="select_option",
        params={
            "dropdown_label": "Industry Vertical",
            "option_label": "Space Exploration",
        },
    )
    result = runner._execute_claude_guided_form(intent, index=0)

    assert result.success is True
    assert result.data.startswith("select:")
    # Two find_form_target calls (one to open, one to pick).
    assert extractor.find_form_target.call_count == 2
    # Two clicks: the dropdown control + the option.
    click_actions = [a for a in env.actions if a.action_type == ActionType.CLICK]
    assert len(click_actions) == 2
    assert click_actions[0].params["x"] == 300  # dropdown
    assert click_actions[1].params["x"] == 320  # option


def test_select_option_option_not_found_dismisses_open_dropdown():
    """Open succeeded but option missing — runner sends Escape to close cleanly."""
    env = _FakeEnv()
    extractor = MagicMock()
    extractor.find_form_target.side_effect = [
        {"x": 300, "y": 100, "action": "click", "value": "", "label": "Industry"},
        None,  # option not found
    ]
    runner = _runner_with_extractor(env, extractor)

    intent = MicroIntent(
        intent="Pick X",
        type="select_option",
        params={"dropdown_label": "Industry", "option_label": "X"},
    )
    result = runner._execute_claude_guided_form(intent, index=0)

    assert result.success is False
    assert result.data == "option_not_found"
    # An Escape key-press should have been dispatched to close the dropdown.
    keys = [
        a.params.get("keys") for a in env.actions
        if a.action_type == ActionType.KEY_PRESS
    ]
    assert "Escape" in keys


# ── _execute_step routes new types to the form dispatch ────────────────


def test_execute_step_routes_fill_field_to_form_dispatch():
    """The wiring: when _execute_step sees fill_field, it dispatches to
    _execute_claude_guided_form (not _execute_holo3_step, not _execute_claude_step)."""
    env = _FakeEnv()
    extractor = MagicMock()
    extractor.find_form_target.return_value = {
        "x": 1, "y": 1, "action": "click", "value": "", "label": "x",
    }
    runner = _runner_with_extractor(env, extractor)

    # Spy on the dispatch so we don't run the actual form logic — just verify it's reached.
    runner._execute_claude_guided_form = MagicMock(  # type: ignore[method-assign]
        return_value=StepResult(step_index=0, intent="x", success=True),
    )

    intent = MicroIntent(intent="x", type="fill_field", params={"label": "x", "value": "y"})
    result = runner._execute_step(intent, index=0)

    runner._execute_claude_guided_form.assert_called_once()
    assert result.success is True


def test_run_loop_preserves_params_on_effective_step():
    """Regression: the run loop builds an `effective_step` per iteration that
    only copies a subset of MicroIntent fields. ``params`` must be in that
    subset — without it, every form step lands at the dispatch with an empty
    params dict and no value to type / button label to find. Caught in
    production E2E against staffai-test-crm where login fired but typed empty
    strings into the User ID and Password fields.
    """
    env = _FakeEnv()
    extractor = MagicMock()
    extractor.find_form_target.return_value = {
        "x": 100, "y": 50, "action": "click", "value": "", "label": "User ID",
    }
    runner = _runner_with_extractor(env, extractor)

    # Capture every dispatch invocation so we can inspect the effective_step.
    dispatched: list[MicroIntent] = []

    def spy(step: MicroIntent, index: int) -> StepResult:
        dispatched.append(step)
        return StepResult(step_index=index, intent=step.intent, success=True)

    runner._execute_claude_guided_form = spy  # type: ignore[method-assign]

    plan = MicroPlan(domain="test")
    plan.steps.append(
        MicroIntent(
            intent="Enter the user ID",
            type="fill_field",
            section="setup",
            required=False,  # avoid required-retry path
            params={"label": "User ID", "value": "sarah.connor"},
        )
    )

    runner.run(plan)

    assert len(dispatched) == 1
    effective = dispatched[0]
    assert effective.params == {"label": "User ID", "value": "sarah.connor"}
