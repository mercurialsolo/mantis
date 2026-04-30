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


def test_microintent_hints_round_trip_through_build_intent():
    """hints is a free-form per-step grounding context the plan supplies to
    the runner. Used to drive the click-dispatch decision without baking
    any layout assumption into the runner. Defaults to {}."""
    default = PlanDecomposer._build_intent({"intent": "x", "type": "click"})
    assert default.hints == {}

    hinted = PlanDecomposer._build_intent({
        "intent": "Click the next lead row",
        "type": "click",
        "hints": {"layout": "listings", "spam_label": "broker"},
    })
    assert hinted.hints == {"layout": "listings", "spam_label": "broker"}


def test_microintent_hints_rejects_non_dict():
    """A misshapen `hints` (string from a buggy decomposer) drops to {}."""
    intent = PlanDecomposer._build_intent({
        "intent": "x", "type": "click", "hints": "not-a-dict",
    })
    assert intent.hints == {}


def test_click_with_layout_single_routes_to_form_dispatch():
    """A click step with hints={"layout": "single"} must NOT use
    find_all_listings (the listings extractor) — it must go through
    find_form_target instead. This is the plan-driven dispatch contract."""
    env = _FakeEnv()
    extractor = MagicMock()
    extractor.find_form_target.return_value = {
        "x": 100, "y": 50, "action": "click", "value": "", "label": "Save Settings",
    }
    runner = _runner_with_extractor(env, extractor)
    runner._ensure_results_filters = MagicMock(return_value=True)  # type: ignore[method-assign]

    intent = MicroIntent(
        intent="Click the Save Settings button",
        type="click",
        budget=4,
        section="setup",  # would normally NOT route to form, but layout hint overrides
        params={"label": "Save Settings"},
        hints={"layout": "single"},
    )
    result = runner._execute_step(intent, index=0)

    assert result.success is True
    extractor.find_form_target.assert_called()
    extractor.find_all_listings.assert_not_called()


def test_click_with_no_hint_in_extraction_section_uses_listings_dispatch():
    """A click step with no layout hint in section=extraction is the
    canonical listings-flow case and must keep routing through
    find_all_listings — no regression on existing BoatTrader-style plans."""
    env = _FakeEnv()
    extractor = MagicMock()
    runner = _runner_with_extractor(env, extractor)
    runner._ensure_results_filters = MagicMock(return_value=True)  # type: ignore[method-assign]
    runner._execute_claude_guided_click = MagicMock(  # type: ignore[method-assign]
        return_value=StepResult(step_index=0, intent="x", success=True),
    )

    intent = MicroIntent(
        intent="Click the next un-extracted listing",
        type="click",
        budget=8,
        section="extraction",
    )
    result = runner._execute_step(intent, index=0)

    runner._execute_claude_guided_click.assert_called_once()
    assert result.success is True


def test_decompose_prompt_template_substitutes_without_format_keyerror():
    """The DECOMPOSE_PROMPT contains literal `params={"label": ...}` JSON
    examples. ``str.format()`` would interpret those `{` as field
    placeholders and raise KeyError on every decompose call. The decomposer
    uses ``str.replace()`` instead.

    This test pins the contract: building the prompt via the documented
    mechanism (replace ``{plan_text}``) must produce the complete prompt
    with the user's plan substituted, AND ``str.format()`` would have
    crashed if used. Caught when a CRM canary failed with
    ``KeyError: '"label"'`` against the deployed v10 decomposer.
    """
    from mantis_agent.plan_decomposer import DECOMPOSE_PROMPT

    plan = "Step 1: do a thing\nStep 2: do another thing"
    rendered = DECOMPOSE_PROMPT.replace("{plan_text}", plan)
    assert plan in rendered
    assert "{plan_text}" not in rendered

    with pytest.raises(KeyError):
        DECOMPOSE_PROMPT.format(plan_text=plan)


def test_submit_scrolls_to_find_below_fold_button():
    """Issue #89 §2: long forms render the primary submit below the fold;
    the previous single-screenshot path declared it missing without ever
    scrolling. The new path Page_Downs and re-asks find_form_target up to
    4 times before giving up."""
    env = _FakeEnv()
    extractor = MagicMock()
    # First two screenshots: button not in viewport. Third: found.
    extractor.find_form_target.side_effect = [
        None,
        None,
        {"x": 640, "y": 480, "action": "click", "value": "", "label": "Update Lead"},
    ]
    runner = _runner_with_extractor(env, extractor)

    intent = MicroIntent(
        intent="Click Update Lead",
        type="submit",
        params={"label": "Update Lead"},
    )
    result = runner._execute_claude_guided_form(intent, index=0)

    assert result.success is True
    assert result.data.startswith("submit:")
    # Three find_form_target calls: initial + 2 after Page_Down.
    assert extractor.find_form_target.call_count == 3
    # Two Page_Down keypresses + one terminal click.
    page_downs = [
        a for a in env.actions
        if a.action_type == ActionType.KEY_PRESS and a.params.get("keys") == "Page_Down"
    ]
    assert len(page_downs) == 2
    clicks = [a for a in env.actions if a.action_type == ActionType.CLICK]
    assert len(clicks) == 1


def test_submit_gives_up_after_max_scrolls():
    """When the button truly isn't on the page, we cap scrolling so the
    runner doesn't loop forever. Cap is 4 scrolls — initial + 4 = 5 total
    find_form_target calls before giving up."""
    env = _FakeEnv()
    extractor = MagicMock()
    extractor.find_form_target.return_value = None  # never found
    runner = _runner_with_extractor(env, extractor)

    intent = MicroIntent(
        intent="Click Update Lead",
        type="submit",
        params={"label": "Update Lead"},
    )
    result = runner._execute_claude_guided_form(intent, index=0)

    assert result.success is False
    assert result.data == "form_target_not_found"
    assert extractor.find_form_target.call_count == 5  # 1 initial + 4 scrolls
    # Final Home press resets scroll for the next step.
    home_presses = [
        a for a in env.actions
        if a.action_type == ActionType.KEY_PRESS and a.params.get("keys") == "Home"
    ]
    assert len(home_presses) == 1


def test_submit_aliases_passed_to_grounder():
    """The decomposer can emit ``params["aliases"]`` for primary submit
    buttons whose copy varies across products. The runner must forward
    these so claude-form can match on any of them."""
    env = _FakeEnv()
    extractor = MagicMock()
    extractor.find_form_target.return_value = {
        "x": 640, "y": 480, "action": "click", "value": "", "label": "Save Changes",
    }
    runner = _runner_with_extractor(env, extractor)

    intent = MicroIntent(
        intent="Click Update Lead",
        type="submit",
        params={
            "label": "Update Lead",
            "aliases": ["Update", "Save", "Save Changes"],
        },
    )
    result = runner._execute_claude_guided_form(intent, index=0)

    assert result.success is True
    # The grounder was told about the aliases.
    call = extractor.find_form_target.call_args
    assert call.kwargs["target_aliases"] == ["Update", "Save", "Save Changes"]


def test_form_aliases_string_normalised_to_list():
    """A misshapen ``aliases: "Save"`` (string instead of list) should still
    work — wrap to a single-element list rather than raising."""
    env = _FakeEnv()
    extractor = MagicMock()
    extractor.find_form_target.return_value = {
        "x": 640, "y": 480, "action": "click", "value": "", "label": "Save",
    }
    runner = _runner_with_extractor(env, extractor)

    intent = MicroIntent(
        intent="Click Update Lead",
        type="submit",
        params={"label": "Update Lead", "aliases": "Save"},
    )
    result = runner._execute_claude_guided_form(intent, index=0)

    assert result.success is True
    assert extractor.find_form_target.call_args.kwargs["target_aliases"] == ["Save"]


def test_read_current_url_prefers_cdp_over_screenshot():
    """Issue #89 §1: the ``(url=)`` empty-string halt was caused by reading
    the address bar from screenshot pixels. _read_current_url must prefer
    env.current_url (CDP) and fall back to screenshot OCR only when CDP
    is unreachable."""
    env = _FakeEnv()
    extractor = MagicMock()
    runner = _runner_with_extractor(env, extractor)

    # CDP path: env.current_url returns the URL, no extractor call needed.
    env.current_url_value = "https://example.com/leads/42"
    type(env).current_url = property(lambda self: getattr(self, "current_url_value", ""))

    url = runner._read_current_url()
    assert url == "https://example.com/leads/42"
    extractor.extract.assert_not_called()


def test_read_current_url_calls_method_property_for_buggy_envs():
    """Some integrations define current_url as a method instead of a
    @property (e.g. ``def current_url(self): return ...``). Attribute
    access then returns the bound method, which is truthy — without this
    guard the helper would return a method object as the URL.

    Regression: post-PR-90 the staffai trace still showed ``(url=)`` empty;
    one viable cause was their VisionClaudeGymEnv defining current_url as
    a method, which short-circuits the truthy check on the bound method.
    """
    env = _FakeEnv()
    runner = _runner_with_extractor(env, extractor=MagicMock())

    # Define current_url as a callable on the env instance.
    def fake_method():
        return "https://example.com/leads/42"

    type(env).current_url = property(lambda self: fake_method)
    url = runner._read_current_url()
    assert url == "https://example.com/leads/42"


def test_read_current_url_falls_back_to_screenshot_when_cdp_empty():
    """When CDP returns empty (port not bound, container without
    --remote-debugging-port), fall back to the existing screenshot-based
    URL extractor. Pins the backward-compatible fallback path."""
    env = _FakeEnv()
    extractor = MagicMock()
    fake_extract = MagicMock()
    fake_extract.url = "https://example.com/leads/42"
    extractor.extract.return_value = fake_extract
    runner = _runner_with_extractor(env, extractor)

    type(env).current_url = property(lambda self: "")

    url = runner._read_current_url(env._img)
    assert url == "https://example.com/leads/42"
    extractor.extract.assert_called_once()


def test_navigate_wait_override_via_params(monkeypatch: pytest.MonkeyPatch):
    """Per-step params["wait_after_load_seconds"] beats the env override.

    Lets a plan say "this navigate hits a slow proxied SPA, wait 35s before
    Holo3 reads the page" without globally bumping the deployment-wide wait.
    """
    env = _FakeEnv()
    runner = _runner_with_extractor(env, extractor=MagicMock())

    captured: list[float] = []
    monkeypatch.setattr("mantis_agent.gym.micro_runner.time.sleep", captured.append)
    # Env override would push to 60 if param were ignored.
    monkeypatch.setenv("MANTIS_NAV_WAIT_SECONDS", "60")

    intent = MicroIntent(
        intent="Go to https://example.com",
        type="navigate",
        params={"wait_after_load_seconds": 35},
    )
    result = runner._execute_navigate(intent, index=0)

    assert result.success is True
    # First sleep is the first-paint wait. Param=35 wins over env=60.
    assert captured[0] == 35.0


def test_navigate_wait_override_via_env(monkeypatch: pytest.MonkeyPatch):
    """MANTIS_NAV_WAIT_SECONDS bumps every navigate when no per-step param.

    Useful for the staffai canary's proxied-CRM cold-start where the splash
    runs longer than 18s. Set once at deploy time, takes effect everywhere.
    """
    env = _FakeEnv()
    runner = _runner_with_extractor(env, extractor=MagicMock())

    captured: list[float] = []
    monkeypatch.setattr("mantis_agent.gym.micro_runner.time.sleep", captured.append)
    monkeypatch.setenv("MANTIS_NAV_WAIT_SECONDS", "30")

    intent = MicroIntent(intent="Go to https://example.com", type="navigate")
    result = runner._execute_navigate(intent, index=0)

    assert result.success is True
    assert captured[0] == 30.0


def test_navigate_wait_default_unchanged(monkeypatch: pytest.MonkeyPatch):
    """No param + no env override → 18s default. Pins the BoatTrader pipeline
    timing so the new override knob doesn't silently shift existing flows."""
    env = _FakeEnv()
    runner = _runner_with_extractor(env, extractor=MagicMock())

    captured: list[float] = []
    monkeypatch.setattr("mantis_agent.gym.micro_runner.time.sleep", captured.append)
    monkeypatch.delenv("MANTIS_NAV_WAIT_SECONDS", raising=False)

    intent = MicroIntent(intent="Go to https://example.com", type="navigate")
    result = runner._execute_navigate(intent, index=0)

    assert result.success is True
    assert captured[0] == 18.0


def test_navigate_wait_clamped_to_safe_range(monkeypatch: pytest.MonkeyPatch):
    """Garbage / extreme values clamp to [0, 120] — no infinite hang from a
    typo'd plan, no zero-wait race from a negative number."""
    env = _FakeEnv()
    runner = _runner_with_extractor(env, extractor=MagicMock())

    captured: list[float] = []
    monkeypatch.setattr("mantis_agent.gym.micro_runner.time.sleep", captured.append)
    monkeypatch.delenv("MANTIS_NAV_WAIT_SECONDS", raising=False)

    intent = MicroIntent(
        intent="Go to https://example.com",
        type="navigate",
        params={"wait_after_load_seconds": 999},
    )
    runner._execute_navigate(intent, index=0)
    assert captured[0] == 120.0

    captured.clear()
    intent = MicroIntent(
        intent="Go to https://example.com",
        type="navigate",
        params={"wait_after_load_seconds": -5},
    )
    runner._execute_navigate(intent, index=0)
    assert captured[0] == 0.0


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
