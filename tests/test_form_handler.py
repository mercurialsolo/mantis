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
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Fill in email", type="fill_field",
        params={"label": "Email", "value": "x@y.com"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is False
    assert result.data == "form_target_not_found"
    # 1 initial label-match + scroll-top probe + 5 scroll-down sweeps +
    # 1 visual-affordance fallback = 8 Claude calls. The scroll-probe
    # path mirrors `submit`'s End/Home/Page_Down sweep so a fill_field
    # whose target is below the visible fold doesn't go straight to
    # the affordance fallback (which historically clicked the form's
    # submit button by mistake on lu.ma-style host-question modals).
    assert runner.costs["claude_extract"] == 8
    # Verify the scroll-probe dispatched mouse-wheel SCROLL actions
    # (not KEY_PRESS) so input focus from a prior fill_field doesn't
    # absorb the page-scroll keystrokes.
    scroll_calls = [
        c for c in env.step.call_args_list
        if c.args[0].action_type == ActionType.SCROLL
    ]
    assert len(scroll_calls) == 6  # 1 scroll-top + 5 scroll-down
    assert scroll_calls[0].args[0].params == {"direction": "up", "amount": 10}
    for c in scroll_calls[1:]:
        assert c.args[0].params == {"direction": "down", "amount": 3}


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


def test_fill_field_scroll_probe_finds_target_after_scroll_down(monkeypatch):
    """When the labelled input isn't in the initial viewport but
    becomes visible after a downward scroll, fill_field's scroll-probe
    locates it and proceeds with the click+type — no affordance
    fallback fires. Mirrors the lu.ma host-question modal where the
    third question is below the visible fold."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform",
        lambda a, b: a,
    )

    runner = _FakeRunner()
    extractor = MagicMock()
    # Initial miss; scroll-top miss; first scroll-down hit.
    extractor.find_form_target.side_effect = [
        None,  # initial
        None,  # scroll-top
        {"x": 600, "y": 480},  # scroll-down-1 hit
    ]
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Fill title", type="fill_field",
        params={"label": "What is your title?", "value": "AI Product Tester"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is True
    assert result.data == "fill:What is your title?"
    # 3 find_form_target calls (initial + scroll-top + scroll-down-1).
    # The affordance fallback must NOT fire when scroll-probe succeeds.
    assert runner.costs["claude_extract"] == 3
    extractor.find_target_by_affordance.assert_not_called()
    # Two SCROLL actions dispatched: scroll-top then scroll-down-1.
    scroll_calls = [
        c for c in env.step.call_args_list
        if c.args[0].action_type == ActionType.SCROLL
    ]
    assert len(scroll_calls) == 2
    assert scroll_calls[0].args[0].params == {"direction": "up", "amount": 10}
    assert scroll_calls[1].args[0].params == {"direction": "down", "amount": 3}


def test_fill_field_affordance_refuses_button_action(monkeypatch):
    """The affordance fallback returning ``action=click`` (button-
    shaped) is the canonical lu.ma failure: with no visible input,
    Claude returns the most prominent thing left — the form's submit
    button. fill_field must refuse and report form_target_not_found
    rather than dispatch a click that submits a half-filled form."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.return_value = None
    extractor.find_target_by_affordance.return_value = {
        "x": 635, "y": 570,
        "action": "click",
        "label": "Register",
        "value": "",
    }
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Fill title", type="fill_field",
        params={"label": "What is your title?", "value": "AI Product Tester"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is False
    assert result.data == "form_target_not_found"
    # No CLICK action dispatched — the submit button was NOT clicked.
    click_calls = [
        c for c in env.step.call_args_list
        if c.args[0].action_type == ActionType.CLICK
    ]
    assert click_calls == []


def test_fill_field_affordance_accepts_type_action(monkeypatch):
    """Counterpart to the refuse test: when the affordance pass
    returns ``action=type`` (correctly identified as a text input by
    visual shape rather than label), fill_field proceeds normally."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform",
        lambda a, b: a,
    )

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.return_value = None
    extractor.find_target_by_affordance.return_value = {
        "x": 400, "y": 250,
        "action": "type",
        "label": "Le titre",  # non-English label, affordance match
        "value": "",
    }
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Fill title", type="fill_field",
        params={"label": "Title", "value": "AI Product Tester"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is True
    assert result.data == "fill:Title"


def test_fill_field_tag_guard_refuses_button_at_click_point(monkeypatch):
    """find_form_target returned coords that land on a BUTTON element
    (most likely cause: stale grounding / a target whose center
    overlaps the modal submit button). The tag-guard probes the
    element at (x, y) via CDP and refuses the click rather than
    dispatching a SoM ``el.click()`` that would submit the form."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.return_value = {"x": 635, "y": 570}
    env = MagicMock()
    # Wire CDP eval to simulate "the element at this point is a BUTTON".
    env.cdp_evaluate = MagicMock(return_value={"tag": "BUTTON", "contentEditable": False})
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Fill title", type="fill_field",
        params={"label": "Title", "value": "AI Product Tester"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is False
    assert result.data == "form_target_not_input:BUTTON"
    # No CLICK / TYPE dispatched.
    blocked_types = {ActionType.CLICK, ActionType.TYPE}
    assert not any(
        c.args[0].action_type in blocked_types
        for c in env.step.call_args_list
    )


def test_fill_field_tag_guard_allows_input_at_click_point(monkeypatch):
    """Counterpart: when the element at (x, y) IS an INPUT, the guard
    passes through and the click+type sequence proceeds."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform",
        lambda a, b: a,
    )

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.return_value = {"x": 200, "y": 300}
    env = MagicMock()
    env.cdp_evaluate = MagicMock(return_value={"tag": "INPUT", "contentEditable": False})
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Fill email", type="fill_field",
        params={"label": "Email", "value": "u@e.com"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is True
    assert result.data == "fill:Email"


def test_fill_field_tag_guard_records_recovery_hint(monkeypatch):
    """#411: when the tag-guard refuses a BUTTON pick, the bad coord +
    tag must land on ``runner._recovery_hints[index]`` so the *next*
    attempt's find_form_target prompt tells the LLM not to re-pick
    the same wrong rectangle. Without this hint, Claude on lu.ma's
    'Your Info' modal keeps returning (618, 587) for the Title field
    on every retry and burns the step budget on identical refusals."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.return_value = {"x": 618, "y": 587}
    env = MagicMock()
    env.cdp_evaluate = MagicMock(return_value={"tag": "BUTTON", "contentEditable": False})
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Fill title", type="fill_field",
        params={"label": "What is your title?", "value": "AI Product Tester"},
    )
    handler = ClaudeGuidedFormHandler(runner)
    result = handler.execute(step, ctx)

    assert result.success is False
    assert result.data == "form_target_not_input:BUTTON"
    # The hint must be recorded on the runner — keyed by the step
    # index supplied via ctx.state["index"] (= 5 in _ctx).
    hints = getattr(runner, "_recovery_hints", {})
    assert 5 in hints
    body = "\n".join(hints[5])
    # Hint mentions the rejected coord, the tag, and gives the LLM
    # something specific to avoid + something specific to look for.
    assert "(618, 587)" in body
    assert "<BUTTON>" in body
    assert "What is your title?" in body
    # Bounding rectangle around the rejected coord is in the hint so
    # the LLM has a concrete region to steer away from.
    assert "(578, 567)" in body or "(578, 567)" in body  # x-40, y-20


def test_fill_field_tag_guard_retry_picks_up_hint(monkeypatch):
    """End-to-end: a prior tag-guard refusal stored a hint. On the
    next call to ``execute`` for the same step (= same ctx index),
    the search prompt passed to find_form_target carries the hint.
    Locks the consumer side of the hint loop so a future refactor
    doesn't silently drop the recovery feedback."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform",
        lambda a, b: a,
    )

    runner = _FakeRunner()
    # Seed the runner with a hint from a previous failed attempt.
    runner._recovery_hints = {5: [
        "Your previous coordinate pick for 'Title' was (618, 587), "
        "which document.elementFromPoint resolves to a <BUTTON> "
        "element. DO NOT return coordinates inside the box (578, 567) "
        "to (658, 607). The input you want is almost certainly ABOVE …"
    ]}
    extractor = MagicMock()
    # Second attempt now finds the *correct* coord above the button.
    extractor.find_form_target.return_value = {"x": 400, "y": 480}
    env = MagicMock()
    env.cdp_evaluate = MagicMock(return_value={"tag": "INPUT", "contentEditable": False})
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Fill title", type="fill_field",
        params={"label": "Title", "value": "AI Product Tester"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is True
    # The hint flowed into find_form_target's search_intent — the
    # extractor saw the augmented prompt, not the bare label.
    sent_intent = extractor.find_form_target.call_args.args[1]
    assert "RECOVERY HINTS" in sent_intent
    assert "(618, 587)" in sent_intent


def test_fill_field_tag_guard_allows_contenteditable_div(monkeypatch):
    """Rich text editors (Quill, ProseMirror, …) render as ``<div
    contenteditable="true">``. The tag-guard must accept them — the
    INPUT_LIKE_TAGS allow-list alone would reject DIV."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform",
        lambda a, b: a,
    )

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.return_value = {"x": 200, "y": 300}
    env = MagicMock()
    env.cdp_evaluate = MagicMock(return_value={"tag": "DIV", "contentEditable": True})
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Fill bio", type="fill_field",
        params={"label": "Bio", "value": "hello"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is True


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


# ── right_click (#373) ──────────────────────────────────────────────


def test_right_click_dispatches_button_right_at_target(monkeypatch):
    """right_click finds the labelled target via find_form_target,
    then dispatches a single CLICK Action with ``button="right"``
    at the target's center — no SoM dispatch, no URL-change check."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform",
        lambda a, b: a,
    )

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.return_value = {"x": 320, "y": 480}
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Right-click on Open Link in New Tab",
        type="right_click",
        params={"label": "Listing card #2"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is True
    assert result.data == "right_click:Listing card #2@(320,480)"
    # Single env.step call: the right-mouse-button click.
    click_calls = [
        c for c in env.step.call_args_list
        if c.args[0].action_type == ActionType.CLICK
    ]
    assert len(click_calls) == 1
    params = click_calls[0].args[0].params
    assert params == {"x": 320, "y": 480, "button": "right"}
    # One find_form_target call billed; no Holo3 grounding.
    assert runner.costs["claude_extract"] == 1
    assert runner.costs["gpu_steps"] == 1


def test_right_click_target_not_found_returns_form_target_not_found(monkeypatch):
    """When find_form_target misses, right_click returns the same
    ``form_target_not_found`` envelope as the other form variants
    so the executor / recovery path can react uniformly."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.return_value = None
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Right-click on Save",
        type="right_click",
        params={"label": "Save"},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is False
    assert result.data == "form_target_not_found"
    # ``selector_miss`` lets the recovery / rewrite path see the
    # specific failure mode (same vocabulary as failure_class.py).
    assert result.failure_class == "selector_miss"
    # No click was dispatched.
    click_calls = [
        c for c in env.step.call_args_list
        if c.args[0].action_type == ActionType.CLICK
    ]
    assert click_calls == []


def test_right_click_uses_intent_string_when_no_label(monkeypatch):
    """right_click without ``params.label`` falls back to the step
    intent verbatim (same shape as the other form variants)."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform",
        lambda a, b: a,
    )

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.return_value = {"x": 50, "y": 60}
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    step = MicroIntent(
        intent="Right-click on the first table row to open its menu",
        type="right_click",
        params={},
    )
    result = ClaudeGuidedFormHandler(runner).execute(step, ctx)

    assert result.success is True
    # find_form_target invoked with the raw intent (no label-templated
    # framing). Inspect the search_intent positional argument.
    call_args = extractor.find_form_target.call_args
    search_intent = call_args.args[1]
    assert "first table row" in search_intent


# ── PR-E: Tab-walk fallback for nav_link kind ────────────────────────


def test_tab_walk_returns_match_when_anchor_found_in_budget(monkeypatch):
    """Tab-walk presses Tab, inspects focused element via cdp_evaluate,
    returns a match dict when an `<a>` element's accessible name matches.
    """
    from mantis_agent.gym.step_handlers.form import _tab_walk_to_nav_link

    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)

    env = MagicMock()
    # Sequence of focused elements per cdp_evaluate call: a few non-matching
    # focusables, then the target Contacted anchor on the 3rd Tab.
    env.cdp_evaluate.side_effect = [
        {"tag": "INPUT", "name": "search"},
        {"tag": "BUTTON", "name": "Logout"},
        {"tag": "A", "name": "Contacted (0)"},
    ]

    match = _tab_walk_to_nav_link(env, "Contacted")

    assert match is not None
    assert match["tabs"] == 3
    assert match["tag"] == "A"
    assert "Contacted" in match["name"]


def test_tab_walk_returns_none_when_no_match_within_budget(monkeypatch):
    """When max_tabs Tabs pass without a matching anchor, return None."""
    from mantis_agent.gym.step_handlers.form import _tab_walk_to_nav_link

    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)

    env = MagicMock()
    env.cdp_evaluate.return_value = {"tag": "INPUT", "name": "anything else"}

    match = _tab_walk_to_nav_link(env, "Contacted", max_tabs=5)

    assert match is None
    # cdp_evaluate called exactly max_tabs times (5)
    assert env.cdp_evaluate.call_count == 5


def test_tab_walk_returns_none_when_env_lacks_cdp_evaluate(monkeypatch):
    """No cdp_evaluate → return None immediately (no fallback to vision)."""
    from mantis_agent.gym.step_handlers.form import _tab_walk_to_nav_link

    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)

    # MagicMock auto-creates attrs; spec= constrains it
    env = MagicMock(spec=[])
    match = _tab_walk_to_nav_link(env, "Contacted")
    assert match is None


def test_tab_walk_empty_label_returns_none(monkeypatch):
    """Empty / whitespace-only label is unwalkable — return None."""
    from mantis_agent.gym.step_handlers.form import _tab_walk_to_nav_link

    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)

    env = MagicMock()
    assert _tab_walk_to_nav_link(env, "") is None
    assert _tab_walk_to_nav_link(env, "   ") is None
    # cdp_evaluate never queried — short-circuit before the loop
    env.cdp_evaluate.assert_not_called()


def test_tab_walk_case_insensitive_substring_match(monkeypatch):
    """Match is case-insensitive and substring: "contacted" matches "Contacted (0)"."""
    from mantis_agent.gym.step_handlers.form import _tab_walk_to_nav_link

    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)

    env = MagicMock()
    env.cdp_evaluate.side_effect = [
        {"tag": "A", "name": "Contacted (0)"},  # matches "contacted" substring
    ]

    match = _tab_walk_to_nav_link(env, "contacted")

    assert match is not None
    assert match["tabs"] == 1


def test_tab_walk_rejects_non_anchor_match(monkeypatch):
    """A BUTTON named "Contacted" doesn't count — only `<a>` elements
    match. Avoids accidentally pressing Enter on a button-shaped item
    that doesn't behave like a nav link.
    """
    from mantis_agent.gym.step_handlers.form import _tab_walk_to_nav_link

    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)

    env = MagicMock()
    env.cdp_evaluate.side_effect = [
        {"tag": "BUTTON", "name": "Contacted"},  # right name, wrong tag
        {"tag": "A", "name": "Contacted (0)"},   # correct
    ]

    match = _tab_walk_to_nav_link(env, "Contacted")

    assert match is not None
    assert match["tabs"] == 2


def test_tab_walk_handles_cdp_evaluate_exception(monkeypatch):
    """cdp_evaluate raising on one iteration shouldn't abort the walk —
    skip the iteration and continue tabbing.
    """
    from mantis_agent.gym.step_handlers.form import _tab_walk_to_nav_link

    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)

    env = MagicMock()
    env.cdp_evaluate.side_effect = [
        RuntimeError("transient CDP read fail"),
        {"tag": "A", "name": "Contacted (0)"},
    ]

    match = _tab_walk_to_nav_link(env, "Contacted")

    assert match is not None
    assert match["tabs"] == 2
