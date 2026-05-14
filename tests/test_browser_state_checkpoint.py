"""Browser-state checkpoint — epic #358 Phase A.

Pins:
- BrowserState dataclass + JSON round-trip on PauseState
- env.capture_browser_state() / env.restore_browser_state() contract
- build_runner_result snapshots browser_state into the PauseState
- MicroPlanRunner.resume + GymRunner.resume call restore_browser_state
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from mantis_agent.gym.checkpoint import BrowserState, FormFieldValue, PauseState


# ── BrowserState defaults + JSON round-trip ─────────────────────────────


def test_browser_state_defaults_are_empty() -> None:
    bs = BrowserState()
    assert bs.url == ""
    assert bs.scroll_x == 0
    assert bs.scroll_y == 0
    assert bs.viewport_w == 0
    assert bs.viewport_h == 0
    assert bs.captured_at == 0.0


def test_pause_state_carries_browser_state_field() -> None:
    bs = BrowserState(url="https://example.com/page", scroll_y=1500, viewport_h=900)
    ps = PauseState(browser_state=bs)
    assert ps.browser_state.url == "https://example.com/page"
    assert ps.browser_state.scroll_y == 1500


def test_pause_state_json_round_trip_with_browser_state() -> None:
    bs = BrowserState(
        url="https://luma.com/discover", scroll_x=0, scroll_y=2400,
        viewport_w=1280, viewport_h=720, captured_at=1747244400.0,
    )
    ps = PauseState(
        run_key="r", plan_signature="sig", session_name="s",
        browser_state=bs,
    )
    payload = json.dumps(ps.to_dict())
    rehydrated = PauseState.from_dict(json.loads(payload))
    assert rehydrated.browser_state.url == bs.url
    assert rehydrated.browser_state.scroll_y == 2400
    assert rehydrated.browser_state.viewport_w == 1280
    assert isinstance(rehydrated.browser_state, BrowserState)


def test_pause_state_legacy_dict_without_browser_state_field() -> None:
    """Snapshots written before Phase A landed must rehydrate
    cleanly with a default empty BrowserState — no AttributeError,
    no KeyError."""
    legacy = {
        "version": 1, "run_key": "r", "plan_signature": "sig",
        "session_name": "s", "step_index": 0, "pending_tool": "",
        "pending_arguments": {}, "pending_reason": "user_input",
        "prompt": "", "step_results": [], "loop_counters": {},
        "listings_on_page": 0, "checkpoint_path": "",
        "timestamp": 0.0, "trajectory_steps": [], "task": "",
        "task_id": "",
        # NO browser_state key.
    }
    ps = PauseState.from_dict(legacy)
    assert ps.browser_state == BrowserState()


def test_pause_state_browser_state_null_value() -> None:
    """A legacy JSON snapshot that explicitly set ``browser_state``
    to None (rare but possible) should rehydrate to the default
    empty BrowserState, not crash."""
    payload = {
        "run_key": "r", "plan_signature": "sig",
        "session_name": "s", "browser_state": None,
    }
    ps = PauseState.from_dict(payload)
    assert ps.browser_state == BrowserState()


# ── env.capture_browser_state (xdotool path) ────────────────────────────


def test_xdotool_capture_browser_state_via_cdp_evaluate() -> None:
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    env = object.__new__(XdotoolGymEnv)
    env.cdp_evaluate = MagicMock(side_effect=[
        {  # URL/scroll/viewport eval
            "url": "https://example.com/page",
            "scroll_x": 0,
            "scroll_y": 1800,
            "viewport_w": 1280,
            "viewport_h": 720,
        },
        [],  # form-field eval (empty page)
    ])
    bs = env.capture_browser_state()
    assert bs.url == "https://example.com/page"
    assert bs.scroll_y == 1800
    assert bs.viewport_w == 1280
    # Two CDP calls: primitives + form-state walk.
    assert env.cdp_evaluate.call_count == 2
    first_expr = env.cdp_evaluate.call_args_list[0].args[0]
    assert "location.href" in first_expr
    assert "window.scrollX" in first_expr


def test_xdotool_capture_handles_cdp_failure() -> None:
    """CDP unreachable / page mid-navigation returns None from the
    URL eval — capture must yield an empty BrowserState, not crash.
    The form-state eval is skipped entirely (no point capturing
    fields when we don't even know the URL)."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    env = object.__new__(XdotoolGymEnv)
    env.cdp_evaluate = MagicMock(return_value=None)
    bs = env.capture_browser_state()
    assert bs.url == ""
    assert bs.captured_at > 0  # still timestamped


# ── env.restore_browser_state (xdotool path) ────────────────────────────


def test_xdotool_restore_calls_reset_and_scrolls() -> None:
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    env = object.__new__(XdotoolGymEnv)
    env.reset = MagicMock()
    env.cdp_evaluate = MagicMock()

    env.restore_browser_state(BrowserState(
        url="https://example.com/page", scroll_x=0, scroll_y=2000,
    ))

    env.reset.assert_called_once()
    reset_kwargs = env.reset.call_args.kwargs
    assert reset_kwargs.get("start_url") == "https://example.com/page"
    # And scrollTo JS evaluation.
    env.cdp_evaluate.assert_called_once()
    js = env.cdp_evaluate.call_args.args[0]
    assert "window.scrollTo(0, 2000)" in js


def test_xdotool_restore_no_op_on_empty_url() -> None:
    """Empty url → no env.reset, no scrolling. The runner picks up
    from whatever URL the env happens to have."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    env = object.__new__(XdotoolGymEnv)
    env.reset = MagicMock()
    env.cdp_evaluate = MagicMock()

    env.restore_browser_state(BrowserState())  # url=""

    env.reset.assert_not_called()
    env.cdp_evaluate.assert_not_called()


def test_xdotool_restore_skips_scroll_when_zero() -> None:
    """No-scroll case: reset still fires, scrollTo doesn't."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    env = object.__new__(XdotoolGymEnv)
    env.reset = MagicMock()
    env.cdp_evaluate = MagicMock()

    env.restore_browser_state(BrowserState(
        url="https://example.com/page", scroll_x=0, scroll_y=0,
    ))

    env.reset.assert_called_once()
    env.cdp_evaluate.assert_not_called()


# ── Helper: _capture_browser_state_safe ────────────────────────────────


def test_capture_helper_returns_empty_when_env_lacks_method() -> None:
    from mantis_agent.gym._runner_helpers import _capture_browser_state_safe

    runner = MagicMock()
    runner.env = MagicMock(spec=[])  # No capture_browser_state.
    bs = _capture_browser_state_safe(runner)
    assert bs == BrowserState()


def test_capture_helper_returns_empty_on_exception() -> None:
    from mantis_agent.gym._runner_helpers import _capture_browser_state_safe

    runner = MagicMock()
    runner.env.capture_browser_state.side_effect = RuntimeError("CDP down")
    bs = _capture_browser_state_safe(runner)
    assert bs == BrowserState()


def test_capture_helper_returns_env_result_when_callable() -> None:
    from mantis_agent.gym._runner_helpers import _capture_browser_state_safe

    runner = MagicMock()
    expected = BrowserState(url="https://x", scroll_y=42)
    runner.env.capture_browser_state.return_value = expected
    assert _capture_browser_state_safe(runner) is expected


# ── Phase B: form-field capture / replay ────────────────────────────────


def test_form_field_value_defaults() -> None:
    ffv = FormFieldValue(kind="text")
    assert ffv.kind == "text"
    assert ffv.value == ""
    assert ffv.masked is False


def test_form_capture_js_masks_passwords_by_default(monkeypatch) -> None:
    """The capture JS expression is built from env state; default
    must produce a literal ``capturePasswords = false`` so secrets
    never leave the page even before they reach Python."""
    monkeypatch.delenv("MANTIS_PAUSE_CAPTURE_PASSWORDS", raising=False)
    from mantis_agent.gym import form_capture_js

    js = form_capture_js.capture_js()
    assert "capturePasswords = false" in js


def test_form_capture_js_opts_in_with_env_flag(monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_PAUSE_CAPTURE_PASSWORDS", "1")
    from mantis_agent.gym import form_capture_js

    js = form_capture_js.capture_js()
    assert "capturePasswords = true" in js


def test_replay_js_embeds_serialized_entries() -> None:
    from mantis_agent.gym import form_capture_js

    payload = json.dumps([
        {"selector": "#email", "kind": "text",
         "value": "a@b.com", "masked": False},
    ])
    js = form_capture_js.replay_js(payload)
    assert "#email" in js
    assert "applied" in js  # returns {applied, skipped, missing}


def test_pause_state_json_round_trip_with_form_state() -> None:
    """form_state survives the JSON round-trip: dict values come
    back as FormFieldValue instances, not raw dicts. Passwords
    stay masked."""
    bs = BrowserState(
        url="https://example.com/checkout",
        form_state={
            "#email": FormFieldValue(kind="text", value="a@b.com"),
            "#country": FormFieldValue(kind="select", value="US"),
            "#tos": FormFieldValue(kind="checkbox", value="true"),
            "#password": FormFieldValue(kind="text", value="", masked=True),
        },
    )
    ps = PauseState(browser_state=bs)
    payload = json.dumps(ps.to_dict())
    rehydrated = PauseState.from_dict(json.loads(payload))
    fs = rehydrated.browser_state.form_state
    assert isinstance(fs["#email"], FormFieldValue)
    assert fs["#email"].value == "a@b.com"
    assert fs["#country"].kind == "select"
    assert fs["#tos"].value == "true"
    assert fs["#password"].masked is True
    assert fs["#password"].value == ""


def test_xdotool_capture_returns_form_state() -> None:
    """The form-state capture call returns a list of dicts; the env
    converts that to a dict[selector, FormFieldValue]."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    env = object.__new__(XdotoolGymEnv)
    env.cdp_evaluate = MagicMock(side_effect=[
        {  # URL/scroll/viewport eval
            "url": "https://example.com/form", "scroll_x": 0,
            "scroll_y": 0, "viewport_w": 1280, "viewport_h": 720,
        },
        [  # form-field eval
            {"selector": "#email", "kind": "text",
             "value": "a@b.com", "masked": False},
            {"selector": "#tos", "kind": "checkbox",
             "value": "true", "masked": False},
        ],
    ])
    bs = env.capture_browser_state()
    assert bs.url == "https://example.com/form"
    assert "#email" in bs.form_state
    assert bs.form_state["#email"].value == "a@b.com"
    assert bs.form_state["#tos"].kind == "checkbox"
    assert env.cdp_evaluate.call_count == 2


def test_xdotool_capture_handles_form_capture_failure() -> None:
    """If the form-state eval raises, URL/scroll capture should
    still succeed — form_state lands empty, the rest is intact."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    env = object.__new__(XdotoolGymEnv)
    env.cdp_evaluate = MagicMock(side_effect=[
        {"url": "https://example.com/form", "scroll_x": 0,
         "scroll_y": 0, "viewport_w": 1280, "viewport_h": 720},
        RuntimeError("CDP dropped mid-eval"),
    ])
    bs = env.capture_browser_state()
    assert bs.url == "https://example.com/form"  # URL part survived
    assert bs.form_state == {}


def test_xdotool_capture_drops_malformed_entries() -> None:
    """Entries without a string selector / kind get skipped, the
    valid ones still flow through."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    env = object.__new__(XdotoolGymEnv)
    env.cdp_evaluate = MagicMock(side_effect=[
        {"url": "https://example.com/form", "scroll_x": 0,
         "scroll_y": 0, "viewport_w": 1280, "viewport_h": 720},
        [
            {"selector": "#good", "kind": "text", "value": "x"},
            {"selector": None, "kind": "text", "value": "y"},
            {"selector": "#bad-kind", "kind": None, "value": "z"},
            "not-a-dict",
        ],
    ])
    bs = env.capture_browser_state()
    assert list(bs.form_state.keys()) == ["#good"]


def test_xdotool_restore_replays_form_state() -> None:
    """When state.form_state is non-empty, restore_browser_state
    should call _replay_form_state after the URL + scroll steps."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    env = object.__new__(XdotoolGymEnv)
    env.reset = MagicMock()
    env.cdp_evaluate = MagicMock(return_value={
        "applied": 1, "skipped": 0, "missing": 0,
    })
    state = BrowserState(
        url="https://example.com/form",
        scroll_x=0, scroll_y=0,
        form_state={"#email": FormFieldValue(kind="text", value="a@b.com")},
    )
    env.restore_browser_state(state)

    env.reset.assert_called_once()
    # scroll skipped (both zero) — single eval for form replay.
    env.cdp_evaluate.assert_called_once()
    js = env.cdp_evaluate.call_args.args[0]
    assert "#email" in js
    assert "a@b.com" in js


def test_playwright_capture_returns_form_state() -> None:
    """Same shape as the xdotool test but exercised through
    PlaywrightGymEnv — page.evaluate is the seam, two evals per
    capture (URL + form fields)."""
    from mantis_agent.gym.playwright_env import PlaywrightGymEnv

    env = object.__new__(PlaywrightGymEnv)
    env._page = MagicMock()
    env._page.evaluate = MagicMock(side_effect=[
        {"url": "https://example.com/form", "scroll_x": 0,
         "scroll_y": 100, "viewport_w": 1280, "viewport_h": 720},
        [
            {"selector": "#email", "kind": "text",
             "value": "x@y.com", "masked": False},
        ],
    ])
    bs = env.capture_browser_state()
    assert bs.url == "https://example.com/form"
    assert bs.form_state["#email"].value == "x@y.com"
    assert env._page.evaluate.call_count == 2


def test_playwright_capture_handles_form_capture_failure() -> None:
    """Form-state eval failure on Playwright path: URL/scroll
    survives, form_state lands empty."""
    from mantis_agent.gym.playwright_env import PlaywrightGymEnv

    env = object.__new__(PlaywrightGymEnv)
    env._page = MagicMock()
    env._page.evaluate = MagicMock(side_effect=[
        {"url": "https://example.com/form", "scroll_x": 0,
         "scroll_y": 0, "viewport_w": 1280, "viewport_h": 720},
        RuntimeError("page closed mid-eval"),
    ])
    bs = env.capture_browser_state()
    assert bs.url == "https://example.com/form"
    assert bs.form_state == {}


def test_playwright_restore_replays_form_state() -> None:
    """restore_browser_state on Playwright env hits page.evaluate
    with the replay JS once the URL+scroll path is done."""
    from mantis_agent.gym.playwright_env import PlaywrightGymEnv

    env = object.__new__(PlaywrightGymEnv)
    env._page = MagicMock()
    env._page.evaluate = MagicMock(return_value={
        "applied": 1, "skipped": 0, "missing": 0,
    })
    env.reset = MagicMock(side_effect=lambda **_: None)

    state = BrowserState(
        url="https://example.com/form",
        scroll_x=0, scroll_y=0,
        form_state={"#email": FormFieldValue(kind="text", value="a@b.com")},
    )
    env.restore_browser_state(state)

    env.reset.assert_called_once()
    # Single eval for the form replay (scroll skipped, both zero).
    env._page.evaluate.assert_called_once()
    js = env._page.evaluate.call_args.args[0]
    assert "#email" in js
    assert "a@b.com" in js


def test_xdotool_replay_skips_when_form_state_empty() -> None:
    """An empty form_state on a state with a real url must skip
    the replay eval entirely (no wasted CDP round-trip)."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    env = object.__new__(XdotoolGymEnv)
    env.reset = MagicMock()
    env.cdp_evaluate = MagicMock()

    env.restore_browser_state(BrowserState(
        url="https://example.com/page",
        scroll_x=0, scroll_y=0,
        form_state={},
    ))

    env.reset.assert_called_once()
    env.cdp_evaluate.assert_not_called()  # no scroll, no form replay
