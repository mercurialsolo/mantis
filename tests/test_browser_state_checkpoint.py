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

from mantis_agent.gym.checkpoint import BrowserState, PauseState


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
    env.cdp_evaluate = MagicMock(return_value={
        "url": "https://example.com/page",
        "scroll_x": 0,
        "scroll_y": 1800,
        "viewport_w": 1280,
        "viewport_h": 720,
    })
    bs = env.capture_browser_state()
    assert bs.url == "https://example.com/page"
    assert bs.scroll_y == 1800
    assert bs.viewport_w == 1280
    # The CDP call happened.
    env.cdp_evaluate.assert_called_once()
    expr = env.cdp_evaluate.call_args.args[0]
    assert "location.href" in expr
    assert "window.scrollX" in expr


def test_xdotool_capture_handles_cdp_failure() -> None:
    """CDP unreachable / page mid-navigation returns None from
    cdp_evaluate — capture must yield an empty BrowserState, not
    crash."""
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
