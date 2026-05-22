"""Tests for #583 — CDP ``window.history.back()`` over xdotool Alt+Left.

Boattrader runs kept ending with ``halt_reason=navigate_back_recovered``
even after #582's tab-count fix — Alt+Left doesn't reliably pop SPA
pushState history. CDP-driven ``window.history.back()`` does.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from mantis_agent.gym.xdotool_env import XdotoolGymEnv


class _StubXdotoolGymEnv(XdotoolGymEnv):
    """Test subclass that overrides ``current_url`` without mutating
    the base class — prevents test pollution across the suite.
    """
    _stubbed_url: str = ""

    @property  # type: ignore[override]
    def current_url(self) -> str:
        return self._stubbed_url


def _make_env(initial_url: str = "https://x.com/detail/123") -> _StubXdotoolGymEnv:
    env = _StubXdotoolGymEnv.__new__(_StubXdotoolGymEnv)
    env._cdp_port = 9222
    env._stubbed_url = initial_url
    return env


def test_cdp_back_succeeds_when_url_changes() -> None:
    env = _make_env(initial_url="https://x.com/detail/123")

    def _evaluate(expr):
        # Simulate that history.back() succeeded — URL changes to the
        # listings page on the next poll.
        env._stubbed_url = "https://x.com/listings/"
        return None

    with patch.object(env, "cdp_evaluate", side_effect=_evaluate):
        assert env.cdp_history_back() is True


def test_cdp_back_returns_false_when_url_unchanged() -> None:
    env = _make_env(initial_url="https://x.com/detail/123")
    # cdp_evaluate succeeds (returns None) but URL never changes —
    # caller should fall back to Alt+Left.
    with patch.object(env, "cdp_evaluate", return_value=None):
        assert env.cdp_history_back(settle_seconds=0.3) is False


def test_cdp_back_returns_false_when_cdp_raises() -> None:
    env = _make_env()
    with patch.object(env, "cdp_evaluate", side_effect=RuntimeError("ws closed")):
        # Defensive: any CDP failure → False so caller falls back.
        assert env.cdp_history_back(settle_seconds=0.2) is False


def test_cdp_back_returns_false_on_empty_url_after() -> None:
    env = _make_env(initial_url="https://x.com/detail/")

    def _evaluate(expr):
        env._stubbed_url = ""  # No URL after — can't verify success.
        return None

    with patch.object(env, "cdp_evaluate", side_effect=_evaluate):
        assert env.cdp_history_back(settle_seconds=0.2) is False


def test_cdp_back_settle_seconds_bounded() -> None:
    """Even with a long settle window, returns quickly when URL changes
    on the first poll."""
    env = _make_env()
    import time as _time
    started = _time.time()

    def _evaluate(expr):
        env._stubbed_url = "https://x.com/listings/"
        return None

    with patch.object(env, "cdp_evaluate", side_effect=_evaluate):
        assert env.cdp_history_back(settle_seconds=5.0) is True
    # Should return well under 1s even with 5s ceiling (we poll at 100ms).
    assert _time.time() - started < 1.0


# ── return_to_results_page wiring ─────────────────────────────────


def test_return_to_results_prefers_cdp_back_over_alt_left() -> None:
    from mantis_agent.gym._runner_helpers import return_to_results_page

    env = MagicMock()
    env.cdp_history_back = MagicMock(return_value=True)
    runner = MagicMock()
    runner.env = env
    runner._opened_detail_in_new_tab = False

    return_to_results_page(runner)

    env.cdp_history_back.assert_called_once()
    # When CDP-back succeeded, xdotool Alt+Left should NOT have been
    # dispatched.
    for call in env.step.call_args_list:
        action = call.args[0] if call.args else call.kwargs.get("action")
        if action is not None:
            params = getattr(action, "params", {}) or {}
            assert params.get("keys") != "alt+Left", \
                "Alt+Left dispatched even though CDP-back succeeded"


def test_return_to_results_falls_back_to_alt_left_when_cdp_fails() -> None:
    from mantis_agent.gym._runner_helpers import return_to_results_page

    env = MagicMock()
    env.cdp_history_back = MagicMock(return_value=False)  # CDP didn't navigate
    runner = MagicMock()
    runner.env = env
    runner._opened_detail_in_new_tab = False

    return_to_results_page(runner)

    env.cdp_history_back.assert_called_once()
    # Should have fallen back to Alt+Left.
    alt_left_dispatched = any(
        getattr(
            (call.args[0] if call.args else call.kwargs.get("action")),
            "params", {},
        ).get("keys") == "alt+Left"
        for call in env.step.call_args_list
    )
    assert alt_left_dispatched


def test_return_to_results_uses_ctrl_w_when_new_tab_flag_set() -> None:
    """The Ctrl+W path is untouched by #583 — new-tab flag still wins."""
    from mantis_agent.gym._runner_helpers import return_to_results_page

    env = MagicMock()
    env.cdp_history_back = MagicMock(return_value=True)
    runner = MagicMock()
    runner.env = env
    runner._opened_detail_in_new_tab = True

    return_to_results_page(runner)

    # CDP-back must NOT be invoked when we're closing a new tab.
    env.cdp_history_back.assert_not_called()
    # Ctrl+W should be dispatched.
    ctrl_w_dispatched = any(
        getattr(
            (call.args[0] if call.args else call.kwargs.get("action")),
            "params", {},
        ).get("keys") == "ctrl+w"
        for call in env.step.call_args_list
    )
    assert ctrl_w_dispatched
    # Flag should be cleared.
    assert runner._opened_detail_in_new_tab is False


def test_return_to_results_falls_back_when_env_has_no_cdp_back() -> None:
    """Envs without cdp_history_back (test stubs, legacy adapters) must
    not crash — fall straight through to Alt+Left."""
    from mantis_agent.gym._runner_helpers import return_to_results_page

    class _EnvNoCdp:
        def __init__(self):
            self.dispatched = []
        def step(self, action):
            self.dispatched.append(action)

    env = _EnvNoCdp()
    runner = MagicMock()
    runner.env = env
    runner._opened_detail_in_new_tab = False

    return_to_results_page(runner)

    assert any(
        getattr(a, "params", {}).get("keys") == "alt+Left"
        for a in env.dispatched
    )
