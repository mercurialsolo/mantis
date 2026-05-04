"""NavigateHandler unit tests — Phase 2 of EPIC #161.

Demonstrates the explicit acceptance criterion: a handler can be unit
tested with a mocked StepContext, no Xvfb, no GymRunner, no real
ClaudeExtractor. The whole point of Phase 2 is making this pattern
possible.

Asserts that the handler:
- Sets results_base_url + required_filter_tokens on the scanner
- Calls env.reset with the right task and url
- Sends Home keypress and 2s settle
- Seeds the dynamic verifier with the page-start record
- Returns success=False when the intent text has no URL (no env calls made)
- Returns success=False when env.reset raises
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from mantis_agent.actions import ActionType
from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.listings_scanner import ListingsScanner
from mantis_agent.gym.step_context import StepContext
from mantis_agent.gym.step_handlers.navigate import NavigateHandler
from mantis_agent.plan_decomposer import MicroIntent


class _FakeRunner:
    """Minimal back-reference. NavigateHandler reads 5 attributes / 2 methods."""

    def __init__(self) -> None:
        self._current_page = 0
        self._last_known_url = ""
        self._results_base_url = ""
        self._required_filter_tokens: tuple[str, ...] = ()
        self.reset_scan_calls = 0
        self.browser_state = MagicMock()

    def _derive_filter_tokens(self, url: str) -> tuple[str, ...]:
        # Match the runner's actual implementation: pull URL path tokens
        # excluding "boats" and page numbers. Tests pin specific inputs.
        import re
        m = re.search(r"https?://[^/]+/([^?#]+)", url)
        if not m:
            return ()
        out = []
        for tok in m.group(1).strip("/").split("/"):
            if not tok or tok in {"boats"} or tok.startswith("page-"):
                continue
            out.append(tok.lower())
        return tuple(out)

    def _reset_results_scan_state(self) -> None:
        self.reset_scan_calls += 1


def _ctx_with_runner_and_env(runner: _FakeRunner, env: Any) -> StepContext:
    scanner = ListingsScanner()
    dynamic_verifier = MagicMock()
    return StepContext(
        env=env,
        brain=None,
        extractor=None,
        grounding=None,
        cost_meter=None,
        dynamic_verifier=dynamic_verifier,
        scanner=scanner,
        site_config=None,
        tool_channel=None,
        extraction_cache=None,
        state={"index": 3},
    )


def _step(intent: str, **params: Any) -> MicroIntent:
    return MicroIntent(intent=intent, type="navigate", params=params or None)


def test_navigate_extracts_url_and_calls_env_reset(monkeypatch):
    monkeypatch.delenv("MANTIS_NAV_WAIT_SECONDS", raising=False)
    monkeypatch.setattr("mantis_agent.gym.step_handlers.navigate.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    env = MagicMock()
    ctx = _ctx_with_runner_and_env(runner, env)
    handler = NavigateHandler(runner)

    result = handler.execute(_step("Go to https://www.example.com/cars"), ctx)

    assert result.success is True
    assert isinstance(result, StepResult)
    assert result.step_index == 3

    env.reset.assert_called_once_with(task="navigate", start_url="https://www.example.com/cars")
    home_call = env.step.call_args_list[0]
    sent_action = home_call.args[0]
    assert sent_action.action_type == ActionType.KEY_PRESS
    assert sent_action.params == {"keys": "Home"}

    # Scanner anchored to the URL; runner state seeded.
    assert ctx.scanner is not None
    assert ctx.scanner.results_base_url == "https://www.example.com/cars"
    assert ctx.scanner.required_filter_tokens == ("cars",)
    assert runner._current_page == 1
    assert runner._last_known_url == "https://www.example.com/cars"
    assert runner.reset_scan_calls == 1
    runner.browser_state.set_scroll_state.assert_called_once()

    # Dynamic verifier seeded with required filter tokens + page start
    ctx.dynamic_verifier.set_required_filter_tokens.assert_called_once_with(())  # runner copy is empty in fake
    ctx.dynamic_verifier.record_page_start.assert_called_once_with(
        page=1, url="https://www.example.com/cars",
    )


def test_navigate_returns_failure_when_intent_lacks_url(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.navigate.time.sleep", lambda *_: None)
    runner = _FakeRunner()
    env = MagicMock()
    ctx = _ctx_with_runner_and_env(runner, env)
    handler = NavigateHandler(runner)

    result = handler.execute(_step("just text, no link here"), ctx)

    assert result.success is False
    env.reset.assert_not_called()
    env.step.assert_not_called()


def test_navigate_returns_failure_when_env_reset_raises(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.navigate.time.sleep", lambda *_: None)
    runner = _FakeRunner()
    env = MagicMock()
    env.reset.side_effect = RuntimeError("Cloudflare timeout")
    ctx = _ctx_with_runner_and_env(runner, env)
    handler = NavigateHandler(runner)

    result = handler.execute(_step("Open https://x.example.com/"), ctx)

    assert result.success is False
    # env.reset was attempted; subsequent calls (env.step Home) didn't happen
    env.reset.assert_called_once()
    env.step.assert_not_called()


def test_navigate_respects_step_param_wait_seconds(monkeypatch):
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.navigate.time.sleep",
        lambda s: sleep_calls.append(s),
    )
    monkeypatch.delenv("MANTIS_NAV_WAIT_SECONDS", raising=False)

    runner = _FakeRunner()
    env = MagicMock()
    ctx = _ctx_with_runner_and_env(runner, env)
    handler = NavigateHandler(runner)

    handler.execute(_step("Go to https://x.example.com/", wait_after_load_seconds=42), ctx)

    # First sleep is the page-load wait (clamped to [0, 120]); second is the Home settle.
    assert sleep_calls[0] == 42.0
    assert sleep_calls[1] == 2


def test_navigate_clamps_wait_seconds_to_max(monkeypatch):
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.navigate.time.sleep",
        lambda s: sleep_calls.append(s),
    )
    monkeypatch.delenv("MANTIS_NAV_WAIT_SECONDS", raising=False)

    runner = _FakeRunner()
    env = MagicMock()
    ctx = _ctx_with_runner_and_env(runner, env)
    handler = NavigateHandler(runner)

    handler.execute(_step("Go to https://x.example.com/", wait_after_load_seconds=999), ctx)

    assert sleep_calls[0] == 120.0  # hard cap


def test_navigate_falls_back_to_env_var_for_wait(monkeypatch):
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.navigate.time.sleep",
        lambda s: sleep_calls.append(s),
    )
    monkeypatch.setenv("MANTIS_NAV_WAIT_SECONDS", "7")

    runner = _FakeRunner()
    env = MagicMock()
    ctx = _ctx_with_runner_and_env(runner, env)
    handler = NavigateHandler(runner)

    handler.execute(_step("Open https://x.example.com/"), ctx)
    assert sleep_calls[0] == 7.0


def test_navigate_uses_18s_default_wait(monkeypatch):
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.navigate.time.sleep",
        lambda s: sleep_calls.append(s),
    )
    monkeypatch.delenv("MANTIS_NAV_WAIT_SECONDS", raising=False)

    runner = _FakeRunner()
    env = MagicMock()
    ctx = _ctx_with_runner_and_env(runner, env)
    handler = NavigateHandler(runner)

    handler.execute(_step("Open https://x.example.com/"), ctx)
    assert sleep_calls[0] == 18.0


def test_navigate_step_type_property_is_navigate():
    handler = NavigateHandler(_FakeRunner())
    assert handler.step_type == "navigate"
