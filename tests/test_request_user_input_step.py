"""Tests for the ``request_user_input`` step type — plan_text accessible
pause/resume hook.

Bug context: plan_text submissions that asked the agent to "pause and
ask the user for X" had no step type to compile to. The decomposer
silently dropped the instruction and the brain made up values.

The fix adds:

* ``request_user_input`` to ``MicroIntent.type`` allowlist + decomposer
  prompt + default section/required.
* A new step handler that bridges to the runner's host tool, which
  raises ``PauseRequested`` on the first entry and returns the staged
  value on resume.

These tests drive the handler in isolation (no Chrome, no Claude).
"""

from __future__ import annotations

from typing import Any

import pytest

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.step_context import StepContext
from mantis_agent.gym.step_handlers.request_user_input import RequestUserInputHandler
from mantis_agent.plan_decomposer import MicroIntent, PlanDecomposer


# ── Decomposer recognises request_user_input ──────────────────────────


def test_decomposer_build_intent_accepts_request_user_input() -> None:
    intent = PlanDecomposer._build_intent({
        "type": "request_user_input",
        "intent": "Pause and ask the operator for the OTP",
        "params": {
            "prompt": "Enter the 6-digit code",
            "reason": "user_input",
        },
    })
    assert intent.type == "request_user_input"
    # Defaults: section=setup, required=True (the staged value
    # feeds a downstream form-fill).
    assert intent.section == "setup"
    assert intent.required is True
    assert intent.params["prompt"] == "Enter the 6-digit code"


def test_decomposer_request_user_input_required_override_respected() -> None:
    """Explicit required=False on the source dict should win — the
    auto-required default is only a default."""
    intent = PlanDecomposer._build_intent({
        "type": "request_user_input",
        "intent": "Maybe ask the operator",
        "required": False,
    })
    assert intent.required is False


# ── Handler protocol shape ────────────────────────────────────────────


def test_handler_step_type_is_canonical() -> None:
    handler = RequestUserInputHandler(runner=_FakeRunner())
    assert handler.step_type == "request_user_input"


# ── Pause path: handler propagates PauseRequested up to runner ──────


class _PauseRequestedStub(Exception):
    """Local PauseRequested stand-in so the test doesn't depend on the
    runtime's actual PauseRequested import wiring."""


class _FakeRunner:
    """Records the host_tools / state mutations the handler performs."""

    def __init__(self, host_tools: dict | None = None) -> None:
        # Defaults to the canonical SDK-flow tool that raises
        # PauseRequested on the first call and returns a staged value
        # on the second (after action=resume).
        self._host_tools: dict[str, Any] = host_tools if host_tools is not None else {
            "request_user_input": self._default_tool,
        }
        self._pause_inputs: list[Any] = []
        self._staged_user_input = None

    def _default_tool(self, args: dict) -> Any:
        if not self._pause_inputs:
            raise _PauseRequestedStub(args.get("prompt", ""))
        return self._pause_inputs.pop(0)


def _step(prompt: str = "Enter OTP", reason: str = "user_input") -> MicroIntent:
    return MicroIntent(
        intent="Pause and ask the operator",
        type="request_user_input",
        params={"prompt": prompt, "reason": reason},
        required=True,
        section="setup",
    )


def _ctx(index: int = 0) -> StepContext:
    ctx = StepContext(env=None, brain=None)
    ctx.state["index"] = index
    return ctx


def test_handler_raises_when_host_tool_raises_pause() -> None:
    """Empty pause inputs → tool raises → handler re-raises through
    its try/except so the runner's PauseRequested catch path owns it."""
    runner = _FakeRunner()
    handler = RequestUserInputHandler(runner)
    with pytest.raises(_PauseRequestedStub):
        handler.execute(_step(), _ctx())


def test_handler_returns_success_and_stashes_staged_value_on_resume() -> None:
    """When the host tool returns a value (i.e. action=resume staged
    one), the handler stashes it on the runner and reports success."""
    runner = _FakeRunner()
    runner._pause_inputs = ["123456"]
    handler = RequestUserInputHandler(runner)
    result = handler.execute(_step(), _ctx())
    assert isinstance(result, StepResult)
    assert result.success is True
    assert result.data == "request_user_input:resumed"
    assert getattr(runner, "_staged_user_input") == "123456"


def test_handler_does_not_leak_secret_into_result_data() -> None:
    """Trace consumers (Augur, ops logs) read ``result.data``. The
    staged value is potentially a secret — the handler MUST keep it
    out of the data field."""
    runner = _FakeRunner()
    runner._pause_inputs = ["super-secret-otp"]
    handler = RequestUserInputHandler(runner)
    result = handler.execute(_step(), _ctx())
    assert "super-secret-otp" not in (result.data or "")


# ── Off-Baseten degradation: no host tool registered ─────────────────


def test_handler_skips_when_no_host_tool_registered() -> None:
    runner = _FakeRunner(host_tools={})
    handler = RequestUserInputHandler(runner)
    result = handler.execute(_step(), _ctx(index=4))
    assert result.success is False
    assert result.skip is True
    assert "no_host_tool" in (result.skip_reason or "")


# ── Registry exposes the handler under the right type ───────────────


def test_default_registry_binds_request_user_input_to_handler() -> None:
    """End-to-end: ``default_registry`` registers the new handler
    so the runner's ``_handler_registry.get('request_user_input')``
    resolves to it."""
    from mantis_agent.gym.step_handlers import default_registry

    # default_registry requires a runner with the brain / env wiring
    # other handlers reach into. We don't need them for the lookup —
    # any handler that fails to construct against a bare-minimum
    # runner would fail HERE, which is itself a useful regression
    # signal (proves our new handler doesn't reach into deep runner
    # internals at construct time).
    class _BareRunner:
        brain = None
        env = None
        extractor = None

    reg = default_registry(_BareRunner())  # type: ignore[arg-type]
    handler = reg.get("request_user_input")
    assert handler is not None
    assert handler.step_type == "request_user_input"
