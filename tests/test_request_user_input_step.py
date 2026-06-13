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

from mantis_agent.gym.checkpoint import PauseRequested, StepResult
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


class _FakeRunner:
    """Wires a real :class:`ToolChannel` exactly the way the Modal /
    Baseten executors do (``runner.register_tool`` → ``tool_channel``),
    so these tests exercise the real lookup path rather than a phantom
    ``_host_tools`` dict. The default tool raises PauseRequested on the
    first call and returns a staged value on the second (resume)."""

    def __init__(self, register_tool: bool = True) -> None:
        from mantis_agent.gym.tool_channel import ToolChannel

        self.tool_channel = ToolChannel()
        self._pause_inputs: list[Any] = []
        self._staged_user_input = None
        if register_tool:
            self.tool_channel.register(
                "request_user_input",
                {"type": "object", "properties": {"prompt": {"type": "string"}}},
                self._default_tool,
            )

    def _default_tool(self, args: dict) -> Any:
        if not self._pause_inputs:
            raise PauseRequested(reason="user_input", prompt=args.get("prompt", ""))
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


def test_handler_stages_pause_on_first_entry() -> None:
    """Empty pause inputs → tool raises PauseRequested → handler stages
    the pending pause on the channel (so the executor's is_paused()
    preamble snapshots status=paused) and returns a non-failing result —
    it does NOT let the exception escape uncaught."""
    runner = _FakeRunner()
    handler = RequestUserInputHandler(runner)
    result = handler.execute(_step(), _ctx())
    assert result.success is True
    assert result.data == "request_user_input:paused"
    # The pause is now visible to the run loop's is_paused() check.
    assert runner.tool_channel.is_paused() is True
    assert runner.tool_channel.pending_pause["tool"] == "request_user_input"


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
    # A runner whose tool_channel has NOT registered request_user_input
    # (bare CLI / unit harness) → non-fatal skip.
    runner = _FakeRunner(register_tool=False)
    handler = RequestUserInputHandler(runner)
    result = handler.execute(_step(), _ctx(index=4))
    assert result.success is False
    assert result.skip is True
    assert "no_host_tool" in (result.skip_reason or "")


def test_handler_skips_when_runner_has_no_tool_channel() -> None:
    """Guards the #882 regression: the handler must look at
    runner.tool_channel, not a phantom attribute. A runner with neither
    a tool_channel nor the tool degrades to a skip, never an AttributeError."""
    class _NoChannelRunner:
        pass

    handler = RequestUserInputHandler(_NoChannelRunner())  # type: ignore[arg-type]
    result = handler.execute(_step(), _ctx(index=2))
    assert result.success is False
    assert result.skip is True


def test_handler_uses_tool_channel_not_phantom_host_tools() -> None:
    """Regression for #882: a runner that only has the legacy
    phantom ``_host_tools`` dict (and no registered tool_channel tool)
    must NOT find the tool — proving the lookup moved to tool_channel."""
    runner = _FakeRunner(register_tool=False)
    # Simulate the old wiring that used to mask the bug.
    runner._host_tools = {"request_user_input": runner._default_tool}  # type: ignore[attr-defined]
    runner._pause_inputs = ["should-not-be-reached"]
    handler = RequestUserInputHandler(runner)
    result = handler.execute(_step(), _ctx())
    # Phantom dict is ignored → skip, because tool_channel has nothing.
    assert result.success is False
    assert result.skip is True


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


# ── #882: {{user_input}} substitution into downstream steps ──


def test_substitute_user_input_replaces_token() -> None:
    from mantis_agent.gym.run_executor import _substitute_user_input

    assert _substitute_user_input("{{user_input}}", "alice") == "alice"
    assert _substitute_user_input(
        "login as {{user_input}} now", "bob",
    ) == "login as bob now"


def test_substitute_user_input_noops_without_staged_value() -> None:
    from mantis_agent.gym.run_executor import _substitute_user_input

    # Nothing staged → the literal token is left untouched (the step
    # simply hasn't been fed a value yet).
    assert _substitute_user_input("{{user_input}}", None) == "{{user_input}}"


def test_substitute_user_input_ignores_non_strings_and_tokenless() -> None:
    from mantis_agent.gym.run_executor import _substitute_user_input

    assert _substitute_user_input(42, "x") == 42
    assert _substitute_user_input(None, "x") is None
    assert _substitute_user_input("no token here", "x") == "no token here"
