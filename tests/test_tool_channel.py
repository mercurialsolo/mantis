"""Tests for #115 step 2 — ToolChannel extracted from MicroPlanRunner."""

from __future__ import annotations

import pytest

from mantis_agent.gym.checkpoint import PauseRequested
from mantis_agent.gym.tool_channel import ToolChannel


# ── Registration ─────────────────────────────────────────────────────────


def test_register_stores_schema_and_handler() -> None:
    tc = ToolChannel()

    def handler(args: dict) -> str:
        return "ok"

    tc.register("ask_user", {"type": "object"}, handler)
    listed = tc.list()
    assert listed == [{"name": "ask_user", "schema": {"type": "object"}}]


def test_register_rejects_non_callable_handler() -> None:
    tc = ToolChannel()
    with pytest.raises(TypeError, match="handler"):
        tc.register("bad", {}, "not-a-function")  # type: ignore[arg-type]


def test_register_overwrite_replaces_handler() -> None:
    tc = ToolChannel()
    tc.register("t", {}, lambda args: "v1")
    tc.register("t", {"k": "v"}, lambda args: "v2")
    assert tc.list() == [{"name": "t", "schema": {"k": "v"}}]
    assert tc.call("t", {}) == "v2"


# ── Direct call ──────────────────────────────────────────────────────────


def test_call_invokes_handler_with_arguments() -> None:
    tc = ToolChannel()
    received: list[dict] = []

    def handler(args: dict) -> str:
        received.append(args)
        return f"ok:{args.get('x')}"

    tc.register("h", {}, handler)
    out = tc.call("h", {"x": 7})
    assert out == "ok:7"
    assert received == [{"x": 7}]


def test_call_unregistered_raises_keyerror() -> None:
    tc = ToolChannel()
    with pytest.raises(KeyError, match="not registered"):
        tc.call("nope", {})


def test_call_handler_exception_propagates() -> None:
    tc = ToolChannel()

    def boom(args: dict) -> None:
        raise RuntimeError("boom")

    tc.register("h", {}, boom)
    with pytest.raises(RuntimeError, match="boom"):
        tc.call("h", {})


# ── Wrapped invoke (loop-internal) ───────────────────────────────────────


def test_invoke_success_returns_ok_payload() -> None:
    tc = ToolChannel()
    tc.register("h", {}, lambda args: "result-value")
    ok, msg = tc.invoke("h", {})
    assert ok is True
    assert msg == "tool:h:ok:result-value"


def test_invoke_truncates_long_results_to_200_chars() -> None:
    tc = ToolChannel()
    tc.register("h", {}, lambda args: "x" * 500)
    ok, msg = tc.invoke("h", {})
    assert ok is True
    # Prefix is "tool:h:ok:" (10 chars) + up to 200 chars of payload.
    assert len(msg) == 10 + 200


def test_invoke_unregistered_returns_failure() -> None:
    tc = ToolChannel()
    ok, msg = tc.invoke("missing", {})
    assert ok is False
    assert msg.startswith("tool:missing:not_registered:")


def test_invoke_handler_error_returns_failure_with_type() -> None:
    tc = ToolChannel()

    def boom(args: dict) -> None:
        raise ValueError("bad input")

    tc.register("h", {}, boom)
    ok, msg = tc.invoke("h", {"a": 1})
    assert ok is False
    assert "tool:h:error:ValueError:bad input" in msg


# ── Pause request flow ──────────────────────────────────────────────────


def test_pause_request_marks_pending_and_returns_ok_pause() -> None:
    tc = ToolChannel()

    def needs_user(args: dict) -> None:
        raise PauseRequested(prompt="approve?", reason="user_input")

    tc.register("ask", {}, needs_user)
    assert tc.is_paused() is False
    ok, msg = tc.invoke("ask", {"context": "deploy"})
    assert ok is True
    assert msg == "tool:ask:pause"
    assert tc.is_paused() is True
    assert tc.pending_pause == {
        "tool": "ask",
        "arguments": {"context": "deploy"},
        "reason": "user_input",
        "prompt": "approve?",
    }


def test_clear_pause_drops_snapshot() -> None:
    tc = ToolChannel()
    tc.register("ask", {}, lambda args: (_ for _ in ()).throw(PauseRequested(prompt="?")))
    tc.invoke("ask", {})
    assert tc.is_paused() is True
    tc.clear_pause()
    assert tc.is_paused() is False
    assert tc.pending_pause is None


def test_pause_arguments_are_copied_not_aliased() -> None:
    """Mutating the args dict after invoke must not bleed into pending_pause."""
    tc = ToolChannel()
    tc.register("ask", {}, lambda args: (_ for _ in ()).throw(PauseRequested(prompt="?")))
    args = {"k": 1}
    tc.invoke("ask", args)
    args["k"] = 999
    assert tc.pending_pause["arguments"] == {"k": 1}


# ── Backward-compat: MicroPlanRunner public surface unchanged ────────────


def test_micro_plan_runner_public_methods_delegate_to_tool_channel() -> None:
    """Ensure the runner's register_tool/list_tools/call_tool still work."""
    from mantis_agent.gym.micro_runner import MicroPlanRunner

    # Build a runner with no env/brain (ToolChannel doesn't need them).
    runner = MicroPlanRunner.__new__(MicroPlanRunner)
    runner.tool_channel = ToolChannel()

    runner.register_tool("h", {"k": "v"}, lambda args: f"ok:{args.get('x')}")
    assert runner.list_tools() == [{"name": "h", "schema": {"k": "v"}}]
    assert runner.call_tool("h", {"x": 9}) == "ok:9"
