"""``request_user_input`` step handler — deterministic pause/resume from plan_text.

Plans authored as free text can ask the runtime to pause for an
operator-supplied value (OTP, password, paste-target) by emitting a
``request_user_input`` step (see the ``DECOMPOSE_PROMPT`` block of
the same name in ``plan_decomposer``). The handler bridges that step
to the host-tool mechanism the SDK pause/resume flow already uses:

* Look up the ``request_user_input`` host tool the runner registered
  in ``baseten_server.runtime`` (``runtime.py:1579``). The tool consumes
  any staged ``user_input`` (set by ``action=resume``) and either:
    - returns the staged value on first-resume → handler types it back
      into ``StepResult.data`` so any downstream ``{{user_input}}`` token
      substitution can pick it up; or
    - raises ``PauseRequested`` if no value is staged → the runner
      catches it and snapshots the run as ``status=paused``.

* When the host tool isn't registered (off-Baseten contexts), the
  handler degrades to a non-fatal skip so a plan that ran fine on
  Baseten doesn't suddenly crash on a deployment that hasn't wired
  the host tool yet.

Wire shape — the decomposer emits::

    {
      "type": "request_user_input",
      "intent": "Pause and ask the operator for the OTP",
      "params": {
        "prompt": "Enter the 6-digit code from your authenticator",
        "reason": "user_input"
      },
      "required": true,
      "section": "setup"
    }

The runner catches ``PauseRequested`` in its execute loop and saves
the :class:`gym.checkpoint.PauseState` blob. The caller polls
``action=status`` → ``status=paused``, then hits ``action=resume``
with ``user_input=<value>``. On resume, the runner re-enters this
step, the host tool returns the staged value, and the handler reports
success — the staged value lives on ``runner._pause_consumed_input``
so subsequent steps that reference ``{{user_input}}`` in their params
or intent can substitute it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..checkpoint import StepResult
from ..step_context import StepContext, StepHandler

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner
    from ...plan_decomposer import MicroIntent

logger = logging.getLogger(__name__)


class RequestUserInputHandler:
    """Plan-text accessible bridge to the runner's ``request_user_input``
    host tool. See module docstring for the round-trip protocol.
    """

    def __init__(self, runner: "MicroPlanRunner") -> None:
        self._runner = runner

    @property
    def step_type(self) -> str:
        return "request_user_input"

    def execute(self, step: "MicroIntent", ctx: StepContext) -> StepResult:
        runner = self._runner
        params = step.params or {}
        prompt = str(params.get("prompt") or step.intent or "user_input")
        reason = str(params.get("reason") or "user_input")
        # Step index isn't a top-level StepContext field — handlers
        # read it from ``ctx.state['index']`` (the runner sets it
        # before dispatch). Fall back to 0 so unit tests that don't
        # plumb the index in still construct a sane StepResult.
        index = int(ctx.state.get("index", 0)) if hasattr(ctx, "state") else 0

        host_tools = getattr(runner, "_host_tools", None) or {}
        tool_fn = host_tools.get("request_user_input")
        if tool_fn is None:
            # No tool registered — surface a non-fatal skip so a
            # plan that ran fine on Baseten doesn't crash on a
            # deployment that hasn't wired the host tool.
            logger.warning(
                "request_user_input step %d: no host tool registered; "
                "downgrade to skip (reason=%s)",
                index, reason,
            )
            return StepResult(
                step_index=index,
                intent=step.intent,
                success=False,
                data="request_user_input:host_tool_missing",
                skip=True,
                skip_reason="request_user_input_no_host_tool",
            )

        # First entry: ``tool_fn`` raises PauseRequested → the runner
        # catches it in its execute loop. Resume entry: ``tool_fn``
        # returns the staged value. Either way the handler doesn't
        # try to suppress — exceptions bubble through the runner's
        # standard recovery machinery.
        staged = tool_fn({"prompt": prompt, "reason": reason})

        # Resume path — stash the value so any downstream step that
        # references ``{{user_input}}`` in its params or intent can
        # substitute it.
        try:
            setattr(runner, "_staged_user_input", staged)
        except Exception:  # noqa: BLE001 — observability only
            logger.warning(
                "request_user_input step %d: stash on runner failed",
                index,
            )

        # ``data`` field is part of the trace surface (Augur, ops
        # logs). Never echo the staged value — it may be a secret.
        return StepResult(
            step_index=index,
            intent=step.intent,
            success=True,
            data="request_user_input:resumed",
            skip=False,
            skip_reason="",
        )


__all__ = ["RequestUserInputHandler"]


# Static type check the protocol — runtime-visible failure if the
# class drifts from the StepHandler shape.
assert isinstance(
    RequestUserInputHandler.__new__(RequestUserInputHandler),  # type: ignore[misc]
    StepHandler,
)
