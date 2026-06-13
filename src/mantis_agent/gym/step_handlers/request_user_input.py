"""``request_user_input`` step handler — deterministic pause/resume from plan_text.

Plans authored as free text can ask the runtime to pause for an
operator-supplied value (OTP, password, paste-target) by emitting a
``request_user_input`` step (see the ``DECOMPOSE_PROMPT`` block of
the same name in ``plan_decomposer``). The handler bridges that step
to the host-tool mechanism the SDK pause/resume flow already uses:

* Look up the ``request_user_input`` host tool via the runner's
  :class:`~..tool_channel.ToolChannel` — the SAME registry both the
  Baseten (``runtime.py``, #344) and Modal (``modal_cua_server.py``,
  #347) executors register the default tool into through
  ``runner.register_tool(...)``. The tool consumes any staged
  ``user_input`` (set by ``action=resume``) and either:
    - returns the staged value on first-resume → handler stashes it on
      the runner so any downstream ``{{user_input}}`` token
      substitution can pick it up; or
    - raises ``PauseRequested`` if no value is staged → the runner
      catches it and snapshots the run as ``status=paused``.

* When the tool isn't registered (off-executor contexts: bare CLI,
  unit harnesses), the handler degrades to a non-fatal skip so a plan
  that pauses fine in production doesn't crash on a deployment that
  hasn't wired the host tool yet.

.. note:: #882 — the handler originally read a phantom
   ``runner._host_tools`` dict that NOTHING in production ever
   populated (the only writer was a test stub), so every
   ``request_user_input`` step on every backend hit the "no host tool"
   skip — it never paused, ``{{user_input}}`` was typed literally, and
   the run burned its budget to ``time_cap``. The real tool lives in
   ``runner.tool_channel``; the lookup now goes there.

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

from ..checkpoint import PauseRequested, StepResult
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

        tool_channel = getattr(runner, "tool_channel", None)
        registered = False
        if tool_channel is not None:
            try:
                registered = any(
                    t.get("name") == "request_user_input"
                    for t in tool_channel.list()
                )
            except Exception:  # noqa: BLE001 — treat a broken channel as absent
                registered = False
        if not registered:
            # No tool registered on this deployment (bare CLI / unit
            # harness) — surface a non-fatal skip so a plan that pauses
            # fine in production doesn't crash where the host tool
            # isn't wired.
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

        # First entry: the tool raises PauseRequested (no value staged
        # yet). We catch it and stage the pending pause on the channel —
        # the SAME slot ``ToolChannel.invoke`` sets — so the executor's
        # ``_tick_preamble`` ``tool_channel.is_paused()`` check snapshots
        # ``status=paused`` on the next iteration. We use ``call`` rather
        # than ``invoke`` so the resume value comes back raw (invoke
        # str-renders + truncates it, which would corrupt a multi-line
        # or structured ``{{user_input}}`` substitution).
        try:
            staged = tool_channel.call(
                "request_user_input", {"prompt": prompt, "reason": reason},
            )
        except PauseRequested as exc:
            tool_channel.stage_pause(
                "request_user_input",
                {"prompt": prompt, "reason": reason},
                getattr(exc, "reason", reason),
                getattr(exc, "prompt", prompt),
            )
            logger.warning(
                "request_user_input step %d: pausing for operator input "
                "(reason=%s)", index, reason,
            )
            return StepResult(
                step_index=index,
                intent=step.intent,
                success=True,
                data="request_user_input:paused",
            )

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
