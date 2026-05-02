"""Host-tool registration channel for MicroPlanRunner — extracted from
micro_runner.py (#115, step 2).

Hosts register callables that the brain can invoke mid-plan ("tools").
This module owns:

* the registry of registered tools and their schemas
* the safe-invoke wrapper that converts handler errors into structured
  step results (never silently swallowed)
* the pause-request slot that a handler can populate by raising
  :class:`PauseRequested` to suspend the run for user input

Behavior is unchanged from the previous in-place implementation in
``MicroPlanRunner``. The runner composes a single :class:`ToolChannel` and
its public ``register_tool`` / ``list_tools`` / ``call_tool`` methods
delegate here.
"""

from __future__ import annotations

import logging
from typing import Any

from .checkpoint import _PauseRequested

logger = logging.getLogger(__name__)


class ToolChannel:
    """Owns registered host tools + pause state for one MicroPlanRunner.

    A registered tool is ``{name: {"schema": dict, "handler": callable}}``.
    The handler signature is ``Callable[[dict[str, Any]], Any]``: the brain's
    parsed kwargs come in, the return value (str-rendered) goes back into the
    step's ``data`` field.

    The :meth:`invoke` wrapper is what the runner calls during a step. It
    catches :class:`PauseRequested` and stores a snapshot in
    :attr:`pending_pause` so the runner's main loop can detect it at the next
    boundary and emit a :class:`PauseState`. Any other exception becomes a
    ``(False, "tool:<n>:error:<type>:<msg>")`` step result.
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self.pending_pause: dict[str, Any] | None = None

    # ── Registration ────────────────────────────────────────────────────

    def register(self, name: str, schema: dict[str, Any], handler: Any) -> None:
        """Register a host-provided tool callable mid-plan.

        Args:
            name: Tool name (matches what the brain emits in its tool_use blocks).
            schema: JSON-schema input definition. Compatible with
                ``GenericToolAdapter.to_params()`` on the host side.
            handler: ``Callable[[dict[str, Any]], Any]``. Invoked with the
                kwargs the brain supplied. Return value is surfaced into the
                step ``data`` field; raised exceptions are caught and surfaced
                as ``success=False`` step results, never silently swallowed.
        """
        if not callable(handler):
            raise TypeError(f"register_tool handler for {name!r} must be callable")
        self._tools[name] = {"schema": dict(schema or {}), "handler": handler}

    def list(self) -> list[dict[str, Any]]:
        """Return registered tools as ``[{"name", "schema"}]`` (for brain prompts)."""
        return [{"name": n, "schema": t["schema"]} for n, t in self._tools.items()]

    # ── Invocation ──────────────────────────────────────────────────────

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Invoke a registered tool. Raises KeyError if not registered.

        Errors from the handler propagate to the caller; :meth:`invoke` is the
        loop-internal wrapper that converts errors into StepResult failures.
        """
        if name not in self._tools:
            raise KeyError(f"tool not registered: {name}")
        return self._tools[name]["handler"](arguments or {})

    def invoke(self, name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """Run a tool, returning ``(success, data_str)`` for the step result."""
        try:
            value = self.call(name, arguments)
        except _PauseRequested as exc:
            # Tool requested pause (#73) — propagate so the run() loop can
            # snapshot and return a PauseState. Don't treat as failure.
            self.pending_pause = {
                "tool": name,
                "arguments": dict(arguments),
                "reason": exc.reason,
                "prompt": exc.prompt,
            }
            return True, f"tool:{name}:pause"
        except KeyError as exc:
            return False, f"tool:{name}:not_registered:{exc}"
        except Exception as exc:  # noqa: BLE001 — surface, never swallow
            logger.exception("tool %s raised", name)
            return False, f"tool:{name}:error:{type(exc).__name__}:{exc}"
        rendered = "" if value is None else str(value)
        return True, f"tool:{name}:ok:{rendered[:200]}"

    # ── Pause lifecycle ─────────────────────────────────────────────────

    def is_paused(self) -> bool:
        return self.pending_pause is not None

    def clear_pause(self) -> None:
        """Drop the pending pause snapshot (called from resume())."""
        self.pending_pause = None


__all__ = ["ToolChannel"]
