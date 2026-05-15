"""StepContext — narrow handle passed to each step handler.

Phase 2 of EPIC #161 (refactor MicroPlanRunner into composable
modules). Defines the dependency-injection bag that each
:class:`StepHandler` will receive in lieu of reaching into runner
internals via ``self.``.

Types-only commit — the protocol is defined but no handler has been
extracted yet. Phase 2 follow-up commits will:

1. Create ``step_handlers/<type>.py`` modules each implementing
   ``StepHandler``.
2. Build a registry mapping step type strings to handler instances.
3. Replace ``MicroPlanRunner._execute_step`` (lines 1342–1426) with a
   single ``registry[step.type].execute(step, ctx)`` call.

The split between ``StepContext`` and ``StepHandler`` lets handlers be
unit-tested with a mock context (no Xvfb, no GymRunner, no
ClaudeExtractor instance required) — that's the explicit acceptance
criterion of EPIC #161.

What lives on ``StepContext`` is the **stable** subset of the runner:
collaborators (env / brain / extractor / grounding / cost meter /
scanner / dynamic verifier / site config / tool channel) plus a small
``state`` dict for handler-private scratch. The recovery decision +
loop-counter / retry-count state stay on the executor side — handlers
are pure: ``(step, ctx) -> StepResult``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .checkpoint import StepResult
from .listings_scanner import ListingsScanner
from ..plan_decomposer import MicroIntent


@dataclass
class StepContext:
    """Collaborators a :class:`StepHandler` needs to execute one step.

    Handlers MUST NOT touch the runner directly. Anything they need is on
    this dataclass; if a handler grows a need for new state, add a field
    here rather than reaching back into ``MicroPlanRunner``. That's how
    the dispatch stays decoupled from the executor.

    The handler may set ``state[key]`` for short-lived scratch shared
    across phases of a single ``execute`` call. The executor does not
    inspect ``state``; it's purely handler-private.
    """

    # Hardware / API collaborators
    env: Any                    # XdotoolGymEnv-like; handlers call .screenshot() / .step(action)
    brain: Any                  # Holo3 brain; handlers may use for tactical micro-actions
    extractor: Any | None = None    # ClaudeExtractor for vision / extraction / verify
    grounding: Any | None = None    # ClaudeGrounding for click coordinate refinement

    # Per-run accounting
    cost_meter: Any | None = None   # CostMeter — handlers call record_step / record_*_extract
    dynamic_verifier: Any | None = None  # DynamicPlanVerifier for record_item_completed

    # Listings-scan state (Phase 1.2) — handlers in Phase 4 will move
    # mutation logic here from run().
    scanner: ListingsScanner | None = None

    # Configuration
    site_config: Any | None = None  # SiteConfig — is_results_page, is_detail_page, etc.

    # #300 follow-up: dispatch policy for clicks (and, later, type/scroll).
    # Handlers consult ``routing_policy.som_for_unstructured_clicks``
    # before falling through to xdotool. Default None = "no policy
    # wired" = handlers skip SoM and execute as before, so legacy
    # callers see no behavior change.
    routing_policy: Any | None = None

    # Optional pluggable channels
    tool_channel: Any | None = None
    extraction_cache: Any | None = None

    # #406: form-target grounding (find_form_target /
    # find_target_by_affordance / verify_dropdown_value) is provider-
    # backed so a runner can swap Claude for Holo3 (or anything else
    # satisfying the :class:`FormTargetProvider` protocol) without
    # touching the form handler. When left ``None`` the form handler
    # lazily builds a :class:`ClaudeFormTargetProvider` from
    # ``extractor`` — preserves legacy behaviour for any caller that
    # constructs a :class:`StepContext` directly.
    form_target_provider: Any | None = None

    # Free-form scratch the handler can use for private state across
    # phases of one execute() call (e.g., the deep-extract pattern keeps
    # screenshot lists here). Executor never reads this.
    state: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class StepHandler(Protocol):
    """Pure-function protocol every per-type handler implements.

    A ``StepHandler`` takes a single :class:`MicroIntent` and a
    :class:`StepContext` and returns a :class:`StepResult`. It does not
    drive the executor's loop — it does not increment ``step_index``,
    advance loops, or persist checkpoints. Side effects on the *world*
    (clicking, scrolling, screenshotting) are fine; side effects on
    the *plan execution* are not.

    The executor is responsible for interpreting the result via
    :class:`~.step_recovery.RecoveryDecision` and applying any
    follow-up — retry, jump to type, halt, persist, etc.

    A handler MAY mutate ``ctx.scanner`` and ``ctx.cost_meter`` (those
    are the per-run accounting collaborators the runner shares with all
    handlers). It MUST NOT mutate any other field on the context.
    """

    @property
    def step_type(self) -> str:
        """Return the ``MicroIntent.type`` string this handler claims."""
        ...

    def execute(self, step: MicroIntent, ctx: StepContext) -> StepResult:
        """Run the step. Pure with respect to plan execution state."""
        ...


class HandlerRegistry:
    """Lookup ``step.type`` → :class:`StepHandler`.

    Phase 2 follow-up will populate the default registry with one entry
    per step type. For now this class is empty so callers can construct
    a registry, register a handler, and resolve it — proving the contract
    works without yet committing the per-type extractions.

    A handler can be registered for multiple types by name (e.g. the
    form handler currently serves ``submit`` / ``fill_field`` /
    ``select_option`` in :meth:`MicroPlanRunner._execute_step`). Use
    :meth:`register_for_types` to bind one handler instance to several
    keys.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, StepHandler] = {}

    def register(self, handler: StepHandler) -> None:
        """Register ``handler`` under its declared :attr:`StepHandler.step_type`."""
        self._handlers[handler.step_type] = handler

    def register_for_types(self, handler: StepHandler, types: tuple[str, ...]) -> None:
        """Bind one handler to multiple step type strings.

        Used by the form / submit / select_option / fill_field handler
        which today is the same code path in :meth:`_execute_claude_guided_form`.
        """
        for t in types:
            self._handlers[t] = handler

    def get(self, step_type: str) -> StepHandler | None:
        return self._handlers.get(step_type)

    def __contains__(self, step_type: str) -> bool:
        return step_type in self._handlers

    def types(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers.keys()))
