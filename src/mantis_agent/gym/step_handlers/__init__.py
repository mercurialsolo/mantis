"""Per-type step handlers — Phase 2 of EPIC #161.

Each module under ``step_handlers/`` implements one
:class:`~..step_context.StepHandler` for a specific
``MicroIntent.type``. The runner's ``_execute_step`` consults the
registry first; types whose handler hasn't been extracted yet fall
through to the legacy if/elif in the runner.

Handlers are unit-testable in isolation: take a :class:`StepContext`
with mocked collaborators (``env``, ``extractor``, ``brain``,
``cost_meter``, ``scanner``, ``dynamic_verifier``, ``site_config``)
and return a :class:`StepResult`. No Xvfb, no GymRunner, no live
brain — that's the explicit acceptance criterion of EPIC #161.

Migration policy:
- One handler per PR (or one PR for a tight cluster like form
  variants — submit / fill_field / select_option).
- Each PR moves the handler body verbatim from the runner to the
  handler module, leaving a delegating shim on the runner so
  external callers (tests, host integrations) keep working.
- After all handlers are migrated, a final cleanup PR rips the
  legacy if/elif and the per-type runner methods entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..step_context import HandlerRegistry
from .form import ClaudeGuidedFormHandler
from .navigate import NavigateHandler

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner


__all__ = ["ClaudeGuidedFormHandler", "NavigateHandler", "default_registry"]


def default_registry(runner: "MicroPlanRunner") -> HandlerRegistry:
    """Build the default registry bound to one runner instance.

    Handlers that need access to runner-only state (e.g.
    ``_current_page``, ``_last_known_url``, ``_reset_results_scan_state``)
    take the runner as a back-reference at construction. This mirrors the
    BrowserState / CheckpointManager pattern from #115 — a thin shim that
    can be tested with a fake parent.

    Returns a registry with the handlers that have been migrated so far.
    Step types not in the returned registry continue to be dispatched by
    the legacy branches in ``MicroPlanRunner._execute_step``.

    Form handler is constructed but NOT registered for fill_field /
    submit / select_option in this PR — the dispatch in
    ``_execute_step`` adds an extra ``time.sleep(2)`` pre-settle on
    those branches that registry-first dispatch would bypass. Merging
    the two settles requires updating the dispatch and the handler in
    one step; that move happens when the click branch's layout-hint
    branching collapses into a router. The form handler is reachable
    today only through the runner shim, which preserves the existing
    dispatch + handler total settle (2s + 2s = 4s).
    """
    reg = HandlerRegistry()
    reg.register(NavigateHandler(runner))
    return reg
