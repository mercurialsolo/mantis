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
from .claude_step import ClaudeStepHandler
from .click import ClaudeGuidedClickHandler
from .collect_urls import CollectUrlsHandler
from .filter import ClaudeGuidedFilterHandler
from .form import ClaudeGuidedFormHandler
from .holo3 import Holo3StepHandler
from .navigate import NavigateHandler
from .navigate_back import MechanicalNavigateBackHandler
from .paginate import PaginateHandler
from .scroll import MechanicalScrollHandler

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner


__all__ = [
    "ClaudeGuidedClickHandler",
    "ClaudeGuidedFilterHandler",
    "ClaudeGuidedFormHandler",
    "ClaudeStepHandler",
    "CollectUrlsHandler",
    "Holo3StepHandler",
    "MechanicalNavigateBackHandler",
    "MechanicalScrollHandler",
    "NavigateHandler",
    "PaginateHandler",
    "default_registry",
]


def default_registry(runner: "MicroPlanRunner") -> HandlerRegistry:
    """Build the default registry bound to one runner instance.

    Every handler that's been extracted under EPIC #161 Phase 2 is
    registered here. Pre-settle sleeps that used to live in
    ``MicroPlanRunner._execute_step``'s if/elif have been moved into
    each handler's ``execute()`` so registry-first dispatch produces
    identical timing behavior.

    Multi-binding (one handler instance for several step types) uses
    ``register_for_types``:

    - ``ClaudeGuidedFormHandler`` → ``fill_field`` / ``submit`` /
      ``select_option`` / ``right_click`` (form-shaped dispatch —
      one labelled target per call to ``find_form_target``; the
      handler picks the button=right path for ``right_click`` and
      a left-click for the other three)
    - ``ClaudeStepHandler`` → ``extract_url`` / ``extract_data``
      (Claude-only steps)
    - ``Holo3StepHandler`` → ``scroll`` / ``navigate_back`` (the only
      types that fall through to Holo3 today)

    The ``click`` type is registered to ``ClaudeGuidedClickHandler``
    BUT the dispatch in ``_execute_step`` keeps a small layout-hint
    router that decides between listings click (this handler) and
    single-element form click (synthesises a ``submit``-typed
    ``MicroIntent`` and dispatches to FormHandler). That router can't
    move into the registry without a meta-handler abstraction; it stays
    on the runner for now.

    Step types not in the returned registry (gate flag, claude_only
    flag, navigate_back-with-detail-tab special case) continue to be
    dispatched inline by ``_execute_step``.
    """
    reg = HandlerRegistry()
    reg.register(NavigateHandler(runner))
    reg.register(ClaudeGuidedClickHandler(runner))
    reg.register(PaginateHandler(runner))
    reg.register(ClaudeGuidedFilterHandler(runner))
    # #615: single-pass URL harvest for the fan-out runner (#616, #617).
    # Stashes urls on runner._collected_urls; safe to run standalone too
    # (sequential plan can use it for a URL-list dump).
    reg.register(CollectUrlsHandler(runner))
    reg.register_for_types(
        ClaudeGuidedFormHandler(runner),
        ("fill_field", "submit", "select_option", "right_click"),
    )
    reg.register_for_types(
        ClaudeStepHandler(runner),
        ("extract_url", "extract_data"),
    )
    # ``scroll`` routes through a dispatcher that prefers the
    # MechanicalScrollHandler when the plan supplies ``params.count``
    # (deterministic, no brain), falling through to Holo3StepHandler
    # for goal-shaped scroll intents that need vision mediation
    # ("scroll until X is visible"). ``navigate_back`` does the same:
    # mechanical CDP-back first (one history.back() + URL-change
    # poll), falling through to Holo3 when CDP back is unavailable,
    # didn't move the URL, or landed on another detail page (#608).
    holo3 = Holo3StepHandler(runner)
    mechanical_scroll = MechanicalScrollHandler(runner)
    mechanical_back = MechanicalNavigateBackHandler(runner)

    class _ScrollDispatcher:
        step_type = "scroll"

        def execute(self, step, ctx):
            if mechanical_scroll.applies_to(step):
                return mechanical_scroll.execute(step, ctx)
            return holo3.execute(step, ctx)

    class _NavigateBackDispatcher:
        step_type = "navigate_back"

        def execute(self, step, ctx):
            if not mechanical_back.applies_to(step):
                return holo3.execute(step, ctx)
            result = mechanical_back.execute(step, ctx)
            if result.success:
                return result
            # Mechanical CDP back didn't work (no history, dispatch
            # failure, landed on another detail page). Let the brain
            # try via its action vocabulary.
            return holo3.execute(step, ctx)

    reg.register(_ScrollDispatcher())
    reg.register(_NavigateBackDispatcher())
    return reg
