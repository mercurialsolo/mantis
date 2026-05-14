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
from .filter import ClaudeGuidedFilterHandler
from .form import ClaudeGuidedFormHandler
from .holo3 import Holo3StepHandler
from .navigate import NavigateHandler
from .paginate import PaginateHandler

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner


__all__ = [
    "ClaudeGuidedClickHandler",
    "ClaudeGuidedFilterHandler",
    "ClaudeGuidedFormHandler",
    "ClaudeStepHandler",
    "Holo3StepHandler",
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
    reg.register_for_types(
        ClaudeGuidedFormHandler(runner),
        ("fill_field", "submit", "select_option", "right_click"),
    )
    reg.register_for_types(
        ClaudeStepHandler(runner),
        ("extract_url", "extract_data"),
    )
    reg.register_for_types(
        Holo3StepHandler(runner),
        ("scroll", "navigate_back"),
    )
    return reg
