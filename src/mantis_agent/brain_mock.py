"""MockBrain — deterministic placeholder for plan authoring (#274).

When ``MANTIS_BRAIN=mock`` the brain registry returns this class instead
of Holo3 / Claude / OpenCUA. Every ``think()`` call returns a
``DONE`` action with zero LLM calls, so plan authors can iterate on
plan **structure** (section transitions, gate predicates, loop
termination) against a real :class:`MicroPlanRunner` without paying
for inference or running a GPU.

**Scope**: this mocks the perception-action brain only. Verification
gates, ClaudeExtractor, and ClaudeGrounding are independent
Claude-backed components and continue to call the Anthropic API
unless they're stubbed separately. Plans that lean heavily on
extraction will still need ``ANTHROPIC_API_KEY``; plans that just
walk a navigate/click/scroll/loop skeleton run free under mock.

The brain has no dependency on torch / transformers / requests /
anthropic. Importing this module is cheap (no module-level side
effects beyond the dataclass / class definitions), so the slim
``mantis-agent`` install can use it without pulling GPU deps.

Example:

.. code-block:: bash

    # Fast structural-only run of a plan — no inference.
    MANTIS_BRAIN=mock mantis plan dry-run plans/example/extract_listings.json

    # Or call the runner with a mock brain from Python:
    >>> import os
    >>> os.environ["MANTIS_BRAIN"] = "mock"
    >>> from mantis_agent.brain_protocol import resolve_brain_from_env
    >>> brain = resolve_brain_from_env()
    >>> brain.load()
    >>> result = brain.think(frames=[], task="click the button")
    >>> result.action.action_type.value
    'done'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .actions import Action, ActionType

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class InferenceResult:
    """Result from a single MockBrain inference cycle.

    Mirrors the public fields the other brains return so the runner
    treats a mock result identically to a real one. ``thinking`` and
    ``tokens_used`` exist for compatibility with downstream code that
    pulls them; they're always empty / zero for mock.
    """

    action: Action
    raw_output: str
    thinking: str = ""
    tokens_used: int = 0


class MockBrain:
    """Always-DONE brain for plan authoring without inference cost.

    Returns a ``DONE`` action on every ``think()`` call so the runner
    accepts the current step as complete and advances to the next.
    This is enough to exercise section / gate / loop semantics in
    :class:`MicroPlanRunner` without burning Holo3 + Claude credits.

    Construction is free — no weights, no clients, no env vars to
    resolve. ``load()`` is a no-op so the brain plugs into the existing
    runner contract.

    Args:
        reasoning: String embedded in every returned ``Action.reasoning``
            so the trace makes it obvious the run was mocked. Override
            for tests that need to distinguish multiple mocks.
    """

    model = "mock"
    model_name = "mock"

    def __init__(self, reasoning: str = "mock brain: structural-only run") -> None:
        self.reasoning = reasoning
        self._think_count = 0

    def load(self) -> None:
        """No-op. Mock has no weights, clients, or external resources."""
        return None

    def think(
        self,
        frames: "list[Image.Image] | None" = None,
        task: str = "",
        action_history: "list[Action] | None" = None,
        screen_size: tuple[int, int] = (1920, 1080),
        **_kwargs: Any,
    ) -> InferenceResult:
        """Return a deterministic ``DONE`` action.

        ``frames`` and ``action_history`` are accepted to match the
        :class:`Brain` protocol but ignored — the mock has no perception
        and no state beyond a ``_think_count`` counter exposed in the
        raw_output for traceability.
        """
        self._think_count += 1
        action = Action(
            action_type=ActionType.DONE,
            params={},
            reasoning=self.reasoning,
        )
        return InferenceResult(
            action=action,
            raw_output=(
                f"[mock think #{self._think_count}] task={task!r} "
                f"frames={len(frames or [])} → DONE"
            ),
        )
