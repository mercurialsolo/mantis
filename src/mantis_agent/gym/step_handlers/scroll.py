"""MechanicalScrollHandler — deterministic scroll dispatch (no brain).

Why this exists: scroll steps routed through ``Holo3StepHandler``
gave the brain the full action vocabulary (click, type, scroll, key)
inside an N-step inner loop. When visible state didn't change
between scroll dispatches (e.g. the page renders everything above
the fold, or Holo3 saw a stale screenshot), the brain would fall
back to clicking visible elements to "make something happen" —
landing on listing cards / ad-links and navigating the tab away
from the anchored URL.

Live repro: boattrader run 20260521_042509_b358f06f. The plan's
"scroll down once" step burned its budget=4 doing
``scroll x3 → scroll x3 → click(983, 316) → click(976, 312)``;
the final click hit the "Boat Loans" ad in the search header and
navigated the tab to ``/boat-loans/``. Subsequent ``extract_data``
then failed against the wrong page, triggering recovery cascades.

This handler bypasses the brain entirely for scroll steps that
specify a concrete ``count``. The runner dispatches ``count`` scroll
notches via the env's existing ``ActionType.SCROLL`` primitive,
verifies the scroll actually moved the viewport via post-action
``window.scrollY`` readback, and returns success/failure
deterministically.

Plans that have a vision-mediated scroll intent (e.g. "scroll until
the Submit button is visible") can OMIT the ``count`` param and the
runner will fall through to ``Holo3StepHandler`` — the brain still
owns goal-shaped scrolling. Mechanical scroll is opt-in via
``params.count`` being set.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ...actions import Action, ActionType
from .. import adaptive_settle
from ..checkpoint import StepResult
from ..step_context import StepContext

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner
    from ...plan_decomposer import MicroIntent

logger = logging.getLogger(__name__)


class MechanicalScrollHandler:
    """Implements :class:`~..step_context.StepHandler` for ``scroll``
    steps that specify an explicit ``params.count`` (and optionally
    ``params.direction``).

    Behaviour:

    * Reads ``params.count`` (int, default falls through to Holo3),
      ``params.direction`` (``"down"`` | ``"up"``, default ``"down"``),
      and ``params.notches_per_count`` (int, default 3 — matches the
      env's ``ActionType.SCROLL`` default amount).
    * Dispatches ``count`` ``ActionType.SCROLL`` actions through the
      env (which fans out to xdotool wheel events under Xvfb).
    * Runs the standard ``settle_after_action`` between dispatches
      so a slow framework can stabilise before the next scroll.
    * Verifies via post-action ``window.scrollY`` readback when the
      env exposes ``cdp_evaluate``: a successful "scroll down" must
      move ``scrollY`` forward by at least 50px total. If the env
      doesn't expose CDP (test stub), success defers to
      ``env.step()`` not raising.

    Returns ``StepResult.success = True`` on verified scroll motion;
    ``False`` with ``failure_class="scroll_no_movement"`` when the
    page didn't move at all (page at top/bottom, overflow:hidden
    body, sub-element scroller swallowing wheel events). The
    deterministic failure class is more actionable to the recovery
    layer than the previous ``brain_loop_exhausted``.
    """

    step_type = "scroll"

    def __init__(self, runner: "MicroPlanRunner") -> None:
        self.parent = runner

    def applies_to(self, step: "MicroIntent") -> bool:
        """Mechanical scroll only takes over when ``params.count`` is
        provided. Plans without an explicit count fall through to
        the legacy Holo3-mediated path so vision-goal scrolling
        (``"scroll until X visible"``) keeps working.
        """
        params = step.params or {}
        count = params.get("count")
        try:
            return count is not None and int(count) > 0
        except (TypeError, ValueError):
            return False

    def execute(self, step: "MicroIntent", ctx: StepContext) -> StepResult:
        index = int(ctx.state.get("index", 0))
        env = ctx.env
        params = step.params or {}

        try:
            count = max(1, int(params.get("count", 1)))
        except (TypeError, ValueError):
            count = 1
        direction = str(params.get("direction") or "down").lower().strip()
        if direction not in ("down", "up"):
            direction = "down"
        try:
            notches_per_count = max(
                1, int(params.get("notches_per_count", 3)),
            )
        except (TypeError, ValueError):
            notches_per_count = 3

        cdp_eval = getattr(env, "cdp_evaluate", None)

        def _read_scroll_y() -> float:
            if not callable(cdp_eval):
                return -1.0  # sentinel — CDP unavailable, skip verification
            try:
                v = cdp_eval(
                    "(window.scrollY || document.documentElement.scrollTop || 0)"
                )
                return float(v) if v is not None else 0.0
            except Exception:  # noqa: BLE001
                return -1.0

        pre_y = _read_scroll_y()
        logger.info(
            "  [scroll] mechanical dispatch: count=%d direction=%s "
            "notches/count=%d pre_scrollY=%s",
            count, direction, notches_per_count,
            f"{pre_y:.0f}" if pre_y >= 0 else "n/a",
        )

        for i in range(count):
            try:
                env.step(Action(
                    action_type=ActionType.SCROLL,
                    params={
                        "direction": direction,
                        "amount": notches_per_count,
                    },
                ))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "  [scroll] env.step failed on iteration %d/%d: %s",
                    i + 1, count, exc,
                )
                return StepResult(
                    step_index=index, intent=step.intent, success=False,
                    data=f"scroll_dispatch_error:{type(exc).__name__}",
                    failure_class="scroll_dispatch_error",
                )
            # Cheap settle between iterations so a slow framework can
            # catch up. Bounded — adaptive_settle picks the right
            # ceiling per page based on previous frame hash deltas.
            adaptive_settle.settle_after_action(env, max_seconds=1.5)

        post_y = _read_scroll_y()
        # Verify via post-action scrollY readback when CDP is wired up.
        if pre_y >= 0 and post_y >= 0:
            delta = post_y - pre_y if direction == "down" else pre_y - post_y
            if delta < 50:
                logger.warning(
                    "  [scroll] mechanical scroll fired but scrollY didn't "
                    "move (pre=%.0f post=%.0f delta=%.0f); failing "
                    "deterministically so recovery sees a real signal",
                    pre_y, post_y, delta,
                )
                return StepResult(
                    step_index=index, intent=step.intent, success=False,
                    data=f"scroll_no_movement:pre={pre_y:.0f}_post={post_y:.0f}",
                    failure_class="scroll_no_movement",
                )
            logger.info(
                "  [scroll] verified scroll motion: scrollY %.0f → %.0f "
                "(delta=%.0f)",
                pre_y, post_y, delta,
            )

        return StepResult(
            step_index=index, intent=step.intent, success=True,
            data=f"scroll:{direction}x{count}",
        )
