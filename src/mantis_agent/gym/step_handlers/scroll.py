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

Backend selection (#643 follow-up): when ``hints.prefer_cdp_scroll``
or ``params.backend == "cdp"`` is set, the handler skips xdotool
wheel events entirely and dispatches scroll via the CDP path used
by ``step_recovery``: ``window.scrollBy + scrollingElement.scrollBy
+ PageDown KeyboardEvent``. The CDP path bypasses any inner-element
wheel-capture handlers (SPA results-panel patterns, sticky
overlays) that swallow xdotool wheel events. Use when v5 boattrader
profiling shows the default xdotool path repeatedly hits
``scroll_no_movement`` on a domain — the CDP backend is a
strictly-more-powerful fallback that doesn't require the brain at
all.
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
        """Mechanical scroll takes over when EITHER ``params.count`` is
        provided OR the plan/operator asked for the CDP backend via
        ``hints.prefer_cdp_scroll`` / ``params.backend == "cdp"``.

        The CDP-backend gate exists so plans on domains where vision
        scroll repeatedly hits ``brain_loop_exhausted`` (boattrader
        listings, observed v5: 173 incidents) can pin the deterministic
        CDP path even when they don't want to specify a concrete
        per-step count. Default ``count=1`` applies in that case — one
        ``window.innerHeight`` scroll per step.
        """
        params = step.params or {}
        hints = step.hints or {}
        count = params.get("count")
        backend = str(params.get("backend") or "").lower()
        prefer_cdp = bool(hints.get("prefer_cdp_scroll"))
        if backend == "cdp" or prefer_cdp:
            return True
        try:
            return count is not None and int(count) > 0
        except (TypeError, ValueError):
            return False

    def execute(self, step: "MicroIntent", ctx: StepContext) -> StepResult:
        index = int(ctx.state.get("index", 0))
        env = ctx.env
        params = step.params or {}
        hints = step.hints or {}

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

        # Backend selection: opt into CDP when the operator pinned it
        # via plan hints. CDP scroll bypasses Chrome's wheel handlers
        # (which inner-element scrollers / sticky overlays sometimes
        # swallow) and routes through document-level scroll mechanisms,
        # matching the recovery-layer fallback in step_recovery.py.
        use_cdp = (
            str(params.get("backend") or "").lower() == "cdp"
            or bool(hints.get("prefer_cdp_scroll"))
        )

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
            "notches/count=%d backend=%s pre_scrollY=%s",
            count, direction, notches_per_count,
            "cdp" if use_cdp else "xdotool",
            f"{pre_y:.0f}" if pre_y >= 0 else "n/a",
        )

        if use_cdp:
            if not callable(cdp_eval):
                # CDP backend requested but env doesn't expose it. Fall
                # through to xdotool so the step still has a chance —
                # better than failing outright when the operator's hint
                # was just optimistic.
                logger.warning(
                    "  [scroll] backend=cdp requested but env has no "
                    "cdp_evaluate; falling back to xdotool wheel"
                )
                use_cdp = False

        if use_cdp:
            # Triple-prong CDP scroll dispatch — same payload the
            # recovery layer uses in step_recovery.py:285. Covers
            # <body>-as-scrolling-root pages, <html>-as-scrolling-root
            # pages, and SPA inner scrollers that subscribe to keyboard
            # events. Dispatched ``count`` times for a count-aware
            # scroll budget; each dispatch advances by one
            # ``window.innerHeight``.
            sign = "" if direction == "down" else "-"
            scroll_js = (
                "(function(){"
                "  var h = " + sign + "window.innerHeight;"
                "  window.scrollBy(0, h);"
                "  if (document.scrollingElement) "
                "    document.scrollingElement.scrollBy(0, h);"
                "  document.dispatchEvent(new KeyboardEvent("
                "    'keydown',"
                "    {key:'" + ("PageDown" if direction == "down" else "PageUp") + "', "
                "     code:'" + ("PageDown" if direction == "down" else "PageUp") + "', "
                "     keyCode:" + ("34" if direction == "down" else "33") + ", "
                "     which:" + ("34" if direction == "down" else "33") + ", "
                "     bubbles:true}"
                "  ));"
                "})()"
            )
            for i in range(count):
                try:
                    cdp_eval(scroll_js)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "  [scroll] CDP backend dispatch failed on "
                        "iteration %d/%d: %s",
                        i + 1, count, exc,
                    )
                    return StepResult(
                        step_index=index, intent=step.intent, success=False,
                        data=f"scroll_cdp_dispatch_error:{type(exc).__name__}",
                        failure_class="scroll_dispatch_error",
                    )
                adaptive_settle.settle_after_action(env, max_seconds=1.5)
        else:
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
