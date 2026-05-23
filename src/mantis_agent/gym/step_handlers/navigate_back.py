"""MechanicalNavigateBackHandler — deterministic CDP back navigation.

Why this exists: ``navigate_back`` steps routed through the Holo3
brain handler gave the model an open-ended N-step inner loop for what
is, on every site, the same mechanical action: pop one history entry.
The brain typically burnt ~8 budget steps trying clicks / keypresses /
scrolls before declaring failure, then ``step_recovery`` did up to 3
CDP-back / Alt+Left attempts (with a Claude extract per attempt to
verify the URL). The brain wasn't adding value — the recovery layer's
mechanical attempts were doing the actual navigation.

Live repro: boattrader run 20260523_041241_b12b4194 — extracted 11
leads in 31 minutes; per-iteration the brain's navigate_back burnt
~$0.04 GPU + 3 claude_extract calls (~$0.036) in recovery, ~8 sec
wall. Over 11 iterations that's ~$0.83 and ~88 seconds (~3 minutes
of the run's wall time) spent on a primitive operation.

This handler bypasses the brain by calling ``env.cdp_history_back()``
directly. The CDP primitive already polls for URL change as success
verification (see ``xdotool_env.cdp_history_back``). On success we
return immediately; on CDP unavailability OR landing on another
detail page (history had multiple detail entries), we return failure
so the dispatcher falls through to the Holo3 brain.

Plans that genuinely need the brain to navigate "back" via a custom
UI control (e.g. an in-app breadcrumb on a SPA without history
support) can still get there via fall-through.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..checkpoint import StepResult
from ..step_context import StepContext

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner
    from ...plan_decomposer import MicroIntent

logger = logging.getLogger(__name__)


class MechanicalNavigateBackHandler:
    """Implements :class:`~..step_context.StepHandler` for ``navigate_back``
    steps that can be served by CDP ``window.history.back()`` — i.e.
    most listings-extraction iteration bodies on most sites.

    Behaviour:

    * Reads the env's ``cdp_history_back`` primitive (added in #583).
      Returns ``applies_to=False`` if the env doesn't expose it OR if
      the runner is in ``_opened_detail_in_new_tab`` mode (existing
      ``execute_close_detail_tab`` handler covers that).
    * Reads the current URL via ``runner._best_effort_current_url()``
      (cheap CDP read, no Claude tokens) so we can verify the result.
    * Calls ``cdp_history_back(settle_seconds=2.0)`` — already polls
      for URL change internally and returns True/False.
    * Verifies the new URL isn't still a detail page (history may have
      had multiple detail entries; landing on another detail page is
      a fall-through condition so the brain can re-attempt).

    Returns ``StepResult.success = True`` with ``data=back_via_cdp:<url>``
    on verified motion to a non-detail page; ``False`` with an explicit
    ``failure_class`` when CDP back didn't move the URL or landed on
    another detail page (caller's dispatcher falls through to Holo3).

    WARNING-level log on success so production logs surface the
    cost-savings path (Modal suppresses INFO).
    """

    step_type = "navigate_back"

    def __init__(self, runner: "MicroPlanRunner") -> None:
        self.parent = runner

    def applies_to(self, step: "MicroIntent") -> bool:
        """Mechanical back applies when CDP back is available AND the
        runner isn't in new-tab mode. Plans / runtime configs can also
        opt out via ``params.brain_required=True`` for sites where
        history.back() is known to fail (SPA with intercepted history).
        """
        runner = self.parent
        # New-tab close path is dispatched BEFORE the handler registry
        # in _runner_helpers; if we're in new-tab mode we shouldn't
        # see this step here, but guard defensively.
        if getattr(runner, "_opened_detail_in_new_tab", False):
            return False
        params = step.params or {}
        if params.get("brain_required"):
            return False
        return callable(getattr(runner.env, "cdp_history_back", None))

    def execute(self, step: "MicroIntent", ctx: StepContext) -> StepResult:
        index = int(ctx.state.get("index", 0))
        runner = self.parent
        env = ctx.env

        # Cheap URL read for the pre-state. The env's CDP read is the
        # same primitive cdp_history_back uses internally; we read it
        # again so we can log the transition for production triage.
        url_before = ""
        try:
            url_before = runner._best_effort_current_url() or ""
        except Exception:  # noqa: BLE001 — fall through on any URL-read failure
            pass

        cdp_back = env.cdp_history_back
        moved = False
        try:
            moved = bool(cdp_back(settle_seconds=2.0))
        except Exception as exc:  # noqa: BLE001 — fall through on dispatch failure
            logger.warning(
                "  [back] mechanical CDP back raised %s; falling through",
                type(exc).__name__,
            )
            return StepResult(
                step_index=index, intent=step.intent, success=False,
                data=f"cdp_back_exception:{type(exc).__name__}",
                failure_class="cdp_back_exception",
            )

        if not moved:
            # cdp_history_back polled settle_seconds and didn't see a
            # URL change. Could mean: history was empty, browser
            # intercepted the call, or the SPA didn't update the URL.
            # Fall through to brain.
            logger.info(
                "  [back] mechanical CDP back: URL did not change "
                "from %s — falling through",
                url_before[:60],
            )
            return StepResult(
                step_index=index, intent=step.intent, success=False,
                data="cdp_back_no_url_change",
                failure_class="cdp_back_no_change",
            )

        url_after = ""
        try:
            url_after = runner._best_effort_current_url() or ""
        except Exception:  # noqa: BLE001
            pass
        if url_after:
            runner._last_known_url = url_after

        site_config = getattr(runner, "site_config", None)
        is_detail = (
            url_after
            and site_config is not None
            and callable(getattr(site_config, "is_detail_page", None))
            and site_config.is_detail_page(url_after)
        )
        if is_detail:
            # URL changed but to ANOTHER detail page (history had
            # multiple detail-page entries). Fall through to brain so
            # it can re-attempt or use a different strategy.
            logger.info(
                "  [back] CDP back landed on another detail page (%s) — "
                "falling through to brain",
                url_after[:60],
            )
            return StepResult(
                step_index=index, intent=step.intent, success=False,
                data=f"back_to_detail_page:{url_after[:80]}",
                failure_class="back_to_detail_page",
            )

        logger.warning(
            "  [back] mechanical CDP back: %s → %s",
            url_before[:60] or "?", url_after[:60] or "?",
        )
        return StepResult(
            step_index=index, intent=step.intent, success=True,
            data=f"back_via_cdp:{url_after[:120]}",
        )
