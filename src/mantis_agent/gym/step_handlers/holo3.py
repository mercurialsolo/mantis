"""Holo3StepHandler — fallback to a fresh GymRunner with Holo3 brain.

Phase 2 of EPIC #161, seventh handler extraction. Lifts
``MicroPlanRunner._execute_holo3_step`` (38 LOC) verbatim into a
standalone class. The runner method becomes a delegating shim.

Holo3 is the tactical brain — used as a last resort when Claude can't
identify a clear coordinate (e.g. ``PaginateHandler`` Layer 3 fallback,
unknown step types). The handler spins up a fresh :class:`GymRunner`
bound to the same env / brain / grounding so post-step verification
has clean state.

Post-step verify uses Claude (not Holo3) when ``step.verify`` is set
and an extractor is available — the runner's ``_check_verify`` helper
holds that string-comparison logic and stays on the runner for now
(it's a 30-LOC pure helper used by other handlers too).

What the handler reads from :class:`StepContext`:

- ``env``         — current_url + screenshot for post-step verify
- ``brain``       — Holo3 model passed to the new GymRunner
- ``grounding``   — optional, conditional on ``step.grounding``
- ``extractor``   — Claude verify when ``step.verify`` is set

What it reads from the runner via parent back-reference:

- ``costs``                                — cost meter aliased dict
- ``on_step``                              — viewer callback forwarded to GymRunner
- ``_update_scroll_state_from_trajectory`` — BrowserState delegate
- ``_last_known_url``                      — updated from env.current_url
- ``_check_verify``                        — Claude string-match against a viable
                                             extraction
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..checkpoint import StepResult
from ..runner import GymRunner
from ..step_context import StepContext

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner
    from ...plan_decomposer import MicroIntent

logger = logging.getLogger(__name__)


class Holo3StepHandler:
    """Implements :class:`~..step_context.StepHandler` for Holo3-driven steps.

    Not registered for a specific step type by default — it's invoked
    by other handlers (PaginateHandler.Layer 3) and by the runner
    dispatch when no Claude-guided path applies (e.g. ``scroll`` /
    ``navigate_back``). The cleanup PR will register it for those step
    types after the recovery dispatch swap lands.
    """

    step_type = "holo3"

    def __init__(self, runner: "MicroPlanRunner") -> None:
        self.parent = runner

    def execute(self, step: "MicroIntent", ctx: StepContext) -> StepResult:
        runner = self.parent
        env = ctx.env
        brain = ctx.brain
        extractor = ctx.extractor
        grounding = ctx.grounding
        index = int(ctx.state.get("index", 0))

        gym_runner = GymRunner(
            brain=brain,
            env=env,
            max_steps=step.budget,
            frames_per_inference=1,
            grounding=grounding if step.grounding else None,
            on_step=runner.on_step,
        )

        # #306 follow-up: when the outer MicroPlanRunner tracks pending
        # form labels across the whole plan, forward them so the inner
        # GymRunner's done-gate can reject ``done(success=True)`` on a
        # sub-step that inadvertently claims whole-task completion
        # while outer credentials remain pending. MicroPlanRunner
        # doesn't track this state today; the kwarg gives us a clean
        # plumbing point for when it does.
        outer_pending_labels = getattr(runner, "pending_form_labels", None)
        result = gym_runner.run(
            task=step.intent,
            task_id=f"step_{index}_{step.type}",
            pending_form_labels=outer_pending_labels,
        )

        success = result.success
        runner._update_scroll_state_from_trajectory(result, context=f"holo3_{step.type}")
        current_url = getattr(env, "current_url", "") or ""
        if current_url:
            runner._last_known_url = current_url

        # Post-step verification using Claude (if extractor available)
        if success and step.verify and extractor:
            screenshot = env.screenshot()
            verify_data = extractor.extract(screenshot)
            runner.costs["claude_extract"] += 1
            if verify_data and getattr(verify_data, "url", ""):
                runner._last_known_url = verify_data.url
            verified = runner._check_verify(step.verify, verify_data, screenshot)
            if not verified:
                logger.warning(
                    f"  [verify] Step {index} claimed success but verification "
                    f"FAILED: {step.verify[:60]}"
                )
                success = False

        # Epic #377 Phase A.2: when the inner GymRunner exits because
        # it hit the step budget OR the loop detector tripped — and
        # without success — that's a structural signal. The intent is
        # almost always goal-shaped instead of mechanical (e.g. lu.ma
        # "Scroll down to reveal title, date, location, host details"
        # which made the brain churn 10 steps trying to satisfy a
        # multi-clause goal). Stamping ``brain_loop_exhausted``
        # surfaces this cleanly so the executor / critic can route
        # the next attempt differently (Phase B will rewrite the
        # intent; Phase C will own the routing decision).
        failure_class = ""
        if not success and result.termination_reason in ("max_steps", "loop"):
            failure_class = "brain_loop_exhausted"
        return StepResult(
            step_index=index,
            intent=step.intent,
            success=success,
            steps_used=result.total_steps,
            duration=result.total_time,
            failure_class=failure_class,
        )
