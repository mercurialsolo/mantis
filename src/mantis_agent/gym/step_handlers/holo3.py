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
from typing import TYPE_CHECKING, Any

from .. import retry_attempts as _retry
from ..checkpoint import StepResult
from ..runner import GymRunner
from ..step_context import StepContext

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner
    from ...plan_decomposer import MicroIntent

logger = logging.getLogger(__name__)


# The shared retry-attempts module owns the window cap + line format
# (#435 item 7: Claude / Fara prompt builders use the same shape).
# Re-export the private helpers here for back-compat with existing
# tests that import them from this module.
_PRIOR_ATTEMPTS_WINDOW = _retry._PRIOR_ATTEMPTS_WINDOW
_format_prior_attempt = _retry.format_prior_attempt


def _build_scoped_task(step: "MicroIntent", runner: Any, step_index: int) -> str:
    """Construct the Holo3 task string with explicit sub-goal scope
    (roadmap P1 #6).

    Replaces the old ``step.intent + hint_block`` concatenation that
    forced Holo3 to infer success criteria, target shape, and prior-
    failure context from a single sentence + opaque ``click(x,y)``
    history. The new shape surfaces:

    * the sub-goal (``step.intent``);
    * the success criterion (``step.verify``) — what Holo3 should
      check before reporting done;
    * structured target hints (``step.params`` keys: ``label``,
      ``kind``, ``aliases``; ``step.hints`` keys: ``region``,
      ``near``, ``layout``) — keeps Holo3 from re-deriving
      semantics the decomposer already extracted;
    * up to ``_PRIOR_ATTEMPTS_WINDOW`` prior attempts at THIS sub-
      goal, each tagged with its failure outcome (no_state_change /
      wrong_target / brain_loop_exhausted) so the brain can refute
      the same coordinates / pattern;
    * any agentic-recovery hints accumulated by the recovery loop
      (Claude's analysis of the failure screenshot — typically the
      most specific signal).

    Each section is omitted when empty so a "fresh" step with no
    structured fields produces a short intent-only string — same
    semantics as the pre-#hub change but with the framing the
    rest of the prompt expects when richer context IS available.
    """
    sections: list[str] = []
    intent = str(getattr(step, "intent", "") or "").strip()
    sections.append(f"Sub-goal: {intent}" if intent else "Sub-goal: (unspecified)")

    verify = str(getattr(step, "verify", "") or "").strip()
    if verify:
        sections.append(f"Success criterion: {verify}")

    params = dict(getattr(step, "params", {}) or {})
    hints = dict(getattr(step, "hints", {}) or {})
    target_lines: list[str] = []
    label = str(params.get("label") or "").strip()
    if label:
        target_lines.append(f"  label: {label}")
    kind = str(params.get("kind") or "").strip()
    if kind:
        target_lines.append(f"  kind: {kind}")
    aliases = params.get("aliases") or []
    if isinstance(aliases, (list, tuple)) and aliases:
        alias_str = ", ".join(str(a).strip() for a in aliases if str(a).strip())
        if alias_str:
            target_lines.append(f"  aliases: {alias_str}")
    region = str(hints.get("region") or "").strip()
    if region:
        target_lines.append(f"  region: {region}")
    near = str(hints.get("near") or "").strip()
    if near:
        target_lines.append(f"  near: {near}")
    layout = str(hints.get("layout") or "").strip()
    if layout:
        target_lines.append(f"  layout: {layout}")
    if target_lines:
        sections.append("Target hints:\n" + "\n".join(target_lines))

    failure_history = (
        runner._step_failure_history.get(step_index, [])
        if hasattr(runner, "_step_failure_history") else []
    )
    if isinstance(failure_history, list) and failure_history:
        recent = [r for r in failure_history if isinstance(r, dict)][-_PRIOR_ATTEMPTS_WINDOW:]
        if recent:
            lines = [f"  {i + 1}. {_format_prior_attempt(r)}" for i, r in enumerate(recent)]
            sections.append(
                "Previous attempts at THIS sub-goal (most recent last) — "
                "do NOT repeat these coordinates / patterns:\n" + "\n".join(lines)
            )

    from .. import recovery_hints as _hints
    hint_block = _hints.get_hint_block(runner, step_index)
    if hint_block:
        # ``get_hint_block`` already includes its own leading header;
        # strip the leading newlines so the section join below stays
        # consistent with the others.
        sections.append(hint_block.lstrip("\n"))

    return "\n\n".join(sections)


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
        # Roadmap P1 #6: build a scoped task string with explicit
        # sub-goal, success criterion, structured target hints, and
        # outcome-tagged prior failures. The old "intent + hint_block"
        # concat made Holo3 infer all of this from a single sentence
        # plus opaque click(x,y) history; with the new shape the brain
        # sees the same context the outer recovery loop already has.
        task = _build_scoped_task(step, runner, index)
        # #435 item 7: also thread the structured failure records through
        # to the inner GymRunner so EVERY brain.think() iteration sees
        # the outcome-tagged ``Recent attempts on this sub-goal`` block
        # — not just the first frame off the scoped task string. The
        # task string still carries it for compat with prompts that
        # didn't migrate; the kwarg is the structured form that the
        # brain adapters render via the shared ``retry_attempts``
        # module.
        prior_attempts = (
            runner._step_failure_history.get(index, [])
            if hasattr(runner, "_step_failure_history") else []
        )
        retry_attempts_list = (
            [r for r in prior_attempts if isinstance(r, dict)]
            if isinstance(prior_attempts, list) else None
        )
        result = gym_runner.run(
            task=task,
            task_id=f"step_{index}_{step.type}",
            pending_form_labels=outer_pending_labels,
            retry_attempts=retry_attempts_list,
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
        # #419 audit-triple: surface the brain's articulated reasoning
        # for the action that drove this StepResult. The inner GymRunner
        # ran some number of think→act iterations; the FINAL trajectory
        # step's thinking is what produced the committed action — that's
        # the reasoning that matters for post-mortem triage. Earlier
        # iterations are visible per-step in the inner trajectory dump.
        # Empty string when the trajectory is empty (paused immediately,
        # no brain inference happened) so pack_step's truthiness check
        # cleanly omits the field.
        reasoning = ""
        if result.trajectory:
            reasoning = result.trajectory[-1].thinking or ""
        return StepResult(
            step_index=index,
            intent=step.intent,
            success=success,
            steps_used=result.total_steps,
            duration=result.total_time,
            failure_class=failure_class,
            reasoning=reasoning,
        )
