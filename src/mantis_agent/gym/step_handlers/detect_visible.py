"""DetectVisibleHandler — vision-only yes/no element existence check.

Why this exists: plans that include optional steps (``click(Show More)``
when Show More may or may not be rendered, ``click(Accept cookies)``
when the banner may already be dismissed, ``select(Older messages)``
when the toggle is only there for long threads) today burn full
brain-budget vision passes trying to LOCATE a target that isn't on
the page, hit ``brain_loop_exhausted``, then skip. Cost: ~$0.10-0.30
per missing-target step. Across a 60-listing run that's ~$10-20 of
pure waste.

This handler takes one screenshot, asks Claude/Holo3 a single
yes/no question ("is this element visible?"), and binds the boolean
to ``runner._state_vars[step.out_var]``. Subsequent steps with
``step.guard`` naming that variable are skipped for free when the
guard evaluates False — no vision call, no env action.

Cost per detect: ~$0.02-0.05 (one Claude verify-shaped call).
Compared to the multi-attempt brain-loop on missing targets, this
is a 3-10× cost reduction on optional paths.

CUA-purity (``feedback_cua_no_dom_access.md``): the detection is one
vision call against the current screenshot. The plan's NL intent
prose describes what to look for ("Is a 'Show More' toggle visible
inside the Description block?"). No DOM access, no selector lookup
— purely screenshot-grounded.

Output is stashed on ``runner._state_vars[out_var]`` as a bool, AND
emitted in ``StepResult.data`` for trace visibility. The detect step
itself is never marked failed when the answer is False — "absent"
is a valid finding, not a failure. The success boolean reflects
whether the vision call returned a parseable result; the answer
(yes/no) lives in ``_state_vars``.
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


class DetectVisibleHandler:
    """Implements :class:`~..step_context.StepHandler` for ``detect_visible``.

    Required step fields:
        * ``intent`` — natural-language description of what to look
          for (e.g. *"Is a 'Show More' toggle visible inside the
          Description block?"*). Used verbatim as the vision-call
          prompt.
        * ``out_var`` — name of the runner state variable that
          receives the boolean. Subsequent steps with ``guard ==
          out_var`` will skip when the variable is False.

    Optional:
        * No params today. Future-proofing slots: ``params.threshold``
          (confidence floor), ``params.invert`` (treat absent as
          True), etc. — add when a real plan needs them.
    """

    step_type = "detect_visible"

    def __init__(self, runner: "MicroPlanRunner") -> None:
        self.parent = runner

    def execute(self, step: "MicroIntent", ctx: StepContext) -> StepResult:
        runner = self.parent
        env = ctx.env
        extractor = ctx.extractor
        index = int(ctx.state.get("index", 0))
        out_var = (step.out_var or "").strip()

        if not out_var:
            logger.warning(
                "  [detect_visible] step has no ``out_var`` — result "
                "would be unreachable; skipping vision call"
            )
            return StepResult(
                step_index=index, intent=step.intent, success=False,
                data="detect_visible:no_out_var", skip=True,
                skip_reason="detect_visible_no_out_var",
            )

        if not extractor:
            logger.warning(
                "  [detect_visible] no extractor wired — defaulting "
                "%s=False so dependent steps skip safely",
                out_var,
            )
            self._bind(runner, out_var, False)
            return StepResult(
                step_index=index, intent=step.intent, success=False,
                data=f"detect_visible:{out_var}=False:no_extractor",
            )

        screenshot = env.screenshot()
        # Reuse ``verify_gate`` as the vision primitive — same
        # input shape (screenshot + NL expected condition), same
        # tool-schema-validated (passed: bool, reason: str) output.
        # The "gate" semantics align: detect_visible is conceptually a
        # gate that doesn't halt; the runner reads the result into a
        # variable instead of treating False as a fatal failure.
        try:
            passed, reason = extractor.verify_gate(
                screenshot, step.intent or "Is the described element visible?",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "  [detect_visible] verify_gate raised — defaulting "
                "%s=False (%s)", out_var, type(exc).__name__,
            )
            self._bind(runner, out_var, False)
            return StepResult(
                step_index=index, intent=step.intent, success=False,
                data=f"detect_visible:{out_var}=False:exception",
            )

        runner.costs["claude_extract"] = (
            runner.costs.get("claude_extract", 0) + 1
        )
        self._bind(runner, out_var, bool(passed))

        logger.warning(
            "  [detect_visible] %s = %s — %s",
            out_var, "True" if passed else "False",
            (reason or "")[:80],
        )

        return StepResult(
            step_index=index, intent=step.intent, success=True,
            data=f"detect_visible:{out_var}={'True' if passed else 'False'}",
        )

    @staticmethod
    def _bind(runner: "MicroPlanRunner", name: str, value: bool) -> None:
        """Stash the boolean on the runner's plan-state variable bag.

        Defaults the bag to empty dict if absent — the runner's init
        creates ``_state_vars`` but defensive in case an older
        runner-shim caller bypassed init.
        """
        state_vars = getattr(runner, "_state_vars", None)
        if state_vars is None:
            state_vars = {}
            runner._state_vars = state_vars
        state_vars[name] = value
