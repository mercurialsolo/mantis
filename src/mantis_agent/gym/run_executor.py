"""RunExecutor — the run-loop body extracted from MicroPlanRunner.run.

Phase 3 of EPIC #161 (refactor MicroPlanRunner into composable
modules). Lifts the ~300-LOC while loop and its surrounding init /
resume / finalize plumbing off ``MicroPlanRunner.run`` into a single
``RunExecutor.execute`` method.

After this PR, ``MicroPlanRunner.run`` becomes a thin wrapper:

    state = RunState.fresh(self.run_key, self.session_name, self.plan_signature)
    self._executor.execute(plan, state, resume=resume)
    self._final_summary(state)
    return state.results

Naming choice: this is a *run* executor (drives ``run()``'s loop), not a
*plan* executor — the existing ``plan_executor.py`` already houses an
unrelated Playwright-based executor. Naming the new module
``run_executor.py`` keeps both modules importable without collision.

The acceptance criterion from EPIC #161 — ``micro_runner.py ≤ 200 LOC
after Phase 3`` — needs Phase 4 (de-leak listings) to fully collapse,
since ClickHandler / PaginateHandler still hold parent back-references
to listings-specific state. This PR brings the runner LOC down
substantially and gets the executor unit-testable with a fake handler
registry.

Architecture mirrors :class:`~.step_recovery.StepRecoveryPolicy` and
:class:`~.checkpoint_manager.CheckpointManager`: thin shim with a
parent ``MicroPlanRunner`` back-reference. The executor reads
collaborators (env, brain, extractor, dynamic_verifier, site_config,
handler_registry, recovery_policy, costs, browser_state) from the
parent. Phase 4 will inject these directly via :class:`StepContext`
so the back-reference can collapse.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..actions import Action, ActionType
from ..plan_decomposer import MicroIntent
from . import step_snapshot
from .checkpoint import RunCheckpoint, StepResult

if TYPE_CHECKING:
    from ..plan_decomposer import MicroPlan
    from .micro_runner import MicroPlanRunner

logger = logging.getLogger("mantis_agent.gym.micro_runner")


@dataclass
class RunState:
    """Mutable per-run state the executor mutates as it advances.

    Replaces the local variables that used to live in ``run()``'s scope
    (``step_index``, ``results``, ``loop_counters``, ``listings_on_page``,
    ``step_retry_counts``). Persisted via ``RunCheckpoint`` between
    steps; the checkpoint manager reads through it.

    Created fresh by ``MicroPlanRunner.run`` per invocation; the
    executor mutates fields in place. Exposed as a dataclass (not
    a closure) so tests can inspect or seed state.
    """

    checkpoint: RunCheckpoint
    step_index: int = 0
    results: list[StepResult] = field(default_factory=list)
    loop_counters: dict[int, int] = field(default_factory=dict)
    listings_on_page: int = 0
    step_retry_counts: dict[Any, int] = field(default_factory=dict)
    max_loop_iterations: int = 200  # safety cap

    @classmethod
    def fresh(cls, run_key: str, session_name: str, plan_signature: str) -> RunState:
        """Build the initial state at the start of ``run()``."""
        return cls(
            checkpoint=RunCheckpoint(
                run_key=run_key,
                plan_signature=plan_signature,
                session_name=session_name,
            ),
        )


class RunExecutor:
    """Drive a :class:`MicroPlan` through completion, halt, pause, or cancel.

    The class owns the while-loop body. Side effects (env.step calls,
    runner-method delegations like ``_execute_step`` /
    ``_persist_checkpoint`` / ``_capture_screenshot_bytes`` /
    ``_record_step_costs`` / ``_log_progress``) happen via the parent
    back-reference. The runner is the only side-effect agent; the
    executor sequences the calls.

    Phase 4 follow-up work removes the parent back-reference by
    injecting the collaborators directly. This PR keeps the back-
    reference so the lift stays mechanical (the same pattern used by
    BrowserState / CheckpointManager / StepRecoveryPolicy).
    """

    def __init__(self, parent: "MicroPlanRunner") -> None:
        self.parent = parent

    # ── Public entry ────────────────────────────────────────────────────

    def execute(
        self,
        plan: "MicroPlan",
        state: RunState,
        *,
        resume: bool = False,
    ) -> list[StepResult]:
        """Drive ``plan`` to completion. Mutates ``state`` in place.

        Returns the list of :class:`StepResult` for callers that want
        to consume them directly; the same list is also accessible on
        ``state.results``.

        ``resume=True`` loads the persisted checkpoint at the runner's
        ``checkpoint_path`` and reconstructs ``state.results`` /
        ``state.loop_counters`` / ``state.listings_on_page`` /
        ``state.step_index`` from it before the loop starts.
        """
        runner = self.parent
        runner._final_status = "running"
        if not runner.plan_signature:
            runner.plan_signature = runner._compute_plan_signature(plan)
            state.checkpoint.plan_signature = runner.plan_signature

        if resume or runner.resume_state:
            self._maybe_resume(plan, state)
            if state.checkpoint.status == "completed":
                logger.info(
                    "Checkpoint already marked complete; returning cached results"
                )
                return state.results

        if not runner._results_base_url and plan.steps:
            runner._results_base_url = runner._extract_url_from_intent(
                plan.steps[0].intent
            )
            runner._required_filter_tokens = runner._derive_filter_tokens(
                runner._results_base_url
            )
            runner.dynamic_verifier.set_required_filter_tokens(
                runner._required_filter_tokens
            )
            if runner._results_base_url:
                runner.dynamic_verifier.record_page_start(
                    page=runner._current_page,
                    url=runner._current_results_page_url() or runner._results_base_url,
                )

        while state.step_index < len(plan.steps):
            if not self._tick_preamble(plan, state):
                break

            step = plan.steps[state.step_index]
            effective_step = self._build_effective_step(step, state)
            logger.info(
                f"  [{state.step_index:2d}] {step.type:15s} "
                f"{effective_step.intent[:60]}"
            )

            if step.type == "loop":
                self._handle_loop_step(plan, step, state)
                continue

            pre_snapshot = self._dispatch_step(plan, state, effective_step)

            self._maybe_demote_form_no_change(state, effective_step, pre_snapshot)

            step_result = state.results[-1]

            if step_result.data and "DUPLICATE" in step_result.data:
                self._handle_duplicate(plan, state)
                continue

            if step_result.success:
                self._handle_success(plan, state, step, step_result)
            else:
                if not self._handle_failure(plan, state, step, step_result):
                    break

        self._finalize(plan, state)
        return state.results

    # ── Resume from checkpoint ──────────────────────────────────────────

    def _maybe_resume(self, plan: "MicroPlan", state: RunState) -> None:
        runner = self.parent
        loaded = RunCheckpoint.load(runner.checkpoint_path)
        if not loaded:
            return
        if (
            loaded.plan_signature
            and runner.plan_signature
            and loaded.plan_signature != runner.plan_signature
        ):
            logger.warning(
                "Checkpoint signature mismatch at %s; starting fresh",
                runner.checkpoint_path,
            )
            return
        state.checkpoint = loaded
        state.results, state.loop_counters, state.listings_on_page = (
            runner._restore_from_checkpoint(loaded)
        )
        state.step_index = loaded.step_index
        logger.info(
            "Resumed from step %s, page %s, %s URLs seen, status=%s",
            loaded.step_index,
            loaded.current_page or loaded.page,
            len(runner._seen_urls),
            loaded.status,
        )
        if loaded.status == "completed":
            return
        reentry_url = (
            loaded.reentry_url
            or loaded.current_url
            or runner._reentry_url_for_step(plan, loaded.step_index)
        )
        if loaded.step_index > 0 and reentry_url:
            runner._resume_browser_state(reentry_url)

    # ── Loop preamble: cancel, pause, budget, time ──────────────────────

    def _tick_preamble(self, plan: "MicroPlan", state: RunState) -> bool:
        """Return False to break the while loop (cancelled / paused / capped)."""
        runner = self.parent
        if runner._is_cancelled():
            logger.info(
                "  CANCEL_EVENT set — stopping at step %s", state.step_index
            )
            print(
                f"  CANCEL: external cancel_event fired — "
                f"stopping at step {state.step_index}"
            )
            self._persist(plan, state, status="cancelled", halt_reason="cancel_event")
            runner._final_status = "cancelled"
            return False

        if runner.tool_channel.is_paused():
            self._persist(plan, state, status="paused", halt_reason="user_input")
            runner._final_status = "paused"
            return False

        elapsed = time.time() - runner._run_start
        _gpu, _claude, _proxy, total_cost = runner._cost_totals()

        if total_cost >= runner.max_cost:
            print(
                f"  BUDGET CAP: ${total_cost:.2f} >= "
                f"${runner.max_cost:.2f} — stopping"
            )
            self._persist(plan, state, status="halted", halt_reason="budget_cap")
            return False
        if elapsed >= runner.max_time:
            print(
                f"  TIME CAP: {elapsed/60:.0f}m >= "
                f"{runner.max_time/60:.0f}m — stopping"
            )
            self._persist(plan, state, status="halted", halt_reason="time_cap")
            return False
        return True

    # ── Effective step + loop step ──────────────────────────────────────

    def _build_effective_step(
        self, step: MicroIntent, state: RunState,
    ) -> MicroIntent:
        """Inject listing-position scroll directive on click steps.

        EPIC #161 Phase 4: the directive string is now built by
        :meth:`ListingsScanner.scroll_directive` so the listings-specific
        wording lives where the scanner does. Called only for ``click``
        steps; other types short-circuit and use ``step.intent`` as-is.
        """
        runner = self.parent
        dynamic_intent = step.intent
        if step.type == "click":
            scanner_directive = (
                runner.scanner.scroll_directive_for(state.listings_on_page)
            )
            if scanner_directive:
                dynamic_intent = scanner_directive
        return MicroIntent(
            intent=dynamic_intent, type=step.type, verify=step.verify,
            budget=step.budget, reverse=step.reverse, grounding=step.grounding,
            claude_only=step.claude_only, loop_target=step.loop_target,
            loop_count=step.loop_count,
            section=step.section, required=step.required, gate=step.gate,
            params=dict(step.params or {}),
            hints=dict(getattr(step, "hints", {}) or {}),
        )

    def _handle_loop_step(
        self, plan: "MicroPlan", step: MicroIntent, state: RunState,
    ) -> None:
        state.loop_counters[state.step_index] = (
            state.loop_counters.get(state.step_index, 0) + 1
        )
        count = state.loop_counters[state.step_index]
        max_count = step.loop_count or state.max_loop_iterations
        if count < max_count:
            target = step.loop_target if step.loop_target >= 0 else state.step_index
            state.step_index = target
            logger.info(
                f"  [loop@{state.step_index}] iteration {count}/{max_count} → "
                f"step {state.step_index}"
            )
        else:
            logger.info("  [loop] max iterations reached")
            state.step_index += 1
        self._persist(plan, state)

    # ── Step dispatch + observability ───────────────────────────────────

    def _dispatch_step(
        self, plan: "MicroPlan", state: RunState, effective_step: MicroIntent,
    ) -> Any:
        """Capture pre-step snapshot, call _execute_step, append result."""
        runner = self.parent
        runner._active_checkpoint_context = {
            "checkpoint": state.checkpoint,
            "plan": plan,
            "results": state.results,
            "loop_counters": state.loop_counters,
            "listings_on_page": state.listings_on_page,
            "step_index": state.step_index,
        }
        pre_snapshot = step_snapshot.capture(runner)
        runner._pre_step_snapshot = pre_snapshot
        try:
            step_result = runner._execute_step(effective_step, state.step_index)
        finally:
            runner._active_checkpoint_context = None
        if step_result.screenshot_png is None:
            step_result.screenshot_png = runner._capture_screenshot_bytes()
        state.results.append(step_result)
        runner._enforce_screenshot_cap(state.results)
        runner._invoke_step_callback(step_result)
        runner._record_step_costs(effective_step, step_result)
        runner._log_progress(step_result, state.results)
        runner._log_step_diff(pre_snapshot, effective_step, step_result)
        _emit_action_metric(runner, effective_step, step_result)
        if not step_result.success:
            runner._dump_debug_screenshot(
                f"step{state.step_index}_post_{effective_step.type}",
                runner._safe_screenshot(),
            )
        return pre_snapshot

    def _maybe_demote_form_no_change(
        self, state: RunState, effective_step: MicroIntent, pre_snapshot: Any,
    ) -> None:
        """Demote submit success → fail when no observable change.
        Preserves the staffcrm-verify follow-up behavior.

        ``select_option`` is intentionally excluded: changing a dropdown
        value almost never produces a URL/scroll/page-counter delta —
        the only delta is the field value itself, which the snapshot
        doesn't track. The select_option handler already verifies that
        the open menu was found and the option clicked; demoting on
        no-state-change there caused valid Industry-Vertical edits on
        staff-crm to retry until the budget exhausted.
        """
        runner = self.parent
        step_result = state.results[-1]
        if not (
            step_result.success
            and effective_step.type == "submit"
        ):
            return
        try:
            post_snapshot = step_snapshot.capture(runner)
            delta = step_snapshot.diff(pre_snapshot, post_snapshot)
        except Exception as exc:  # noqa: BLE001 — never break runs
            logger.debug("post-submit diff capture failed: %s", exc)
            delta = None
        if delta is not None and not delta.has_changes:
            # Before demoting, consult the SPA-aware visual verifier:
            # CRM-style logins (staff-crm is the canonical case) replace
            # the login form with a dashboard at the SAME URL, so the
            # runner-state snapshot sees no delta even though the
            # submit succeeded. The submit handler stashes the
            # pre-click screenshot at ``runner._last_submit_pre_screenshot``;
            # take a post-screenshot now and ask the extractor to
            # compare. Pattern mirrors the click-handler's
            # ``verify_post_click_navigation`` path from PR #222.
            if self._submit_visually_changed(runner, effective_step):
                logger.info(
                    "  [%d] submit had no runner-state delta but visual "
                    "verifier confirms UI change — keeping success",
                    state.step_index,
                )
                # Clear the stash so the next step starts fresh.
                runner._last_submit_pre_screenshot = None
                return
            logger.warning(
                "  [%d] %s reported success but no observable state "
                "change — demoting to failure (will retry)",
                state.step_index, effective_step.type,
            )
            step_result.success = False
            step_result.data = (step_result.data or "") + ":no_state_change"
            # Record the failed click target into the per-step failure
            # history so the next retry's ``find_form_target`` call
            # avoids the same broken target. Without this, retries
            # blindly re-pick the same coordinates and fail
            # identically — the canonical case is staff-crm's "Click
            # Qualified" step where the label-text matched a status
            # pill rather than the row link.
            self._record_failure_for_retry(
                runner=runner,
                step_index=state.step_index,
                kind="no_state_change",
                reason="snapshot diff and visual verifier both saw no UI change",
            )
            # Pattern-match the failure history for handler escalation.
            # When the same step has accumulated 2+ same-kind failures,
            # the default handler is locked on the wrong element class
            # (e.g. text-matching a status pill instead of a row link).
            # The next retry should route through a different handler
            # — currently Holo3StepHandler, which uses brain grounding
            # over the intent prose rather than text matching.
            self._maybe_set_handler_override(
                runner=runner, step_index=state.step_index,
            )
        # Always clear the per-step stashes — they're only valid for
        # the immediately following demotion check.
        if hasattr(runner, "_last_submit_pre_screenshot"):
            runner._last_submit_pre_screenshot = None
        if hasattr(runner, "_last_submit_target"):
            runner._last_submit_target = None

    @staticmethod
    def _record_failure_for_retry(
        *,
        runner: Any,
        step_index: int,
        kind: str,
        reason: str,
    ) -> None:
        """Append a failure record for the agentic retry path.

        The record carries the (x, y) coordinates and label the
        previous attempt clicked, so on retry the form handler
        can tell ``find_form_target`` "avoid this target." Records
        accumulate per step_index and are cleared on success
        (see ``_handle_success``).
        """
        target = getattr(runner, "_last_submit_target", None)
        if not target:
            # The form handler didn't surface a click target — nothing
            # to feed back. Common reason: ``form_target_not_found``
            # already-failed step; the retry path's no-coordinate
            # fallback handles it.
            return
        if not hasattr(runner, "_step_failure_history"):
            return
        history = runner._step_failure_history.setdefault(step_index, [])
        history.append({
            "x": target.get("x"),
            "y": target.get("y"),
            "label": target.get("label", ""),
            "matched_label": target.get("matched_label", ""),
            "kind": kind,
            "reason": reason,
        })

    @staticmethod
    def _maybe_set_handler_override(
        *, runner: Any, step_index: int,
    ) -> None:
        """Decide whether the next retry should route through a
        different handler based on the failure history pattern.

        The escalation triggers when the per-step failure history
        accumulates 2+ records of the same ``kind`` (currently only
        ``no_state_change`` triggers escalation — that's the strongest
        signal that the default handler is finding text matches that
        don't actually navigate). Sets
        ``runner._step_handler_override[step_index] = "holo3"`` so the
        dispatcher routes the next attempt through ``Holo3StepHandler``
        — the brain-grounded loop that operates on intent prose
        rather than label text matching.

        General-purpose: triggered purely by observed failure pattern,
        no hardcoded plan content. Cleared on step success
        (see :meth:`_handle_success`).

        Side effects: mutates ``runner._step_handler_override`` only.
        Idempotent — calling twice with the same trigger produces
        the same override; the dispatcher consumes it once per
        retry attempt.
        """
        if not hasattr(runner, "_step_handler_override"):
            return
        history = (
            runner._step_failure_history.get(step_index, [])
            if hasattr(runner, "_step_failure_history") else []
        )
        # Count kinds eligible for escalation. Only no_state_change
        # signals "the click happened but the page didn't change" —
        # the canonical wrong-handler symptom. Other kinds
        # (form_target_not_found, fill_error, etc.) indicate
        # different problems where Holo3 wouldn't necessarily help.
        no_state_changes = sum(
            1 for r in history if r.get("kind") == "no_state_change"
        )
        if no_state_changes >= 2 and runner.brain is not None:
            runner._step_handler_override[step_index] = "holo3"
            logger.warning(
                "  [escalation] step %d: %d×no_state_change → next "
                "retry will route through Holo3StepHandler (brain "
                "grounding) instead of the default text-match handler",
                step_index, no_state_changes,
            )

    def _submit_visually_changed(
        self, runner: Any, effective_step: MicroIntent,
    ) -> bool:
        """Return True iff a pre/post visual diff confirms the submit
        actually changed the page UI. Used as the SPA-aware escape hatch
        for ``_maybe_demote_form_no_change`` so same-URL form submits
        (CRM logins, in-page settings saves) aren't falsely demoted.

        Returns False on any unavailable input (no extractor / no
        pre-screenshot / API error) so the calling code falls through
        to the existing snapshot-based demotion — the verifier is an
        additive override, never a regression of the existing behavior.
        """
        pre = getattr(runner, "_last_submit_pre_screenshot", None)
        extractor = getattr(runner, "extractor", None)
        if pre is None or extractor is None:
            return False
        verify = getattr(extractor, "verify_post_click_navigation", None)
        if verify is None:
            return False
        try:
            post = runner._safe_screenshot()
        except Exception:
            return False
        if post is None:
            return False
        try:
            nav = verify(pre, post, effective_step.intent)
        except Exception as exc:  # noqa: BLE001 — never break runs
            logger.debug("submit visual verifier raised: %s", exc)
            return False
        # The verifier returns kinds: ``url_change`` / ``modal`` /
        # ``no_change`` / ``wrong_target``. Only the first two count
        # as evidence of a successful submit; ``no_change`` and
        # ``wrong_target`` should still demote.
        if not isinstance(nav, dict):
            return False
        if not nav.get("navigated"):
            return False
        kind = str(nav.get("kind", ""))
        if kind in ("url_change", "modal"):
            runner.costs["claude_extract"] += 1
            return True
        return False

    # ── Outcome handlers ────────────────────────────────────────────────

    def _handle_duplicate(self, plan: "MicroPlan", state: RunState) -> None:
        """extract_url returned DUPLICATE → return to results, jump to loop."""
        runner = self.parent
        logger.info(f"  [{state.step_index}] DEDUP — skipping to next listing")
        try:
            runner._return_to_results_page()
        except Exception:
            pass
        target = self._first_step_of_type(plan, state.step_index + 1, "loop")
        state.step_index = (
            target if target is not None else state.step_index + 1
        )
        state.listings_on_page += 1
        self._persist(plan, state, halt_reason="duplicate_listing")

    def _handle_success(
        self,
        plan: "MicroPlan",
        state: RunState,
        step: MicroIntent,
        step_result: StepResult,
    ) -> None:
        """Success path: paginate reset, navigate_back verify, advance."""
        del step_result  # signature-stable; success values not consumed here
        runner = self.parent
        state.step_retry_counts.pop(state.step_index, None)
        # Clear the agentic-retry failure history for this step on
        # success so warnings don't bleed into a later step that
        # happens to share the same step_index (loops, resumed plans).
        if hasattr(runner, "_step_failure_history"):
            runner._step_failure_history.pop(state.step_index, None)
        # Same for the handler override — once a step has succeeded,
        # the override is consumed and a future occurrence of the
        # same step_index starts with the default routing.
        if hasattr(runner, "_step_handler_override"):
            runner._step_handler_override.pop(state.step_index, None)

        if step.type == "paginate":
            # Phase 4: listings-scan reset is one method on the scanner now,
            # rather than 7 lines of mutation across runner properties.
            state.listings_on_page = 0
            runner.scanner.on_page_change()
            for k in list(state.loop_counters.keys()):
                if k != state.step_index:
                    state.loop_counters[k] = 0
            try:
                runner.env.step(Action(
                    action_type=ActionType.KEY_PRESS, params={"keys": "Home"},
                ))
                time.sleep(8)
            except Exception:
                pass
            logger.info("  [paginate] Success — reset to top of new page")
            runner._last_known_url = (
                runner._current_results_page_url() or runner._last_known_url
            )

        if step.type == "navigate_back" and runner.extractor:
            time.sleep(2)
            screenshot = runner.env.screenshot()
            check = runner.extractor.extract(screenshot)
            runner.costs["claude_extract"] += 1
            url = check.url if check else ""
            if url:
                runner._last_known_url = url
            if url and runner.site_config.is_detail_page(url):
                recovery_intent = step.reverse or "Go back to the previous page."
                logger.warning(
                    f"  [back-verify] Still on detail page — "
                    f"CUA recovery: {recovery_intent[:50]}"
                )
                recovery = runner._execute_holo3_step(
                    MicroIntent(
                        intent=recovery_intent,
                        type="navigate_back",
                        budget=8,
                        grounding=True,
                    ),
                    state.step_index,
                )
                runner.costs["gpu_steps"] += recovery.steps_used

        state.step_index += 1
        self._persist(plan, state)

    def _handle_failure(
        self,
        plan: "MicroPlan",
        state: RunState,
        step: MicroIntent,
        step_result: StepResult,
    ) -> bool:
        """Delegate to StepRecoveryPolicy; returns False if the run should halt."""
        runner = self.parent
        outcome = runner._recovery_policy.handle_failure(
            step=step,
            step_result=step_result,
            plan=plan,
            step_index=state.step_index,
            step_retry_counts=state.step_retry_counts,
            loop_counters=state.loop_counters,
            max_retries=runner.max_retries,
            listings_on_page=state.listings_on_page,
        )
        state.step_index = outcome.step_index
        if outcome.halt:
            self._persist(plan, state, status="halted", halt_reason=outcome.halt_reason)
            return False
        self._persist(plan, state, halt_reason=outcome.halt_reason)
        return True

    # ── Finalize ────────────────────────────────────────────────────────

    def _finalize(self, plan: "MicroPlan", state: RunState) -> None:
        runner = self.parent
        logger.info(f"MicroPlan complete: {len(state.results)} steps executed")
        if state.step_index >= len(plan.steps):
            self._persist(plan, state, status="completed")
            runner._final_status = "completed"
        elif runner._final_status == "running":
            self._persist(plan, state, status="halted", halt_reason="stopped")
            runner._final_status = "halted"

        # Stash final loop counters / progress so resume() / run_with_status()
        # can reconstruct PauseState without re-walking the loop.
        runner._last_run_step_index = state.step_index
        runner._last_loop_counters = dict(state.loop_counters)
        runner._last_listings_on_page = state.listings_on_page

        # #156 loop-termination observability — one increment per run.
        try:
            from ..metrics import LOOP_TERMINATION_TOTAL
            LOOP_TERMINATION_TOTAL.labels(
                tenant_id=getattr(runner, "tenant_id", "") or "",
                reason=runner._final_status or "unknown",
            ).inc()
        except Exception as exc:  # noqa: BLE001 — telemetry must not break runs
            logger.debug("loop termination metric emit failed: %s", exc)

        # #155 step 1 — production trace export (gated on MANTIS_TRACE_EXPORT_DIR).
        # No-op when the env var is unset, so legacy deployments pay nothing.
        try:
            from .trace_exporter import TraceExporter
            TraceExporter.from_env().maybe_export(
                runner, state.results, status=runner._final_status,
            )
        except Exception as exc:  # noqa: BLE001 — telemetry never breaks runs
            logger.debug("trace export failed: %s", exc)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _persist(
        self,
        plan: "MicroPlan",
        state: RunState,
        *,
        status: str = "running",
        halt_reason: str = "",
    ) -> None:
        runner = self.parent
        runner._persist_checkpoint(
            checkpoint=state.checkpoint,
            plan=plan,
            results=state.results,
            loop_counters=state.loop_counters,
            listings_on_page=state.listings_on_page,
            next_step_index=state.step_index,
            status=status,
            halt_reason=halt_reason,
        )

    @staticmethod
    def _first_step_of_type(
        plan: "MicroPlan", start: int, step_type: str,
    ) -> int | None:
        for j in range(start, len(plan.steps)):
            if plan.steps[j].type == step_type:
                return j
        return None


# ── #156 per-action observability ──────────────────────────────────────


def _emit_action_metric(
    runner: Any, step: MicroIntent, step_result: StepResult,
) -> None:
    """Emit ``mantis_action_total`` once per dispatched step.

    Outcome bucketing matches the executor's downstream branching:
    ``duplicate`` → handled by ``_handle_duplicate``; ``filters_not_applied``
    → ``_ensure_results_filters`` short-circuit; ``success`` /  ``failed``
    → the standard happy / sad path. Any new bucket the executor learns
    should also surface here so dashboards stay coherent.
    """
    try:
        from ..metrics import ACTION_TOTAL
        data = (step_result.data or "")
        if "DUPLICATE" in data:
            outcome = "duplicate"
        elif data == "filters_not_applied":
            outcome = "filters_not_applied"
        elif step_result.success:
            outcome = "success"
        else:
            outcome = "failed"
        ACTION_TOTAL.labels(
            tenant_id=getattr(runner, "tenant_id", "") or "",
            step_kind=step.type or "unknown",
            outcome=outcome,
        ).inc()
    except Exception as exc:  # noqa: BLE001 — telemetry must not break runs
        logger.debug("action metric emit failed: %s", exc)
