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
from . import failure_class as failure_class_mod
from . import step_snapshot
from .checkpoint import RunCheckpoint, StepResult

if TYPE_CHECKING:
    from ..plan_decomposer import MicroPlan
    from .micro_runner import MicroPlanRunner

logger = logging.getLogger("mantis_agent.gym.micro_runner")


def _pending_form_labels(plan: "MicroPlan", current_step_index: int) -> list[str]:
    """Audit item 2 — collect ``params.label`` from every ``fill_field``
    step in the plan at or after ``current_step_index``.

    The Holo3StepHandler forwards this list to its inner GymRunner so
    the done-gate can reject ``done(success=True)`` on a sub-step that
    inadvertently claims whole-plan completion while form values are
    still pending elsewhere. Without it the gate had no signal and
    Holo3 could short-circuit the run mid-form.

    Deduplicated + ordered by appearance. Labels with empty / whitespace
    text are dropped. Includes the current step itself when its type is
    ``fill_field`` — the sub-runner hasn't completed it yet at the
    moment this helper runs.
    """
    seen: set[str] = set()
    out: list[str] = []
    steps = getattr(plan, "steps", []) or []
    for step in steps[max(0, int(current_step_index)):]:
        if str(getattr(step, "type", "") or "") != "fill_field":
            continue
        params = getattr(step, "params", {}) or {}
        label = str(params.get("label") or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


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
            self._maybe_demote_click_no_change(state, effective_step, pre_snapshot)
            self._maybe_demote_wrong_target(state, effective_step)

            step_result = state.results[-1]

            if step_result.data and "DUPLICATE" in step_result.data:
                self._handle_duplicate(plan, state)
                continue

            if step_result.success:
                self._handle_success(plan, state, step, step_result)
                continued = True
            else:
                continued = self._handle_failure(plan, state, step, step_result)
                if not continued:
                    break

            # Phase C of epic #377: critic observes the post-step state
            # and can emit directives the runner applies. v1 covers
            # navigate_back → InsertStep(navigate). Future capabilities
            # plug in via the same hook.
            self._consult_critic(plan, state, step, step_result, continued)

        self._finalize(plan, state)
        return state.results

    def _consult_critic(
        self,
        plan: "MicroPlan",
        state: RunState,
        step: MicroIntent,
        step_result: StepResult,
        continued: bool,
    ) -> None:
        """Run the ExecutionCritic and apply any returned directive.

        Wrapped in try/except — the critic is observability +
        opportunistic correction; a critic exception must never
        break a run that the recovery policy already decided to
        continue.
        """
        runner = self.parent
        critic = getattr(runner, "_critic", None)
        if critic is None:
            return
        try:
            from . import critic as _critic_mod
            directive = critic.observe_step(
                plan, state, step, step_result,
                recovery_continued=continued,
            )
            if directive is not None:
                _critic_mod.apply_directive(runner, plan, state, directive)
        except Exception as exc:  # noqa: BLE001 — never break runs
            logger.debug("critic raised: %s", exc)

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
        """Inject listing-position scroll directive on click steps, or
        the IntentRewriter's proposed mechanical intent (epic #377
        Phase B) when the prior attempt's failure_class triggered a
        rewrite.

        Override priority:

        1. ``_step_intent_overrides[step_index]`` — set by the
           IntentRewriter when the previous attempt's failure_class
           was in ``REWRITE_TRIGGERING_CLASSES``. Wins over the
           scanner directive because the rewrite was proposed in
           light of a specific failure.
        2. Scanner scroll directive — listings-specific positional
           hint for click steps.
        3. ``step.intent`` as-authored.
        """
        runner = self.parent
        dynamic_intent = step.intent
        # Defensive: tests that hand the executor a MagicMock runner
        # see ``_step_intent_overrides`` auto-created as a Mock.
        # Require a real dict + a non-empty string override before we
        # swap the intent.
        overrides = getattr(runner, "_step_intent_overrides", None)
        override = overrides.get(state.step_index) if isinstance(overrides, dict) else None
        if isinstance(override, str) and override:
            dynamic_intent = override
        elif step.type == "click":
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
        # #audit item 2: populate ``pending_form_labels`` from the
        # remaining plan steps so the Holo3 sub-runner's done-gate
        # can reject ``done(success=True)`` while outer ``fill_field``
        # steps remain. Without this the gate had no signal and
        # Holo3 could claim whole-plan completion mid-step.
        runner.pending_form_labels = _pending_form_labels(plan, state.step_index)
        pre_snapshot = step_snapshot.capture(runner)
        runner._pre_step_snapshot = pre_snapshot
        # Epic #377 follow-up (#381): read the browser's URL directly
        # from the env BEFORE the step executes. The snapshot's ``url``
        # field reads ``runner._last_known_url``, which the click
        # handler can self-mutate on its SPA-aware verify path —
        # making the snapshot diff a misleading proxy for "did the
        # world actually change?". The pre-URL stash here is the
        # ground truth the click-demotion path consults instead.
        runner._pre_step_env_url = _read_env_url(runner.env)
        meter = getattr(runner, "time_meter", None)
        # Publish the dispatch context for the duration of this step.
        # Deep helpers (adaptive_settle, navigate handler, env.step /
        # env.screenshot inside the GymRunner, ClaudeExtractor /
        # ClaudeGrounding) read this context and credit their own
        # buckets — no outer ``measure`` is needed. Time spent in
        # runner orchestration outside any helper falls into
        # ``overhead`` automatically via :meth:`TimeMeter.breakdown`.
        from . import time_meter as _tm
        try:
            with _tm.publish_dispatch(meter, state.step_index):
                step_result = runner._execute_step(effective_step, state.step_index)
        finally:
            runner._active_checkpoint_context = None
        if step_result.screenshot_png is None:
            # The screenshot helper itself credits ``perceive`` via the
            # env-level wrapper; we still publish the dispatch context
            # so the credit routes to this step.
            with _tm.publish_dispatch(meter, state.step_index):
                step_result.screenshot_png = runner._capture_screenshot_bytes()
        if not step_result.success:
            _stamp_failure_context(step_result, runner.env)
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
            step_result.failure_class = "no_state_change"
            from . import healing_events
            healing_events.record_demotion(
                runner,
                step_index=state.step_index,
                step_type=effective_step.type,
                reason="snapshot diff and visual verifier both saw no UI change",
                source="demote_form",
            )
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

    # Step types that should produce some observable state change (URL,
    # page, scroll, focus, …) when their action lands. A success on one
    # of these with a zero-delta snapshot is a strong signal the action
    # missed — even when the handler reported success. Excludes form-
    # field types (fill_field / select_option) because their natural
    # delta — the field value — isn't tracked by the snapshot, and
    # excludes the existing submit path because that has its own
    # SPA-aware visual escape hatch in :meth:`_maybe_demote_form_no_change`.
    _CLICK_DEMOTE_STEP_TYPES: frozenset[str] = frozenset({
        "click", "navigate_back",
    })

    def _maybe_demote_click_no_change(
        self, state: RunState, effective_step: MicroIntent, pre_snapshot: Any,
    ) -> None:
        """Demote click / navigate_back success → fail when no observable
        change. Self-healing primitive — epic #377 Phase A.

        **URL-first signal** (#381): reads ``env.current_url`` directly
        for the pre-snapshot URL and the current URL. ``runner._last_known_url``
        is unreliable because the click handler's SPA-aware verify path
        writes it optimistically even when the browser's URL didn't
        change. ``env.current_url`` (CDP / Playwright) is the actual
        browser state — the handler can't lie about it without genuinely
        navigating.

        Falls through to the original snapshot-diff signal when the
        env doesn't expose ``current_url`` (legacy adapters / test
        stubs) so the primitive doesn't lose coverage where the
        ground-truth URL isn't available.

        Generic by design: triggers on ``step.type`` membership in
        :attr:`_CLICK_DEMOTE_STEP_TYPES` (currently ``click`` /
        ``navigate_back``). No plan, URL, or domain content reads in.

        After demoting, records the failure into the per-step retry
        history (so the next attempt can avoid the same target) and
        consults the handler-escalation policy (so repeated misses
        route through a different handler).
        """
        runner = self.parent
        step_result = state.results[-1]
        if not step_result.success:
            return
        if step_result.skip:
            # Recipe-rejection / skip envelopes are an intentional
            # halt path — not a missed action. Don't demote.
            return
        if effective_step.type not in self._CLICK_DEMOTE_STEP_TYPES:
            return

        pre_url = getattr(runner, "_pre_step_env_url", None)
        post_url = _read_env_url(runner.env)

        # Prefer the direct URL signal when we have both ends (#381).
        # The handler can't self-mutate ``env.current_url``, so URL-
        # unchanged here is a strong signal the click didn't actually
        # accomplish anything.
        if isinstance(pre_url, str) and isinstance(post_url, str) and pre_url and post_url:
            if pre_url == post_url:
                # URL is unchanged — but a true SPA-modal click would
                # also have no URL change. Disambiguate via the
                # runner-state snapshot's NON-handler-mutated fields:
                # ``focused_input_signature``, ``viewport_stage``,
                # ``current_page``. Excludes ``scroll_signature`` and
                # ``last_extracted_url`` since those are exactly what
                # the click handler optimistically mutates.
                try:
                    post_snapshot = step_snapshot.capture(runner)
                    non_handler_changed = (
                        pre_snapshot.viewport_stage != post_snapshot.viewport_stage
                        or pre_snapshot.current_page != post_snapshot.current_page
                        or pre_snapshot.focused_input_signature
                            != post_snapshot.focused_input_signature
                    )
                except Exception as exc:  # noqa: BLE001 — never break runs
                    logger.debug("post-click snapshot capture failed: %s", exc)
                    non_handler_changed = False
                if non_handler_changed:
                    return
                # URL same + nothing-the-handler-shouldn't-have-touched
                # changed → demote.
                logger.warning(
                    "  [%d] %s reported success but URL stayed at %s "
                    "and no non-handler state changed — demoting "
                    "(handler self-mutation isn't real navigation)",
                    state.step_index, effective_step.type, post_url[:80],
                )
                # Fall through to the demotion stamping below.
            else:
                # URL changed → real navigation. Don't demote.
                return
        else:
            # Direct URL signal unavailable (legacy adapter / test
            # stub). Fall back to the original snapshot-diff check.
            try:
                post_snapshot = step_snapshot.capture(runner)
                delta = step_snapshot.diff(pre_snapshot, post_snapshot)
            except Exception as exc:  # noqa: BLE001 — never break runs
                logger.debug("post-click diff capture failed: %s", exc)
                return
            if delta is None or delta.has_changes:
                return
            logger.warning(
                "  [%d] %s reported success but no observable state "
                "change — demoting to failure (will retry)",
                state.step_index, effective_step.type,
            )
        step_result.success = False
        step_result.data = (step_result.data or "") + ":no_state_change"
        step_result.failure_class = "no_state_change"
        from . import healing_events
        healing_events.record_demotion(
            runner,
            step_index=state.step_index,
            step_type=effective_step.type,
            reason="env URL unchanged and no non-handler state changed",
            source="demote_click",
        )
        # Feed the failed click coordinates back into the retry history
        # so ``find_form_target`` / ``find_click_target`` on retry can
        # AVOID the same target. The click handler stashes its last
        # target on the runner via the same protocol the submit handler
        # uses (``_last_submit_target``); falling back to no coords is
        # fine — the no-coordinate retry path just doesn't get the
        # "avoid these" hint.
        self._record_failure_for_retry(
            runner=runner,
            step_index=state.step_index,
            kind="no_state_change",
            reason="post-click snapshot saw no URL / page / scroll change",
        )
        self._maybe_set_handler_override(
            runner=runner, step_index=state.step_index,
        )

    # Step types that own an action whose post-URL is meaningful for
    # wrong-target detection. ``select_option`` is excluded — picking
    # an option doesn't usually change the URL (the value lives in
    # form state), and a strict URL check there would mis-demote
    # valid edits. ``fill_field`` excluded for the same reason.
    _WRONG_TARGET_STEP_TYPES: frozenset[str] = frozenset({
        "submit", "click", "navigate", "navigate_back",
    })

    def _maybe_demote_wrong_target(
        self, state: RunState, effective_step: MicroIntent,
    ) -> None:
        """Demote success → fail when the step succeeded (state DID
        change) but the post-click URL doesn't match the plan-supplied
        expectation (roadmap D).

        Plan hints recognised:

        * ``hints.expect_url_contains`` — str or list[str]. Every
          listed substring must appear in ``env.current_url`` after
          the action. The canonical staff-crm-long step-6 case: the
          plan says "click Contacted → URL should include
          ``status=Contacted``"; if the click landed on a status pill
          instead of the sidebar link, the URL won't update and we
          demote to ``wrong_target`` rather than reporting OK.
        * ``hints.expect_url_excludes`` — str or list[str]. Listed
          substrings must NOT appear in the post-URL. Covers the
          "click drifted to a row-detail page" failure mode (step
          6 must NOT land on ``/leads/<id>``).

        Order matters: this runs AFTER ``_maybe_demote_form_no_change``
        / ``_maybe_demote_click_no_change``. Those handle the
        "nothing happened" failure mode; this one handles
        "something happened but the wrong thing." When neither hint
        is present on the step (decomposer didn't emit one — most
        plans today), this is a no-op.

        Failure class is ``wrong_target`` (already in the runner's
        rewrite-triggering set) so the retry routes through the
        intent rewriter / agentic recovery loop with full prior-
        attempt context.
        """
        runner = self.parent
        step_result = state.results[-1]
        if not step_result.success:
            return
        if effective_step.type not in self._WRONG_TARGET_STEP_TYPES:
            return
        hints = dict(getattr(effective_step, "hints", {}) or {})
        expect_contains_raw = hints.get("expect_url_contains") or []
        expect_excludes_raw = hints.get("expect_url_excludes") or []
        if isinstance(expect_contains_raw, str):
            expect_contains: list[str] = [expect_contains_raw]
        else:
            expect_contains = [str(s) for s in expect_contains_raw if str(s).strip()]
        if isinstance(expect_excludes_raw, str):
            expect_excludes: list[str] = [expect_excludes_raw]
        else:
            expect_excludes = [str(s) for s in expect_excludes_raw if str(s).strip()]
        if not expect_contains and not expect_excludes:
            return
        post_url = _read_env_url(runner.env) or ""
        missing = [s for s in expect_contains if s not in post_url]
        forbidden = [s for s in expect_excludes if s in post_url]
        if not missing and not forbidden:
            return
        reason_parts: list[str] = []
        if missing:
            reason_parts.append(
                f"URL missing expected substring(s): {missing}"
            )
        if forbidden:
            reason_parts.append(
                f"URL contains forbidden substring(s): {forbidden}"
            )
        url_clip = post_url[:120] + ("…" if len(post_url) > 120 else "")
        reason = "; ".join(reason_parts) + f" (post-URL: {url_clip})"
        logger.warning(
            "  [%d] %s reported success but URL postcondition failed — "
            "demoting wrong_target: %s",
            state.step_index, effective_step.type, reason,
        )
        step_result.success = False
        step_result.data = (step_result.data or "") + ":wrong_target"
        step_result.failure_class = "wrong_target"
        from . import healing_events
        healing_events.record_demotion(
            runner,
            step_index=state.step_index,
            step_type=effective_step.type,
            reason=reason,
            source="demote_wrong_target",
        )
        self._record_failure_for_retry(
            runner=runner,
            step_index=state.step_index,
            kind="wrong_target",
            reason=reason,
        )

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
        # PR-H (Option 1): enrich retry context with the SoM
        # diagnostic from the most recent click. Tells the next brain
        # prompt "you clicked at (x, y) and the element at that
        # pixel was <elv_tag> with text <elv_text>" — the same signal
        # a human gets from cursor state, lets the brain adjust its
        # pixel choice on retry without DOM-target derivation.
        som_diag = getattr(getattr(runner, "env", None), "_last_som_diag", None) or {}
        record = {
            "x": target.get("x"),
            "y": target.get("y"),
            "label": target.get("label", ""),
            "matched_label": target.get("matched_label", ""),
            "kind": kind,
            "reason": reason,
        }
        # Only attach SoM data if it's for THIS click's coords. Stale
        # diagnostics from an earlier step's click would mislead the
        # retry prompt.
        if (
            som_diag.get("x") == target.get("x")
            and som_diag.get("y") == target.get("y")
        ):
            elv_tag = str(som_diag.get("elv_tag") or "").strip()
            elv_text = str(som_diag.get("elv_text") or "").strip()
            if elv_tag or elv_text:
                record["som_elv_tag"] = elv_tag
                record["som_elv_text"] = elv_text[:60]
                logger.warning(
                    "  [retry-history] step %d: attached SoM diag "
                    "elv=%s elv_text=%r to failure record (kind=%s)",
                    step_index, elv_tag, elv_text[:40], kind,
                )
            else:
                logger.warning(
                    "  [retry-history] step %d: SoM diag at "
                    "(%s,%s) had empty elv (probably off-screen) — "
                    "no enrichment",
                    step_index, som_diag.get("x"), som_diag.get("y"),
                )
        else:
            logger.warning(
                "  [retry-history] step %d: SoM diag coords "
                "(%s,%s) don't match target (%s,%s) — no enrichment",
                step_index,
                som_diag.get("x"), som_diag.get("y"),
                target.get("x"), target.get("y"),
            )
        history.append(record)

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
            from . import healing_events
            healing_events.record_handler_escalation(
                runner,
                step_index=step_index,
                from_handler="default",
                to_handler="holo3",
                trigger=f"{no_state_changes}x_no_state_change",
            )
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
        # Same for agentic-recovery hints — once the step has
        # succeeded, accumulated hints have served their purpose;
        # don't bleed them into a future occurrence of the same
        # step_index (loops, resumed plans).
        if hasattr(runner, "_recovery_hints"):
            runner._recovery_hints.pop(state.step_index, None)
        if hasattr(runner, "_recovery_attempts_per_step"):
            runner._recovery_attempts_per_step.pop(state.step_index, None)
        # Epic #377 Phase B: rewriter state is per-step. Clear on
        # success so a loop iteration / resumed plan that re-enters
        # the same step_index starts with the original intent and a
        # fresh budget.
        if hasattr(runner, "_step_intent_overrides"):
            runner._step_intent_overrides.pop(state.step_index, None)
        if hasattr(runner, "_step_rewrite_attempts"):
            runner._step_rewrite_attempts.pop(state.step_index, None)

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
        # Capture the failed step_index BEFORE recovery_policy mutates
        # state.step_index — the rewriter keys overrides off the index
        # of the step that just failed, not whatever the policy
        # decides to advance to.
        failed_step_index = state.step_index
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
            # Issue #250: opt-in skip envelope on navigation-primitive
            # halts. When a host configures
            # ``navigation_primitives_emit_skip`` and the halted step
            # type is in that set, re-stamp the last StepResult with
            # ``skip=True, skip_reason='navigation_failed'`` so the
            # host's tool surface promotes the halt to a successful
            # tool result and the orchestrator advances past the
            # listing instead of retrying the same intent. Recipe
            # rejections that already populated ``skip_reason`` (#246
            # — dealer / spam / incomplete_required) win over this
            # generic stamp; we only stamp when ``last_result.skip``
            # is False.
            nav_set = getattr(runner, "navigation_primitives_emit_skip", None)
            if (
                nav_set
                and step.type in nav_set
                and state.results
                and not state.results[-1].skip
            ):
                state.results[-1].skip = True
                state.results[-1].skip_reason = "navigation_failed"
            self._persist(plan, state, status="halted", halt_reason=outcome.halt_reason)
            return False
        # Epic #377 Phase B: when the policy decides retry AND the
        # failure_class is in the rewrite-triggering set, ask Claude to
        # propose a more mechanical / specific intent for the next
        # attempt. Bounded by the per-step budget in
        # ``_step_rewrite_attempts``.
        self._maybe_rewrite_intent_for_retry(
            step=step, step_result=step_result, step_index=failed_step_index,
        )
        self._persist(plan, state, halt_reason=outcome.halt_reason)
        return True

    def _maybe_rewrite_intent_for_retry(
        self,
        *,
        step: MicroIntent,
        step_result: StepResult,
        step_index: int,
    ) -> None:
        """Phase B of epic #377 — Claude-backed intent rewriting.

        Calls :func:`intent_rewriter.propose_rewrite` when the just-
        failed step's ``failure_class`` is in
        :data:`REWRITE_TRIGGERING_CLASSES` and the per-step budget
        hasn't been spent. On a successful rewrite, stashes the new
        intent in ``runner._step_intent_overrides[step_index]`` — the
        next retry's ``_build_effective_step`` reads it.

        Never breaks the run. Any exception from the Claude call (or
        ``requests`` not installed, or no API key) just skips the
        rewrite path; the recovery policy's retry proceeds with the
        original intent.
        """
        runner = self.parent
        failure_class = getattr(step_result, "failure_class", "") or ""
        # #433: unconditional entry log so a step that halts without
        # any rewriter signal can be definitively distinguished from
        # one where this function was never reached. Logged at INFO
        # so it doesn't drown normal runs but survives modal-server's
        # default capture (which is also why we use the explicit
        # ``mantis_agent.gym.micro_runner`` logger established at
        # module top — same logger that successfully emits
        # ``[rewriter] step N intent rewritten`` warnings).
        logger.info(
            "  [%d] _maybe_rewrite_intent_for_retry: entered "
            "(failure_class=%r, runner=%s)",
            step_index, failure_class, type(runner).__name__,
        )
        try:
            from . import intent_rewriter
        except ImportError as exc:  # noqa: BLE001 — never break runs
            logger.warning(
                "  [%d] rewriter_skipped: failed to import intent_rewriter (%s)",
                step_index, exc,
            )
            return
        attempts_used = (
            runner._step_rewrite_attempts.get(step_index, 0)
            if hasattr(runner, "_step_rewrite_attempts") else 0
        )
        if not intent_rewriter.should_attempt_rewrite(
            failure_class, attempts_used=attempts_used,
        ):
            # #431: visibility on every skip path. The two reasons
            # ``should_attempt_rewrite`` returns False — empty /
            # untriggering failure_class, or per-step budget spent —
            # are otherwise silent and look the same as "rewriter not
            # wired up at all" in logs.
            if not failure_class:
                logger.warning(
                    "  [%d] rewriter_skipped: failure_class is empty — "
                    "set on step_result by the handler that produced "
                    "the failure to enable rewrites on this path",
                    step_index,
                )
            elif failure_class not in intent_rewriter.REWRITE_TRIGGERING_CLASSES:
                logger.warning(
                    "  [%d] rewriter_skipped: failure_class=%r not in "
                    "REWRITE_TRIGGERING_CLASSES=%s",
                    step_index, failure_class,
                    sorted(intent_rewriter.REWRITE_TRIGGERING_CLASSES),
                )
            else:
                logger.warning(
                    "  [%d] rewriter_skipped: per-step budget exhausted "
                    "(attempts_used=%d)",
                    step_index, attempts_used,
                )
            return
        try:
            failures = [intent_rewriter.FailureContext(
                failure_class=failure_class,
                data=getattr(step_result, "data", "") or "",
                page_title=getattr(step_result, "page_title", "") or "",
                final_url=getattr(step_result, "final_url", "") or "",
                screenshot_png=getattr(step_result, "screenshot_png", None),
            )]
            new_intent = intent_rewriter.propose_rewrite(step.intent, failures)
        except Exception as exc:  # noqa: BLE001 — never break runs on rewrite
            logger.warning(
                "  [%d] rewriter_skipped: propose_rewrite raised %s",
                step_index, exc,
            )
            return
        if not new_intent or new_intent == step.intent:
            # The rewriter ran (Claude was called) but returned no
            # actionable rewrite — KEEP, empty, or identical to the
            # original intent. Surface so a halt with no rewrite is
            # distinguishable from a halt where the rewriter was
            # never even invoked.
            logger.info(
                "  [%d] rewriter_no_change: Claude returned no "
                "actionable rewrite (failure_class=%r) — retry will "
                "use the original intent",
                step_index, failure_class,
            )
            return
        if not hasattr(runner, "_step_intent_overrides"):
            runner._step_intent_overrides = {}
        if not hasattr(runner, "_step_rewrite_attempts"):
            runner._step_rewrite_attempts = {}
        runner._step_intent_overrides[step_index] = new_intent
        runner._step_rewrite_attempts[step_index] = attempts_used + 1
        # Record so the operator can audit what the framework did
        # (epic #377 Phase C accounting).
        from . import healing_events
        healing_events.record_rewrite(
            runner,
            step_index=step_index,
            from_intent=step.intent,
            to_intent=new_intent,
            source="intent_rewriter",
            failure_class=failure_class,
        )
        logger.warning(
            "  [rewriter] step %d intent rewritten from %r → %r "
            "(failure_class=%s, attempts_used=%d)",
            step_index, step.intent[:80], new_intent[:80],
            failure_class, attempts_used + 1,
        )

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
        # #audit item 4: surface halt_reason on the runner so
        # ``build_micro_result`` can include it in the result envelope
        # (the detached-status writer needs it to distinguish
        # ``budget_cap`` / ``time_cap`` / generic ``halted``).
        if halt_reason:
            runner._final_halt_reason = halt_reason
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


# ── Failure diagnostics ────────────────────────────────────────────────


def _read_env_url(env: Any) -> str:
    """Best-effort read of the browser's actual URL from the env.

    Both ``XdotoolGymEnv`` and ``PlaywrightGymEnv`` expose ``current_url``
    as a property. The xdotool path reads CDP's ``/json/list``; the
    Playwright path reads ``self._page.url``. Both return ``""`` on
    any failure (CDP unreachable, page not initialised).

    Returns ``""`` when the env doesn't expose the attribute, the read
    raises, or the returned value isn't a string. Lets callers
    distinguish "ground-truth URL unavailable" (fall through to
    snapshot-based logic) from "URL is known to be X" (use it).

    Used by the click-demotion path (#381) because ``runner._last_known_url``
    can be self-mutated by the click handler's SPA-aware verify and
    isn't a reliable proxy for "did the browser actually navigate?".
    """
    try:
        value = getattr(env, "current_url", None)
    except Exception as exc:  # noqa: BLE001 — never break runs
        logger.debug("env.current_url access raised: %s", exc)
        return ""
    return value if isinstance(value, str) else ""


def _stamp_failure_context(step_result: StepResult, env: Any) -> None:
    """Snapshot URL + page title + failure class onto a failed StepResult.

    Best-effort: every probe is wrapped in ``try/except`` because the
    failure path must not raise — even if CDP is unreachable or the
    Playwright page is in a half-torn-down state, the StepResult still
    needs to land.

    **Preserves any handler-stamped ``failure_class``.** Step handlers
    (``Holo3StepHandler`` for ``brain_loop_exhausted``, click handler
    for ``wrong_target`` / ``no_state_change``, executor demotions for
    ``no_state_change``) write the canonical class directly on the
    StepResult. The classifier here is the **fallback** for handlers
    that haven't been wired yet — it reads ``data`` prose and produces
    a best-guess. Clobbering an already-stamped class would neuter the
    self-healing wiring (epic #377): the IntentRewriter / critic key
    off ``failure_class``, and an "unknown" overwrite means the
    rewriter never sees the signal the handler took the trouble to
    surface.
    """
    try:
        url, title = failure_class_mod.read_failure_context(env)
    except Exception as exc:
        logger.debug("failure context probe raised: %s", exc)
        url, title = "", ""
    step_result.final_url = url
    step_result.page_title = title
    # Honor a handler-stamped class. Only fall through to the
    # classifier when nothing was stamped.
    if step_result.failure_class:
        return
    try:
        step_result.failure_class = failure_class_mod.classify(
            step_result.data or "", title,
        )
    except Exception as exc:
        logger.debug("failure classifier raised: %s", exc)
        step_result.failure_class = "unknown"


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
