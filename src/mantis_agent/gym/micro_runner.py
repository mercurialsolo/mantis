"""MicroPlanRunner — execute decomposed micro-intents with checkpoint/verify/reverse.

Takes a MicroPlan (list of MicroIntent) and executes each step:
  - Holo3 steps: fresh GymRunner per intent (3-8 steps, 1 sentence)
  - Claude steps: screenshot → ClaudeExtractor reads data
  - Verify: Claude checks before/after screenshots
  - Reverse: undo failed step (Escape, Alt+Left, Ctrl+W)
  - Checkpoint: save state after each verified step
  - Loop: repeat step sequences (e.g., extract listings on each page)

Usage:
    from mantis_agent.plan_decomposer import PlanDecomposer
    from mantis_agent.gym.micro_runner import MicroPlanRunner

    plan = PlanDecomposer().decompose_text("Extract jobs from ...")
    runner = MicroPlanRunner(brain=brain, env=env, ...)
    results = runner.run(plan)
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, ClassVar, TYPE_CHECKING

from ..actions import Action, ActionType
from ..cost_config import CostConfig
from .browser_state import BrowserState
from .checkpoint_manager import CheckpointManager
from .cost_meter import CostMeter
from .listing_dedup import ListingDedup
from . import step_snapshot
from .checkpoint import (
    REVERSE_ACTIONS,
    PauseRequested,
    PauseState,
    RunCheckpoint,
    RunnerResult,
    StepResult,
    _PauseRequested,
)
from .runner import GymRunner
from .tool_channel import ToolChannel

from ..plan_decomposer import MicroIntent, MicroPlan
from ..site_config import SiteConfig
from ..verification.dynamic_plan_verifier import DynamicPlanVerifier

if TYPE_CHECKING:
    from ..extraction import ClaudeExtractor

logger = logging.getLogger(__name__)


# Re-export every persisted/data type from the new checkpoint module so
# external callers (modal_cua_server, baseten_server, host integrations,
# tests) keep working unchanged. The implementation lives in
# :mod:`mantis_agent.gym.checkpoint` (#115).
__all__ = [
    "MicroPlanRunner",
    "REVERSE_ACTIONS",
    "PauseRequested",
    "PauseState",
    "RunCheckpoint",
    "RunnerResult",
    "StepResult",
    "_PauseRequested",
]


class MicroPlanRunner:
    """Execute a MicroPlan step-by-step with verify/reverse/checkpoint.

    Args:
        brain: Holo3 brain for executing micro-intents.
        env: XdotoolGymEnv with screenshot() method.
        grounding: ClaudeGrounding for click steps.
        extractor: ClaudeExtractor for data extraction steps.
        on_step: Optional viewer callback.
        max_retries: Max retries per step before skip.
        checkpoint_path: Path for checkpoint file.
    """

    def __init__(
        self,
        brain: Any,
        env: Any,
        grounding: Any = None,
        extractor: ClaudeExtractor | None = None,
        on_step: Any = None,
        max_retries: int = 2,
        checkpoint_path: str = "/data/checkpoints/micro_run.json",
        run_key: str = "",
        session_name: str = "",
        plan_signature: str = "",
        resume_state: bool = False,
        on_checkpoint: Any = None,
        dynamic_verifier: DynamicPlanVerifier | None = None,
        max_cost: float = 10.0,     # Stop if total cost exceeds this
        max_time_minutes: int = 180, # Stop if runtime exceeds this (3 hours)
        site_config: SiteConfig | None = None,
        step_callback: Any = None,           # Callable[[int, str, Action|None, bool], None]
        keep_screenshots: int | None = None,  # cap on retained screenshot bytes (None=all)
        cancel_event: Any = None,            # threading.Event-like (.is_set()) or callable for #76
        cost_config: CostConfig | None = None,  # rate overrides for cost reporting (#122)
        tenant_id: str = "",                 # label for Prometheus inflight cost gauge (#122)
    ):
        self.brain = brain
        self.env = env
        self.grounding = grounding
        self.extractor = extractor
        self.on_step = on_step
        self.max_retries = max_retries
        self.checkpoint_path = checkpoint_path
        self.run_key = run_key
        self.session_name = session_name
        self.plan_signature = plan_signature
        self.resume_state = resume_state
        self.on_checkpoint = on_checkpoint
        self.dynamic_verifier = dynamic_verifier or DynamicPlanVerifier(plan_name=session_name)
        self.site_config = site_config or SiteConfig.default_boattrader()
        self._seen_urls: set[str] = set()
        self._extracted_titles: list[str] = []  # Exact titles Claude returned, for skip list
        self._page_listings: list[tuple[int, int, str]] = []  # Cached card coords for current viewport
        self._page_listing_index: int = 0  # Next card to click from cache
        self._viewport_stage: int = 0  # 0=Home, 1=Page_Down, 2=Page_Down×2
        self._max_viewport_stages: int = 6
        self._results_base_url: str = ""
        self._required_filter_tokens: tuple[str, ...] = ()
        self._current_page: int = 1
        self._last_known_url: str = ""
        self._scroll_state: dict[str, Any] = {}
        self._last_extracted: dict[str, Any] = {}
        self._opened_detail_in_new_tab: bool = False
        self._active_checkpoint_context: dict[str, Any] | None = None
        # #121 step 2: snapshot of runner state taken right before
        # _execute_step, used by _reverse_step to skip undo actions
        # when the diff shows nothing meaningful changed.
        self._pre_step_snapshot: step_snapshot.StepStateSnapshot | None = None
        self._final_status: str = "running"
        self.max_cost = max_cost
        self.max_time = max_time_minutes * 60
        self.step_callback = step_callback
        self.keep_screenshots = keep_screenshots
        self.cancel_event = cancel_event
        # #115: tool registry + pause state extracted into ToolChannel.
        # Public ``register_tool``/``list_tools``/``call_tool`` below remain on
        # the runner and delegate here so external callers don't change.
        self.tool_channel = ToolChannel()

        # #115 step 3: cost counters + rate config + Prometheus inflight
        # gauges live on the CostMeter. ``self.costs`` aliases the meter's
        # canonical dict so the 60+ scattered ``self.costs["..."] += N``
        # mutation sites in this file continue to work unchanged.
        # ``self.cost_config`` and ``self.tenant_id`` stay as runner attrs
        # for any external reader; they mirror the meter's view.
        self.cost_meter = CostMeter(cost_config=cost_config, tenant_id=tenant_id)
        self.cost_config = self.cost_meter.cost_config
        self.tenant_id = self.cost_meter.tenant_id
        self.costs = self.cost_meter.costs
        self._run_start = self.cost_meter.run_start

        # #115 step 4: scroll/viewport/url helpers live on BrowserState.
        # State attributes (_scroll_state, _viewport_stage, _current_page,
        # _last_known_url, _results_base_url, _required_filter_tokens) stay
        # on the runner — there are 170+ scattered access sites; migrating
        # them all in one PR would be unreviewable. Helper methods below
        # delegate here so the bodies live in one place.
        self.browser_state = BrowserState(self)

        # #115 step 6: checkpoint persist/restore + plan-signature flow
        # live on CheckpointManager. Reads 14 different runner attributes
        # via self.parent — same back-reference pattern as BrowserState.
        self.checkpoint_manager = CheckpointManager(self)

    def dynamic_verification_report(self, status: str | None = None) -> dict[str, Any]:
        return self.dynamic_verifier.report(status=status or self._final_status)

    # ── #71 Tool channel — public API delegates to self.tool_channel ────
    def register_tool(self, name: str, schema: dict[str, Any], handler: Any) -> None:
        """Register a host-provided tool callable mid-plan. See :class:`ToolChannel`."""
        self.tool_channel.register(name, schema, handler)

    def list_tools(self) -> list[dict[str, Any]]:
        """Return registered tools as ``[{"name", "schema"}]`` (for brain prompts)."""
        return self.tool_channel.list()

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Invoke a registered tool. Raises KeyError if not registered."""
        return self.tool_channel.call(name, arguments)

    def _invoke_tool(self, name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """Run a tool, returning ``(success, data_str)`` for the step result."""
        return self.tool_channel.invoke(name, arguments)

    # ── #74 Observability ──────────────────────────────────────────────
    def _capture_screenshot_bytes(self) -> bytes | None:
        """Capture a PNG snapshot for observability. Returns None on failure."""
        env = self.env
        if env is None or not hasattr(env, "screenshot"):
            return None
        try:
            img = env.screenshot()
        except Exception as exc:
            logger.debug("screenshot capture failed: %s", exc)
            return None
        try:
            import io
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
        except Exception as exc:
            logger.debug("screenshot encode failed: %s", exc)
            return None

    def _enforce_screenshot_cap(self, results: list[StepResult]) -> None:
        """Drop oldest screenshot bytes once `keep_screenshots` is exceeded."""
        cap = self.keep_screenshots
        if cap is None or cap < 0:
            return
        kept = 0
        # Walk newest→oldest, retaining the most-recent ``cap`` screenshots.
        for r in reversed(results):
            if r.screenshot_png is None:
                continue
            if kept >= cap:
                r.screenshot_png = None
            else:
                kept += 1

    def _invoke_step_callback(self, result: StepResult) -> None:
        """Invoke step_callback (#74). Errors are logged, never raised."""
        cb = self.step_callback
        if cb is None:
            return
        try:
            cb(result.step_index, result.intent, result.last_action, result.success)
        except Exception as exc:  # noqa: BLE001 — observability must not break runs
            logger.warning("step_callback raised: %s", exc)

    # ── #76 External cancellation ──────────────────────────────────────
    def _is_cancelled(self) -> bool:
        """True if external cancel hook fired (#76)."""
        ev = self.cancel_event
        if ev is None:
            return False
        # Support threading.Event-style and plain callables.
        try:
            if callable(ev):
                return bool(ev())
            return bool(ev.is_set())
        except Exception:
            return False

    @staticmethod
    def _compute_plan_signature(plan: MicroPlan) -> str:
        """Backward-compat shim — delegates to :class:`CheckpointManager`."""
        return CheckpointManager.compute_plan_signature(plan)

    def _cost_totals(self) -> tuple[float, float, float, float]:
        """Backward-compat shim — delegates to :meth:`CostMeter.totals`."""
        return self.cost_meter.totals()

    @classmethod
    def _unique_leads_from_results(cls, results: list[StepResult]) -> list[str]:
        """Backward-compat shim — delegates to :class:`ListingDedup`."""
        return ListingDedup.unique_leads_from_results(results)

    def _record_step_costs(self, step: MicroIntent, step_result: StepResult) -> None:
        """Backward-compat shim — delegates to :meth:`CostMeter.record_step`."""
        self.cost_meter.record_step(step, step_result)

    def _log_progress(self, step_result: StepResult, results: list[StepResult]) -> None:
        gpu_cost, claude_cost, proxy_cost, total_cost = self._cost_totals()
        unique_leads, phone_leads = self._lead_counts(results)
        elapsed = time.time() - self._run_start
        cost_per_lead = total_cost / max(unique_leads, 1)
        cost_per_phone_lead = total_cost / max(phone_leads, 1)
        print(
            f"  [{step_result.step_index:2d}] {'OK' if step_result.success else 'FAIL'} "
            f"| {unique_leads} leads ({phone_leads} phone) | ${total_cost:.2f} total "
            f"(${cost_per_lead:.2f}/lead, ${cost_per_phone_lead:.2f}/phone lead) | "
            f"GPU ${gpu_cost:.2f} Claude ${claude_cost:.2f} Proxy ${proxy_cost:.2f} | "
            f"{elapsed/60:.0f}m"
        )
        self._emit_cost_gauges(gpu_cost, claude_cost, proxy_cost, total_cost)

    def _emit_cost_gauges(
        self, gpu_cost: float, claude_cost: float, proxy_cost: float, total_cost: float
    ) -> None:
        """Backward-compat shim — delegates to :meth:`CostMeter.emit_inflight_gauges`."""
        self.cost_meter.emit_inflight_gauges(gpu_cost, claude_cost, proxy_cost, total_cost)

    def _log_step_diff(
        self,
        pre_snapshot: "step_snapshot.StepStateSnapshot",
        step: MicroIntent,
        step_result: StepResult,
    ) -> None:
        """Compute pre/post step diff and log it (#121 step 1).

        Observation-only for this PR — the next PR in #121 uses the diff
        to drive plan-aware reverse decisions. Logging here lets us
        validate the diff is correct on real traces before changing
        recovery behavior.
        """
        try:
            post_snapshot = step_snapshot.capture(self)
            delta = step_snapshot.diff(pre_snapshot, post_snapshot)
        except Exception as exc:  # noqa: BLE001 — observability must not break runs
            logger.debug("step diff capture failed: %s", exc)
            return
        outcome = "ok" if step_result.success else "fail"
        logger.info(
            "  [diff] %s/%s step=%s: %s",
            step.type, outcome, step_result.step_index, delta.summary(),
        )

    # ── Adaptive submit settle ────────────────────────────────────────

    # Maximum total seconds to wait for a submit click's navigation /
    # state change. Tuned to cover slow CRM logins (3-5s typical) plus
    # headroom for high-latency tenants. Login redirects faster than
    # this break out of the polling loop early; only genuinely slow
    # actions pay the full budget.
    _SUBMIT_SETTLE_MAX_SECONDS: ClassVar[float] = 8.0
    _SUBMIT_SETTLE_POLL_SECONDS: ClassVar[float] = 0.5
    _SUBMIT_SETTLE_MIN_SECONDS: ClassVar[float] = 1.0

    def _best_effort_current_url(self) -> str:
        """Fetch the env's current URL without raising. Empty string if
        the env doesn't expose ``current_url`` or the read fails — the
        adaptive settle then falls through to the timeout-based wait."""
        env = self.env
        try:
            return str(getattr(env, "current_url", "") or "")
        except Exception as exc:  # noqa: BLE001 — telemetry must never break runs
            logger.debug("current_url read failed: %s", exc)
            return ""

    def _safe_screenshot(self) -> Any:
        """Capture one screenshot without raising. Returns None when the
        env doesn't expose ``screenshot()`` or capture fails."""
        env = self.env
        try:
            return env.screenshot() if hasattr(env, "screenshot") else None
        except Exception as exc:  # noqa: BLE001 — observability never breaks runs
            logger.debug("safe screenshot failed: %s", exc)
            return None

    def _dump_debug_screenshot(self, name_stem: str, screenshot: Any) -> None:
        """Write a screenshot to ``$MANTIS_DEBUG_DUMP_DIR/<run_key>/<stem>.png``
        when the env var is set. No-op otherwise. Useful for diagnosing
        submit-failure cases (login click that never navigates) without
        rerunning the full Modal pipeline.

        Filenames carry the run key + caller-supplied stem so files from
        concurrent runs don't collide.
        """
        if screenshot is None:
            return
        dump_dir = os.environ.get("MANTIS_DEBUG_DUMP_DIR", "").strip()
        if not dump_dir:
            return
        try:
            target_dir = os.path.join(
                dump_dir, self.run_key or self.session_name or "default",
            )
            os.makedirs(target_dir, exist_ok=True)
            target = os.path.join(target_dir, f"{name_stem}.png")
            screenshot.save(target, format="PNG", optimize=True)
            logger.debug("dumped debug screenshot: %s", target)
        except Exception as exc:  # noqa: BLE001 — observability never breaks runs
            logger.debug("debug screenshot dump failed: %s", exc)

    def _adaptive_submit_settle(self, *, url_before: str) -> float:
        """Wait for a submit click's effect, breaking early on URL change.

        Returns the actual seconds slept so the cost meter can record
        accurate GPU-seconds.

        Pure-observational: polls the env's CDP-backed ``current_url``,
        no LLM call, no heuristic on intent text. If the URL never
        changes the runner's separate state-change verifier (PR #150)
        will demote the step to fail and the retry loop will kick in
        with a fresh attempt.
        """
        deadline = time.time() + self._SUBMIT_SETTLE_MAX_SECONDS
        # Always wait at least the minimum — covers fast-redirect SPAs
        # where the URL changes within a few hundred ms but the page
        # still needs DOM time to settle for the next find_form_target.
        time.sleep(self._SUBMIT_SETTLE_MIN_SECONDS)
        elapsed = self._SUBMIT_SETTLE_MIN_SECONDS

        while time.time() < deadline:
            current = self._best_effort_current_url()
            if current and url_before and current != url_before:
                # URL changed — submit triggered navigation, settle done.
                logger.debug(
                    "  [settle] url changed after %.1fs: %s → %s",
                    elapsed, url_before[:40], current[:40],
                )
                return elapsed
            time.sleep(self._SUBMIT_SETTLE_POLL_SECONDS)
            elapsed += self._SUBMIT_SETTLE_POLL_SECONDS

        # Hit the budget without observing a URL change. The state-change
        # verifier downstream will catch this and trigger a retry.
        logger.debug(
            "  [settle] no url change within %.1fs (url=%s)",
            self._SUBMIT_SETTLE_MAX_SECONDS, url_before[:60],
        )
        return self._SUBMIT_SETTLE_MAX_SECONDS

    def _current_results_page_url(self) -> str:
        """Backward-compat shim — delegates to :class:`BrowserState`."""
        return self.browser_state.current_results_page_url()

    def _reentry_url_for_step(self, plan: MicroPlan, next_step_index: int) -> str:
        """Backward-compat shim — delegates to :class:`BrowserState`."""
        return self.browser_state.reentry_url_for_step(plan, next_step_index)

    def _persist_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        plan: MicroPlan,
        results: list[StepResult],
        loop_counters: dict[int, int],
        listings_on_page: int,
        next_step_index: int,
        status: str = "running",
        halt_reason: str = "",
    ) -> None:
        """Backward-compat shim — delegates to :class:`CheckpointManager`."""
        self.checkpoint_manager.persist(
            checkpoint=checkpoint,
            plan=plan,
            results=results,
            loop_counters=loop_counters,
            listings_on_page=listings_on_page,
            next_step_index=next_step_index,
            status=status,
            halt_reason=halt_reason,
        )

    def _restore_from_checkpoint(
        self,
        checkpoint: RunCheckpoint,
    ) -> tuple[list[StepResult], dict[int, int], int]:
        """Backward-compat shim — delegates to :class:`CheckpointManager`."""
        return self.checkpoint_manager.restore(checkpoint)

    def _checkpoint_active_progress(self, halt_reason: str = "step_progress") -> None:
        """Backward-compat shim — delegates to :class:`CheckpointManager`."""
        self.checkpoint_manager.save_active_progress(halt_reason)

    def _set_scroll_state(
        self,
        *,
        context: str,
        url: str = "",
        page_downs: int | None = None,
        wheel_downs: int | None = None,
        viewport_stage: int | None = None,
        label: str = "",
        flush: bool = False,
    ) -> None:
        """Backward-compat shim — delegates to :class:`BrowserState`."""
        self.browser_state.set_scroll_state(
            context=context,
            url=url,
            page_downs=page_downs,
            wheel_downs=wheel_downs,
            viewport_stage=viewport_stage,
            label=label,
            flush=flush,
        )

    def _update_scroll_state_from_trajectory(self, result: Any, context: str) -> None:
        """Backward-compat shim — delegates to :class:`BrowserState`."""
        self.browser_state.update_scroll_state_from_trajectory(result, context)

    def _restore_scroll_position(self) -> None:
        """Backward-compat shim — delegates to :class:`BrowserState`."""
        self.browser_state.restore_scroll_position()

    def _resume_browser_state(self, url: str) -> bool:
        """Backward-compat shim — delegates to :class:`BrowserState`."""
        return self.browser_state.resume_browser_state(url)

    def run(self, plan: MicroPlan, resume: bool = False) -> list[StepResult]:
        """Execute the full micro-plan.

        Args:
            plan: Decomposed plan with ordered micro-intents.
            resume: If True, load checkpoint and resume from last step.

        Returns:
            List of StepResult for each executed step.
        """
        self._final_status = "running"
        if not self.plan_signature:
            self.plan_signature = self._compute_plan_signature(plan)

        results: list[StepResult] = []
        loop_counters: dict[int, int] = {}
        listings_on_page = 0  # Track how many listings processed on current page
        checkpoint = RunCheckpoint(
            run_key=self.run_key,
            plan_signature=self.plan_signature,
            session_name=self.session_name,
        )

        should_resume = resume or self.resume_state
        if should_resume:
            loaded = RunCheckpoint.load(self.checkpoint_path)
            if loaded:
                if (
                    loaded.plan_signature
                    and self.plan_signature
                    and loaded.plan_signature != self.plan_signature
                ):
                    logger.warning(
                        "Checkpoint signature mismatch at %s; starting fresh",
                        self.checkpoint_path,
                    )
                else:
                    checkpoint = loaded
                    results, loop_counters, listings_on_page = self._restore_from_checkpoint(checkpoint)
                    logger.info(
                        "Resumed from step %s, page %s, %s URLs seen, status=%s",
                        checkpoint.step_index,
                        checkpoint.current_page or checkpoint.page,
                        len(self._seen_urls),
                        checkpoint.status,
                    )
                    if checkpoint.status == "completed":
                        logger.info("Checkpoint already marked complete; returning cached results")
                        return results

                    reentry_url = (
                        checkpoint.reentry_url
                        or checkpoint.current_url
                        or self._reentry_url_for_step(plan, checkpoint.step_index)
                    )
                    if checkpoint.step_index > 0 and reentry_url:
                        self._resume_browser_state(reentry_url)

        step_index = checkpoint.step_index
        step_retry_counts: dict[int, int] = {}
        max_loop_iterations = 200  # Safety cap

        if not self._results_base_url and plan.steps:
            self._results_base_url = self._extract_url_from_intent(plan.steps[0].intent)
            self._required_filter_tokens = self._derive_filter_tokens(self._results_base_url)
            self.dynamic_verifier.set_required_filter_tokens(self._required_filter_tokens)
            if self._results_base_url:
                self.dynamic_verifier.record_page_start(
                    page=self._current_page,
                    url=self._current_results_page_url() or self._results_base_url,
                )

        def persist(next_step_index: int, status: str = "running", halt_reason: str = "") -> None:
            self._persist_checkpoint(
                checkpoint=checkpoint,
                plan=plan,
                results=results,
                loop_counters=loop_counters,
                listings_on_page=listings_on_page,
                next_step_index=next_step_index,
                status=status,
                halt_reason=halt_reason,
            )

        while step_index < len(plan.steps):
            # External cancellation (#76) — check at every step boundary.
            if self._is_cancelled():
                logger.info("  CANCEL_EVENT set — stopping at step %s", step_index)
                print(f"  CANCEL: external cancel_event fired — stopping at step {step_index}")
                persist(step_index, status="cancelled", halt_reason="cancel_event")
                self._final_status = "cancelled"
                break

            # Pending pause (#73) — surface as paused, build PauseState in
            # _build_runner_result.
            if self.tool_channel.is_paused():
                persist(step_index, status="paused", halt_reason="user_input")
                self._final_status = "paused"
                break

            # Budget + time checks
            elapsed = time.time() - self._run_start
            _gpu_cost, _claude_cost, _proxy_cost, total_cost = self._cost_totals()

            if total_cost >= self.max_cost:
                print(f"  BUDGET CAP: ${total_cost:.2f} >= ${self.max_cost:.2f} — stopping")
                persist(step_index, status="halted", halt_reason="budget_cap")
                break
            if elapsed >= self.max_time:
                print(f"  TIME CAP: {elapsed/60:.0f}m >= {self.max_time/60:.0f}m — stopping")
                persist(step_index, status="halted", halt_reason="time_cap")
                break

            step = plan.steps[step_index]

            # Dynamic intent: inject listing position for click steps
            dynamic_intent = step.intent
            if step.type == "click" and listings_on_page > 0:
                dynamic_intent = (
                    f"Scroll down past the first {listings_on_page} listings. "
                    f"Then click the next listing title text below a photo."
                )
            effective_step = MicroIntent(
                intent=dynamic_intent, type=step.type, verify=step.verify,
                budget=step.budget, reverse=step.reverse, grounding=step.grounding,
                claude_only=step.claude_only, loop_target=step.loop_target,
                loop_count=step.loop_count,
                section=step.section, required=step.required, gate=step.gate,
                params=dict(step.params or {}),  # form-vocab fill_field/submit/select_option payload
                hints=dict(getattr(step, "hints", {}) or {}),  # plan-driven grounding hints
            )

            logger.info(f"  [{step_index:2d}] {step.type:15s} {dynamic_intent[:60]}")

            # Handle loop steps — each loop step has its own counter
            if step.type == "loop":
                loop_counters[step_index] = loop_counters.get(step_index, 0) + 1
                count = loop_counters[step_index]
                max_count = step.loop_count or max_loop_iterations
                if count < max_count:
                    target = step.loop_target if step.loop_target >= 0 else step_index
                    step_index = target
                    logger.info(f"  [loop@{step_index}] iteration {count}/{max_count} → step {step_index}")
                    persist(step_index, status="running")
                    continue
                else:
                    logger.info("  [loop] max iterations reached")
                    step_index += 1
                    persist(step_index, status="running")
                    continue

            # Execute step
            self._active_checkpoint_context = {
                "checkpoint": checkpoint,
                "plan": plan,
                "results": results,
                "loop_counters": loop_counters,
                "listings_on_page": listings_on_page,
                "step_index": step_index,
            }
            # #121 step 1+2: capture pre-step state. Step 1 logged it for
            # validation; step 2 stashes it on self so _reverse_step can
            # use the diff to skip undo work that wasn't needed.
            pre_snapshot = step_snapshot.capture(self)
            self._pre_step_snapshot = pre_snapshot
            try:
                step_result = self._execute_step(effective_step, step_index)
            finally:
                self._active_checkpoint_context = None
            # Observability extras (#74): capture screenshot + invoke callback.
            if step_result.screenshot_png is None:
                step_result.screenshot_png = self._capture_screenshot_bytes()
            results.append(step_result)
            self._enforce_screenshot_cap(results)
            self._invoke_step_callback(step_result)
            self._record_step_costs(effective_step, step_result)
            self._log_progress(step_result, results)
            self._log_step_diff(pre_snapshot, effective_step, step_result)
            # Debug screenshot dump (#152 follow-up) — when
            # MANTIS_DEBUG_DUMP_DIR is set, save the post-step screenshot
            # so failing runs can be inspected without re-running the
            # whole Modal pipeline.
            if not step_result.success:
                self._dump_debug_screenshot(
                    f"step{step_index}_post_{effective_step.type}",
                    self._safe_screenshot(),
                )

            # Verify form-shape steps actually produced an observable
            # state change (staffcrm verify follow-up). The handler can
            # report success because the click fired, but if NOTHING in
            # the runner's snapshot changed (URL, focus, scroll, page,
            # viewport, extraction, new URLs seen) the click missed or
            # the page rejected it silently — common login failure mode.
            # Demote to fail so the existing retry loop kicks in.
            #
            # Pure-observational: uses the #121 step_snapshot diff, no
            # regex / heuristic / vision call. Skips fill_field (no
            # state change is the normal case there — the field just
            # gains focus).
            if (
                step_result.success
                and effective_step.type in ("submit", "select_option")
            ):
                try:
                    post_snapshot = step_snapshot.capture(self)
                    delta = step_snapshot.diff(pre_snapshot, post_snapshot)
                except Exception as exc:  # noqa: BLE001 — never break runs
                    logger.debug("post-submit diff capture failed: %s", exc)
                    delta = None
                if delta is not None and not delta.has_changes:
                    logger.warning(
                        "  [%d] %s reported success but no observable state "
                        "change — demoting to failure (will retry)",
                        step_index, effective_step.type,
                    )
                    step_result.success = False
                    step_result.data = (
                        step_result.data or ""
                    ) + ":no_state_change"

            # Handle dedup: extract_url returned DUPLICATE → skip to loop
            if step_result.data and "DUPLICATE" in step_result.data:
                logger.info(f"  [{step_index}] DEDUP — skipping to next listing")
                # Go back to results page first
                try:
                    self._return_to_results_page()
                except Exception:
                    pass
                # Jump to loop step
                for j in range(step_index + 1, len(plan.steps)):
                    if plan.steps[j].type == "loop":
                        step_index = j
                        break
                else:
                    step_index += 1
                listings_on_page += 1  # Count it as "processed" for scroll-past
                persist(step_index, status="running", halt_reason="duplicate_listing")
                continue

            if step_result.success:
                step_retry_counts.pop(step_index, None)
                # Track listing progress
                if step.type == "paginate":
                    listings_on_page = 0
                    self._listings_on_page = 0
                    self._extracted_titles = []  # New page = new listings
                    self._page_listings = []   # Reset card cache
                    self._page_listing_index = 0
                    self._viewport_stage = 0  # Start from Home on new page
                    # Reset inner loop counters — new page means fresh listing loop
                    for k in list(loop_counters.keys()):
                        if k != step_index:  # Don't reset the outer loop's own counter
                            loop_counters[k] = 0
                    # Scroll to top of new page + wait for load
                    try:
                        self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
                        time.sleep(8)
                    except Exception:
                        pass
                    logger.info("  [paginate] Success — reset to top of new page")
                    self._last_known_url = self._current_results_page_url() or self._last_known_url

                # Verify navigate_back: check if we left the detail page
                if step.type == "navigate_back" and self.extractor:
                    time.sleep(2)
                    screenshot = self.env.screenshot()
                    check = self.extractor.extract(screenshot)
                    self.costs["claude_extract"] += 1
                    url = check.url if check else ""
                    if url:
                        self._last_known_url = url
                    if url and self.site_config.is_detail_page(url):
                        # Still on detail page — give the CUA a recovery task
                        # Use the plan's reverse intent, not hardcoded site knowledge
                        recovery_intent = step.reverse or "Go back to the previous page."
                        logger.warning(f"  [back-verify] Still on detail page — CUA recovery: {recovery_intent[:50]}")
                        recovery = self._execute_holo3_step(
                            MicroIntent(
                                intent=recovery_intent,
                                type="navigate_back",
                                budget=8,
                                grounding=True,
                            ),
                            step_index,
                        )
                        self.costs["gpu_steps"] += recovery.steps_used

                step_index += 1
                persist(step_index, status="running")
            else:
                # Check required/gate constraints FIRST
                if step.required:
                    attempt = step_retry_counts.get(step_index, 0) + 1
                    if attempt <= self.max_retries:
                        step_retry_counts[step_index] = attempt
                        logger.warning(f"  [{step_index}] REQUIRED step failed — retry {attempt}/{self.max_retries}")
                        persist(step_index, status="running", halt_reason=f"required_retry:{step.type}:{attempt}")
                        time.sleep(3)
                        continue  # Retry the same step
                    else:
                        logger.error(f"  [{step_index}] REQUIRED step failed after {self.max_retries} retries — HALTING")
                        print(f"  HALT: Required step '{step.intent[:50]}' failed. Cannot proceed.")
                        persist(step_index, status="halted", halt_reason=f"required_failed:{step.type}")
                        break

                if step.gate:
                    # If Cloudflare/anti-bot detected, retry navigate + gate once
                    gate_data = step_result.data or ""
                    gate_retry_key = f"gate_retry_{step_index}"
                    if (
                        "cloudflare" in gate_data.lower()
                        or "blocked" in gate_data.lower()
                        or "security" in gate_data.lower()
                        or "something went wrong" in gate_data.lower()
                        or "request fail" in gate_data.lower()
                    ):
                        if not step_retry_counts.get(gate_retry_key):
                            step_retry_counts[gate_retry_key] = 1
                            print("  [gate] Anti-bot detected — waiting 15s and retrying from navigate")
                            time.sleep(15)
                            # Re-run navigate step (step 0) then retry gate
                            nav_step = plan.steps[0] if plan.steps[0].type == "navigate" else None
                            if nav_step:
                                self._execute_navigate(nav_step, 0)
                                time.sleep(5)
                            persist(step_index, status="running", halt_reason="gate_retry")
                            continue  # Retry the gate step
                    logger.error(f"  [{step_index}] GATE FAILED: {step.verify[:60]} — HALTING")
                    print(f"  HALT: Gate verification '{step.verify[:50]}' failed. Setup incomplete.")
                    persist(step_index, status="halted", halt_reason="gate_failed")
                    break

                # Handle failure based on step type
                if step.type in ("navigate",):
                    logger.error(f"  [{step_index}] NAVIGATE FAILED — cannot proceed")
                    self._reverse_step(step)
                    persist(step_index, status="halted", halt_reason="navigate_failed")
                    break
                elif step.type in ("click",):
                    # Check if page exhausted (no more listings)
                    if step_result.data == "page_exhausted":
                        logger.info(f"  [{step_index}] PAGE EXHAUSTED — jumping to paginate")
                        # Find paginate step and jump there
                        for j in range(step_index + 1, len(plan.steps)):
                            if plan.steps[j].type == "paginate":
                                step_index = j
                                break
                        else:
                            # No paginate step — find next loop
                            for j in range(step_index + 1, len(plan.steps)):
                                if plan.steps[j].type == "loop":
                                    step_index = j
                                    break
                            else:
                                step_index += 1
                        persist(step_index, status="running", halt_reason="page_exhausted")
                        continue
                    if step_result.data in ("scan_error", "page_blocked"):
                        attempt = step_retry_counts.get(step_index, 0) + 1
                        if attempt <= self.max_retries:
                            step_retry_counts[step_index] = attempt
                            wait_s = 12 if step_result.data == "page_blocked" else 4
                            logger.warning(
                                f"  [{step_index}] {step_result.data.upper()} — "
                                f"waiting {wait_s}s and retrying ({attempt}/{self.max_retries})"
                            )
                            persist(step_index, status="running", halt_reason=f"{step_result.data}_retry:{attempt}")
                            time.sleep(wait_s)
                            continue
                        logger.warning(f"  [{step_index}] {step_result.data.upper()} — retry budget exhausted")
                        if step_result.data == "page_blocked":
                            reload_key = f"page_blocked_reload_{step_index}"
                            reload_attempt = step_retry_counts.get(reload_key, 0) + 1
                            if reload_attempt <= 1 and self._ensure_results_filters(
                                step_index, force_reload=True
                            ):
                                step_retry_counts[reload_key] = reload_attempt
                                step_retry_counts[step_index] = 0
                                logger.warning(
                                    f"  [{step_index}] PAGE_BLOCKED — reloaded filtered URL, retrying click"
                                )
                                persist(step_index, status="running", halt_reason="page_blocked_reload")
                                continue
                            logger.error(
                                f"  [{step_index}] PAGE_BLOCKED after filtered reload — halting"
                            )
                            print("  HALT: Filtered results page is blocked/erroring.")
                            persist(step_index, status="halted", halt_reason="page_blocked")
                            break
                    # Click failed — skip extraction cycle to loop
                    logger.warning(f"  [{step_index}] CLICK FAILED — skipping to next")
                    try:
                        self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Escape"}))
                        time.sleep(0.5)
                    except Exception:
                        pass
                    for j in range(step_index + 1, len(plan.steps)):
                        if plan.steps[j].type == "loop":
                            step_index = j
                            break
                    else:
                        step_index += 1
                    persist(step_index, status="running", halt_reason="click_failed")
                    continue
                elif step.type in ("filter",):
                    # Filter failure is non-fatal — skip and try next filter
                    logger.warning(f"  [{step_index}] FILTER FAILED — skipping")
                    self._reverse_step(step)
                    step_index += 1
                    persist(step_index, status="running", halt_reason="filter_failed")
                elif step.type in ("scroll",):
                    # Scroll "failure" usually means the model didn't call done()
                    # but the page DID scroll — treat as success
                    logger.info(f"  [{step_index}] Scroll completed (no done() but page changed)")
                    step_index += 1
                    persist(step_index, status="running", halt_reason="scroll_no_done")
                elif step.type in ("navigate_back",):
                    # Back failure — try multiple times and verify
                    logger.warning(f"  [{step_index}] BACK FAILED — retrying Alt+Left")
                    for back_attempt in range(3):
                        try:
                            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "alt+Left"}))
                            time.sleep(3)
                        except Exception:
                            pass
                        # Verify: check if URL is back on search results
                        if self.extractor:
                            screenshot = self.env.screenshot()
                            check = self.extractor.extract(screenshot)
                            url = check.url if check else ""
                            if url:
                                self._last_known_url = url
                            if url and self.site_config.is_results_page(url) and not self.site_config.is_detail_page(url):
                                logger.info(f"  [back] Verified on results page after {back_attempt+1} attempts")
                                break
                    step_index += 1
                    persist(step_index, status="running", halt_reason="navigate_back_recovered")
                elif step.type in ("paginate",):
                    # Paginate failed — no new page loaded, stop the pipeline
                    logger.warning(f"  [{step_index}] PAGINATE FAILED — no more pages, ending")
                    # Exhaust the outer loop so it doesn't restart on the same page
                    for k in list(loop_counters.keys()):
                        loop_counters[k] = 999999
                    step_index += 1
                    persist(step_index, status="running", halt_reason="paginate_exhausted")
                elif step.type in ("extract_url", "extract_data"):
                    # Claude-only step failed — skip
                    step_index += 1
                    persist(step_index, status="running", halt_reason=f"{step.type}_failed")
                else:
                    # Generic failure — reverse and skip
                    self._reverse_step(step)
                    step_result.reversed = True
                    logger.warning(f"  [{step_index}] FAILED + reversed — skipping")
                    step_index += 1
                    persist(step_index, status="running", halt_reason=f"{step.type}_failed")

        logger.info(f"MicroPlan complete: {len(results)} steps executed")
        # Final cost summary
        if step_index >= len(plan.steps):
            persist(step_index, status="completed")
            self._final_status = "completed"
        elif self._final_status == "running":
            persist(step_index, status="halted", halt_reason="stopped")
            self._final_status = "halted"

        # Stash final loop counters / progress so resume() / run_with_status()
        # can reconstruct PauseState without re-walking the loop.
        self._last_run_step_index = step_index
        self._last_loop_counters = dict(loop_counters)
        self._last_listings_on_page = listings_on_page

        gpu_cost, claude_cost, proxy_cost, total_cost = self._cost_totals()
        viable_count, phone_leads = self._lead_counts(results)
        elapsed = time.time() - self._run_start

        print(f"\n{'='*60}")
        print("MICRO-PLAN COMPLETE")
        print(f"  Time:     {elapsed/60:.0f}m")
        print(f"  Steps:    {len(results)}")
        print(f"  Leads:    {viable_count}")
        print(f"  Phone:    {phone_leads}")
        print(
            f"  Cost:     ${total_cost:.2f} total "
            f"(${total_cost/max(viable_count,1):.2f}/lead, "
            f"${total_cost/max(phone_leads,1):.2f}/phone lead)"
        )
        print(f"    GPU:    ${gpu_cost:.2f} ({self.costs['gpu_steps']} steps)")
        print(f"    Claude: ${claude_cost:.2f} ({self.costs['claude_extract']} extract + {self.costs['claude_grounding']} grounding)")
        print(f"    Proxy:  ${proxy_cost:.2f} ({self.costs['proxy_mb']:.0f} MB)")
        print(f"{'='*60}")

        # Attach costs to results for saving
        self._final_costs = {
            "total": round(total_cost, 3),
            "gpu": round(gpu_cost, 3),
            "claude": round(claude_cost, 3),
            "proxy": round(proxy_cost, 3),
            "leads": viable_count,
            "leads_with_phone": phone_leads,
            "per_lead": round(total_cost / max(viable_count, 1), 3),
            "per_phone_lead": round(total_cost / max(phone_leads, 1), 3),
            "status": self._final_status,
            "checkpoint_path": self.checkpoint_path,
        }

        return results

    def run_with_status(self, plan: MicroPlan, resume: bool = False) -> RunnerResult:
        """Same as ``run(plan)``, but returns the rich :class:`RunnerResult`.

        Carries cancellation / pause state alongside the step list so callers
        wiring the host backend don't have to read ``self._final_status``.
        """
        steps = self.run(plan, resume=resume)
        return self._build_runner_result(plan, steps)

    def resume(
        self,
        state: PauseState | dict[str, Any],
        *,
        user_input: Any = None,
        plan: MicroPlan | None = None,
    ) -> RunnerResult:
        """Resume a paused run with the supplied user input.

        ``state`` is the :class:`PauseState` previously returned via
        ``RunnerResult.pause_state``. ``user_input`` is fed back to the paused
        tool handler as its return value (the tool simulates re-invocation
        and reads the staged value via :meth:`consume_pause_input`).
        ``plan`` is required because the runner doesn't keep the plan around;
        callers persist the plan alongside the PauseState.
        """
        if isinstance(state, dict):
            state = PauseState.from_dict(state)
        if plan is None:
            raise ValueError("resume() requires the original plan; pass plan=...")
        # Validate the plan matches the snapshot.
        signature = self._compute_plan_signature(plan)
        if state.plan_signature and state.plan_signature != signature:
            raise ValueError(
                "PauseState plan_signature mismatch — can't resume a different plan"
            )

        # Replay results from the snapshot so cost/lead totals stay continuous.
        replayed = [StepResult.from_dict(d) for d in state.step_results]
        self.tool_channel.clear_pause()

        # Rehydrate runner state. The simplest correct approach: load the
        # checkpoint that the original run() persisted (state.checkpoint_path
        # mirrors self.checkpoint_path) and ride the existing resume codepath.
        self.checkpoint_path = state.checkpoint_path or self.checkpoint_path
        self.run_key = state.run_key or self.run_key
        self.session_name = state.session_name or self.session_name
        self.plan_signature = signature
        self.resume_state = True

        # Inject the user-supplied value as the next return of the paused tool.
        # The host's handler simply re-runs and reads the pre-injected value
        # via runner.consume_pause_input(). One-shot — cleared after read.
        self._pause_input = user_input
        try:
            steps_continued = self.run(plan, resume=True)
        finally:
            self._pause_input = None

        all_steps = replayed + steps_continued[len(replayed):]
        return self._build_runner_result(plan, all_steps)

    def consume_pause_input(self, default: Any = None) -> Any:
        """Read (and clear) the value supplied to the most recent ``resume()``.

        Tools that previously called ``raise PauseRequested(...)`` should call
        this on the next invocation to retrieve the user's input. Returns
        ``default`` if no input is staged.
        """
        value = getattr(self, "_pause_input", None)
        self._pause_input = None
        return default if value is None else value

    def _build_runner_result(
        self, plan: MicroPlan, steps: list[StepResult]
    ) -> RunnerResult:
        status = self._final_status or "completed"
        cancelled = status == "cancelled"
        paused = status == "paused" and self.tool_channel.is_paused()
        pause_state: PauseState | None = None
        if paused:
            pp = self.tool_channel.pending_pause or {}
            pause_state = PauseState(
                run_key=self.run_key,
                plan_signature=self.plan_signature
                or self._compute_plan_signature(plan),
                session_name=self.session_name,
                step_index=getattr(self, "_last_run_step_index", 0),
                pending_tool=str(pp.get("tool", "")),
                pending_arguments=dict(pp.get("arguments", {})),
                pending_reason=str(pp.get("reason", "user_input")),
                prompt=str(pp.get("prompt", "")),
                step_results=[s.to_dict() for s in steps],
                loop_counters={str(k): v for k, v in getattr(self, "_last_loop_counters", {}).items()},
                listings_on_page=getattr(self, "_last_listings_on_page", 0),
                checkpoint_path=self.checkpoint_path,
                timestamp=time.time(),
            )
        return RunnerResult(
            steps=steps,
            status=status,
            cancelled=cancelled,
            paused=paused,
            pause_state=pause_state,
            halt_reason="" if status == "completed" else status,
        )

    @staticmethod
    def _successful_lead_data(results: list[StepResult]) -> list[str]:
        """Backward-compat shim — delegates to :class:`ListingDedup`."""
        return ListingDedup.successful_lead_data(results)

    @staticmethod
    def _lead_key(data: str) -> str:
        """Backward-compat shim — delegates to :class:`ListingDedup`."""
        return ListingDedup.lead_key(data)

    @staticmethod
    def _lead_has_phone(data: str) -> bool:
        """Backward-compat shim — delegates to :class:`ListingDedup`."""
        return ListingDedup.lead_has_phone(data)

    @classmethod
    def _lead_counts(cls, results: list[StepResult]) -> tuple[int, int]:
        """Backward-compat shim — delegates to :class:`ListingDedup`."""
        return ListingDedup.lead_counts(results)

    @staticmethod
    def _extract_url_from_intent(intent: str) -> str:
        match = re.search(r'https?://[^\s"]+', intent)
        return match.group() if match else ""

    @staticmethod
    def _derive_filter_tokens(url: str) -> tuple[str, ...]:
        """Derive path tokens that must remain present on result pages."""
        match = re.search(r'https?://[^/]+/([^?#]+)', url)
        if not match:
            return ()
        tokens = []
        for token in match.group(1).strip("/").split("/"):
            if not token or token in {"boats"} or token.startswith("page-"):
                continue
            tokens.append(token.lower())
        return tuple(tokens)

    def _url_has_required_filters(self, url: str) -> bool:
        url_lower = url.lower()
        is_results = self.site_config.is_results_page(url) if self.site_config.results_page_pattern else bool(url_lower)
        is_detail = self.site_config.is_detail_page(url) if self.site_config.detail_page_pattern else False
        return (
            bool(url_lower)
            and is_results
            and not is_detail
            and all(token in url_lower for token in self._required_filter_tokens)
        )

    def _reset_results_scan_state(self) -> None:
        self._page_listings = []
        self._page_listing_index = 0
        self._viewport_stage = 0

    def _ensure_results_filters(self, index: int, force_reload: bool = False) -> bool:
        """Keep result-page actions on the canonical filtered result URL."""
        if not self.extractor or not self._results_base_url or not self._required_filter_tokens:
            return True

        url = ""
        screenshot = None
        try:
            screenshot = self.env.screenshot()
            data = self.extractor.extract(screenshot)
            self.costs["claude_extract"] += 1
            url = data.url if data else ""
        except Exception as e:
            logger.warning("  [filters] URL verification failed: %s", e)

        if not force_reload and self._url_has_required_filters(url):
            self._last_known_url = url
            self.dynamic_verifier.record_filter_check(
                page=self._current_page,
                url=url,
                passed=True,
                reason="url_contains_required_filters",
            )
            return True

        if not force_reload and not url and screenshot is not None:
            gate_prefix = self.site_config.gate_verify_prompt or "Page is a filtered results page with these active filters: "
            requirement = (
                gate_prefix
                + ", ".join(self._required_filter_tokens)
            )
            try:
                passed, reason = self.extractor.verify_gate(screenshot, requirement)
                self.costs["claude_extract"] += 1
                if passed:
                    logger.info("  [filters] Visual filter gate passed despite unreadable URL")
                    self._last_known_url = self._current_results_page_url() or self._results_base_url
                    self.dynamic_verifier.record_filter_check(
                        page=self._current_page,
                        url=self._last_known_url,
                        passed=True,
                        reason="visual_gate_passed",
                    )
                    return True
                logger.warning("  [filters] Visual filter gate failed: %s", reason[:120])
                self.dynamic_verifier.record_filter_check(
                    page=self._current_page,
                    url=url,
                    passed=False,
                    reason=reason[:200],
                )
            except Exception as e:
                logger.warning("  [filters] Visual filter gate errored: %s", e)

        logger.warning(
            "  [filters] Reloading canonical filtered results before step %s "
            "(current url=%s, required=%s)",
            index,
            url[:120],
            ",".join(self._required_filter_tokens),
        )
        try:
            reload_url = self._current_results_page_url() or self._results_base_url
            self.env.reset(task="navigate", start_url=reload_url)
            time.sleep(12)
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(2)
            self._reset_results_scan_state()
            self._last_known_url = reload_url
            self._set_scroll_state(context="results_top", url=reload_url, page_downs=0, wheel_downs=0)
            self.dynamic_verifier.record_filter_check(
                page=self._current_page,
                url=reload_url,
                passed=True,
                reason="reloaded_canonical_filtered_results",
            )
            return True
        except Exception as e:
            logger.error("  [filters] Failed to reload filtered results: %s", e)
            self.dynamic_verifier.record_filter_check(
                page=self._current_page,
                url=url,
                passed=False,
                reason=f"reload_failed:{e}",
            )
            return False

    def _execute_step(self, step: MicroIntent, index: int) -> StepResult:
        """Execute a single micro-intent."""

        # Navigate steps: use env.reset() with URL instead of Holo3
        if step.type == "navigate":
            return self._execute_navigate(step, index)

        # Click steps: dispatch by hints["layout"] (plan-driven, see issue #86).
        # Default for click in an "extraction" section = listings click. Pages
        # outside the extraction section, or steps that explicitly hint
        # layout="single", route through find_form_target instead.
        if step.type == "click" and self.extractor:
            layout_hint = (step.hints or {}).get("layout", "")
            is_listings = (
                layout_hint == "listings"
                or (not layout_hint and step.section == "extraction")
            )
            if is_listings:
                if not self._ensure_results_filters(index):
                    return StepResult(
                        step_index=index, intent=step.intent, success=False,
                        data="filters_not_applied",
                    )
                # Brief settle — page may still be loading after navigate/paginate
                time.sleep(2)
                return self._execute_claude_guided_click(step, index)
            # Single-element click (nav link, button, anything labelled).
            time.sleep(2)
            return self._execute_claude_guided_form(
                MicroIntent(
                    intent=step.intent, type="submit",  # reuse submit dispatch
                    budget=step.budget, section=step.section,
                    required=step.required,
                    params={"label": (step.params or {}).get("label", "")},
                ),
                index,
            )

        # Gate steps: dedicated verifier (not extract_data)
        if step.gate and self.extractor:
            print(f"  [gate] Verifying: {(step.verify or step.intent)[:80]}")
            time.sleep(2)
            screenshot = self.env.screenshot()
            passed, reason = self.extractor.verify_gate(screenshot, step.verify or step.intent)
            self.costs["claude_extract"] += 1
            print(f"  [gate] Result: {'PASS' if passed else 'FAIL'} — {reason[:80]}")
            return StepResult(
                step_index=index, intent=step.intent,
                success=passed, data=f"gate:{'PASS' if passed else 'FAIL'}:{reason[:100]}",
            )

        # Claude-only steps (extract_url, extract_data)
        if step.claude_only:
            # Brief settle — page may still be rendering after scroll
            time.sleep(1)
            return self._execute_claude_step(step, index)

        # Paginate: layered strategy
        # 1. URL-based (fastest, most reliable if URL pattern known)
        # 2. Claude-guided with End→Page_Up viewport
        # 3. Holo3 with calculated scroll (fallback)
        if step.type == "paginate":
            if not self._ensure_results_filters(index):
                return StepResult(
                    step_index=index, intent=step.intent, success=False,
                    data="filters_not_applied",
                )
            result = self._execute_paginate_layered(step, index)
            return result

        # Filter steps: Claude identifies target → direct click/type (Holo3 can't handle sidebar)
        if step.type == "filter" and self.extractor:
            time.sleep(3)  # Longer wait — page filters may lazy-load
            return self._execute_claude_guided_filter(step, index)

        # Form-shaped steps (issue #80): login forms, edit pages, dropdowns,
        # single labelled buttons. Use find_form_target instead of find_all_listings
        # so non-listings pages don't return "0 cards".
        if step.type in ("fill_field", "submit", "select_option") and self.extractor:
            time.sleep(2)  # form pages may finish hydrating
            return self._execute_claude_guided_form(step, index)

        if step.type == "navigate_back" and self._opened_detail_in_new_tab:
            return self._execute_close_detail_tab(step, index)

        # Holo3 steps (scroll, navigate_back, paginate)
        return self._execute_holo3_step(step, index)

    def _return_to_results_page(self) -> None:
        if self._opened_detail_in_new_tab:
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "ctrl+w"}))
            self._opened_detail_in_new_tab = False
        else:
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "alt+Left"}))
        time.sleep(2)

    def _execute_close_detail_tab(self, step: MicroIntent, index: int) -> StepResult:
        try:
            self._return_to_results_page()
            if self.extractor:
                screenshot = self.env.screenshot()
                check = self.extractor.extract(screenshot)
                self.costs["claude_extract"] += 1
                url = check.url if check else ""
                if url:
                    self._last_known_url = url
                if url and self.site_config.is_detail_page(url):
                    return StepResult(step_index=index, intent=step.intent, success=False)
            return StepResult(step_index=index, intent=step.intent, success=True, steps_used=1)
        except Exception as exc:
            logger.warning("  [back] Failed closing detail tab: %s", exc)
            return StepResult(step_index=index, intent=step.intent, success=False)

    def _read_current_url(self, screenshot=None) -> str:
        """Resolve the active tab's URL, preferring CDP over screenshot OCR.

        Issue #89 §1: the screenshot-only verification was returning empty
        URLs (``(url=)``) on SPA navigations because the address bar text
        wasn't yet repainted, or the page used ``history.pushState`` and
        the runner read a stale screenshot. CDP's ``/json/list`` is the
        ground truth for the active page's URL.

        Falls back to ``ClaudeExtractor.extract`` when CDP is unreachable
        (Modal hosts where the port wasn't bound, older Truss images, etc.)
        so existing screenshot-based behaviour still works.

        Diagnostic logging: the path taken (cdp / ocr) is logged at INFO so
        run traces show which source produced the URL — needed because
        post-PR-90 the host canary still showed ``(url=)`` empty and we
        couldn't tell whether CDP was unreachable or returned empty itself.
        Also handles the case where ``env.current_url`` is a *method*
        instead of a property (a common integration mistake) by calling
        it when callable.
        """
        cdp_url = ""
        cdp_error: Exception | None = None
        try:
            raw = getattr(self.env, "current_url", "")
            # Tolerate integrations that defined current_url() as a method
            # rather than a @property — call it instead of returning the
            # bound-method object (which would be truthy and short-circuit).
            if callable(raw):
                raw = raw()
            cdp_url = (raw or "").strip() if isinstance(raw, str) else ""
        except Exception as exc:
            cdp_error = exc

        if cdp_url:
            logger.info(f"  [url] cdp={cdp_url[:80]}")
            return cdp_url

        if cdp_error is not None:
            logger.info(f"  [url] cdp unavailable ({type(cdp_error).__name__}); falling back to OCR")
        elif screenshot is not None:
            logger.info("  [url] cdp empty; falling back to OCR")

        if screenshot is not None and self.extractor:
            try:
                verify_data = self.extractor.extract(screenshot)
            except Exception as exc:
                logger.info(f"  [url] ocr extract failed: {exc}")
                return ""
            self.costs["claude_extract"] += 1
            ocr_url = (verify_data.url or "") if verify_data else ""
            logger.info(f"  [url] ocr={ocr_url[:80] or '<empty>'}")
            return ocr_url
        return ""

    def _execute_navigate(self, step: MicroIntent, index: int) -> StepResult:
        """Navigate to a URL using env.reset() — no Holo3 steps needed.

        Waits for page load and handles Cloudflare challenges (auto-solve in 5-10s).

        First-paint wait resolution order (first hit wins):
          1. step.params["wait_after_load_seconds"] — plan-driven override
          2. env MANTIS_NAV_WAIT_SECONDS                — deployment-wide override
          3. 18s default                                — covers Cloudflare auto-solve
        """
        import re
        url_match = re.search(r'https?://[^\s"]+', step.intent)
        url = url_match.group() if url_match else ""

        if not url:
            logger.warning(f"  [navigate] No URL found in intent: {step.intent[:60]}")
            return StepResult(step_index=index, intent=step.intent, success=False)

        try:
            wait_seconds = float(
                (step.params or {}).get("wait_after_load_seconds")
                or os.environ.get("MANTIS_NAV_WAIT_SECONDS")
                or 18
            )
        except (TypeError, ValueError):
            wait_seconds = 18.0
        wait_seconds = max(0.0, min(wait_seconds, 120.0))

        logger.info(f"  [navigate] Loading {url} (first-paint wait {wait_seconds:.0f}s)")
        try:
            self.env.reset(task="navigate", start_url=url)
            # Wait for Cloudflare challenge to auto-solve + page render
            time.sleep(wait_seconds)
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(2)
            # Store as base URL for pagination (results page, not detail page)
            self._results_base_url = url
            self._required_filter_tokens = self._derive_filter_tokens(url)
            self._current_page = 1
            self._last_known_url = url
            self._reset_results_scan_state()
            self._set_scroll_state(context="results_top", url=url, page_downs=0, wheel_downs=0)
            self.dynamic_verifier.set_required_filter_tokens(self._required_filter_tokens)
            self.dynamic_verifier.record_page_start(page=self._current_page, url=url)
            return StepResult(step_index=index, intent=step.intent, success=True)
        except Exception as e:
            logger.error(f"  [navigate] Failed: {e}")
            return StepResult(step_index=index, intent=step.intent, success=False)

    def _execute_claude_guided_click(self, step: MicroIntent, index: int) -> StepResult:
        """Claude finds ALL listings once per page, clicks them one by one.

        First call on a page: find_all_listings() → cache coordinates (1 Claude call)
        Subsequent calls: pop next from cache (0 Claude calls)
        Page exhausted when cache is empty.
        """
        # If no cached listings, scan the page (one Claude call for ALL cards)
        if self._page_listing_index >= len(self._page_listings):
            # Staged per-viewport scan: scan ONE viewport, cache its cards.
            # When cache empties, advance to next viewport (Page_Down).
            # Only page_exhausted after all viewport stages return empty.
            while self._viewport_stage < self._max_viewport_stages:
                # Reconstruct the current viewport deterministically:
                # always start from Home, then Page_Down N times.
                try:
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
                    time.sleep(0.5)
                    for _ in range(self._viewport_stage):
                        self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                        time.sleep(0.5)
                    self._set_scroll_state(
                        context="results_scan",
                        url=self._current_results_page_url() or self._results_base_url,
                        page_downs=self._viewport_stage,
                        wheel_downs=0,
                        viewport_stage=self._viewport_stage,
                    )
                except Exception:
                    pass

                screenshot = self.env.screenshot()
                scan_result = self.extractor.find_all_listings(screenshot)
                self.costs["claude_extract"] += 1

                scan_status = "ok"
                if isinstance(scan_result, tuple):
                    status = scan_result[0]
                    if status == "blocked":
                        # Could be a real error/anti-bot page OR a transient
                        # proxy/CDN loading splash. Wait + re-scan the SAME
                        # viewport once before giving up — proxy-loading
                        # screens typically resolve in 5-15s. Caught when
                        # Chromium's first-paint splash misled find_all_listings
                        # into reporting blocked on a CRM that loaded fine
                        # 30s later.
                        already_retried = getattr(self, "_blocked_retry_done", False)
                        if not already_retried:
                            logger.warning(
                                f"  [claude-click] Viewport {self._viewport_stage}: "
                                f"blocked/error page — waiting 12s and rescanning"
                            )
                            self._blocked_retry_done = True
                            time.sleep(12)
                            screenshot = self.env.screenshot()
                            scan_result = self.extractor.find_all_listings(screenshot)
                            self.costs["claude_extract"] += 1
                            if isinstance(scan_result, tuple) and scan_result[0] == "blocked":
                                logger.warning(
                                    f"  [claude-click] Viewport {self._viewport_stage}: "
                                    f"still blocked after rescan — halting"
                                )
                            else:
                                # Recovered — fall through to normal processing.
                                self._blocked_retry_done = False
                                if isinstance(scan_result, tuple):
                                    status = scan_result[0]
                                else:
                                    status = "ok"
                        if status == "blocked":
                            self._blocked_retry_done = False
                            logger.warning(f"  [claude-click] Viewport {self._viewport_stage}: blocked/error page")
                            self.dynamic_verifier.record_viewport_scan(
                                page=self._current_page,
                                viewport_stage=self._viewport_stage,
                                cards=[],
                                new_cards=[],
                                status="blocked",
                                url=self._current_results_page_url() or self._last_known_url,
                            )
                            return StepResult(
                                step_index=index,
                                intent=step.intent,
                                success=False,
                                data="page_blocked",
                            )
                    if status == "error":
                        logger.warning(f"  [claude-click] Viewport {self._viewport_stage}: parse/API failure")
                        self.dynamic_verifier.record_viewport_scan(
                            page=self._current_page,
                            viewport_stage=self._viewport_stage,
                            cards=[],
                            new_cards=[],
                            status="error",
                            url=self._current_results_page_url() or self._last_known_url,
                        )
                        return StepResult(
                            step_index=index,
                            intent=step.intent,
                            success=False,
                            data="scan_error",
                        )
                    scan_status = status
                    cards = []
                else:
                    cards = scan_result

                # Filter out already-extracted titles
                skip = set(t.lower() for t in self._extracted_titles)
                filtered = [(x, y, t) for x, y, t in cards
                           if t.lower() not in skip and t != "unknown"]
                unknown_cards = [(x, y, t) for x, y, t in cards if t == "unknown"]
                filtered.extend(unknown_cards)
                filtered.sort(key=lambda c: c[1])
                self.dynamic_verifier.record_viewport_scan(
                    page=self._current_page,
                    viewport_stage=self._viewport_stage,
                    cards=cards,
                    new_cards=filtered,
                    status=scan_status,
                    url=self._current_results_page_url() or self._last_known_url,
                )

                logger.info(f"  [claude-click] Viewport {self._viewport_stage}: {len(cards)} cards, {len(filtered)} new")

                if filtered:
                    self._page_listings = filtered
                    self._page_listing_index = 0
                    break  # Found cards in this viewport — click them
                else:
                    self._viewport_stage += 1  # Try next viewport

            if not self._page_listings or self._page_listing_index >= len(self._page_listings):
                logger.info(f"  [claude-click] All {self._max_viewport_stages} viewports exhausted")
                self.dynamic_verifier.record_page_exhausted(
                    page=self._current_page,
                    reason=f"all_{self._max_viewport_stages}_viewports_exhausted",
                )
                return StepResult(step_index=index, intent=step.intent, success=False,
                                data="page_exhausted")

        # Pop next card from cache — scroll to the viewport where it was found
        x, y, title = self._page_listings[self._page_listing_index]
        self._page_listing_index += 1
        title_for_verification = (
            title
            if title.strip().lower() != "unknown"
            else f"unknown@v{self._viewport_stage}:{x},{y}"
        )
        self._last_click_title = title_for_verification
        self.dynamic_verifier.record_item_attempt(
            page=self._current_page,
            item=title_for_verification,
            viewport_stage=self._viewport_stage,
        )

        # Scroll to the correct viewport (Home + N Page_Downs)
        try:
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(0.5)
            for _ in range(self._viewport_stage):
                self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                time.sleep(0.5)
            self._set_scroll_state(
                context="results_click",
                url=self._current_results_page_url() or self._results_base_url,
                page_downs=self._viewport_stage,
                wheel_downs=0,
                viewport_stage=self._viewport_stage,
            )
        except Exception:
            pass

        logger.info(
            f"  [claude-click] Card {self._page_listing_index}/{len(self._page_listings)}: "
            f"'{title_for_verification[:40]}' at ({x}, {y}) viewport={self._viewport_stage}"
        )

        # Delay before the final screenshot so grounding sees the frame we will actually click.
        import random
        time.sleep(random.uniform(1.5, 3.5))

        # Grounding refines — but only accept if the delta is small
        if self.grounding and title.strip().lower() != "unknown":
            screenshot = self.env.screenshot()
            grounding_result = self.grounding.ground(screenshot, title, x, y)
            self.costs["claude_grounding"] += 1
            dx = abs(grounding_result.x - x)
            dy = abs(grounding_result.y - y)
            if grounding_result.confidence > 0.5 and dx < 200 and dy < 200:
                x, y = grounding_result.x, grounding_result.y
                logger.info(f"  [grounding] refined to ({x}, {y}) delta=({dx},{dy})")
            else:
                logger.info(f"  [grounding] rejected: delta=({dx},{dy}) conf={grounding_result.confidence}")
        elif title.strip().lower() == "unknown":
            logger.info("  [grounding] skipped for unknown-title card; using scan coordinates")

        # Click
        try:
            self.env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
            self.costs["gpu_steps"] += 1
            self.costs["gpu_seconds"] += 3
            self.costs["proxy_mb"] += 5.0
        except Exception as e:
            logger.warning(f"  [claude-click] Click failed: {e}")
            self.dynamic_verifier.record_item_completed(
                page=self._current_page,
                item=getattr(self, "_last_click_title", "") or title,
                success=False,
                reason=f"click_failed:{e}",
            )
            return StepResult(step_index=index, intent=step.intent, success=False)

        # Verify: are we on a detail page? Retry once (page may still load)
        # Prefer CDP over screenshot URL extraction — issue #89 §1.
        for verify_attempt in range(2):
            time.sleep(3 + verify_attempt * 3)  # 3s first, 6s retry
            url = self._read_current_url()
            if not url:
                # CDP unavailable — fall back to screenshot OCR.
                after = self.env.screenshot()
                url = self._read_current_url(after)

            if url and self.site_config.is_detail_page(url):
                logger.info(f"  [claude-click] Verified on detail page: {url[:60]}")
                self._last_known_url = url
                self.dynamic_verifier.record_item_opened(
                    page=self._current_page,
                    item=getattr(self, "_last_click_title", "") or title,
                    url=url,
                )
                self._last_extracted = {
                    **self._last_extracted,
                    "last_clicked_title": getattr(self, "_last_click_title", ""),
                    "last_attempted_url": url,
                    "last_attempted_at": time.time(),
                    "last_attempted_step": index,
                }
                self._set_scroll_state(context="detail_top", url=url, page_downs=0, wheel_downs=0)
                self._listings_on_page += 1
                # Store the exact title Claude found for skip list
                if hasattr(self, '_last_click_title') and self._last_click_title:
                    self._extracted_titles.append(self._last_click_title)
                return StepResult(step_index=index, intent=step.intent, success=True,
                                steps_used=1, duration=3.0 + verify_attempt * 3)

            if verify_attempt == 0:
                logger.info(f"  [claude-click] Not on detail page yet (url={url[:40]}) — retrying verify")

        logger.info("  [claude-click] Plain click did not navigate — trying middle-click fallback")
        try:
            self.env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y, "button": "middle"}))
            self.costs["gpu_steps"] += 1
            self.costs["gpu_seconds"] += 3
            self.costs["proxy_mb"] += 5.0
            time.sleep(2)

            for switch_attempt in range(2):
                url = self._read_current_url()
                if not url:
                    after = self.env.screenshot()
                    url = self._read_current_url(after)
                if url and self.site_config.is_detail_page(url):
                    logger.info(f"  [claude-click] Middle-click fallback opened detail: {url[:60]}")
                    self._opened_detail_in_new_tab = True
                    self._last_known_url = url
                    self.dynamic_verifier.record_item_opened(
                        page=self._current_page,
                        item=getattr(self, "_last_click_title", "") or title,
                        url=url,
                    )
                    self._last_extracted = {
                        **self._last_extracted,
                        "last_clicked_title": getattr(self, "_last_click_title", ""),
                        "last_attempted_url": url,
                        "last_attempted_at": time.time(),
                        "last_attempted_step": index,
                    }
                    self._set_scroll_state(context="detail_top", url=url, page_downs=0, wheel_downs=0)
                    self._listings_on_page += 1
                    if hasattr(self, '_last_click_title') and self._last_click_title:
                        self._extracted_titles.append(self._last_click_title)
                    return StepResult(step_index=index, intent=step.intent, success=True,
                                    steps_used=2 + switch_attempt, duration=9.0)

                if switch_attempt == 0:
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "ctrl+Tab"}))
                    time.sleep(2)
        except Exception as e:
            logger.warning(f"  [claude-click] Middle-click fallback failed: {e}")

        logger.info("  [claude-click] Middle-click did not verify — trying card-area click probes")
        probe_points = [
            ("image_center", x, y - 145),
            ("image_lower", x, y - 90),
            ("title_lower", x, y + 28),
            ("title_left", x - 90, y),
            ("title_right", x + 90, y),
        ]
        tried_points: set[tuple[int, int]] = set()
        for label, probe_x, probe_y in probe_points:
            probe_x = max(1, min(int(probe_x), self.env.screen_size[0] - 2))
            probe_y = max(1, min(int(probe_y), self.env.screen_size[1] - 2))
            if (probe_x, probe_y) in tried_points:
                continue
            tried_points.add((probe_x, probe_y))

            try:
                self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Escape"}))
                self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
                time.sleep(0.3)
                for _ in range(self._viewport_stage):
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                    time.sleep(0.3)
                logger.info(
                    "  [claude-click] Probe %s at (%s, %s)",
                    label,
                    probe_x,
                    probe_y,
                )
                self.env.step(Action(action_type=ActionType.CLICK, params={"x": probe_x, "y": probe_y}))
                self.costs["gpu_steps"] += 1
                self.costs["gpu_seconds"] += 3
                self.costs["proxy_mb"] += 5.0
                time.sleep(3)

                url = self._read_current_url()
                if not url:
                    after = self.env.screenshot()
                    url = self._read_current_url(after)
                if url and self.site_config.is_detail_page(url):
                    logger.info(
                        "  [claude-click] Probe %s opened detail: %s",
                        label,
                        url[:60],
                    )
                    self._last_known_url = url
                    self.dynamic_verifier.record_item_opened(
                        page=self._current_page,
                        item=getattr(self, "_last_click_title", "") or title,
                        url=url,
                    )
                    self._last_extracted = {
                        **self._last_extracted,
                        "last_clicked_title": getattr(self, "_last_click_title", ""),
                        "last_attempted_url": url,
                        "last_attempted_at": time.time(),
                        "last_attempted_step": index,
                    }
                    self._set_scroll_state(context="detail_top", url=url, page_downs=0, wheel_downs=0)
                    self._listings_on_page += 1
                    if hasattr(self, '_last_click_title') and self._last_click_title:
                        self._extracted_titles.append(self._last_click_title)
                    return StepResult(step_index=index, intent=step.intent, success=True,
                                    steps_used=3, duration=12.0)
            except Exception as e:
                logger.warning(f"  [claude-click] Probe {label} failed: {e}")

        logger.warning(f"  [claude-click] Failed verification after retries (url={url[:40]})")
        self.dynamic_verifier.record_item_completed(
            page=self._current_page,
            item=getattr(self, "_last_click_title", "") or title,
            url=url,
            success=False,
            reason="detail_page_not_verified",
        )
        # Mark title as tried so we don't re-attempt the same card
        if hasattr(self, '_last_click_title') and self._last_click_title:
            self._extracted_titles.append(self._last_click_title)
        return StepResult(step_index=index, intent=step.intent, success=False)

    def _execute_claude_guided_filter(self, step: MicroIntent, index: int) -> StepResult:
        """Claude identifies filter element → direct click/type via env.step().

        Holo3 is 0% reliable on sidebar filters (clicks wrong elements).
        Claude reads the screenshot, identifies exact coordinates and action type,
        then we execute directly — no Holo3 involved.

        If not found in current viewport, scrolls down and retries.
        """
        import random

        # Reset sidebar to top before each filter step (scroll persists between steps).
        # Filters are spread across the sidebar: Location near top, Seller Type near bottom.
        try:
            for _ in range(10):
                self.env.step(Action(action_type=ActionType.SCROLL,
                                   params={"direction": "up", "amount": 5,
                                           "x": 150, "y": 400}))
                time.sleep(0.3)
        except Exception:
            pass
        time.sleep(1)

        # Scan sidebar top-to-bottom with small scroll increments.
        # Check each viewport position for the target filter element.
        target = None
        for scroll_attempt in range(8):
            if scroll_attempt > 0:
                # Scroll sidebar down in small increments (3 clicks ≈ ~100px)
                try:
                    self.env.step(Action(action_type=ActionType.SCROLL,
                                       params={"direction": "down", "amount": 3,
                                               "x": 150, "y": 400}))
                    time.sleep(1)
                except Exception:
                    pass

            screenshot = self.env.screenshot()
            target = self.extractor.find_filter_target(screenshot, step.intent)
            self.costs["claude_extract"] += 1

            if target:
                break
            print(f"  [claude-filter] Not found in viewport {scroll_attempt}, scrolling sidebar")

        if not target:
            logger.warning("  [claude-filter] Could not find filter element")
            return StepResult(step_index=index, intent=step.intent, success=False)

        x, y = target["x"], target["y"]
        action = target["action"]
        value = target["value"]
        label = target["label"]

        # Grounding refines coordinates (bounded delta)
        if self.grounding:
            grounding_result = self.grounding.ground(screenshot, label or step.intent, x, y)
            self.costs["claude_grounding"] += 1
            dx = abs(grounding_result.x - x)
            dy = abs(grounding_result.y - y)
            if grounding_result.confidence > 0.5 and dx < 150 and dy < 150:
                x, y = grounding_result.x, grounding_result.y
                logger.info(f"  [grounding] filter refined to ({x},{y}) delta=({dx},{dy})")
            else:
                logger.info(f"  [grounding] filter rejected: delta=({dx},{dy}) conf={grounding_result.confidence}")

        # Human-like delay before interaction
        time.sleep(random.uniform(0.5, 1.5))

        try:
            if action == "click":
                # Simple click — checkbox, radio, toggle
                self.env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                self.costs["gpu_steps"] += 1
                time.sleep(2)  # Wait for filter to apply

            elif action == "type":
                # Click input → clear → type value → Enter
                self.env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                time.sleep(0.5)
                # Triple-click to select all existing text
                self.env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                time.sleep(0.1)
                self.env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                time.sleep(0.3)
                # Select all and delete
                self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "ctrl+a"}))
                time.sleep(0.2)
                self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Delete"}))
                time.sleep(0.3)
                # Type the value
                if value:
                    self.env.step(Action(action_type=ActionType.TYPE, params={"text": value}))
                    time.sleep(0.5)
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Return"}))
                    time.sleep(3)  # Wait for results to update
                self.costs["gpu_steps"] += 1

            elif action == "select":
                # Click dropdown to open → wait → screenshot → find option → click
                self.env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                self.costs["gpu_steps"] += 1
                time.sleep(1.5)

                # Take new screenshot with dropdown open
                dropdown_shot = self.env.screenshot()
                # Ask Claude to find the specific option in the dropdown
                option_target = self.extractor.find_filter_target(
                    dropdown_shot,
                    f"Find and click the option '{value}' in the open dropdown menu"
                )
                self.costs["claude_extract"] += 1

                if option_target:
                    ox, oy = option_target["x"], option_target["y"]
                    time.sleep(random.uniform(0.3, 0.8))
                    self.env.step(Action(action_type=ActionType.CLICK, params={"x": ox, "y": oy}))
                    time.sleep(2)
                else:
                    # Dropdown option not found — close dropdown
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Escape"}))
                    time.sleep(0.5)
                    logger.warning(f"  [claude-filter] Dropdown option '{value}' not found")
                    return StepResult(step_index=index, intent=step.intent, success=False)

            else:
                # Unknown action — fall back to click
                self.env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                self.costs["gpu_steps"] += 1
                time.sleep(2)

        except Exception as e:
            logger.warning(f"  [claude-filter] Action failed: {e}")
            return StepResult(step_index=index, intent=step.intent, success=False)

        logger.info(f"  [claude-filter] {action}@({x},{y}) '{label[:30]}' value='{value[:20]}'")
        self._last_known_url = self._current_results_page_url() or self._results_base_url
        self._set_scroll_state(
            context="results_after_filter",
            url=self._last_known_url,
            page_downs=0,
            wheel_downs=0,
        )
        return StepResult(
            step_index=index, intent=step.intent, success=True,
            steps_used=1, duration=3.0,
        )

    # ── Form-shaped step types (issue #80) ────────────────────────────
    def _execute_claude_guided_form(self, step: MicroIntent, index: int) -> StepResult:
        """Form-shaped dispatch: fill_field, submit, select_option.

        Uses ``ClaudeExtractor.find_form_target`` (single labelled element)
        instead of ``find_all_listings`` (the listings extractor that returns
        zero cards on a login form). Operates on whatever page is currently
        loaded — does not assume listings/results semantics, does not call
        ``_ensure_results_filters``.

        ``MicroIntent.params`` carries the structured payload:
        - fill_field    : {"label", "value"}
        - submit        : {"label"}
        - select_option : {"dropdown_label", "option_label"}

        The runner trusts ``params`` over the prose ``intent`` when both are
        present.
        """
        import random

        params = dict(getattr(step, "params", {}) or {})
        # Brief settle — form pages frequently finish hydrating after the
        # navigate that brought us here.
        time.sleep(2)
        screenshot = self.env.screenshot()

        if step.type == "fill_field":
            label = str(params.get("label") or "").strip()
            value = str(params.get("value") or "").strip()
            aliases = params.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            aliases = [str(a).strip() for a in aliases if str(a).strip()]
            search_intent = (
                f"Click the input field labelled '{label}' so we can type into it"
                if label
                else step.intent
            )
            target = self.extractor.find_form_target(
                screenshot,
                search_intent,
                target_label=label,
                target_value=value,
                target_aliases=aliases,
            )
            self.costs["claude_extract"] += 1
            if not target:
                logger.warning(f"  [claude-form] fill_field: target '{label}' not found")
                return StepResult(step_index=index, intent=step.intent, success=False, data="form_target_not_found")
            x, y = target["x"], target["y"]
            type_value = value or target.get("value") or ""
            try:
                # Click the field, clear any pre-filled value, then type.
                time.sleep(random.uniform(0.3, 0.8))
                self.env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                self.costs["gpu_steps"] += 1
                time.sleep(0.4)
                # Triple-click to select existing text (more reliable than ctrl+a in some inputs)
                self.env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                time.sleep(0.05)
                self.env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                time.sleep(0.2)
                self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "ctrl+a"}))
                time.sleep(0.15)
                self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Delete"}))
                time.sleep(0.2)
                if type_value:
                    self.env.step(Action(action_type=ActionType.TYPE, params={"text": type_value}))
                    time.sleep(0.4)
                logger.info(f"  [claude-form] fill_field '{label[:40]}' = '{type_value[:30]}'")
                return StepResult(
                    step_index=index, intent=step.intent, success=True,
                    steps_used=2, duration=2.0,
                    data=f"fill:{label[:40]}",
                )
            except Exception as e:
                logger.warning(f"  [claude-form] fill_field failed: {e}")
                return StepResult(step_index=index, intent=step.intent, success=False, data=f"fill_error:{e}")

        if step.type == "submit":
            label = str(params.get("label") or "").strip()
            aliases = params.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            aliases = [str(a).strip() for a in aliases if str(a).strip()]
            search_intent = (
                f"Click the '{label}' button to submit the form" if label else step.intent
            )
            # Scroll-and-rescan loop — issue #89 §2. Long forms (CRMs,
            # settings panels) often render the primary submit button below
            # the fold; the previous single-screenshot path declared the
            # button missing without ever checking lower viewports.
            target = self.extractor.find_form_target(
                screenshot, search_intent,
                target_label=label, target_aliases=aliases,
            )
            self.costs["claude_extract"] += 1
            scroll_steps = 0
            max_scrolls = 4
            while target is None and scroll_steps < max_scrolls:
                logger.info(
                    f"  [claude-form] submit '{label}' not in viewport — "
                    f"scrolling Page_Down ({scroll_steps + 1}/{max_scrolls})"
                )
                try:
                    self.env.step(Action(
                        action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"},
                    ))
                except Exception:
                    break
                time.sleep(0.6)
                screenshot = self.env.screenshot()
                target = self.extractor.find_form_target(
                    screenshot, search_intent,
                    target_label=label, target_aliases=aliases,
                )
                self.costs["claude_extract"] += 1
                scroll_steps += 1
            if not target:
                logger.warning(
                    f"  [claude-form] submit: button '{label}' not found "
                    f"after {scroll_steps} scroll(s)"
                )
                # Reset scroll position so the next step starts at top.
                try:
                    self.env.step(Action(
                        action_type=ActionType.KEY_PRESS, params={"keys": "Home"},
                    ))
                except Exception:
                    pass
                return StepResult(step_index=index, intent=step.intent, success=False, data="form_target_not_found")
            x, y = target["x"], target["y"]
            try:
                time.sleep(random.uniform(0.4, 0.9))
                # Snapshot URL before click so the adaptive settle below
                # can detect the moment the page actually navigates.
                url_before = self._best_effort_current_url()
                self._dump_debug_screenshot(
                    f"submit_step{index}_pre_click", screenshot,
                )
                self.env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
                self.costs["gpu_seconds"] += self._adaptive_submit_settle(
                    url_before=url_before,
                )
                self.costs["gpu_steps"] += 1

                # Enter-key fallback: HTML forms whose JS swallows the click
                # event still submit on Return in a focused input (the
                # browser's native form behaviour). When the click + adaptive
                # settle didn't produce navigation, fire Enter and give it
                # a short additional window. Common reason this is needed:
                # the click landed on the right pixel but the button's
                # onclick handler is conditioned on something we can't see
                # from the screenshot (CSRF token, validation state).
                url_after_click = self._best_effort_current_url()
                if url_before and url_after_click == url_before:
                    logger.info(
                        "  [claude-form] click did not navigate — trying "
                        "Enter-key fallback on focused field"
                    )
                    try:
                        self.env.step(Action(
                            action_type=ActionType.KEY_PRESS,
                            params={"keys": "Return"},
                        ))
                    except Exception as enter_exc:  # noqa: BLE001
                        logger.debug("Enter fallback failed: %s", enter_exc)
                    else:
                        self.costs["gpu_seconds"] += self._adaptive_submit_settle(
                            url_before=url_before,
                        )
                        self.costs["gpu_steps"] += 1
                    self._dump_debug_screenshot(
                        f"submit_step{index}_post_enter",
                        self._safe_screenshot(),
                    )
                else:
                    self._dump_debug_screenshot(
                        f"submit_step{index}_post_click",
                        self._safe_screenshot(),
                    )

                logger.info(f"  [claude-form] submit '{label[:40]}'")
                return StepResult(
                    step_index=index, intent=step.intent, success=True,
                    steps_used=1, duration=3.0,
                    data=f"submit:{label[:40]}",
                )
            except Exception as e:
                logger.warning(f"  [claude-form] submit failed: {e}")
                return StepResult(step_index=index, intent=step.intent, success=False, data=f"submit_error:{e}")

        if step.type == "select_option":
            dropdown = str(params.get("dropdown_label") or params.get("label") or "").strip()
            option = str(params.get("option_label") or params.get("value") or "").strip()
            # Phase 1: open the dropdown.
            open_intent = (
                f"Click the '{dropdown}' dropdown to open its option list"
                if dropdown else step.intent
            )
            target = self.extractor.find_form_target(
                screenshot, open_intent, target_label=dropdown,
            )
            self.costs["claude_extract"] += 1
            if not target:
                logger.warning(f"  [claude-form] select_option: dropdown '{dropdown}' not found")
                return StepResult(step_index=index, intent=step.intent, success=False, data="form_target_not_found")
            try:
                time.sleep(random.uniform(0.3, 0.8))
                self.env.step(Action(action_type=ActionType.CLICK, params={"x": target["x"], "y": target["y"]}))
                self.costs["gpu_steps"] += 1
                time.sleep(1.5)  # Allow option list to render
            except Exception as e:
                logger.warning(f"  [claude-form] select_option open failed: {e}")
                return StepResult(step_index=index, intent=step.intent, success=False, data=f"select_open_error:{e}")

            # Phase 2: pick the option. Re-screenshot so Claude sees the open menu.
            opened_shot = self.env.screenshot()
            pick_intent = (
                f"Click the '{option}' option in the open dropdown menu"
                if option else step.intent
            )
            option_target = self.extractor.find_form_target(
                opened_shot, pick_intent, target_label=option,
            )
            self.costs["claude_extract"] += 1
            if not option_target:
                # Close the dropdown to keep the page in a clean state.
                try:
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Escape"}))
                except Exception:
                    pass
                logger.warning(f"  [claude-form] select_option: option '{option}' not found in open menu")
                return StepResult(step_index=index, intent=step.intent, success=False, data="option_not_found")
            try:
                time.sleep(random.uniform(0.2, 0.6))
                self.env.step(Action(action_type=ActionType.CLICK, params={"x": option_target["x"], "y": option_target["y"]}))
                self.costs["gpu_steps"] += 1
                time.sleep(1.5)
                logger.info(f"  [claude-form] select_option '{dropdown[:30]}' = '{option[:30]}'")
                return StepResult(
                    step_index=index, intent=step.intent, success=True,
                    steps_used=2, duration=4.0,
                    data=f"select:{dropdown[:30]}={option[:30]}",
                )
            except Exception as e:
                logger.warning(f"  [claude-form] select_option pick failed: {e}")
                return StepResult(step_index=index, intent=step.intent, success=False, data=f"select_pick_error:{e}")

        # Unknown form type — shouldn't reach here.
        logger.warning(f"  [claude-form] unsupported form step type: {step.type}")
        return StepResult(step_index=index, intent=step.intent, success=False)

    def _execute_paginate_layered(self, step: MicroIntent, index: int) -> StepResult:
        """Layered pagination: URL-based → Claude-guided → Holo3 fallback.

        Layer 1: URL-based — if we can detect the current page URL pattern,
                 construct page N+1 URL and navigate directly. Fastest, no
                 risk of clicking sidebar filters.
        Layer 2: Claude-guided — End key then Page_Up to get pagination bar
                 in view. Claude finds Next button coordinates.
        Layer 3: Holo3 — simple 1-sentence task as last resort.
        """

        # Track current page number
        if not hasattr(self, '_current_page'):
            self._current_page = 1
        current_page = self._current_page

        # ── Layer 1: URL-based pagination ──
        # Use the stored results base URL (from initial navigate), NOT the current page URL
        # (which might be a detail page after extraction)
        base_url = getattr(self, '_results_base_url', '')
        if base_url and self.site_config.pagination_format:
            next_page = self._current_page + 1
            next_url = self.site_config.paginated_url(base_url, next_page)

            # Ensure full URL
            if not next_url.startswith("http"):
                next_url = f"https://www.{next_url}"

            logger.info(f"  [paginate] Layer 1: URL-based → {next_url[:80]}")
            try:
                self.env.reset(task="paginate_url", start_url=next_url)
                time.sleep(10)
                self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
                time.sleep(2)
                self._current_page = next_page
                self._last_known_url = next_url
                self._set_scroll_state(context="results_top", url=next_url, page_downs=0, wheel_downs=0)
                self.dynamic_verifier.record_pagination(
                    page=current_page,
                    success=True,
                    method="url",
                    next_url=next_url,
                )
                self.dynamic_verifier.record_page_start(page=next_page, url=next_url)
                return StepResult(step_index=index, intent=step.intent, success=True,
                                steps_used=0, data=f"url_paginate_page{next_page}")
            except Exception as e:
                logger.warning(f"  [paginate] Layer 1 failed: {e}")
                self.dynamic_verifier.record_pagination(
                    page=current_page,
                    success=False,
                    method="url",
                    next_url=next_url,
                    reason=f"url_navigation_failed:{e}",
                )

        # ── Layer 2: Claude-guided ──
        logger.info("  [paginate] Layer 2: Claude-guided (End → Page_Up)")
        claude_result = self._execute_claude_guided_paginate(step, index)
        if claude_result.success:
            self._current_page += 1
            self._last_known_url = self._current_results_page_url()
            self._set_scroll_state(context="results_top", url=self._last_known_url, page_downs=0, wheel_downs=0)
            self.dynamic_verifier.record_pagination(
                page=current_page,
                success=True,
                method="claude_guided",
                next_url=self._last_known_url,
            )
            self.dynamic_verifier.record_page_start(page=self._current_page, url=self._last_known_url)
            return claude_result

        # ── Layer 3: Holo3 fallback ──
        logger.info("  [paginate] Layer 3: Holo3 fallback")
        # Scroll to a calculated position: End then 2x Page_Up to avoid sidebar
        try:
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(0.5)
            # Scroll down ~80% of the page (past listings, before footer/sidebar bottom)
            for _ in range(6):
                self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                time.sleep(0.5)
        except Exception:
            pass

        holo_result = self._execute_holo3_step(
            MicroIntent(
                intent="Click the Next page button or the next page number.",
                type="paginate",
                budget=8,
                grounding=True,
            ),
            index,
        )
        if holo_result.success:
            self._current_page += 1
            self._last_known_url = self._current_results_page_url()
            self._set_scroll_state(context="results_top", url=self._last_known_url, page_downs=0, wheel_downs=0)
            self.dynamic_verifier.record_pagination(
                page=current_page,
                success=True,
                method="holo3",
                next_url=self._last_known_url,
            )
            self.dynamic_verifier.record_page_start(page=self._current_page, url=self._last_known_url)
        else:
            self.dynamic_verifier.record_pagination(
                page=current_page,
                success=False,
                method="all_layers",
                reason="next_control_not_found",
            )
        return holo_result

    def _execute_claude_guided_paginate(self, step: MicroIntent, index: int) -> StepResult:
        """Claude finds Next button → Holo3 clicks it.

        Scrolls near the bottom, Claude finds pagination, retry on error, bounded grounding.
        """
        # Clear focus traps such as open menus or overlays before repositioning.
        try:
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Escape"}))
            time.sleep(0.5)
        except Exception:
            pass

        # Go to bottom first so the pagination bar is likely on screen.
        try:
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "End"}))
            time.sleep(3)
        except Exception:
            pass

        # Find pagination target with retry. On retry, move slightly up so the
        # pagination bar is not flush with the screen edge or hidden by footer UI.
        target = None
        screenshot = None
        for attempt in range(3):
            if attempt == 1:
                try:
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Up"}))
                    time.sleep(1.5)
                except Exception:
                    pass
            elif attempt == 2:
                try:
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "End"}))
                    time.sleep(2)
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Up"}))
                    time.sleep(1.5)
                except Exception:
                    pass

            screenshot = self.env.screenshot()
            target = self.extractor.find_paginate_target(screenshot)
            self.costs["claude_extract"] += 1

            if isinstance(target, tuple) and len(target) == 3:
                break
            if isinstance(target, tuple) and target[0] == "not_found":
                logger.warning(f"  [claude-paginate] no control visible on attempt {attempt+1}/3")
                continue
            if isinstance(target, tuple) and target[0] == "error":
                logger.warning(f"  [claude-paginate] parse/error on attempt {attempt+1}/3")
                continue

            logger.warning(f"  [claude-paginate] empty response on attempt {attempt+1}/3")

        if not isinstance(target, tuple) or len(target) != 3:
            logger.info("  [claude-paginate] No Next control found after retries")
            return StepResult(step_index=index, intent=step.intent, success=False)

        x, y, label = target

        # Grounding with delta bound
        if self.grounding:
            grounding_result = self.grounding.ground(screenshot, f"pagination control {label or 'Next'}", x, y)
            self.costs["claude_grounding"] += 1
            dx = abs(grounding_result.x - x)
            dy = abs(grounding_result.y - y)
            if grounding_result.confidence > 0.5 and dx < 150 and dy < 150:
                x, y = grounding_result.x, grounding_result.y
                logger.info(f"  [grounding] paginate refined to ({x}, {y}) delta=({dx},{dy})")
            else:
                logger.info(f"  [grounding] paginate rejected: delta=({dx},{dy}) conf={grounding_result.confidence}")

        # Click
        try:
            self.env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
            self.costs["gpu_steps"] += 1
            self.costs["gpu_seconds"] += 4
            self.costs["proxy_mb"] += 5.0
        except Exception as e:
            logger.warning(f"  [claude-paginate] Click failed: {e}")
            return StepResult(step_index=index, intent=step.intent, success=False)

        # Wait for page load, then scroll to top
        time.sleep(8)
        try:
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(2)
        except Exception:
            pass

        logger.info(f"  [claude-paginate] Clicked '{label[:20]}' at ({x}, {y})")
        self._listings_on_page = 0  # Reset for new page
        self._set_scroll_state(context="pagination_clicked", page_downs=0, wheel_downs=0)
        return StepResult(step_index=index, intent=step.intent, success=True, steps_used=1)

    @property
    def _listings_on_page(self):
        """Track listings processed on current page."""
        if not hasattr(self, '_page_listing_count'):
            self._page_listing_count = 0
        return self._page_listing_count

    @_listings_on_page.setter
    def _listings_on_page(self, value):
        self._page_listing_count = value

    def _execute_holo3_step(self, step: MicroIntent, index: int) -> StepResult:
        """Execute a Holo3 micro-intent with fresh GymRunner."""
        runner = GymRunner(
            brain=self.brain,
            env=self.env,
            max_steps=step.budget,
            frames_per_inference=1,
            grounding=self.grounding if step.grounding else None,
            on_step=self.on_step,
        )

        result = runner.run(task=step.intent, task_id=f"step_{index}_{step.type}")

        success = result.success
        self._update_scroll_state_from_trajectory(result, context=f"holo3_{step.type}")
        current_url = getattr(self.env, "current_url", "") or ""
        if current_url:
            self._last_known_url = current_url

        # Post-step verification using Claude (if extractor available)
        if success and step.verify and self.extractor:
            screenshot = self.env.screenshot()
            verify_data = self.extractor.extract(screenshot)
            self.costs["claude_extract"] += 1
            if verify_data and getattr(verify_data, "url", ""):
                self._last_known_url = verify_data.url
            verified = self._check_verify(step.verify, verify_data, screenshot)
            if not verified:
                logger.warning(f"  [verify] Step {index} claimed success but verification FAILED: {step.verify[:60]}")
                success = False

        return StepResult(
            step_index=index,
            intent=step.intent,
            success=success,
            steps_used=result.total_steps,
            duration=result.total_time,
        )

    def _check_verify(self, verify_condition: str, extract_data, screenshot) -> bool:
        """Check if a step's verify condition is met using Claude extraction data.

        Simple heuristic checks based on the verify string and extracted data.
        Falls back to True if check is ambiguous (don't over-block).
        """
        v = verify_condition.lower()
        url = extract_data.url if extract_data else ""

        # Click verification: should be on a detail page, not search results
        if "detail page" in v or "page opens" in v:
            if url and self.site_config.is_detail_page(url):
                return True
            if url and self.site_config.is_results_page(url) and not self.site_config.is_detail_page(url):
                logger.info("  [verify] Still on search page, not detail page")
                return False
            # If no URL extracted, check if page looks different
            if not url:
                return True  # Can't verify, assume OK

        # Filter verification: check heading/URL for filter signals
        if "selected" in v or "highlighted" in v or "filter" in v:
            # Can't easily verify filter state from extraction — pass through
            return True

        # Pagination verification: check for new listings
        if "new listings" in v:
            return True  # Pagination success already checked by runner

        # Default: trust the runner's success signal
        return True

    def _current_item_label(self, data: Any = None) -> str:
        title = (
            self._last_extracted.get("last_clicked_title")
            or getattr(self, "_last_click_title", "")
        )
        if title:
            return str(title)
        if data is not None:
            year = getattr(data, "year", "") or ""
            make = getattr(data, "make", "") or ""
            model = getattr(data, "model", "") or ""
            label = " ".join(part for part in (year, make, model) if part).strip()
            if label:
                return label
            url = getattr(data, "url", "") or ""
            if url:
                return url
        return self._last_extracted.get("last_attempted_url") or "unknown"

    def _execute_claude_step(self, step: MicroIntent, index: int) -> StepResult:
        """Execute a Claude-only step (screenshot → API → data)."""
        if not self.extractor:
            return StepResult(step_index=index, intent=step.intent, success=False)

        screenshot = self.env.screenshot()

        if step.type == "extract_url":
            data = self.extractor.extract(screenshot)
            url = data.url if data else ""

            # Dedup check
            if url and url in self._seen_urls:
                logger.info(f"  [dedup] Already seen: {url[:50]}")
                self.dynamic_verifier.record_item_completed(
                    page=self._current_page,
                    item=self._current_item_label(data),
                    url=url,
                    success=True,
                    reason="duplicate_url_skipped",
                )
                return StepResult(
                    step_index=index, intent=step.intent,
                    success=False, data=f"DUPLICATE|{url}",
                )
            if url:
                self._seen_urls.add(url)
                self._last_known_url = url
                self._last_extracted = {
                    **self._last_extracted,
                    "last_attempted_url": url,
                    "last_attempted_step": index,
                    "last_attempted_at": time.time(),
                }

            return StepResult(
                step_index=index, intent=step.intent,
                success=bool(url), data=f"URL:{url}" if url else "",
            )

        elif step.type == "extract_data":
            data, _actions_used = self._extract_listing_data_deep(screenshot)
            item_label = self._current_item_label(data)
            if data and getattr(data, "url", ""):
                self._last_known_url = data.url
                self._last_extracted = {
                    **self._last_extracted,
                    "last_attempted_url": data.url,
                    "last_attempted_step": index,
                    "last_attempted_at": time.time(),
                }
            if data and data.is_viable():
                summary = data.to_summary()
                self._last_extracted = {
                    **self._last_extracted,
                    "last_completed_url": data.url,
                    "last_completed_key": self._lead_key(summary),
                    "last_completed_summary": summary,
                    "last_completed_has_phone": self._lead_has_phone(summary),
                    "last_completed_step": index,
                    "last_completed_at": time.time(),
                }
                self.dynamic_verifier.record_item_completed(
                    page=self._current_page,
                    item=item_label,
                    url=data.url,
                    success=True,
                    reason="viable_lead",
                )
                return StepResult(
                    step_index=index, intent=step.intent,
                    success=True, data=summary,
                )
            if data and data.dealer_reason():
                reason = data.dealer_reason()
                logger.info("  [extract] Rejected non-private listing: %s", reason)
                self._last_extracted = {
                    **self._last_extracted,
                    "last_rejected_url": data.url,
                    "last_rejected_reason": f"dealer:{reason}",
                    "last_rejected_step": index,
                    "last_rejected_at": time.time(),
                }
                self.dynamic_verifier.record_item_completed(
                    page=self._current_page,
                    item=item_label,
                    url=data.url,
                    success=True,
                    reason=f"rejected_dealer:{reason}",
                )
                return StepResult(
                    step_index=index, intent=step.intent,
                    success=False,
                    data=f"REJECTED_DEALER|{reason}|{data.to_summary()[:160]}",
                )
            if data and data.missing_required_reason():
                reason = data.missing_required_reason()
                logger.info("  [extract] Rejected incomplete lead: %s", reason)
                self._last_extracted = {
                    **self._last_extracted,
                    "last_rejected_url": data.url,
                    "last_rejected_reason": f"incomplete:{reason}",
                    "last_rejected_step": index,
                    "last_rejected_at": time.time(),
                }
                self.dynamic_verifier.record_item_completed(
                    page=self._current_page,
                    item=item_label,
                    url=data.url,
                    success=True,
                    reason=f"rejected_incomplete:{reason}",
                )
                return StepResult(
                    step_index=index, intent=step.intent,
                    success=False,
                    data=f"REJECTED_INCOMPLETE|{reason}|{data.to_summary()[:160]}",
                )
            self.dynamic_verifier.record_item_completed(
                page=self._current_page,
                item=item_label,
                url=getattr(data, "url", "") if data else "",
                success=False,
                reason="extract_data_incomplete",
            )
            return StepResult(
                step_index=index, intent=step.intent,
                success=False, data=data.raw_response[:100] if data else "",
            )

        return StepResult(step_index=index, intent=step.intent, success=False)

    def _extract_listing_data_deep(self, initial_screenshot):
        """Capture top, expanded description, and lower detail viewports.

        Private-seller phones often appear inside seller-written
        descriptions, and those descriptions can be collapsed. This routine is
        the execution-time policy for dynamic pages: inspect each viewport,
        click only safe reveal controls, then ask Claude to extract from the
        complete screenshot set.
        """
        screenshots = []
        labels = []
        controls_clicked = 0
        clicked_keys: set[str] = set()
        max_screenshots = 12
        max_viewports = 6

        def capture(label: str):
            if len(screenshots) >= max_screenshots:
                return None
            try:
                shot = self.env.screenshot()
                screenshots.append(shot)
                labels.append(label)
                return shot
            except Exception as e:
                logger.warning(f"  [deep-extract] screenshot failed: {e}")
                return None

        if initial_screenshot is not None:
            screenshots.append(initial_screenshot)
            labels.append("initial extraction viewport")

        # Start from the top so the final prompt sees title, price, seller card,
        # and any safe contact/phone reveal controls before scanning details.
        try:
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(1.5)
        except Exception:
            pass
        self._set_scroll_state(
            context="detail_extract",
            url=self._last_known_url,
            page_downs=0,
            wheel_downs=0,
            label="top/contact area",
            flush=True,
        )
        top_shot = capture("top/contact area")

        for viewport in range(max_viewports):
            self._set_scroll_state(
                context="detail_extract",
                url=self._last_known_url,
                page_downs=viewport,
                wheel_downs=0,
                viewport_stage=viewport,
                label=f"detail viewport {viewport + 1}",
                flush=True,
            )
            if viewport == 0 and top_shot is not None:
                shot = top_shot
            else:
                shot = capture(f"detail viewport {viewport + 1}")
            if shot is None:
                break

            target = self.extractor.find_listing_content_control(shot)
            self.costs["claude_extract"] += 1

            if target:
                key = (
                    f"{target.get('action', '')}:{target.get('label', '').lower()}:"
                    f"{target['x'] // 25}:{target['y'] // 25}"
                )
                if key not in clicked_keys:
                    clicked_keys.add(key)
                    try:
                        self.env.step(Action(
                            action_type=ActionType.CLICK,
                            params={"x": target["x"], "y": target["y"]},
                        ))
                        controls_clicked += 1
                        time.sleep(2)
                        capture(
                            f"after {target.get('action', 'expand')} "
                            f"{target.get('label', '')[:40]}"
                        )
                        self._set_scroll_state(
                            context="detail_extract",
                            url=self._last_known_url,
                            page_downs=viewport,
                            wheel_downs=0,
                            viewport_stage=viewport,
                            label=(
                                f"after {target.get('action', 'expand')} "
                                f"{target.get('label', '')[:40]}"
                            ),
                            flush=True,
                        )
                        logger.info(
                            "  [deep-extract] clicked %s '%s'",
                            target.get("action", ""),
                            target.get("label", "")[:60],
                        )
                    except Exception as e:
                        logger.warning(f"  [deep-extract] reveal click failed: {e}")

            if viewport < max_viewports - 1:
                try:
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                    time.sleep(1)
                except Exception:
                    break

        data = self.extractor.extract_multi(screenshots, labels=labels)
        self._set_scroll_state(
            context="detail_extract_complete",
            url=self._last_known_url,
            page_downs=max(0, min(len(labels), max_viewports) - 1),
            wheel_downs=0,
            viewport_stage=max(0, min(len(labels), max_viewports) - 1),
            label=f"captured {len(labels)} screenshots, controls_clicked={controls_clicked}",
            flush=True,
        )
        if data and data.is_viable():
            return data, controls_clicked

        # Fallback to legacy single-screenshot extraction if the multi-shot JSON
        # parse fails or somehow loses the core listing identity.
        fallback_shot = screenshots[-1] if screenshots else initial_screenshot
        if fallback_shot is not None:
            fallback = self.extractor.extract(fallback_shot)
            self.costs["claude_extract"] += 1
            return fallback, controls_clicked

        return data, controls_clicked

    def _reverse_step(self, step: MicroIntent):
        """Undo a failed step. Uses the pre-step snapshot from #121 step 1
        to skip undo actions when the diff shows nothing meaningful
        actually changed — preserving partial form-fill progress instead
        of blanket-firing Escape + Alt+Left.

        Decision tree:
          • No snapshot available (legacy callers / object.__new__):
              fall back to the static REVERSE_ACTIONS map exactly as
              before. Same behavior as pre-#121.
          • URL did not change AND nothing else meaningful changed:
              skip reverse entirely. Step "failed" but the page state
              is identical to what we entered with — there is nothing
              to undo.
          • URL did not change AND focus changed AND no scroll/extract
              progress: light-touch Escape only. Likely a stuck modal
              or autocomplete dropdown — Alt+Left would dismiss the
              underlying form.
          • URL changed: keep the legacy plan (Escape then Alt+Left)
              because we need to navigate back.
          • Otherwise: legacy plan.
        """
        actions = self._plan_aware_reverse_actions(step)
        for action_type, keys in actions:
            try:
                self.env.step(Action(
                    action_type=ActionType.KEY_PRESS,
                    params={"keys": keys},
                ))
                time.sleep(0.5)
            except Exception:
                pass
        logger.info(f"  [reverse] {len(actions)} actions applied")

    def _plan_aware_reverse_actions(
        self, step: MicroIntent
    ) -> list[tuple[str, str]]:
        """Decide which keystrokes to fire based on the pre/post-step diff.

        Pure helper — easy to unit test. The runner calls this and then
        executes whatever it returns.
        """
        # #121 step 2: use the diff when available to skip unneeded work.
        pre = self._pre_step_snapshot
        if pre is not None:
            try:
                post = step_snapshot.capture(self)
                delta = step_snapshot.diff(pre, post)
            except Exception as exc:  # noqa: BLE001 — telemetry never breaks runs
                logger.debug("plan-aware reverse diff failed: %s", exc)
                delta = None
            if delta is not None:
                decision = self._reverse_decision_from_diff(step, delta)
                if decision is not None:
                    logger.info(
                        "  [reverse:plan-aware] %s — diff: %s",
                        "skip" if not decision else f"{len(decision)} action(s)",
                        delta.summary(),
                    )
                    return decision

        # No snapshot or diff couldn't be computed — fall back to the
        # pre-#121 static map. Identical behavior for legacy callers.
        actions = list(REVERSE_ACTIONS.get(step.type, []))
        if step.reverse:
            if "escape" in step.reverse.lower():
                actions = [("key_press", "Escape")] + actions
            if "alt+left" in step.reverse.lower():
                actions.append(("key_press", "alt+Left"))
        return actions

    @staticmethod
    def _reverse_decision_from_diff(
        step: MicroIntent,
        delta: "step_snapshot.StepDiff",
    ) -> list[tuple[str, str]] | None:
        """Translate a step diff into reverse keystrokes, or None to mean
        "no plan-aware decision applicable — use the legacy fallback".

        Returning ``[]`` (empty list) means "skip reverse entirely" —
        nothing visibly changed so there's nothing to undo.
        """
        # Type-specific overrides go first. The runner's own ``step.reverse``
        # hint is preserved by the legacy fallback path; here we only
        # short-circuit when the diff is informative enough to act safely.
        if delta.url_changed:
            # Navigation happened (intentionally or not). Going back is
            # the safe undo regardless of step type.
            return [("key_press", "alt+Left")]

        if delta.extraction_added or delta.new_urls_seen:
            # Forward progress was made even though the step was marked
            # "failed". Reverting would destroy the work; skip.
            return []

        if not delta.has_changes:
            # Identical state pre/post — nothing happened. No-op skip.
            return []

        if delta.focus_changed and not (
            delta.scroll_changed or delta.viewport_changed or delta.page_changed
        ):
            # Likely a stuck modal / autocomplete / focused-but-empty input.
            # Escape dismisses without dropping form context.
            return [("key_press", "Escape")]

        # Diff was non-trivial but doesn't match a clear pattern — let
        # the legacy fallback handle it.
        return None
