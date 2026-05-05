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
from .listings_scanner import ListingsScanner
from .run_reporter import RunReporter
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
from .tool_channel import ToolChannel

from ..plan_decomposer import MicroIntent, MicroPlan
from ..site_config import SiteConfig
from ..verification.dynamic_plan_verifier import DynamicPlanVerifier

if TYPE_CHECKING:
    from ..extraction import ClaudeExtractor
    from .step_context import StepContext

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
        extraction_cache: Any = None,        # ExtractionCache | None — see extraction.cache
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
        # Per-request extraction cache. None = legacy behavior (no cache).
        # When set, the extract_data branch consults it BEFORE the deep-extract
        # Claude call to short-circuit on previously-seen URLs.
        self.extraction_cache = extraction_cache
        # All listings-scan state lives on a single dataclass so the
        # eight related fields (seen URLs, viewport stage, cached cards,
        # results base URL, required filter tokens) can be tested,
        # persisted, and eventually have their mutation logic extracted
        # together (EPIC #161 Phase 4). Property delegates below preserve
        # the legacy ``self._seen_urls`` / ``self._viewport_stage`` /
        # … access pattern for the 70+ internal call sites and external
        # readers (CheckpointManager, BrowserState, tests).
        self.scanner = ListingsScanner()
        # Pre-seed seen-URLs with cache contents so the deep-extract dedup
        # short-circuits on previously-cached URLs even when cache_read is
        # off and only cache_write is on (rare but coherent: warm cache for
        # later runs without using existing entries this run).
        if extraction_cache is not None and extraction_cache.read_enabled:
            for url in extraction_cache.known_urls():
                self.scanner.seen_urls.add(url)
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

        # #161 Phase 2: per-type step handler registry. Dispatch in
        # ``_execute_step`` consults this first; types without a handler
        # fall through to the legacy if/elif. Handlers are migrated one
        # at a time as separate PRs land.
        from .step_handlers import default_registry as _default_registry
        self._handler_registry = _default_registry(self)

        # #161 Phase 1.3: failure-recovery dispatch lifted off run() into
        # StepRecoveryPolicy.handle_failure. The runner's else-branch is
        # now a thin "ask the policy, persist, break-or-continue" wrapper.
        from .step_recovery import StepRecoveryPolicy
        self._recovery_policy = StepRecoveryPolicy(self)

        # #161 Phase 3: while-loop body lifted off run() into
        # RunExecutor.execute. The runner's run() collapses to:
        #   build state → drive executor → run reporter → return
        from .run_executor import RunExecutor
        self._executor = RunExecutor(self)

    # ── Listings-scan state — delegates to ``self.scanner`` (#161 Phase 1.2) ──
    # 70+ internal call sites, external readers (checkpoint_manager,
    # browser_state), and existing test fixtures (test_plan_aware_reverse,
    # test_checkpoint_manager) keep using the legacy underscore-prefixed
    # attribute names. Phase 4 will extract behaviour into ListingsScanner
    # methods and move callers off these property shims.
    #
    # Tests construct runners via ``MicroPlanRunner.__new__(MicroPlanRunner)``
    # which skips ``__init__``, so the scanner may not exist yet on attribute
    # access. ``_ensure_scanner`` lazily creates one — same defaults as the
    # __init__ path so a partially-constructed runner behaves identically.
    def _ensure_scanner(self) -> ListingsScanner:
        scanner = self.__dict__.get("scanner")
        if scanner is None:
            scanner = ListingsScanner()
            self.__dict__["scanner"] = scanner
        return scanner

    @property
    def _seen_urls(self) -> set[str]:
        return self._ensure_scanner().seen_urls

    @_seen_urls.setter
    def _seen_urls(self, value: set[str]) -> None:
        self._ensure_scanner().seen_urls = value

    @property
    def _extracted_titles(self) -> list[str]:
        return self._ensure_scanner().extracted_titles

    @_extracted_titles.setter
    def _extracted_titles(self, value: list[str]) -> None:
        self._ensure_scanner().extracted_titles = value

    @property
    def _page_listings(self) -> list[tuple[int, int, str]]:
        return self._ensure_scanner().page_listings

    @_page_listings.setter
    def _page_listings(self, value: list[tuple[int, int, str]]) -> None:
        self._ensure_scanner().page_listings = value

    @property
    def _page_listing_index(self) -> int:
        return self._ensure_scanner().page_listing_index

    @_page_listing_index.setter
    def _page_listing_index(self, value: int) -> None:
        self._ensure_scanner().page_listing_index = value

    @property
    def _viewport_stage(self) -> int:
        return self._ensure_scanner().viewport_stage

    @_viewport_stage.setter
    def _viewport_stage(self, value: int) -> None:
        self._ensure_scanner().viewport_stage = value

    @property
    def _max_viewport_stages(self) -> int:
        return self._ensure_scanner().max_viewport_stages

    @_max_viewport_stages.setter
    def _max_viewport_stages(self, value: int) -> None:
        self._ensure_scanner().max_viewport_stages = value

    @property
    def _results_base_url(self) -> str:
        return self._ensure_scanner().results_base_url

    @_results_base_url.setter
    def _results_base_url(self, value: str) -> None:
        self._ensure_scanner().results_base_url = value

    @property
    def _required_filter_tokens(self) -> tuple[str, ...]:
        return self._ensure_scanner().required_filter_tokens

    @_required_filter_tokens.setter
    def _required_filter_tokens(self, value: tuple[str, ...]) -> None:
        self._ensure_scanner().required_filter_tokens = value

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
        elapsed = time.time() - self._run_start
        print(RunReporter.step_progress_line(
            step_index=step_result.step_index,
            success=step_result.success,
            results=results,
            gpu_cost=gpu_cost,
            claude_cost=claude_cost,
            proxy_cost=proxy_cost,
            total_cost=total_cost,
            elapsed_seconds=elapsed,
        ))
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

        Phase 3 of EPIC #161: the while-loop body and its surrounding
        init / resume / finalize plumbing live in
        :class:`~.run_executor.RunExecutor`. This method is now a thin
        wrapper: build state, drive the executor, run the reporter,
        return.

        Args:
            plan: Decomposed plan with ordered micro-intents.
            resume: If True, load checkpoint and resume from last step.

        Returns:
            List of StepResult for each executed step.
        """
        from .run_executor import RunState

        if not self.plan_signature:
            self.plan_signature = self._compute_plan_signature(plan)
        state = RunState.fresh(
            run_key=self.run_key,
            session_name=self.session_name,
            plan_signature=self.plan_signature,
        )
        self._executor.execute(plan, state, resume=resume)
        self._final_summary(state.results)
        return state.results

    def _final_summary(self, results: list[StepResult]) -> None:
        """Print the MICRO-PLAN COMPLETE block and stash _final_costs.

        Called once per run, right before ``run`` returns. Kept on the
        runner (rather than the executor) because callers like
        ``run_with_status`` and the result-builder expect ``_final_costs``
        as a runner attribute.
        """
        gpu_cost, claude_cost, proxy_cost, total_cost = self._cost_totals()
        elapsed = time.time() - self._run_start

        # Final summary block — leading newline preserves pre-refactor output.
        print()
        for line in RunReporter.final_summary_lines(
            results=results,
            gpu_cost=gpu_cost,
            claude_cost=claude_cost,
            proxy_cost=proxy_cost,
            total_cost=total_cost,
            elapsed_seconds=elapsed,
            gpu_steps=int(self.costs.get("gpu_steps", 0)),
            claude_extract_calls=int(self.costs.get("claude_extract", 0)),
            claude_grounding_calls=int(self.costs.get("claude_grounding", 0)),
            proxy_mb=float(self.costs.get("proxy_mb", 0.0)),
        ):
            print(line)

        # Attach costs to results for saving (consumed by build_micro_result).
        self._final_costs = RunReporter.final_costs_dict(
            results=results,
            gpu_cost=gpu_cost,
            claude_cost=claude_cost,
            proxy_cost=proxy_cost,
            total_cost=total_cost,
            final_status=self._final_status,
            checkpoint_path=self.checkpoint_path,
        )

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

    def _build_step_context(self, index: int) -> "StepContext":
        """Build a :class:`~.step_context.StepContext` for the next step.

        Handlers are pure functions of (step, ctx); the runner is the
        only thing that knows the current step index. We tuck it into
        ``ctx.state["index"]`` so the handler signature stays
        ``execute(step, ctx) -> StepResult`` per the protocol.
        """
        from .step_context import StepContext
        return StepContext(
            env=self.env,
            brain=self.brain,
            extractor=self.extractor,
            grounding=self.grounding,
            cost_meter=self.cost_meter,
            dynamic_verifier=self.dynamic_verifier,
            scanner=self.scanner,
            site_config=self.site_config,
            tool_channel=self.tool_channel,
            extraction_cache=self.extraction_cache,
            state={"index": index},
        )

    def _execute_step(self, step: MicroIntent, index: int) -> StepResult:
        """Execute a single micro-intent — registry-first dispatch.

        EPIC #161 Phase 2 cleanup: every per-type handler (Navigate,
        Click, Form, ClaudeStep, Paginate, Filter, Holo3) is registered
        in ``self._handler_registry`` and reachable by step type. The
        if/elif fan-out collapsed to four explicit special cases:

        1. **Click layout router** — type="click" branches to listings
           click (ClaudeGuidedClickHandler) or single-element click
           (FormHandler with a synthesised submit MicroIntent) based
           on ``step.hints["layout"]`` and ``step.section``.
        2. **Gate verify** — ``step.gate=True`` runs Claude's
           verify_gate inline (10 LOC, not worth a handler module).
        3. **claude_only flag** — promotes any step type to
           ClaudeStepHandler regardless of step.type.
        4. **navigate_back close-tab** — when a previous click opened
           a detail in a new tab, navigate_back closes the tab instead
           of going through history.

        Pre-settle sleeps that used to live in this method's branches
        have been moved into each handler's ``execute()`` so the timing
        is identical.
        """
        registry = self._handler_registry

        # ── Special case: click layout router ────────────────────────────
        # Listings click vs single-element click can't both bind to "click"
        # in the registry, so the runner keeps the layout decision. Both
        # branches end at a registered handler.
        if step.type == "click" and self.extractor:
            layout_hint = (step.hints or {}).get("layout", "")
            is_listings = (
                layout_hint == "listings"
                or (not layout_hint and step.section == "extraction")
            )
            ctx = self._build_step_context(index)
            if is_listings:
                if not self._ensure_results_filters(index):
                    return StepResult(
                        step_index=index, intent=step.intent, success=False,
                        data="filters_not_applied",
                    )
                return registry.get("click").execute(step, ctx)
            # Single-element click — synthesise a submit MicroIntent and
            # dispatch to the form handler (same code path the form-shaped
            # types use natively).
            synthesised = MicroIntent(
                intent=step.intent, type="submit",
                budget=step.budget, section=step.section,
                required=step.required,
                params={"label": (step.params or {}).get("label", "")},
            )
            return registry.get("submit").execute(synthesised, ctx)

        # ── Special case: gate verify ────────────────────────────────────
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

        # ── Special case: claude_only flag promotes to ClaudeStepHandler ──
        if step.claude_only:
            ctx = self._build_step_context(index)
            return registry.get("extract_url").execute(step, ctx)

        # ── Paginate guard: ensure_results_filters before handler ────────
        if step.type == "paginate":
            if not self._ensure_results_filters(index):
                return StepResult(
                    step_index=index, intent=step.intent, success=False,
                    data="filters_not_applied",
                )
            ctx = self._build_step_context(index)
            return registry.get("paginate").execute(step, ctx)

        # ── Special case: navigate_back close-tab ─────────────────────────
        if step.type == "navigate_back" and self._opened_detail_in_new_tab:
            return self._execute_close_detail_tab(step, index)

        # ── Registry-first for the remaining types ────────────────────────
        handler = registry.get(step.type)
        if handler is not None:
            ctx = self._build_step_context(index)
            return handler.execute(step, ctx)

        # ── Fallback: unknown step type → Holo3 ──────────────────────────
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
        """Backwards-compat shim — delegates to :class:`NavigateHandler`.

        Kept on the runner so external callers (tests, host integrations)
        that invoke ``runner._execute_navigate(step, index)`` directly
        continue to work. Phase 2 cleanup PR will remove this once all
        callers go through the registry.
        """
        handler = self._handler_registry.get("navigate")
        if handler is None:
            # Registry didn't include navigate — shouldn't happen in production
            # but keep a safe fallback for tests that strip the registry.
            from .step_handlers.navigate import NavigateHandler
            handler = NavigateHandler(self)
        ctx = self._build_step_context(index)
        return handler.execute(step, ctx)

    def _execute_claude_guided_click(self, step: MicroIntent, index: int) -> StepResult:
        """Backwards-compat shim — delegates to :class:`ClaudeGuidedClickHandler`.

        Phase 2 of EPIC #161: the implementation lives in
        ``step_handlers/click.py`` so it can be unit-tested with a
        mocked StepContext (no Xvfb, no real GymRunner). The dispatch in
        ``_execute_step`` keeps its layout-hint branching that decides
        between listings click (this handler) and single-element form
        click (``_execute_claude_guided_form``); when FormHandler also
        lands, both bind to the "click" type and the dispatch logic
        moves into a small router.
        """
        from .step_handlers.click import ClaudeGuidedClickHandler
        handler = ClaudeGuidedClickHandler(self)
        ctx = self._build_step_context(index)
        return handler.execute(step, ctx)

    def _execute_claude_guided_filter(self, step: MicroIntent, index: int) -> StepResult:
        """Backwards-compat shim — delegates to :class:`ClaudeGuidedFilterHandler`.

        Phase 2 of EPIC #161: sidebar filter dispatch lives in
        ``step_handlers/filter.py`` so it's unit-testable in isolation.
        """
        from .step_handlers.filter import ClaudeGuidedFilterHandler
        handler = ClaudeGuidedFilterHandler(self)
        ctx = self._build_step_context(index)
        return handler.execute(step, ctx)

    # ── Form-shaped step types (issue #80) ────────────────────────────
    def _execute_claude_guided_form(self, step: MicroIntent, index: int) -> StepResult:
        """Backwards-compat shim — delegates to :class:`ClaudeGuidedFormHandler`.

        The implementation lives in ``step_handlers/form.py`` so it can
        be unit-tested with a mocked StepContext. The dispatch in
        ``_execute_step`` keeps its layout-hint branching for the
        synthesised "click → submit" path; native fill_field / submit /
        select_option steps go through the registry directly.
        """
        from .step_handlers.form import ClaudeGuidedFormHandler
        handler = ClaudeGuidedFormHandler(self)
        ctx = self._build_step_context(index)
        return handler.execute(step, ctx)

    def _execute_paginate_layered(self, step: MicroIntent, index: int) -> StepResult:
        """Backwards-compat shim — delegates to :class:`PaginateHandler`.

        Phase 2 of EPIC #161: layered pagination (URL-based →
        Claude-guided → Holo3 fallback) lives in
        ``step_handlers/paginate.py`` so it's unit-testable in
        isolation. The runner shim stays so external callers (test
        fixtures, host integrations) keep working unchanged.
        """
        from .step_handlers.paginate import PaginateHandler
        handler = PaginateHandler(self)
        ctx = self._build_step_context(index)
        return handler.execute(step, ctx)

    def _execute_claude_guided_paginate(self, step: MicroIntent, index: int) -> StepResult:
        """Backwards-compat shim — delegates to :meth:`PaginateHandler._claude_guided_paginate`.

        Layer 2 of the layered pagination strategy. Some tests
        previously exercised this directly; the shim keeps that path
        working while the body is unit-testable on the handler.
        """
        from .step_handlers.paginate import PaginateHandler
        handler = PaginateHandler(self)
        ctx = self._build_step_context(index)
        return handler._claude_guided_paginate(step, ctx, index)

    @property
    def _listings_on_page(self):
        """Track listings processed on current page.

        Phase 4 of EPIC #161: storage lives on the scanner now
        (``scanner.listings_attempted``). Property kept for the 6+
        runner / handler call sites that read/write it via
        ``self._listings_on_page``; the canonical state owner is the
        scanner.
        """
        return self._ensure_scanner().listings_attempted

    @_listings_on_page.setter
    def _listings_on_page(self, value):
        self._ensure_scanner().listings_attempted = value

    def _execute_holo3_step(self, step: MicroIntent, index: int) -> StepResult:
        """Backwards-compat shim — delegates to :class:`Holo3StepHandler`.

        Phase 2 of EPIC #161: Holo3 micro-intent execution (fresh
        GymRunner + post-step Claude verify) lives in
        ``step_handlers/holo3.py`` so it's unit-testable in isolation.
        ``PaginateHandler`` Layer 3 calls this shim until the cleanup
        PR registers Holo3StepHandler in the registry directly.
        """
        from .step_handlers.holo3 import Holo3StepHandler
        handler = Holo3StepHandler(self)
        ctx = self._build_step_context(index)
        return handler.execute(step, ctx)

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
        """Backwards-compat shim — delegates to :class:`ClaudeStepHandler`.

        Phase 2 of EPIC #161: extract_url + extract_data + the
        deep-extract subroutine all live in ``step_handlers/claude_step.py``
        so they can be unit-tested with a mocked StepContext (no Xvfb,
        no real GymRunner). The dispatch in ``_execute_step`` continues
        to consult ``step.claude_only`` rather than the registry for
        these types; the registry swap happens in the cleanup PR after
        every step type has migrated.
        """
        from .step_handlers.claude_step import ClaudeStepHandler
        handler = ClaudeStepHandler(self)
        ctx = self._build_step_context(index)
        return handler.execute(step, ctx)

    def _extract_listing_data_deep(self, initial_screenshot):
        """Backwards-compat shim — multi-viewport deep-extract.

        Implementation lives on :class:`ClaudeStepHandler`. Some
        external callers (test fixtures, host integrations) still call
        ``runner._extract_listing_data_deep`` directly; this shim keeps
        them working while the body is unit-testable in isolation.
        """
        from .step_handlers.claude_step import ClaudeStepHandler
        handler = ClaudeStepHandler(self)
        # The deep-extract subroutine doesn't carry a step_index; pass 0.
        ctx = self._build_step_context(0)
        return handler._extract_listing_data_deep(initial_screenshot, ctx)

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
