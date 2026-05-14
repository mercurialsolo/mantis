"""MicroPlanRunner — execute decomposed micro-intents.

EPIC #161 final: this module is a thin coordinator. The orchestration
loop lives in :class:`~.run_executor.RunExecutor`; per-step strategies
in :mod:`~.step_handlers`; failure recovery in
:class:`~.step_recovery.StepRecoveryPolicy`; listings state in
:class:`~.listings_scanner.ListingsScanner`; persistence in
:class:`~.checkpoint_manager.CheckpointManager`; cost in
:class:`~.cost_meter.CostMeter`; URL/scroll in
:class:`~.browser_state.BrowserState`. Pure helpers live in
:mod:`._runner_helpers`. The runner wires them together and exposes
the public API (``run``, ``run_with_status``, ``resume``).
"""

from __future__ import annotations

import functools
import logging
import random
import time  # noqa: F401 — re-exported for tests that monkeypatch micro_runner.time.sleep
from typing import TYPE_CHECKING, Any

from ..cost_config import CostConfig
from ..plan_decomposer import MicroPlan
from ..site_config import SiteConfig
from ..verification.dynamic_plan_verifier import DynamicPlanVerifier
from . import _runner_helpers as _h
from .browser_state import BrowserState
from .checkpoint import (
    REVERSE_ACTIONS, PauseRequested, PauseState, RunCheckpoint,
    RunnerResult, StepResult, _PauseRequested,
)
from .checkpoint_manager import CheckpointManager
from .cost_meter import CostMeter
from .listing_dedup import ListingDedup
from .listings_scanner import ListingsScanner
from .tool_channel import ToolChannel

if TYPE_CHECKING:
    from ..extraction import ClaudeExtractor

logger = logging.getLogger(__name__)

__all__ = [
    "MicroPlanRunner", "REVERSE_ACTIONS", "PauseRequested", "PauseState",
    "RunCheckpoint", "RunnerResult", "StepResult", "_PauseRequested",
]


class MicroPlanRunner:
    """Execute a MicroPlan step-by-step with verify/reverse/checkpoint."""

    # ── Settle constants (also live on _runner_helpers — kept here for tests) ─
    _SUBMIT_SETTLE_MAX_SECONDS = _h._SUBMIT_SETTLE_MAX_SECONDS
    _SUBMIT_SETTLE_POLL_SECONDS = _h._SUBMIT_SETTLE_POLL_SECONDS
    _SUBMIT_SETTLE_MIN_SECONDS = _h._SUBMIT_SETTLE_MIN_SECONDS

    # ── Static back-compat shims (callers use both Class.x and runner.x) ──
    _compute_plan_signature = staticmethod(CheckpointManager.compute_plan_signature)
    _unique_leads_from_results = staticmethod(ListingDedup.unique_leads_from_results)
    _successful_lead_data = staticmethod(ListingDedup.successful_lead_data)
    _lead_key = staticmethod(ListingDedup.lead_key)
    _lead_has_phone = staticmethod(ListingDedup.lead_has_phone)
    _lead_counts = staticmethod(ListingDedup.lead_counts)
    _extract_url_from_intent = staticmethod(_h.extract_url_from_intent)
    _derive_filter_tokens = staticmethod(_h.derive_filter_tokens)
    _reverse_decision_from_diff = staticmethod(_h.reverse_decision_from_diff)

    def __init__(
        self, brain: Any, env: Any, grounding: Any = None,
        extractor: ClaudeExtractor | None = None, on_step: Any = None,
        max_retries: int = 2,
        checkpoint_path: str = "/data/checkpoints/micro_run.json",
        run_key: str = "", session_name: str = "", plan_signature: str = "",
        resume_state: bool = False, on_checkpoint: Any = None,
        dynamic_verifier: DynamicPlanVerifier | None = None,
        max_cost: float = 10.0, max_time_minutes: int = 180,
        site_config: SiteConfig | None = None, step_callback: Any = None,
        keep_screenshots: int | None = None, cancel_event: Any = None,
        cost_config: CostConfig | None = None, tenant_id: str = "",
        extraction_cache: Any = None,
        navigation_primitives_emit_skip: set[str] | None = None,
        context_budget: Any = None,  # ContextBudget | None — typed in body to avoid import cycle
        seen_url_predicate: Any = None,  # Callable[[str], bool] | None
        routing_policy: Any = None,  # RoutingPolicy | None — typed in body to avoid import cycle
        seed: int | None = None,
    ):
        # Seed the global RNG so per-action human_speed delays
        # (random.uniform / random.randint in playwright_env.py +
        # xdotool_env.py + step_handlers/*) are reproducible across
        # runs of the same plan. ``None`` preserves the previous
        # non-deterministic behavior.
        if seed is not None:
            random.seed(seed)
        self.seed = seed
        self.brain, self.env, self.grounding, self.extractor = (
            brain, env, grounding, extractor
        )
        # Issue #250: opt-in set of step types whose terminal halt
        # should re-stamp the last StepResult with ``skip=True,
        # skip_reason='navigation_failed'``. Mirror of issue #246's
        # recipe-rejection skip envelope, but triggered runner-side
        # on navigation primitives (click / submit / scroll / navigate
        # / gate). Default ``None`` preserves today's behavior — host
        # gets ``status='halted'`` as before. Recipe rejections that
        # already populated ``skip_reason`` (#246) win over this
        # generic stamp; the executor checks ``last_result.skip``
        # before re-stamping.
        self.navigation_primitives_emit_skip: set[str] | None = (
            set(navigation_primitives_emit_skip)
            if navigation_primitives_emit_skip is not None
            else None
        )
        # Issue #254: per-context sub-goal budget. Opt-in. When set,
        # the runner tracks how many ``run()`` calls have executed
        # against each URL anchor; an over-budget run short-circuits
        # with a synthetic StepResult carrying ``skip=True,
        # skip_reason='listing_budget_exceeded'``. Default ``None``
        # preserves today's unbounded behavior.
        self.context_budget = context_budget
        self._sub_goal_count_by_url: dict[str, int] = {}
        self._sub_goal_count_total: int = 0
        # Issue #255: host-injected predicate for cross-session
        # dedup. When set and the runner is about to deep-extract a
        # detail-page URL, the predicate is consulted; if it returns
        # True the runner short-circuits with
        # ``skip_reason='already_seen'`` without invoking
        # ClaudeExtractor. Mantis owns the timing window
        # (post-navigate / pre-extract); the host owns the dedup
        # policy (URL set, content hash, CRM lookup, …) entirely.
        # Default ``None`` preserves today's behavior.
        self.seen_url_predicate = seen_url_predicate
        # #300 follow-up: dispatch policy for unstructured CLICK actions
        # in step handlers. Default ``None`` resolves to
        # :meth:`RoutingPolicy.from_env`, which reads
        # ``MANTIS_ROUTE_SOM_CLICKS`` so a deploy can flip the policy
        # without a code change. Tests / explicit callers pass an
        # instance to bypass the env override.
        if routing_policy is None:
            from .runner import RoutingPolicy
            routing_policy = RoutingPolicy.from_env()
        self.routing_policy = routing_policy
        self.on_step, self.max_retries = on_step, max_retries
        self.checkpoint_path, self.run_key = checkpoint_path, run_key
        self.session_name, self.plan_signature = session_name, plan_signature
        self.resume_state, self.on_checkpoint = resume_state, on_checkpoint
        self.dynamic_verifier = (
            dynamic_verifier or DynamicPlanVerifier(plan_name=session_name)
        )
        self.site_config = site_config or SiteConfig.default_boattrader()
        self.extraction_cache = extraction_cache
        self.scanner = ListingsScanner()
        if extraction_cache is not None and extraction_cache.read_enabled:
            for url in extraction_cache.known_urls():
                self.scanner.seen_urls.add(url)
        self._current_page, self._last_known_url = 1, ""
        self._scroll_state, self._last_extracted = {}, {}
        self._opened_detail_in_new_tab = False
        self._active_checkpoint_context = None
        self._pre_step_snapshot, self._final_status = None, "running"
        # Stash for the SPA-aware submit demotion check. The form
        # handler's submit branch sets this just before clicking the
        # submit button; ``run_executor._maybe_demote_form_no_change``
        # consumes it on the very next step then clears it.
        self._last_submit_pre_screenshot: Any = None
        # Stash for the most recently clicked submit target (x, y,
        # label). The submit handler writes it just before clicking;
        # ``_maybe_demote_form_no_change`` reads it when recording a
        # demotion to ``_step_failure_history``. Cleared after the
        # demote check.
        self._last_submit_target: dict[str, Any] | None = None
        # Per-step failure history — issue #224 follow-up: when a
        # required step fails repeatedly on the same broken click,
        # the form handler reads this on retry and tells
        # ``find_form_target`` to AVOID the previously-tried targets.
        # Keyed by step_index so multiple steps can independently
        # accumulate their own histories without cross-talk. Cleared
        # by the executor on step success (so a successful retry
        # doesn't bleed warnings into the next step).
        self._step_failure_history: dict[int, list[dict[str, Any]]] = {}
        # Per-step handler-routing override — set by the executor when
        # a step has accumulated enough same-kind failures that the
        # default handler is clearly the wrong tool. Currently the only
        # value is ``"brain_grounded_click"`` which routes a submit /
        # fill step through the click handler with brain grounding
        # (Holo3) instead of Claude text-matching. General-purpose:
        # triggered purely by observed failure pattern, not plan
        # content. Cleared on step success.
        self._step_handler_override: dict[int, str] = {}
        # Per-step intent override — set by the IntentRewriter when a
        # step fails with a class that suggests the intent itself is
        # the problem (epic #377 Phase B). Holds the rewritten intent
        # string for the NEXT retry attempt; consumed by
        # ``_build_effective_step``. Cleared on step success.
        # ``_step_rewrite_attempts`` is the per-step budget tracker
        # (default 1 rewrite per step per run); the rewriter consults
        # it via ``intent_rewriter.should_attempt_rewrite``.
        self._step_intent_overrides: dict[int, str] = {}
        self._step_rewrite_attempts: dict[int, int] = {}
        # Per-run audit log of self-healing actions (epic #377 Phase C —
        # ``healing_events.record_*`` helpers append here; surfaced in
        # ``result.json`` via ``build_micro_result``). Each entry is a
        # dict with ``kind`` / ``step_index`` / ``source`` / ``at`` plus
        # kind-specific fields. Empty list on plans that didn't need
        # any healing.
        self._healing_events: list[dict] = []
        # Agentic-recovery state (issue #224 follow-up). When the
        # recovery loop returns ``mode=add_hint``, the hint is
        # appended to ``_recovery_hints[step_index]`` and surfaced in
        # the next attempt's search prompt. Budget tracking (per-step
        # + per-run) prevents infinite recovery loops. Cleared on
        # step success along with failure history / handler override.
        self._recovery_hints: dict[int, list[str]] = {}
        self._recovery_attempts_per_step: dict[int, int] = {}
        self._total_recovery_attempts: int = 0
        self.max_cost, self.max_time = max_cost, max_time_minutes * 60
        self.step_callback, self.keep_screenshots = step_callback, keep_screenshots
        self.cancel_event = cancel_event
        self.tool_channel = ToolChannel()
        self.cost_meter = CostMeter(cost_config=cost_config, tenant_id=tenant_id)
        self.cost_config, self.tenant_id = (
            self.cost_meter.cost_config, self.cost_meter.tenant_id
        )
        self.costs, self._run_start = self.cost_meter.costs, self.cost_meter.run_start
        # Sibling to CostMeter: same lifecycle, surfaces wall-time
        # accounting per bucket. Phase B (#365) will lift this onto
        # the result envelope; Phase A wires it only at the executor's
        # step dispatch so the foundation lands before the surface.
        from .time_meter import TimeMeter
        self.time_meter = TimeMeter(tenant_id=tenant_id)
        self.browser_state = BrowserState(self)
        self.checkpoint_manager = CheckpointManager(self)
        from .step_handlers import default_registry
        from .step_recovery import StepRecoveryPolicy
        from .run_executor import RunExecutor
        from .critic import ExecutionCritic
        self._handler_registry = default_registry(self)
        self._recovery_policy = StepRecoveryPolicy(self)
        self._executor = RunExecutor(self)
        # Phase C of epic #377: ExecutionCritic runs after every step
        # and can emit directives (currently: InsertStep for
        # navigate_back budget-burn recovery). The executor calls
        # critic.observe_step() and applies any returned directive.
        self._critic = ExecutionCritic(self)

    # ── Public API ────────────────────────────────────────────────────

    def run(self, plan: MicroPlan, resume: bool = False) -> list[StepResult]:
        from .run_executor import RunState

        # Issue #254: per-context sub-goal budget gate. Check BEFORE
        # executor.execute so an over-budget run never enters the
        # plan body — the skip envelope is the whole response.
        # Resume calls bypass the gate (they're continuations, not
        # fresh sub-goals).
        if not resume and self.context_budget is not None:
            envelope = self._check_and_emit_context_budget(plan)
            if envelope is not None:
                return envelope

        if not self.plan_signature:
            self.plan_signature = self._compute_plan_signature(plan)
        state = RunState.fresh(
            run_key=self.run_key, session_name=self.session_name,
            plan_signature=self.plan_signature,
        )
        self._executor.execute(plan, state, resume=resume)
        self._final_summary(state.results)
        return state.results

    def _check_and_emit_context_budget(
        self, plan: MicroPlan,
    ) -> list[StepResult] | None:
        """Increment the per-URL counter and either short-circuit
        with a skip envelope, halt the runner, or pass through with
        a warning depending on the configured ``on_exceeded`` mode.

        Returns ``None`` when the run should proceed normally;
        returns a list of StepResult(s) when the run should
        short-circuit (the caller treats the list as the run's
        full output).
        """
        cb = self.context_budget
        if cb is None:
            return None

        anchor = self._resolve_context_anchor(plan)
        # Increment counters BEFORE checking — so a budget of N
        # allows exactly N runs, not N-1. The N+1-th call sees
        # ``count > max`` and trips.
        self._sub_goal_count_total += 1
        count = self._sub_goal_count_by_url.get(anchor, 0) + 1
        self._sub_goal_count_by_url[anchor] = count

        per_url_exceeded = (
            cb.max_sub_goals_per_url is not None
            and count > cb.max_sub_goals_per_url
        )
        total_exceeded = (
            cb.max_sub_goals_per_iteration is not None
            and self._sub_goal_count_total > cb.max_sub_goals_per_iteration
        )

        if not (per_url_exceeded or total_exceeded):
            return None

        reason_bound = "per_url" if per_url_exceeded else "per_iteration"

        if cb.on_exceeded == "log_only":
            logger.warning(
                "context budget %s exceeded (anchor=%s count=%d) — log_only "
                "mode, executing anyway",
                reason_bound, anchor[:80], count,
            )
            return None

        if cb.on_exceeded == "halt":
            logger.warning(
                "context budget %s exceeded (anchor=%s count=%d) — halting",
                reason_bound, anchor[:80], count,
            )
            self._final_status = "halted"
            return []

        # Default: emit_skip — return a synthetic skip envelope.
        intent_text = (
            plan.steps[0].intent if plan.steps else ""
        ) or "context budget exceeded"
        envelope_data = (
            f"listing_budget_exceeded:bound={reason_bound}:"
            f"url={anchor[:120]}:count={count}"
        )
        logger.warning(
            "context budget %s exceeded — emitting skip envelope "
            "(anchor=%s count=%d)",
            reason_bound, anchor[:80], count,
        )
        return [StepResult(
            step_index=0, intent=intent_text, success=False,
            data=envelope_data,
            skip=True, skip_reason="listing_budget_exceeded",
        )]

    @staticmethod
    def _resolve_context_anchor_from_plan(plan: MicroPlan) -> str:
        """Pull the navigate-step URL from a plan, or the empty
        string if no navigate is present. Static helper exposed so
        tests can pin the resolution rule independently of the
        runner state."""
        for step in plan.steps:
            if step.type == "navigate":
                url = (step.params or {}).get("url", "")
                if isinstance(url, str) and url:
                    return url
        return ""

    def _resolve_context_anchor(self, plan: MicroPlan) -> str:
        """Decide which URL bucket a given ``run(plan)`` invocation
        belongs to. Preference order:

        1. The first ``navigate`` step's ``params.url`` — canonical
           because that's the URL the sub-goal is *targeted at*,
           not whatever was in the address bar.
        2. ``self._last_known_url`` — for sub-plans that operate on
           the page a prior sub-goal opened.
        3. The fixed sentinel ``"_no_anchor_"`` — keeps the counter
           working for cold runners with neither signal (rare).
        """
        url = self._resolve_context_anchor_from_plan(plan)
        if url:
            return url
        if getattr(self, "_last_known_url", ""):
            return self._last_known_url
        return "_no_anchor_"

    def run_with_status(
        self, plan: MicroPlan, resume: bool = False,
    ) -> RunnerResult:
        return self._build_runner_result(plan, self.run(plan, resume=resume))

    def run_with_exploration(
        self,
        *,
        plan_variants: list[MicroPlan],
        recipe_variants: list[Any] | None = None,
        budget_per_variant: Any = None,
    ) -> list[Any]:
        """Run plan + recipe variants sequentially and return a
        per-variant outcome bundle (issue #248).

        v1 ships the *substrate*: each variant is dispatched through
        the existing :meth:`run` infrastructure, the runtime stamps the
        outcome with rejection histogram + URL coverage derived from
        the step results, and a per-variant
        :class:`ExplorationBudget` enforces a cost / wall-clock cap.
        Concrete deviation strategies that emit
        :class:`ExperimentEvent` records land in follow-up PRs against
        this surface.

        Args:
            plan_variants: List of :class:`MicroPlan` variants to run.
                Each is dispatched once per recipe variant (if any).
            recipe_variants: Optional list of
                :class:`ExtractionSchema` to swap onto
                ``self.extractor.schema`` for the duration of each
                variant. ``None`` (default) runs every plan variant
                once with the extractor's existing schema.
            budget_per_variant: Per-variant
                :class:`ExplorationBudget`. ``None`` uses defaults
                (3 USD / 10 min). The runtime checks the wall-clock
                cap before starting each variant; a variant that
                trips the budget aborts cleanly with
                ``terminal_status='budget_exceeded'`` and an empty
                step_results list (it never started).

        Returns:
            ``list[VariantOutcome]``, one per variant in iteration
            order (plan_variants × recipe_variants).
        """
        # Imports inside the method so the substrate module is opt-in
        # — modules that never call this entrypoint don't pay the
        # exploration import cost.
        from .exploration import (
            ExplorationBudget,
            VariantOutcome,
            rejection_histogram_from_steps,
            url_coverage_from_steps,
        )

        if not plan_variants:
            return []
        recipe_list: list[Any] = list(recipe_variants) if recipe_variants else [None]
        budget = budget_per_variant or ExplorationBudget()

        # Snapshot the extractor's current schema so we can restore it
        # after the run — recipe swap must not leak into legacy
        # callers that share this runner instance.
        original_schema = (
            getattr(self.extractor, "schema", None)
            if self.extractor is not None
            else None
        )

        outcomes: list[Any] = []
        run_started = time.time()
        wall_budget_seconds = budget.max_minutes * 60.0

        for plan_idx, plan in enumerate(plan_variants):
            for recipe_idx, recipe in enumerate(recipe_list):
                variant_id = f"plan{plan_idx}"
                if recipe is not None:
                    variant_id += f"_recipe{recipe_idx}"

                # Wall-clock budget check before each variant.
                # ``run`` itself can overshoot — by-design, since the
                # underlying execution path is best-effort. The budget
                # is a *don't start the next variant if we're already
                # over* signal, which matches the refinement-agent's
                # cost-control intent.
                elapsed = time.time() - run_started
                if elapsed > wall_budget_seconds:
                    outcomes.append(VariantOutcome(
                        variant_id=variant_id,
                        terminal_status="budget_exceeded",
                        wall_time_s=elapsed,
                    ))
                    continue

                # Swap recipe onto extractor for this variant. Done
                # outside the try/finally so any exception during the
                # swap (mis-typed recipe variant) surfaces immediately
                # rather than after a half-run.
                if recipe is not None and self.extractor is not None:
                    self.extractor.schema = recipe

                variant_start = time.time()
                pre_cost = float(self.costs.get("claude_extract", 0.0))
                try:
                    step_results = self.run(plan)
                finally:
                    # Restore the original schema between variants.
                    if recipe is not None and self.extractor is not None:
                        self.extractor.schema = original_schema

                variant_elapsed = time.time() - variant_start
                terminal = getattr(self, "_final_status", "completed") or "completed"
                if terminal == "running":  # underlying runner didn't set it
                    terminal = "completed"

                outcomes.append(VariantOutcome(
                    variant_id=variant_id,
                    terminal_status=terminal,
                    step_results=list(step_results or []),
                    recipe_rejection_histogram=rejection_histogram_from_steps(
                        step_results or []
                    ),
                    url_coverage=url_coverage_from_steps(step_results or []),
                    cost_total=float(self.costs.get("claude_extract", 0.0)) - pre_cost,
                    wall_time_s=variant_elapsed,
                ))

        return outcomes

    def resume(
        self, state: PauseState | dict[str, Any], *,
        user_input: Any = None, plan: MicroPlan | None = None,
    ) -> RunnerResult:
        if isinstance(state, dict):
            state = PauseState.from_dict(state)
        if plan is None:
            raise ValueError("resume() requires the original plan; pass plan=...")
        signature = self._compute_plan_signature(plan)
        if state.plan_signature and state.plan_signature != signature:
            raise ValueError(
                "PauseState plan_signature mismatch — can't resume a different plan"
            )
        replayed = [StepResult.from_dict(d) for d in state.step_results]
        self.tool_channel.clear_pause()
        self.checkpoint_path = state.checkpoint_path or self.checkpoint_path
        self.run_key = state.run_key or self.run_key
        self.session_name = state.session_name or self.session_name
        self.plan_signature, self.resume_state = signature, True
        self._pause_input = user_input
        try:
            steps_continued = self.run(plan, resume=True)
        finally:
            self._pause_input = None
        return self._build_runner_result(
            plan, replayed + steps_continued[len(replayed):],
        )

    def consume_pause_input(self, default: Any = None) -> Any:
        value = getattr(self, "_pause_input", None)
        self._pause_input = None
        return default if value is None else value

    def dynamic_verification_report(
        self, status: str | None = None,
    ) -> dict[str, Any]:
        return self.dynamic_verifier.report(status=status or self._final_status)

    # ── Delegation magic — tables live in :mod:`._runner_helpers` ─────

    def __getattr__(self, name: str) -> Any:
        if name in _h.SCANNER_FIELDS:
            scanner = self.__dict__.get("scanner") or ListingsScanner()
            self.__dict__.setdefault("scanner", scanner)
            return getattr(scanner, _h.SCANNER_FIELDS[name])
        if name in _h.COLLAB_METHODS:
            obj, method = _h.COLLAB_METHODS[name]
            return getattr(getattr(self, obj), method)
        if name in _h.HELPER_METHODS:
            return functools.partial(getattr(_h, _h.HELPER_METHODS[name]), self)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _h.SCANNER_FIELDS:
            scanner = self.__dict__.get("scanner") or ListingsScanner()
            self.__dict__.setdefault("scanner", scanner)
            setattr(scanner, _h.SCANNER_FIELDS[name], value)
            return
        super().__setattr__(name, value)
