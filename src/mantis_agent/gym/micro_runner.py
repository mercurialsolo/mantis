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
    ):
        self.brain, self.env, self.grounding, self.extractor = (
            brain, env, grounding, extractor
        )
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
        self.browser_state = BrowserState(self)
        self.checkpoint_manager = CheckpointManager(self)
        from .step_handlers import default_registry
        from .step_recovery import StepRecoveryPolicy
        from .run_executor import RunExecutor
        self._handler_registry = default_registry(self)
        self._recovery_policy = StepRecoveryPolicy(self)
        self._executor = RunExecutor(self)

    # ── Public API ────────────────────────────────────────────────────

    def run(self, plan: MicroPlan, resume: bool = False) -> list[StepResult]:
        from .run_executor import RunState
        if not self.plan_signature:
            self.plan_signature = self._compute_plan_signature(plan)
        state = RunState.fresh(
            run_key=self.run_key, session_name=self.session_name,
            plan_signature=self.plan_signature,
        )
        self._executor.execute(plan, state, resume=resume)
        self._final_summary(state.results)
        return state.results

    def run_with_status(
        self, plan: MicroPlan, resume: bool = False,
    ) -> RunnerResult:
        return self._build_runner_result(plan, self.run(plan, resume=resume))

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
