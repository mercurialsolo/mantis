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

    plan = PlanDecomposer().decompose("plans/boattrader/extract_only.txt")
    runner = MicroPlanRunner(brain=brain, env=env, ...)
    results = runner.run(plan)
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, TYPE_CHECKING

from ..actions import Action, ActionType
from .runner import GymRunner

from ..plan_decomposer import MicroIntent, MicroPlan
from ..site_config import SiteConfig
from ..verification.dynamic_plan_verifier import DynamicPlanVerifier

if TYPE_CHECKING:
    from ..extraction import ClaudeExtractor

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Outcome of executing one micro-intent."""
    step_index: int
    intent: str
    success: bool
    data: str = ""
    steps_used: int = 0
    duration: float = 0.0
    reversed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StepResult:
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in allowed})


@dataclass
class RunCheckpoint:
    """Persistent logical run state for cross-session resume."""
    version: int = 2
    run_key: str = ""
    plan_signature: str = ""
    session_name: str = ""
    status: str = "running"
    halt_reason: str = ""
    step_index: int = 0
    page: int = 1
    current_url: str = ""
    reentry_url: str = ""
    seen_urls: list = field(default_factory=list)
    extracted_leads: list = field(default_factory=list)
    step_results: list = field(default_factory=list)
    loop_counters: dict = field(default_factory=dict)
    listings_on_page: int = 0
    extracted_titles: list = field(default_factory=list)
    page_listings: list = field(default_factory=list)
    page_listing_index: int = 0
    viewport_stage: int = 0
    current_page: int = 1
    results_base_url: str = ""
    required_filter_tokens: list = field(default_factory=list)
    scroll_state: dict = field(default_factory=dict)
    last_extracted: dict = field(default_factory=dict)
    costs: dict = field(default_factory=dict)
    dynamic_coverage: dict = field(default_factory=dict)
    timestamp: float = 0.0

    def save(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = asdict(self)
        payload["timestamp"] = time.time()
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> RunCheckpoint | None:
        try:
            with open(path) as f:
                d = json.load(f)
            allowed = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in d.items() if k in allowed})
        except Exception:
            return None


# Reverse actions for each step type
REVERSE_ACTIONS = {
    "click": [("key_press", "Escape"), ("key_press", "alt+Left")],
    "scroll": [("key_press", "Home")],
    "navigate": [("key_press", "alt+Left")],
    "navigate_back": [],  # Already going back
    "filter": [("key_press", "alt+Left")],
    "paginate": [("key_press", "alt+Left")],
}


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
        self._final_status: str = "running"
        self.max_cost = max_cost
        self.max_time = max_time_minutes * 60

        # Cost tracking
        self.costs = {
            "gpu_steps": 0,        # Total Holo3 steps
            "gpu_seconds": 0.0,    # Approx GPU time
            "claude_extract": 0,   # Claude extraction calls
            "claude_grounding": 0, # Claude grounding calls
            "proxy_mb": 0.0,       # Estimated proxy bandwidth
        }
        self._run_start = time.time()

    def dynamic_verification_report(self, status: str | None = None) -> dict[str, Any]:
        return self.dynamic_verifier.report(status=status or self._final_status)

    @staticmethod
    def _compute_plan_signature(plan: MicroPlan) -> str:
        payload = [
            {
                "intent": step.intent,
                "type": step.type,
                "verify": step.verify,
                "budget": step.budget,
                "reverse": step.reverse,
                "grounding": step.grounding,
                "claude_only": step.claude_only,
                "loop_target": step.loop_target,
                "loop_count": step.loop_count,
                "section": step.section,
                "required": step.required,
                "gate": step.gate,
            }
            for step in plan.steps
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _cost_totals(self) -> tuple[float, float, float, float]:
        gpu_cost = (self.costs["gpu_seconds"] / 3600) * 3.25
        claude_cost = (self.costs["claude_extract"] * 0.003) + (self.costs["claude_grounding"] * 0.003)
        proxy_cost = (self.costs["proxy_mb"] / 1024) * 5.0
        total_cost = gpu_cost + claude_cost + proxy_cost
        return gpu_cost, claude_cost, proxy_cost, total_cost

    @classmethod
    def _unique_leads_from_results(cls, results: list[StepResult]) -> list[str]:
        unique: dict[str, str] = {}
        for lead in cls._successful_lead_data(results):
            unique[cls._lead_key(lead)] = lead
        return list(unique.values())

    def _record_step_costs(self, step: MicroIntent, step_result: StepResult) -> None:
        if step_result.steps_used > 0:
            self.costs["gpu_steps"] += step_result.steps_used
            self.costs["gpu_seconds"] += step_result.steps_used * 3  # ~3s per step
        if step.claude_only:
            self.costs["claude_extract"] += 1
        if step.grounding:
            self.costs["claude_grounding"] += step_result.steps_used  # ~1 grounding per click
        if step.type in ("click", "navigate", "paginate"):
            self.costs["proxy_mb"] += 5.0  # ~5MB per page load
        elif step.type == "scroll":
            self.costs["proxy_mb"] += 0.5  # minimal for scroll

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

    def _current_results_page_url(self) -> str:
        if not self._results_base_url:
            return ""
        if self._current_page <= 1:
            base_clean = re.sub(
                self.site_config.pagination_strip_pattern or r"/page-\d+/?$",
                "",
                self._results_base_url.rstrip("/"),
            )
            return f"{base_clean}/"
        return self.site_config.paginated_url(self._results_base_url, self._current_page)

    def _reentry_url_for_step(self, plan: MicroPlan, next_step_index: int) -> str:
        next_step = plan.steps[next_step_index] if 0 <= next_step_index < len(plan.steps) else None
        results_url = self._current_results_page_url() or self._results_base_url
        if next_step and next_step.type in {"click", "paginate", "loop", "filter"}:
            return results_url or self._last_known_url
        if next_step and next_step.type in {"extract_url", "scroll", "extract_data", "navigate_back"}:
            return self._last_known_url or results_url
        return self._last_known_url or results_url

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
        checkpoint.run_key = self.run_key
        checkpoint.plan_signature = self.plan_signature
        checkpoint.session_name = self.session_name
        checkpoint.status = status
        checkpoint.halt_reason = halt_reason
        checkpoint.step_index = next_step_index
        checkpoint.page = getattr(self, "_current_page", 1)
        checkpoint.current_url = self._last_known_url
        checkpoint.reentry_url = self._reentry_url_for_step(plan, next_step_index)
        checkpoint.seen_urls = sorted(self._seen_urls)
        checkpoint.extracted_leads = self._unique_leads_from_results(results)
        checkpoint.step_results = [result.to_dict() for result in results]
        checkpoint.loop_counters = {str(k): v for k, v in loop_counters.items()}
        checkpoint.listings_on_page = listings_on_page
        checkpoint.extracted_titles = list(self._extracted_titles)
        checkpoint.page_listings = [list(item) for item in self._page_listings]
        checkpoint.page_listing_index = self._page_listing_index
        checkpoint.viewport_stage = self._viewport_stage
        checkpoint.current_page = getattr(self, "_current_page", 1)
        checkpoint.results_base_url = self._results_base_url
        checkpoint.required_filter_tokens = list(self._required_filter_tokens)
        checkpoint.scroll_state = dict(self._scroll_state)
        checkpoint.last_extracted = dict(self._last_extracted)
        checkpoint.costs = dict(self.costs)
        checkpoint.dynamic_coverage = self.dynamic_verification_report(status=status)
        checkpoint.save(self.checkpoint_path)
        self._final_status = status
        if self.on_checkpoint:
            try:
                self.on_checkpoint()
            except Exception as e:
                logger.warning("  [checkpoint] external commit failed: %s", e)

    def _restore_from_checkpoint(
        self,
        checkpoint: RunCheckpoint,
    ) -> tuple[list[StepResult], dict[int, int], int]:
        self._seen_urls = set(checkpoint.seen_urls)
        self._extracted_titles = list(checkpoint.extracted_titles)
        self._page_listings = [tuple(item) for item in checkpoint.page_listings]
        self._page_listing_index = checkpoint.page_listing_index
        self._viewport_stage = checkpoint.viewport_stage
        self._results_base_url = checkpoint.results_base_url
        self._required_filter_tokens = tuple(checkpoint.required_filter_tokens)
        self._current_page = checkpoint.current_page or checkpoint.page or 1
        self._last_known_url = checkpoint.current_url or checkpoint.reentry_url
        self._scroll_state = dict(checkpoint.scroll_state or {})
        self._last_extracted = dict(checkpoint.last_extracted or {})
        self.costs.update(checkpoint.costs or {})
        if checkpoint.dynamic_coverage:
            self.dynamic_verifier.load_report(checkpoint.dynamic_coverage)
        self._listings_on_page = checkpoint.listings_on_page
        results = [StepResult.from_dict(item) for item in checkpoint.step_results]
        loop_counters = {int(k): int(v) for k, v in (checkpoint.loop_counters or {}).items()}
        return results, loop_counters, checkpoint.listings_on_page

    def _checkpoint_active_progress(self, halt_reason: str = "step_progress") -> None:
        ctx = self._active_checkpoint_context
        if not ctx:
            return
        self._persist_checkpoint(
            checkpoint=ctx["checkpoint"],
            plan=ctx["plan"],
            results=ctx["results"],
            loop_counters=ctx["loop_counters"],
            listings_on_page=ctx["listings_on_page"],
            next_step_index=ctx["step_index"],
            status="running",
            halt_reason=halt_reason,
        )

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
        state = dict(self._scroll_state)
        state["context"] = context
        state["url"] = url or self._last_known_url or self._current_results_page_url()
        state["updated_at"] = time.time()
        if page_downs is not None:
            state["page_downs"] = max(0, page_downs)
        if wheel_downs is not None:
            state["wheel_downs"] = max(0, wheel_downs)
        if viewport_stage is not None:
            state["viewport_stage"] = max(0, viewport_stage)
        if label:
            state["label"] = label
        self._scroll_state = state
        if flush:
            self._checkpoint_active_progress(f"scroll_state:{context}")

    def _update_scroll_state_from_trajectory(self, result: Any, context: str) -> None:
        page_downs = int(self._scroll_state.get("page_downs", 0) or 0)
        wheel_downs = int(self._scroll_state.get("wheel_downs", 0) or 0)
        for item in getattr(result, "trajectory", []) or []:
            action = getattr(item, "action", None)
            if not action:
                continue
            if action.action_type == ActionType.KEY_PRESS:
                keys = str(action.params.get("keys") or action.params.get("key") or "").lower()
                if "home" in keys:
                    page_downs = 0
                    wheel_downs = 0
                elif "page_down" in keys or "pagedown" in keys:
                    page_downs += 1
                elif "page_up" in keys or "pageup" in keys:
                    page_downs = max(0, page_downs - 1)
                elif keys == "end":
                    self._scroll_state["end_reached"] = True
            elif action.action_type == ActionType.SCROLL:
                direction = str(action.params.get("direction", "down")).lower()
                amount = int(action.params.get("amount", 3) or 0)
                if direction == "down":
                    wheel_downs += amount
                elif direction == "up":
                    wheel_downs = max(0, wheel_downs - amount)
        self._set_scroll_state(
            context=context,
            page_downs=page_downs,
            wheel_downs=wheel_downs,
            viewport_stage=self._viewport_stage,
        )

    def _restore_scroll_position(self) -> None:
        """Replay logical scroll depth after URL re-entry in screen-only envs."""
        if not self._scroll_state:
            return
        state_url = str(self._scroll_state.get("url") or "")
        current_url = self._last_known_url or self._current_results_page_url()
        if state_url and current_url:
            def normalize(url: str) -> str:
                return re.sub(r"^https?://(www\.)?", "", url).rstrip("/")

            if normalize(state_url) != normalize(current_url):
                logger.info(
                    "  [resume] Skipping scroll restore for different URL "
                    "(state=%s current=%s)",
                    state_url[:80],
                    current_url[:80],
                )
                return
        page_downs = int(self._scroll_state.get("page_downs", 0) or 0)
        wheel_downs = int(self._scroll_state.get("wheel_downs", 0) or 0)
        if page_downs <= 0 and wheel_downs <= 0:
            return
        try:
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(0.5)
            for _ in range(min(page_downs, 12)):
                self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                time.sleep(0.3)
            if wheel_downs:
                self.env.step(Action(
                    action_type=ActionType.SCROLL,
                    params={"direction": "down", "amount": min(wheel_downs, 40)},
                ))
                time.sleep(0.5)
            logger.info(
                "  [resume] Restored scroll depth page_downs=%s wheel_downs=%s context=%s",
                page_downs,
                wheel_downs,
                self._scroll_state.get("context", ""),
            )
        except Exception as e:
            logger.warning("  [resume] Failed to restore scroll position: %s", e)

    def _resume_browser_state(self, url: str) -> bool:
        if not url:
            return False
        logger.info("  [resume] Re-entering browser state at %s", url[:140])
        try:
            self.env.reset(task="resume", start_url=url)
            time.sleep(12)
            try:
                self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
                time.sleep(1)
            except Exception:
                pass
            self._last_known_url = url
            self._restore_scroll_position()
            return True
        except Exception as e:
            logger.warning("  [resume] Failed to restore browser at %s: %s", url[:120], e)
            return False

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
            try:
                step_result = self._execute_step(effective_step, step_index)
            finally:
                self._active_checkpoint_context = None
            results.append(step_result)
            self._record_step_costs(effective_step, step_result)
            self._log_progress(step_result, results)

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
        elif self._final_status == "running":
            persist(step_index, status="halted", halt_reason="stopped")

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

    @staticmethod
    def _successful_lead_data(results: list[StepResult]) -> list[str]:
        return [
            r.data for r in results
            if r.success and (r.data or "").startswith("VIABLE")
        ]

    @staticmethod
    def _lead_key(data: str) -> str:
        url_match = re.search(r"URL:\s*([^|]+)", data)
        if url_match:
            return url_match.group(1).strip()
        return data[:100]

    @staticmethod
    def _lead_has_phone(data: str) -> bool:
        phone_match = re.search(r"Phone:\s*([^|]+)", data, flags=re.IGNORECASE)
        if not phone_match:
            return False
        phone = phone_match.group(1).strip().lower()
        if phone in {"", "none", "n/a", "na", "unknown", "not visible", "not shown"}:
            return False
        return len(re.sub(r"\D", "", phone)) >= 10

    @classmethod
    def _lead_counts(cls, results: list[StepResult]) -> tuple[int, int]:
        leads_by_key = {}
        for data in cls._successful_lead_data(results):
            leads_by_key[cls._lead_key(data)] = data
        total = len(leads_by_key)
        with_phone = sum(1 for data in leads_by_key.values() if cls._lead_has_phone(data))
        return total, with_phone

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

        # Click steps: Claude finds target → Holo3 clicks coordinates
        if step.type == "click" and self.extractor:
            if not self._ensure_results_filters(index):
                return StepResult(
                    step_index=index, intent=step.intent, success=False,
                    data="filters_not_applied",
                )
            # Brief settle — page may still be loading after navigate/paginate
            time.sleep(2)
            return self._execute_claude_guided_click(step, index)

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

    def _execute_navigate(self, step: MicroIntent, index: int) -> StepResult:
        """Navigate to a URL using env.reset() — no Holo3 steps needed.

        Waits for page load and handles Cloudflare challenges (auto-solve in 5-10s).
        """
        import re
        url_match = re.search(r'https?://[^\s"]+', step.intent)
        url = url_match.group() if url_match else ""

        if not url:
            logger.warning(f"  [navigate] No URL found in intent: {step.intent[:60]}")
            return StepResult(step_index=index, intent=step.intent, success=False)

        logger.info(f"  [navigate] Loading {url}")
        try:
            self.env.reset(task="navigate", start_url=url)
            # Wait for Cloudflare challenge to auto-solve + page render
            time.sleep(18)
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
        for verify_attempt in range(2):
            time.sleep(3 + verify_attempt * 3)  # 3s first, 6s retry
            after = self.env.screenshot()
            verify_data = self.extractor.extract(after)
            self.costs["claude_extract"] += 1
            url = verify_data.url if verify_data else ""

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
                after = self.env.screenshot()
                verify_data = self.extractor.extract(after)
                self.costs["claude_extract"] += 1
                url = verify_data.url if verify_data else ""
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

                after = self.env.screenshot()
                verify_data = self.extractor.extract(after)
                self.costs["claude_extract"] += 1
                url = verify_data.url if verify_data else ""
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
        """Undo a failed step using predefined reverse actions."""
        actions = REVERSE_ACTIONS.get(step.type, [])

        # Use step-specific reverse if provided
        if step.reverse:
            # Parse "Press Escape then Alt+Left" into actions
            if "escape" in step.reverse.lower():
                actions = [("key_press", "Escape")] + actions
            if "alt+left" in step.reverse.lower():
                actions.append(("key_press", "alt+Left"))

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
