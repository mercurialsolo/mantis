"""Checkpoint persistence flow for MicroPlanRunner — extracted from
micro_runner.py (#115, step 6).

Owns the *flow* of taking the runner's mutable state and serializing it
into a :class:`RunCheckpoint`, plus the inverse hydration pass. The
:class:`RunCheckpoint` data structure itself lives in
:mod:`mantis_agent.gym.checkpoint` (extracted in step 1).

Responsibilities:

* :meth:`persist` — snapshot every runner attribute that round-trips
  through resume into the checkpoint, then save to disk + invoke the
  optional ``on_checkpoint`` host callback.
* :meth:`restore` — replay a loaded checkpoint into the runner's state
  attributes, returning the persisted step results / loop counters /
  page bookkeeping for the run loop.
* :meth:`save_active_progress` — persist mid-step progress using the
  pending context the runner stashes during long-running steps (so a
  cancellation halfway through a loop doesn't lose the partially
  completed work).
* :meth:`compute_plan_signature` — the SHA-256 over the plan's
  step-defining fields, used to detect "different plan, can't resume"
  on checkpoint load.

The runner keeps the same public methods as 1-line shims for backward
compat with any external caller that subclasses ``MicroPlanRunner`` or
constructs runners via ``object.__new__``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from .checkpoint import RunCheckpoint, StepResult

if TYPE_CHECKING:
    from ..plan_decomposer import MicroPlan
    from .micro_runner import MicroPlanRunner

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Owns the runner's checkpoint persistence flow.

    Holds a back-reference to the :class:`MicroPlanRunner` so it can
    read/write the 14 runner attributes that round-trip through resume.
    Subsequent steps in the runner-split could migrate ownership of those
    attributes into this class via property descriptors, but for now this
    extraction just consolidates the flow methods.
    """

    def __init__(self, parent: "MicroPlanRunner") -> None:
        self.parent = parent

    # ── Plan signature (pure / static) ──────────────────────────────────

    @staticmethod
    def compute_plan_signature(plan: "MicroPlan") -> str:
        """SHA-256 hex over the plan's step-defining fields.

        Used to fail fast on resume when the on-disk checkpoint was
        produced by a different plan shape — silently resuming with a
        mismatched plan would corrupt step indices.
        """
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

    # ── Persist ─────────────────────────────────────────────────────────

    def persist(
        self,
        checkpoint: RunCheckpoint,
        plan: "MicroPlan",
        results: list[StepResult],
        loop_counters: dict[int, int],
        listings_on_page: int,
        next_step_index: int,
        status: str = "running",
        halt_reason: str = "",
    ) -> None:
        """Snapshot runner state into ``checkpoint`` and save to disk.

        Mirrors the pre-#115 ``MicroPlanRunner._persist_checkpoint``
        exactly: every attribute read here was read by the original
        method.
        """
        p = self.parent
        checkpoint.run_key = p.run_key
        checkpoint.plan_signature = p.plan_signature
        checkpoint.session_name = p.session_name
        checkpoint.status = status
        checkpoint.halt_reason = halt_reason
        checkpoint.step_index = next_step_index
        checkpoint.page = getattr(p, "_current_page", 1)
        checkpoint.current_url = p._last_known_url
        checkpoint.reentry_url = p.browser_state.reentry_url_for_step(plan, next_step_index)
        checkpoint.seen_urls = sorted(p._seen_urls)
        checkpoint.extracted_leads = p._unique_leads_from_results(results)
        checkpoint.step_results = [result.to_dict() for result in results]
        checkpoint.loop_counters = {str(k): v for k, v in loop_counters.items()}
        checkpoint.listings_on_page = listings_on_page
        checkpoint.extracted_titles = list(p._extracted_titles)
        checkpoint.page_listings = [list(item) for item in p._page_listings]
        checkpoint.page_listing_index = p._page_listing_index
        checkpoint.viewport_stage = p._viewport_stage
        checkpoint.current_page = getattr(p, "_current_page", 1)
        checkpoint.results_base_url = p._results_base_url
        checkpoint.required_filter_tokens = list(p._required_filter_tokens)
        checkpoint.scroll_state = dict(p._scroll_state)
        checkpoint.last_extracted = dict(p._last_extracted)
        checkpoint.costs = dict(p.costs)
        checkpoint.dynamic_coverage = p.dynamic_verification_report(status=status)
        # #127: snapshot prompt SHAs so a regression can be attributed to
        # a specific prompt edit even after the registry changes.
        try:
            from ..prompts import current_prompt_versions as _cpv
            checkpoint.prompt_versions = _cpv()
        except Exception as exc:  # noqa: BLE001 — telemetry must not abort saves
            logger.debug("prompt_versions snapshot skipped: %s", exc)
            checkpoint.prompt_versions = {}
        checkpoint.save(p.checkpoint_path)
        p._final_status = status
        if p.on_checkpoint:
            try:
                p.on_checkpoint()
            except Exception as e:
                logger.warning("  [checkpoint] external commit failed: %s", e)

    # ── Restore ─────────────────────────────────────────────────────────

    def restore(
        self, checkpoint: RunCheckpoint
    ) -> tuple[list[StepResult], dict[int, int], int]:
        """Replay ``checkpoint`` into the runner. Mirrors the pre-#115
        ``_restore_from_checkpoint`` exactly.

        Returns ``(step_results, loop_counters, listings_on_page)`` for
        the run loop to seed itself with.
        """
        p = self.parent
        p._seen_urls = set(checkpoint.seen_urls)
        p._extracted_titles = list(checkpoint.extracted_titles)
        p._page_listings = [tuple(item) for item in checkpoint.page_listings]
        p._page_listing_index = checkpoint.page_listing_index
        p._viewport_stage = checkpoint.viewport_stage
        p._results_base_url = checkpoint.results_base_url
        p._required_filter_tokens = tuple(checkpoint.required_filter_tokens)
        p._current_page = checkpoint.current_page or checkpoint.page or 1
        p._last_known_url = checkpoint.current_url or checkpoint.reentry_url
        p._scroll_state = dict(checkpoint.scroll_state or {})
        p._last_extracted = dict(checkpoint.last_extracted or {})
        # Preserves the pre-#115 contract: callers that bypass __init__
        # (e.g. test_private_seller_filter uses object.__new__ + manual
        # ``runner.costs = {...}``) keep working. CostMeter.restore()
        # exists for clean composition but the runner's path uses the
        # in-place dict update so it doesn't require a cost_meter to
        # be present on the instance.
        p.costs.update(checkpoint.costs or {})
        if checkpoint.dynamic_coverage:
            p.dynamic_verifier.load_report(checkpoint.dynamic_coverage)
        p._listings_on_page = checkpoint.listings_on_page
        results = [StepResult.from_dict(item) for item in checkpoint.step_results]
        loop_counters = {
            int(k): int(v) for k, v in (checkpoint.loop_counters or {}).items()
        }
        return results, loop_counters, checkpoint.listings_on_page

    # ── Mid-step progress ───────────────────────────────────────────────

    def save_active_progress(self, halt_reason: str = "step_progress") -> None:
        """Persist mid-step progress using the pending context the runner
        stashes during long-running operations.

        Skipped if no context is active (e.g. between top-level steps).
        """
        ctx: dict[str, Any] | None = self.parent._active_checkpoint_context
        if not ctx:
            return
        self.persist(
            checkpoint=ctx["checkpoint"],
            plan=ctx["plan"],
            results=ctx["results"],
            loop_counters=ctx["loop_counters"],
            listings_on_page=ctx["listings_on_page"],
            next_step_index=ctx["step_index"],
            status="running",
            halt_reason=halt_reason,
        )


__all__ = ["CheckpointManager"]
