"""Run-time observer that proposes corrections after each step.

Phase C of epic #377. Skeleton + one concrete capability:

* **Skeleton** (v1): :class:`ExecutionCritic` runs after every step
  via :meth:`observe_step`. Reads ``failure_class`` + step context +
  recovery-policy outcome and decides whether to emit a directive.
  Future phases plug additional capabilities (Claude-vision obstacle
  detection, brain-ladder policy, intent-rewriter unification) into
  the same hook.

* **One concrete capability**: when a ``navigate_back`` step fails
  with ``failure_class=brain_loop_exhausted`` (the brain spent its
  budget trying to drive the browser back, typically on SPAs whose
  history stack is broken), emit
  :class:`InsertStep` with type=``navigate`` to the runner's
  ``_results_base_url``. The runner inserts the step ahead of the
  next iteration, restoring the agent to the expected page without
  relying on the browser's back button.

Why generic: every signal the critic reads is framework-level
(``step.type``, ``failure_class``, ``runner._results_base_url``).
Zero plan / URL / domain content reads in.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from ..plan_decomposer import MicroIntent

if TYPE_CHECKING:
    from .checkpoint import StepResult
    from .micro_runner import MicroPlanRunner
    from .run_executor import RunState

logger = logging.getLogger(__name__)


# How many rewrite-triggering failures on the same step before the
# frontier-model capability fires. Set to 2 so the first failure
# still goes through the existing recovery loop (cheap rewriter,
# label-match retries) — only the persistent miss escalates to a
# Claude call.
#
# The gate covers any class in :data:`intent_rewriter.REWRITE_TRIGGERING_CLASSES`
# (``wrong_target`` / ``no_state_change`` / ``brain_loop_exhausted``).
# The original wrong_target-only gate left the canonical staff-crm
# sidebar cascade unreachable: clicks LAND on real ``<a>`` elements
# (so the snapshot-diff demote stamps ``no_state_change``, not
# ``wrong_target``) yet the page doesn't navigate — exactly the
# pattern the frontier critic exists to break.
_FRONTIER_PERSISTENT_FAILURE_THRESHOLD: int = 2


def _frontier_enabled() -> bool:
    """``MANTIS_CRITIC_FRONTIER=enabled`` flips the opt-in.

    Default off so deployments that don't want the extra Claude
    spend behave exactly as before. The lone existing rule
    (``navigate_back`` + ``brain_loop_exhausted``) keeps working
    regardless of this gate.
    """
    return os.environ.get("MANTIS_CRITIC_FRONTIER", "").strip().lower() == "enabled"


@dataclass(frozen=True)
class InsertStep:
    """Critic directive: splice a new step into the plan at the
    current ``state.step_index`` (shifting subsequent steps right).
    The runner records the insertion as a healing event and the
    inserted step runs next.
    """

    intent: str
    step_type: str
    reason: str
    params: dict[str, Any] | None = None


@dataclass(frozen=True)
class ReplaceStep:
    """Critic directive: replace the step at the current
    ``state.step_index`` IN PLACE — does NOT shift subsequent steps.
    The new step occupies the same plan slot and runs on the next
    loop iteration.

    Used when the frontier-model observer (#435 item 8) decides the
    current step's structure is wrong rather than its execution — a
    sidebar-link click cascade should become a direct ``navigate``
    to the filtered URL, for example. The runner records the
    replacement as a healing event and resets the per-step retry
    history so the new step starts with a clean slate.
    """

    intent: str
    step_type: str
    reason: str
    params: dict[str, Any] | None = None
    hints: dict[str, Any] | None = None


# Union of every directive type ``observe_step`` may return — used
# in apply_directive's dispatch and for type hints on the public
# surface.
Directive = Union[InsertStep, ReplaceStep]


class ExecutionCritic:
    """Observer that proposes corrections after each step.

    v1 capabilities:
    - Detect ``navigate_back`` + ``brain_loop_exhausted`` → emit
      ``InsertStep`` for a direct ``navigate`` to the results base URL.

    Future capabilities to plug in via the same ``observe_step`` hook:
    - Claude-vision modal / banner detection → ``InsertStep`` for
      ``dismiss_overlay``.
    - Cross-attempt URL drift → directive to the IntentRewriter with
      drift context.
    - Brain-ladder policy: when N failures across all steps have
      escalated to Holo3 without progress, force fallback to Claude.

    The critic does NOT replace ``_recovery_policy`` or the existing
    rewriters. It runs AFTER ``_handle_failure`` and adds capabilities
    that the recovery policy doesn't cover.
    """

    def __init__(self, runner: "MicroPlanRunner") -> None:
        self.runner = runner

    def observe_step(
        self,
        plan: Any,  # MicroPlan — typed via TYPE_CHECKING to avoid import cycle
        state: "RunState",
        step: MicroIntent,
        step_result: "StepResult",
        *,
        recovery_continued: bool,
    ) -> Optional[Directive]:
        """Called after every step dispatch + recovery handling.

        ``recovery_continued`` is True when the recovery policy
        decided retry/advance (i.e. ``_handle_failure`` returned
        True), False on halt. The critic only emits directives when
        the run is continuing — there's no point mutating the plan
        ahead of a halted run.

        Returns one of:

        * :class:`InsertStep` — splice a NEW step at the current
          index (existing capability: navigate_back + brain loop).
        * :class:`ReplaceStep` — replace the step in place (new in
          #435 item 8: frontier-model observer on persistent
          ``wrong_target`` failures).
        * ``None`` — let the recovery policy continue as-is.

        Capabilities are checked cheapest-first. Rule-based
        capabilities run unconditionally; the frontier-model
        capability is opt-in via ``MANTIS_CRITIC_FRONTIER``.
        """
        if not recovery_continued:
            return None
        if step_result.success:
            return None

        # Rule-based: navigate_back loop exhaustion → direct nav.
        rule_directive = self._maybe_recover_navigate_back(step, step_result)
        if rule_directive is not None:
            return rule_directive

        # Rule-based: plan-supplied fallback_url for stuck clicks.
        # Deterministic — fires BEFORE the frontier-model capability
        # so a plan author who knows the structural alternative can
        # short-circuit the Claude consultation entirely.
        fallback_directive = self._maybe_use_fallback_url(state, step, step_result)
        if fallback_directive is not None:
            return fallback_directive

        # Rule-based: DOM-derived href for row-link clicks Holo3
        # can't ground visually. Same shape as fallback_url but the
        # destination URL comes from CDP querying the live DOM
        # instead of a plan-author hint — useful when the row's <id>
        # isn't predictable at plan time (the v30 DECOMPOSE_PROMPT
        # explicitly skips fallback_url for those cases).
        row_link_directive = self._maybe_use_row_link_dom_href(
            state, step, step_result,
        )
        if row_link_directive is not None:
            return row_link_directive

        # Frontier-model: persistent rewrite-triggering failure → ask Claude.
        return self._maybe_frontier_recover_persistent_failure(
            plan, state, step, step_result,
        )

    # ── Capabilities ────────────────────────────────────────────────────

    def _maybe_recover_navigate_back(
        self,
        step: MicroIntent,
        step_result: "StepResult",
    ) -> Optional[InsertStep]:
        """When ``navigate_back`` exhausts the brain budget, the
        browser is stuck on the wrong URL. Insert a direct
        ``navigate`` to the results base URL — agents don't need
        the back button if we can name the destination.

        Generic: the trigger is ``step.type == "navigate_back"`` AND
        ``failure_class == "brain_loop_exhausted"`` AND the runner
        has a non-empty ``_results_base_url``. Any plan with a
        navigate_back step on any site benefits.
        """
        if step.type != "navigate_back":
            return None
        if (step_result.failure_class or "") != "brain_loop_exhausted":
            return None
        base_url = getattr(self.runner, "_results_base_url", "") or ""
        if not base_url:
            # No base URL recorded — without a destination, we can't
            # propose an alternative. Fall through to whatever the
            # recovery policy decided.
            return None
        # Reasoning-trace event so viewer overlays see the
        # deterministic recovery alongside the frontier-critic events.
        from . import reasoning_trace as _trace
        _trace.record(
            self.runner, layer="critic-navigate-back-recovery", kind="fire",
            summary=f"inserting direct navigate to {base_url[:80]}",
            base_url=base_url[:200],
        )
        return InsertStep(
            intent=f"Navigate to {base_url}",
            step_type="navigate",
            reason=(
                "navigate_back hit brain_loop_exhausted — replacing "
                "back-button recovery with a direct navigate to the "
                "results base URL"
            ),
            params={"url": base_url},
        )


    # ── Rule-based capability — plan-supplied fallback_url ──────────────

    def _maybe_use_fallback_url(
        self,
        state: "RunState",
        step: MicroIntent,
        step_result: "StepResult",
    ) -> Optional[Directive]:
        """When a click / submit step accumulates 2+ rewrite-triggering
        failures AND the plan author supplied ``hints.fallback_url``,
        replace the step with a direct ``navigate`` to that URL.

        Deterministic — no Claude call. Fires BEFORE the frontier
        capability so a plan that ALREADY names the structural
        alternative skips the cost + non-determinism of asking Claude.

        Why this is the highest-leverage rule for sites where vision
        keeps misclicking small targets: the plan author often knows
        the URL pattern that achieves the same post-state. ``Click
        Contacted in sidebar`` → ``hints.fallback_url:
        "/leads?status=Contacted"``. After two demotes the runner
        navigates directly.

        Trigger:

        * ``step.type`` in ``{"submit", "click"}``
        * ``step_result.failure_class`` is rewrite-triggering
          (``no_state_change`` / ``wrong_target`` / ``brain_loop_exhausted``)
        * ``step.hints.fallback_url`` is non-empty
        * ``_step_failure_history[step_index]`` has at least 2 entries
          (so we let the cheap retry path try first)
        """
        from . import intent_rewriter
        step_type = str(getattr(step, "type", "") or "")
        if step_type not in ("submit", "click"):
            return None
        failure_class = str(getattr(step_result, "failure_class", "") or "")
        # The deterministic rule is MORE permissive than the frontier
        # capability's class set on purpose: the LLM rewriter is gated
        # narrowly because asking Claude has a cost. This rule has no
        # LLM cost — it's a literal navigate to a URL the plan author
        # wrote. So we also accept ``selector_miss`` (Holo3 / SoM /
        # CSS-target misses) and ``unknown`` (any uncategorised retry
        # failure on a click/submit) — when the plan author named the
        # structural alternative, we use it regardless of failure
        # shape.
        _TRIGGERING = intent_rewriter.REWRITE_TRIGGERING_CLASSES | {
            "selector_miss", "unknown",
        }
        if failure_class not in _TRIGGERING:
            return None
        hints = dict(getattr(step, "hints", {}) or {})
        fallback_url = str(hints.get("fallback_url") or "").strip()
        if not fallback_url:
            return None
        # Plan-author-supplied fallback_url is conventionally written
        # as a path-relative URL ("/leads?status=Active") because the
        # plan doesn't always know the origin. Resolve it against the
        # browser's current page origin so the navigate handler gets
        # a full URL — Modal's navigate dispatcher requires
        # ``http(s)://...`` and fails on bare paths.
        if not fallback_url.startswith(("http://", "https://")):
            from urllib.parse import urljoin, urlparse
            origin_url = ""
            env = getattr(self.runner, "env", None)
            if env is not None:
                origin_url = str(getattr(env, "current_url", "") or "")
            if not origin_url:
                origin_url = str(getattr(self.runner, "_results_base_url", "") or "")
            if not origin_url:
                origin_url = str(getattr(self.runner, "start_url", "") or "")
            if origin_url:
                parsed = urlparse(origin_url)
                if parsed.scheme and parsed.netloc:
                    base = f"{parsed.scheme}://{parsed.netloc}/"
                    fallback_url = urljoin(base, fallback_url.lstrip("/"))
            # If we still don't have a scheme, fall through — let the
            # navigate handler surface its own error. Better than
            # silently no-op on the only recovery path the plan named.
        # Read history under the FAILED step's index, NOT
        # ``state.step_index``. The recovery policy may have advanced
        # ``state.step_index`` past the failed step before this critic
        # runs (canonical case: ``handle_failure`` returns an
        # ``outcome.step_index`` further along, then ``_consult_critic``
        # fires); reading under the advanced index misses the history
        # the demote / centralized-failure path recorded against the
        # ACTUAL failed step. Symptom: ``[retry-history] step N`` keeps
        # firing while ``[critic-frontier] step M`` reports
        # ``failure_count=0`` forever (N != M).
        history_key = int(getattr(step_result, "step_index", state.step_index))
        history = (
            self.runner._step_failure_history.get(history_key, [])
            if hasattr(self.runner, "_step_failure_history") else []
        )
        if not isinstance(history, list) or len(history) < 2:
            return None
        logger.warning(
            "  [critic] step %d: using plan-supplied fallback_url=%r "
            "(after %d prior failures of class %s)",
            history_key, fallback_url, len(history), failure_class,
        )
        # Reasoning-trace event so viewer overlays see the
        # deterministic replacement alongside the frontier-critic
        # events the Claude-based capability emits.
        from . import reasoning_trace as _trace
        _trace.record(
            self.runner, layer="critic-fallback-url", kind="fire",
            summary=(
                f"replaced step with navigate to {fallback_url[:80]} "
                f"(after {len(history)} {failure_class} failures)"
            ),
            step_index=int(state.step_index),
            failure_class=failure_class,
            failure_count=len(history),
            fallback_url=fallback_url[:200],
        )
        return ReplaceStep(
            intent=f"Navigate to {fallback_url} (fallback for stuck click)",
            step_type="navigate",
            params={"url": fallback_url},
            reason=(
                f"plan-supplied fallback_url after {len(history)} prior "
                f"{failure_class} failures — deterministic rule, no "
                f"Claude consultation needed"
            ),
        )

    # ── Rule-based capability — DOM-derived row-link href ────────────────

    def _maybe_use_row_link_dom_href(
        self,
        state: "RunState",
        step: MicroIntent,
        step_result: "StepResult",
    ) -> Optional[Directive]:
        """For row-link click/submit steps that Holo3 can't ground
        visually, extract the row's first matching ``<a href>`` from
        the live DOM via CDP and promote the step to a direct
        ``navigate``.

        Holo3 + SoM grounding consistently fails on small table-row
        hyperlinks (returns 0,0 coords; claude-director substitutes
        arbitrary coords; per-step recovery budget exhausts before
        a click ever lands on the link). The DOM, however, knows
        exactly where each row link points — one ``querySelectorAll``
        away. This capability extracts that href and converts the
        step to a navigate, parallel to the plan-supplied
        :meth:`_maybe_use_fallback_url` rule but with the destination
        computed at runtime from the visible table rather than
        provided up-front by the plan author.

        Why the plan can't pre-supply this: the v30 DECOMPOSE_PROMPT
        explicitly tells Opus to skip ``fallback_url`` on clicks for
        dynamic data (record rows, comment replies, etc.) because the
        target ``<id>`` isn't predictable at decompose time. That's
        correct — but it leaves a gap when Holo3 can't visually
        ground the row. This rule closes that gap.

        Trigger:

        * ``step.type`` in ``{"submit", "click"}``
        * ``params.kind == "row_link"`` — the plan explicitly tagged
          this click as a record-row-link click
          (DECOMPOSE_PROMPT's ``kind="row_link"`` taxonomy)
        * ``failure_class`` is target-identification family
          (``selector_miss`` / ``unknown`` / ``wrong_target``)
        * ``hints.expect_url_contains`` is non-empty — provides the
          URL pattern the destination must contain
        * ``runner.env`` exposes ``cdp_evaluate``

        Note: unlike :meth:`_maybe_use_fallback_url`, this rule does
        NOT gate on ``_step_failure_history`` length. The brain-grounded
        loop's internal SoM-click failures don't propagate into the
        per-step failure history (live observation: run
        20260518_171310_4b792be2 — the critic was invoked on a failed
        step 8 with ``history=0``). For a step the plan has explicitly
        tagged as a row-link click and supplied a URL-pattern hint
        for, "the visual loop failed once" is signal enough — there's
        no value in waiting for a second observe_step call when the
        intermediate retry just burns more budget.
        """
        # Unconditional entry diagnostic — fires every time the
        # method is invoked, regardless of whether any gate passes.
        # If this log line is absent from a Modal run after step
        # 8 halts, the critic chain isn't reaching this method at
        # all (deploy not picked up by the executor container, or
        # wiring removed). With this line present, the subsequent
        # diagnostic skip-lines reveal which gate fails.
        logger.warning(
            "  [critic-row-link] step %d: ENTER — step.type=%r, "
            "params=%s, hints=%s, failure_class=%r",
            state.step_index,
            getattr(step, "type", None),
            list((getattr(step, "params", {}) or {}).keys()),
            list((getattr(step, "hints", {}) or {}).keys()),
            getattr(step_result, "failure_class", None),
        )

        step_type = str(getattr(step, "type", "") or "")
        if step_type not in ("submit", "click"):
            logger.debug(
                "  [critic-row-link] step %d: skip — step.type=%r not in {submit, click}",
                state.step_index, step_type,
            )
            return None
        params = dict(getattr(step, "params", {}) or {})
        if params.get("kind") != "row_link":
            logger.warning(
                "  [critic-row-link] step %d: skip — params.kind=%r != 'row_link' (params=%s)",
                state.step_index, params.get("kind"), list(params.keys()),
            )
            return None
        failure_class = str(getattr(step_result, "failure_class", "") or "")
        # Include ``no_state_change`` — the canonical "click ok=True but
        # page didn't transition" class. This is the shape produced when
        # a row-link <a> is clicked but the click-event chain doesn't
        # actually navigate (broken onclick interceptor, JS-routed SPA,
        # or vision-grounding hitting a row cell that's not the
        # navigable anchor). Live observation from run
        # 20260518_181523_7bce4437: failure_class='no_state_change' on
        # step 8 with kind=row_link.
        if failure_class not in {
            "selector_miss", "unknown", "wrong_target", "no_state_change",
        }:
            logger.warning(
                "  [critic-row-link] step %d: skip — failure_class=%r not in target-id family",
                state.step_index, failure_class,
            )
            return None
        hints = dict(getattr(step, "hints", {}) or {})
        patterns = list(hints.get("expect_url_contains") or [])
        if not patterns:
            logger.warning(
                "  [critic-row-link] step %d: skip — hints.expect_url_contains is empty",
                state.step_index,
            )
            return None
        env = getattr(self.runner, "env", None)
        if env is None:
            logger.warning("  [critic-row-link] step %d: skip — runner.env is None", state.step_index)
            return None
        eval_fn = getattr(env, "cdp_evaluate", None)
        if not callable(eval_fn):
            logger.warning(
                "  [critic-row-link] step %d: skip — env has no cdp_evaluate (env type=%s)",
                state.step_index, type(env).__name__,
            )
            return None
        logger.warning(
            "  [critic-row-link] step %d: entering DOM lookup (patterns=%s, failure_class=%s)",
            state.step_index, patterns, failure_class,
        )

        import json as _json
        pattern_js = _json.dumps([str(p) for p in patterns])
        js = (
            "(function() {"
            f"  var patterns = {pattern_js};"
            "  var rows = document.querySelectorAll('tbody tr, table tr');"
            "  for (var i = 0; i < rows.length; i++) {"
            "    var links = rows[i].querySelectorAll('a[href]');"
            "    for (var j = 0; j < links.length; j++) {"
            "      var h = links[j].href;"
            "      for (var k = 0; k < patterns.length; k++) {"
            "        if (h.indexOf(patterns[k]) !== -1) return h;"
            "      }"
            "    }"
            "  }"
            "  return '';"
            "})()"
        )
        try:
            href = eval_fn(js) or ""
        except Exception as exc:  # pragma: no cover — env-specific
            logger.warning(
                "  [critic] step %d: row-link DOM lookup failed: %s",
                state.step_index, exc,
            )
            return None
        if not isinstance(href, str) or not href.startswith(("http://", "https://")):
            logger.warning(
                "  [critic-row-link] step %d: skip — DOM lookup returned no matching href "
                "(returned=%r, patterns=%s)",
                state.step_index, str(href)[:100], patterns,
            )
            return None

        logger.warning(
            "  [critic-row-link] step %d: FIRE — using DOM-derived row href=%r "
            "(failure_class=%s; patterns=%s)",
            state.step_index, href, failure_class, patterns,
        )
        from . import reasoning_trace as _trace
        _trace.record(
            self.runner, layer="critic-row-link-dom-href", kind="fire",
            summary=(
                f"replaced row-link click with navigate to {href[:80]} "
                f"(failure_class={failure_class})"
            ),
            step_index=int(state.step_index),
            failure_class=failure_class,
            dom_href=href[:200],
            patterns=patterns,
        )
        return ReplaceStep(
            intent=f"Navigate to {href} (DOM-derived row-link fallback)",
            step_type="navigate",
            params={"url": href},
            reason=(
                f"DOM-derived row href={href[:80]} for {step_type} kind=row_link "
                f"on {failure_class} failure — no Claude consultation needed"
            ),
        )

    # ── Frontier-model capability (#435 item 8) ─────────────────────────

    def _maybe_frontier_recover_persistent_failure(
        self,
        plan: Any,
        state: "RunState",
        step: MicroIntent,
        step_result: "StepResult",
    ) -> Optional[Directive]:
        """Ask the frontier model (Claude via :mod:`agentic_recovery`)
        for a mid-run plan change when a rewrite-triggering failure
        pattern persists on the same step.

        Trigger conditions (all generic — no plan / URL / domain
        content reads in):

        * ``MANTIS_CRITIC_FRONTIER=enabled`` (opt-in).
        * ``step_result.failure_class`` is in
          :data:`intent_rewriter.REWRITE_TRIGGERING_CLASSES`
          (``wrong_target`` / ``no_state_change`` /
          ``brain_loop_exhausted``).
        * At least :data:`_FRONTIER_PERSISTENT_FAILURE_THRESHOLD`
          rewrite-triggering records in ``_step_failure_history[step_index]``
          (skip the first miss — that's still in the cheap retry zone).
        * The frontier model hasn't already been consulted for this
          step (tracked on ``runner._critic_frontier_fired_steps``).
        * The shared recovery budget (per-step + per-run, from
          :mod:`agentic_recovery`) isn't exhausted.

        The gate covers all three rewrite-triggering classes (not
        just ``wrong_target``) because the staff-crm sidebar cascade
        produces ``no_state_change`` failures even when SoM clicks
        resolve to real ``<a>`` elements — the page doesn't navigate
        so snapshot-diff fires first and stamps no_state_change
        before the URL postcondition can stamp wrong_target. A
        wrong_target-only gate would never trigger on that pattern.

        Mode mapping:

        * ``add_hint`` — append to ``_recovery_hints[step_index]`` and
          return ``None`` (no directive — the next retry's prompt
          carries the hint and we don't disturb the plan).
        * ``edit_step`` — :class:`ReplaceStep` with the model's
          ``intent`` / ``type`` / ``params``. Missing fields preserve
          the original step's values.
        * ``insert_steps`` — first inserted step becomes an
          :class:`InsertStep` directive (MVP simplification — multi-
          step inserts come from the terminal-failure path that
          already supports splicing).
        * ``halt`` — return ``None`` so the recovery loop's terminal
          path handles it.

        On any fallback (no API key, API error, decision schema
        violation, exception) the method returns ``None`` and the
        existing recovery flow continues unchanged.
        """
        if not _frontier_enabled():
            return None
        from . import intent_rewriter
        failure_class = str(getattr(step_result, "failure_class", "") or "")
        runner = self.runner
        # Use the FAILED step's index, not ``state.step_index``. The
        # recovery policy may have advanced state.step_index past the
        # failed step before this critic runs, so reading
        # ``_step_failure_history[state.step_index]`` misses the
        # records the demote / centralized-failure paths recorded
        # under the actual failed step. The boattrader run
        # ``20260522_080738_ac8962a8`` surfaced this: every iteration
        # logged ``[retry-history] step 2`` (correct, failed_step=2)
        # alongside ``[critic-frontier] step 7`` ``failure_count=0``
        # — different keys, history lookup missed forever, escalation
        # never fired, time_cap halt with 0 leads.
        step_index = int(getattr(step_result, "step_index", state.step_index))

        # Diagnostic: surface every gate decision at WARNING so the
        # trace shows which guard closed the door. This makes the
        # "critic never fired" failure mode visible from Modal logs
        # alone (Modal suppresses INFO+DEBUG).
        if failure_class not in intent_rewriter.REWRITE_TRIGGERING_CLASSES:
            logger.warning(
                "  [critic-frontier] step %d: skipped — failure_class=%r "
                "not in REWRITE_TRIGGERING_CLASSES",
                step_index, failure_class,
            )
            return None

        history = (
            runner._step_failure_history.get(step_index, [])
            if hasattr(runner, "_step_failure_history") else []
        )
        failure_count = sum(
            1 for r in history
            if isinstance(r, dict)
            and str(r.get("kind") or "") in intent_rewriter.REWRITE_TRIGGERING_CLASSES
        )
        if failure_count < _FRONTIER_PERSISTENT_FAILURE_THRESHOLD:
            logger.warning(
                "  [critic-frontier] step %d: skipped — failure_count=%d "
                "below threshold %d",
                step_index, failure_count, _FRONTIER_PERSISTENT_FAILURE_THRESHOLD,
            )
            return None

        # Growth-gated re-fire: track the failure_count at last fire
        # and allow re-firing only after THRESHOLD more failures have
        # accumulated. The prior "fire-once-ever" guard was too strict
        # for loop-iterated plans (e.g. boattrader scrapes that hit
        # the same plan step on each listing iteration) — Claude got
        # one shot at the same step_index regardless of how many fresh
        # failures arrived, then was locked out until time_cap. The
        # per-step + per-run budget caps below already bound total
        # Claude cost, so the binary fire-once was belt-and-suspenders
        # at the cost of recovery opportunity.
        fired = getattr(runner, "_critic_frontier_fired_steps", None)
        if not isinstance(fired, set):
            fired = set()
            runner._critic_frontier_fired_steps = fired
        last_fire_counts = getattr(
            runner, "_critic_frontier_last_fire_counts", None,
        )
        if not isinstance(last_fire_counts, dict):
            last_fire_counts = {}
            runner._critic_frontier_last_fire_counts = last_fire_counts
        if step_index in fired:
            prev_count = int(last_fire_counts.get(step_index, 0))
            new_failures = failure_count - prev_count
            if new_failures < _FRONTIER_PERSISTENT_FAILURE_THRESHOLD:
                logger.warning(
                    "  [critic-frontier] step %d: skipped — only %d new "
                    "failures since last fire (count=%d, last_fire=%d); "
                    "threshold=%d",
                    step_index, new_failures, failure_count, prev_count,
                    _FRONTIER_PERSISTENT_FAILURE_THRESHOLD,
                )
                return None

        # Reuse the existing recovery budget pool so the critic's
        # frontier call and step_recovery's terminal call share one
        # pot — keeps the total Claude spend bounded by the same
        # per-step / per-run caps.
        per_step_dict = getattr(runner, "_recovery_attempts_per_step", None)
        total_attempts = getattr(runner, "_total_recovery_attempts", None)
        if not isinstance(per_step_dict, dict) or not isinstance(total_attempts, int):
            logger.warning(
                "  [critic-frontier] step %d: skipped — runner missing budget "
                "trackers (per_step=%s, total=%s)",
                step_index, type(per_step_dict).__name__, type(total_attempts).__name__,
            )
            return None
        # #567: per-run override via runtime fields; fallback to DEFAULT_*.
        try:
            from ..agentic_recovery import effective_max_recoveries
        except Exception as exc:  # noqa: BLE001 — never break runs
            logger.warning(
                "  [critic-frontier] step %d: skipped — agentic_recovery "
                "import failed: %s", step_index, exc,
            )
            return None
        max_per_step, max_per_run = effective_max_recoveries(self.runner)
        if per_step_dict.get(step_index, 0) >= max_per_step:
            logger.warning(
                "  [critic-frontier] step %d: skipped — per-step budget "
                "exhausted (%d/%d) before critic could fire",
                step_index, per_step_dict.get(step_index, 0), max_per_step,
            )
            return None
        if total_attempts >= max_per_run:
            logger.warning(
                "  [critic-frontier] step %d: skipped — per-run budget "
                "exhausted (%d/%d)",
                step_index, total_attempts, max_per_run,
            )
            return None
        # All gates passed — Claude call below logs result.
        logger.warning(
            "  [critic-frontier] step %d: gate passed (failure_class=%s, "
            "failures=%d) — calling analyse_failure_and_recover",
            step_index, failure_class, failure_count,
        )
        from . import reasoning_trace as _trace
        _trace.record(
            runner, layer="critic-frontier", kind="fire",
            summary=f"gate passed on {failure_class} (×{failure_count})",
            step_index=step_index,
            failure_class=failure_class, failure_count=failure_count,
        )

        # Capture the post-failure screenshot for Claude's analysis.
        # Same shape ``step_recovery._try_agentic_recovery`` uses.
        env = getattr(runner, "env", None)
        screenshot = None
        if env is not None and hasattr(env, "screenshot"):
            try:
                screenshot = env.screenshot()
            except Exception as exc:  # noqa: BLE001
                logger.debug("  [critic-frontier] env.screenshot failed: %s", exc)

        plan_context = [
            f"step {i}: type={s.type}, intent={s.intent[:60]}"
            for i, s in enumerate(getattr(plan, "steps", []) or [])
        ]
        failure_data = (
            f"failure_class={failure_class}; "
            f"prior_failures={failure_count}; "
            f"data={(step_result.data or '')[:160]}"
        )

        # H8: surface accumulated hints so Claude sees when add_hint
        # loops are already burning attempts. The runner stores hints
        # in ``_recovery_hints[step_index]``; the prompt's HINT-LOOP
        # DETECTION section reads from PRIOR HINTS TEXT directly.
        prior_hints: list[str] = []
        hint_map = getattr(runner, "_recovery_hints", None)
        if isinstance(hint_map, dict):
            stored = hint_map.get(step_index, [])
            if isinstance(stored, list):
                prior_hints = [str(h) for h in stored if str(h).strip()]
        try:
            from ..agentic_recovery import analyse_failure_and_recover
            decision = analyse_failure_and_recover(
                step=step,
                failure_data=failure_data,
                screenshot=screenshot,
                plan_context=plan_context,
                attempts=failure_count,
                prior_hints=prior_hints,
            )
        except Exception as exc:  # noqa: BLE001 — never break runs
            logger.warning("  [critic-frontier] Claude call raised: %s", exc)
            return None

        # Mark fired BEFORE applying — even if Claude returned halt /
        # invalid, we don't want to retry the consultation on the same
        # step (the policy budget exists for that). The companion
        # ``last_fire_counts`` dict gates re-fires on growth — see the
        # guard above.
        fired.add(step_index)
        last_fire_counts[step_index] = failure_count

        if decision is None:
            # Claude call failed (no key / API error / parse fallback).
            # Don't burn the shared recovery budget — the terminal
            # path may yet succeed on its own call. Mark fired so we
            # don't retry, but keep budget intact for downstream.
            logger.warning(
                "  [critic-frontier] step %d: Claude returned no decision "
                "(API/key/parse fallback) — leaving recovery budget intact "
                "for the terminal path",
                step_index,
            )
            _trace.record(
                runner, layer="critic-frontier", kind="result",
                summary="Claude returned no decision (API fallback)",
                step_index=step_index, decision_mode="none",
            )
            return None

        # Real decision received — count this as a recovery consultation
        # against the shared per-step / per-run budget so the terminal
        # path doesn't double-spend.
        per_step_dict[step_index] = per_step_dict.get(step_index, 0) + 1
        runner._total_recovery_attempts = total_attempts + 1
        logger.warning(
            "  [critic-frontier] step %d: consumed recovery budget "
            "(per_step=%d/%d, per_run=%d/%d) — decision.mode=%s",
            step_index, per_step_dict[step_index], max_per_step,
            runner._total_recovery_attempts, max_per_run, decision.mode,
        )
        if decision.mode == "halt":
            logger.warning(
                "  [critic-frontier] step %d: halt — Claude says no plan "
                "tweak helps (%s)",
                step_index, decision.reasoning[:120],
            )
            _trace.record(
                runner, layer="critic-frontier", kind="result",
                summary=f"halt: {decision.reasoning[:120]}",
                step_index=step_index, decision_mode="halt",
                reasoning=decision.reasoning[:300],
            )
            return None
        if decision.mode == "add_hint" and decision.hint:
            from . import recovery_hints
            recovery_hints.add_hint(runner, step_index, decision.hint)
            logger.warning(
                "  [critic-frontier] step %d: add_hint — %s",
                step_index, decision.hint[:120],
            )
            _trace.record(
                runner, layer="critic-frontier", kind="result",
                summary=f"add_hint: {decision.hint[:120]}",
                step_index=step_index, decision_mode="add_hint",
                hint=decision.hint[:300],
                reasoning=decision.reasoning[:300],
            )
            return None
        if decision.mode == "edit_step":
            edited = decision.edited_step or {}
            new_intent = (edited.get("intent") or step.intent or "").strip()
            new_type = (edited.get("type") or step.type or "").strip()
            new_params = dict(edited.get("params") or step.params or {})
            if not new_intent or not new_type:
                return None
            logger.warning(
                "  [critic-frontier] step %d: edit_step → ReplaceStep "
                "(type=%s, intent=%s)",
                step_index, new_type, new_intent[:80],
            )
            _trace.record(
                runner, layer="critic-frontier", kind="result",
                summary=f"edit_step → ReplaceStep (type={new_type})",
                step_index=step_index, decision_mode="edit_step",
                new_intent=new_intent[:200], new_type=new_type,
                reasoning=decision.reasoning[:300],
            )
            return ReplaceStep(
                intent=new_intent,
                step_type=new_type,
                params=new_params,
                hints=dict(getattr(step, "hints", {}) or {}),
                reason=(
                    f"frontier critic replaced step on persistent "
                    f"{failure_class} ({failure_count} prior): "
                    f"{decision.reasoning[:120]}"
                ),
            )
        if decision.mode == "insert_steps":
            steps = decision.inserted_steps or []
            if not steps:
                return None
            first = steps[0]
            intent = str(first.get("intent") or "").strip()
            step_type = str(first.get("type") or "").strip()
            if not intent or not step_type:
                return None
            logger.warning(
                "  [critic-frontier] step %d: insert_steps[0] → InsertStep "
                "(type=%s, intent=%s)",
                step_index, step_type, intent[:80],
            )
            _trace.record(
                runner, layer="critic-frontier", kind="result",
                summary=f"insert_steps[0] → InsertStep (type={step_type})",
                step_index=step_index, decision_mode="insert_steps",
                inserted_intent=intent[:200], inserted_type=step_type,
                reasoning=decision.reasoning[:300],
            )
            return InsertStep(
                intent=intent,
                step_type=step_type,
                params=dict(first.get("params") or {}),
                reason=(
                    f"frontier critic insert on persistent "
                    f"{failure_class} ({failure_count} prior): "
                    f"{decision.reasoning[:120]}"
                ),
            )
        return None


def apply_directive(
    runner: "MicroPlanRunner",
    plan: Any,
    state: "RunState",
    directive: Directive,
) -> None:
    """Mutate ``plan.steps`` per a critic-emitted directive.

    Dispatches on directive type:

    * :class:`InsertStep` — splice a new step at ``state.step_index``
      so it runs on the next loop iteration.
    * :class:`ReplaceStep` — replace the step at ``state.step_index``
      in place (no shift). Resets ``_step_failure_history`` for the
      slot so the new step doesn't inherit the old one's retry
      pressure.

    Stays separate from :class:`ExecutionCritic` so the critic stays
    pure (returns directives, doesn't mutate state) — the runner
    owns plan mutation.
    """
    if isinstance(directive, InsertStep):
        _apply_insert_step(runner, plan, state, directive)
        return
    if isinstance(directive, ReplaceStep):
        _apply_replace_step(runner, plan, state, directive)
        return


def _apply_insert_step(
    runner: "MicroPlanRunner",
    plan: Any,
    state: "RunState",
    directive: InsertStep,
) -> None:
    new_step = MicroIntent(
        intent=directive.intent,
        type=directive.step_type,
        params=dict(directive.params or {}),
        budget=3,
        required=False,
        section="recovery",
    )
    insert_at = max(0, min(state.step_index, len(plan.steps)))
    plan.steps.insert(insert_at, new_step)

    from . import healing_events
    healing_events.record_insert_step(
        runner,
        after_step_index=insert_at - 1,
        inserted_intent=directive.intent,
        inserted_type=directive.step_type,
        reason=directive.reason,
    )
    logger.warning(
        "  [critic] inserting recovery step %d: %s (%s)",
        insert_at, directive.intent[:80], directive.reason[:80],
    )


def _apply_replace_step(
    runner: "MicroPlanRunner",
    plan: Any,
    state: "RunState",
    directive: ReplaceStep,
) -> None:
    idx = int(state.step_index)
    if idx < 0 or idx >= len(plan.steps):
        return
    original = plan.steps[idx]
    original_type = str(getattr(original, "type", "") or "")
    new_step = MicroIntent(
        intent=directive.intent,
        type=directive.step_type,
        params=dict(directive.params or {}),
        hints=dict(directive.hints or {}),
        budget=int(getattr(original, "budget", 3) or 3),
        required=bool(getattr(original, "required", False)),
        section=str(getattr(original, "section", "") or ""),
        gate=bool(getattr(original, "gate", False)),
    )
    plan.steps[idx] = new_step

    # Reset the per-step retry history so the replacement runs with a
    # clean slate. Inheriting the original step's failure pattern
    # would defeat the point of replacing the step.
    if hasattr(runner, "_step_failure_history"):
        try:
            runner._step_failure_history.pop(idx, None)
        except Exception:  # noqa: BLE001
            pass
    # Same for the recovery hint accumulator — the new step likely
    # has a different label / shape and the old hints don't apply.
    if hasattr(runner, "_recovery_hints"):
        try:
            runner._recovery_hints.pop(idx, None)
        except Exception:  # noqa: BLE001
            pass

    from . import healing_events
    healing_events.record_replace_step(
        runner,
        step_index=idx,
        original_type=original_type,
        new_intent=directive.intent,
        new_type=directive.step_type,
        reason=directive.reason,
    )
    logger.warning(
        "  [critic] replacing step %d in place: %s → %s (%s)",
        idx, original_type or "?",
        directive.step_type, directive.reason[:120],
    )


__all__ = [
    "ExecutionCritic",
    "InsertStep",
    "ReplaceStep",
    "Directive",
    "apply_directive",
]
