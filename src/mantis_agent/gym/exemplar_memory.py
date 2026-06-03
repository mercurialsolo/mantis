"""S1 injection seam — stamp a worked-step exemplar onto a plan.

S0's :func:`mantis_agent.gym.hint_memory.apply_hint_overlay` injects a
grounding *anchor* (where to click). This is the S1 parallel: it injects
a worked *procedure* — the action that succeeded on this sub-goal before
and the outcome it produced — so the brain pattern-matches against a
known-good example instead of re-deriving the policy.

The two rungs are deliberately different signals, which is what lets the
Learning Allocator tell them apart on different failure clusters:

* **S0 (retrieval)** answers *where* — a visual anchor for a hard-to-
  locate target (the ``knowledge`` cluster, BT02's buried Caterpillar).
* **S1 (exemplar)** answers *what worked* — the action→outcome procedure
  for a non-obvious policy (the ``policy`` cluster, BT03's by-owner phone
  reveal).

CUA purity (``feedback_cua_no_dom_access``): the exemplar's recorded
``last_action`` carries the old screen coordinate, but the overlay never
injects it. A coordinate is position-specific — replaying it would defeat
the whole point of an exemplar that has to survive layout drift. We inject
only the *action type* and the *outcome*; the brain re-grounds the target
from the current screenshot by sight (the holo3 prompt says so explicitly).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Tokens too generic to carry intent signal — matching on them alone would
# stamp the wrong step (every plan step "clicks" or "applies" something).
_STOPWORDS = frozenset({
    "the", "a", "an", "to", "on", "of", "in", "and", "or", "for", "with",
    "click", "open", "go", "this", "that", "it", "is", "be", "at", "by",
})


def _tokens(text: object) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall(str(text or "").lower())
        if len(t) > 1 and t not in _STOPWORDS
    }


def _action_label(last_action: object) -> str:
    """Human-readable action verb from a recorded ``last_action``.

    Never returns coordinates — only the action *type* (``click``,
    ``type``, ``scroll`` …). Defends against the three shapes the trace
    exporter emits: a dict, a bare string, or ``None``.
    """
    if isinstance(last_action, dict):
        at = str(last_action.get("action_type") or "").strip()
        return at or "an action"
    if isinstance(last_action, str) and last_action.strip():
        # A bare string could itself carry a coordinate; keep only the
        # leading verb token to stay coordinate-free.
        head = last_action.strip().split()[0]
        return head if head.isalpha() else "an action"
    return "an action"


def _replay_strings(ex: dict[str, Any]) -> tuple[str, str]:
    """The ``(replay, source_run)`` pair an exemplar contributes — identical
    formatting whether the exemplar NUDGES an existing step or is INJECTED as a
    new one. Coordinate-free by construction (see :func:`_action_label`)."""
    action = _action_label(ex.get("last_action"))
    outcome = str(ex.get("observed_outcome") or "").strip()
    if outcome:
        replay = f"a {action} produced '{outcome}'"
    else:
        replay = f"a {action} succeeded on this sub-goal"
    source_run = str(ex.get("source_run") or "").strip()
    return replay, source_run


def _apply_nudges(plan: object, exemplars: list[dict[str, Any]]) -> int:
    """Stamp ``exemplar_replay`` onto each EXISTING step a nudge exemplar
    matches (same ``type`` + largest non-empty intent token-overlap). This is
    the per-step hint frozen and S1 share a plan for; only S1 gets the nudge.
    Author-set ``exemplar_replay`` hints are never overwritten.
    """
    if not exemplars:
        return 0

    applied = 0
    for step in getattr(plan, "steps", []) or []:
        step_type = str(getattr(step, "type", "") or "")
        step_tokens = _tokens(getattr(step, "intent", ""))
        if not step_tokens:
            continue

        best: dict[str, Any] | None = None
        best_overlap = 0
        for ex in exemplars:
            if str(ex.get("type", "") or "") != step_type:
                continue
            overlap = len(step_tokens & _tokens(ex.get("intent", "")))
            if overlap > best_overlap:
                best_overlap = overlap
                best = ex
        if best is None or best_overlap < 1:
            continue

        hints = dict(getattr(step, "hints", None) or {})
        if "exemplar_replay" in hints:
            continue  # author/operator wins

        replay, source_run = _replay_strings(best)
        hints["exemplar_replay"] = replay
        if source_run:
            hints["exemplar_source_run"] = source_run
        step.hints = hints
        applied += 1
        logger.warning(
            "  [exemplar-overlay] nudge step type=%s intent=%r ← replay=%r (src=%s, overlap=%d)",
            step_type,
            str(getattr(step, "intent", "") or "")[:60],
            replay[:80],
            source_run or "?",
            best_overlap,
        )
    return applied


def _apply_injections(plan: object, exemplars: list[dict[str, Any]]) -> int:
    """Insert each inject exemplar as a NEW required step immediately before the
    first step whose intent token-overlaps the exemplar's ``inject_before``
    successor. This is the lever that lets S1 supply a missing non-obvious
    prerequisite the base plan omits — frozen runs plan-as-is and skips it.

    Skips (with a warning) any exemplar whose successor isn't in the plan: with
    no anchor there's nowhere to position the step, and a free-floating insert
    would reorder the flow unpredictably. The new step is built by cloning the
    anchor's class (duck-typed ``type(anchor)(...)``) so it works for both the
    real ``MicroIntent`` dataclass and test stubs.
    """
    if not exemplars:
        return 0
    steps = getattr(plan, "steps", None)
    if steps is None:
        return 0

    injected = 0
    for ex in exemplars:
        successor_tokens = _tokens(ex.get("inject_before", ""))
        anchor_idx = None
        for i, step in enumerate(steps):
            if _tokens(getattr(step, "intent", "")) & successor_tokens:
                anchor_idx = i
                break
        if anchor_idx is None:
            logger.warning(
                "  [exemplar-overlay] inject skipped — no step matches inject_before=%r (src=%s)",
                str(ex.get("inject_before") or "")[:60],
                str(ex.get("source_run") or "?"),
            )
            continue

        replay, source_run = _replay_strings(ex)
        # The injected step is a hand-authored "missing step" spec: it may carry
        # the executable knobs a fresh step needs (params, grounding, extra hints).
        # The procedural ``exemplar_replay`` is stamped on top — ``setdefault`` so
        # an author-provided replay wins, mirroring the nudge path.
        hints: dict[str, Any] = dict(ex.get("hints") or {})
        hints.setdefault("exemplar_replay", replay)
        if source_run:
            hints.setdefault("exemplar_source_run", source_run)

        anchor = steps[anchor_idx]
        new_step = type(anchor)(
            intent=str(ex.get("intent", "") or ""),
            type=str(ex.get("type", "") or ""),
            params=dict(ex.get("params") or {}),
            hints=hints,
            required=bool(ex.get("required", True)),
            grounding=bool(ex.get("grounding", True)),
        )
        steps.insert(anchor_idx, new_step)
        # Loop steps jump via ABSOLUTE indices (``loop_target``). The insert
        # pushed every step at/after ``anchor_idx`` one slot later, so any
        # loop_target pointing there must follow it — otherwise an exemplar
        # injected inside a loop body silently rewinds to the wrong step.
        # Mirrors agentic_recovery.splice_inserted_steps.
        for s in steps:
            lt = getattr(s, "loop_target", -1)
            if lt is not None and lt >= anchor_idx:
                try:
                    s.loop_target = lt + 1
                except AttributeError:
                    pass
        injected += 1
        logger.warning(
            "  [exemplar-overlay] inject step type=%s intent=%r BEFORE %r ← replay=%r (src=%s)",
            str(ex.get("type", "") or ""),
            str(ex.get("intent", "") or "")[:60],
            str(getattr(anchor, "intent", "") or "")[:40],
            replay[:80],
            source_run or "?",
        )
    return injected


def apply_exemplar_overlay(
    plan: object,
    exemplars: list[dict[str, Any]],
    *,
    plan_signature: str = "",
) -> int:
    """Overlay worked-step exemplars onto a plan: NUDGE matching steps and
    INJECT unmatched know-how as new steps.

    Two exemplar shapes, distinguished by the ``inject_before`` key:

    * **Nudge** (no ``inject_before``): stamp ``exemplar_replay`` on the
      existing step with the same ``type`` and the largest intent
      token-overlap. The frozen and S1 arms share a decomposed plan; only S1
      runs this overlay, so the nudge is S1's per-step lever. See
      :func:`_apply_nudges`.

    * **Inject** (carries ``inject_before``): the worked sub-goal has NO
      matching plan step because the base plan omits it. Insert it as a new
      required step before the first step whose intent token-overlaps
      ``inject_before``. This lets S1 supply a missing non-obvious
      prerequisite that frozen (plan-as-is) skips — a clean binary separator.
      See :func:`_apply_injections`.

    Mutates ``plan.steps`` in place. Returns stamped + injected count.
    """
    if not exemplars:
        return 0

    inject_exemplars = [
        ex for ex in exemplars if str(ex.get("inject_before") or "").strip()
    ]
    nudge_exemplars = [
        ex for ex in exemplars if not str(ex.get("inject_before") or "").strip()
    ]

    # Nudge existing steps first, then inject — so the nudge pass only sees the
    # author's original steps and never re-stamps a freshly injected one.
    applied = _apply_nudges(plan, nudge_exemplars)
    injected = _apply_injections(plan, inject_exemplars)
    return applied + injected


__all__ = ["apply_exemplar_overlay"]
