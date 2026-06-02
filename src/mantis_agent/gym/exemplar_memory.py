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


def apply_exemplar_overlay(
    plan: object,
    exemplars: list[dict[str, Any]],
    *,
    plan_signature: str = "",
) -> int:
    """Pre-flight: stamp ``exemplar_replay`` on each step a positive
    exemplar matches.

    Mutates ``plan.steps[i].hints`` in place. A step matches the exemplar
    with the same ``type`` and the largest intent token-overlap (a
    non-empty overlap is required — :func:`_tokens` drops stopwords so the
    overlap reflects real sub-goal nouns, not shared filler). Author-set
    ``exemplar_replay`` hints are never overwritten — the store is a
    fallback, not an override, exactly as in :func:`apply_hint_overlay`.

    Returns the count of steps stamped.
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

        action = _action_label(best.get("last_action"))
        outcome = str(best.get("observed_outcome") or "").strip()
        if outcome:
            replay = f"a {action} produced '{outcome}'"
        else:
            replay = f"a {action} succeeded on this sub-goal"
        hints["exemplar_replay"] = replay
        source_run = str(best.get("source_run") or "").strip()
        if source_run:
            hints["exemplar_source_run"] = source_run
        step.hints = hints
        applied += 1
        logger.warning(
            "  [exemplar-overlay] step type=%s intent=%r ← replay=%r (src=%s, overlap=%d)",
            step_type,
            str(getattr(step, "intent", "") or "")[:60],
            replay[:80],
            source_run or "?",
            best_overlap,
        )
    return applied


__all__ = ["apply_exemplar_overlay"]
