"""Exploration-mode runtime substrate (issue #248).

Today the production runner — :meth:`MicroPlanRunner.run_with_status` —
is strictly an *execution* mode: it follows the decomposed plan, emits
:class:`StepResult` per intent, halts on failure, and surfaces nothing
about *what could have worked instead*. That's the right behaviour for
production traffic, but the *plan-and-recipe refinement loop*
(picking the right ``spam_indicators``, finding the right sub-goal
phrasing, discovering a site's right-click DOM is unreliable on tile
#N) needs a substrate of its own.

This module is that substrate. The data types here are deliberately
simple so a host-side refinement agent can serialise outcomes for
offline analysis the same way checkpoints serialise step results.

The exploration runtime entrypoint
(:meth:`MicroPlanRunner.run_with_exploration`) takes a list of plan /
recipe variants, runs them sequentially against the runner's env,
enforces a per-variant :class:`ExplorationBudget`, and returns one
:class:`VariantOutcome` per variant.

**v1 scope (this module).**

- Data shapes for experiment events, exploration budget, and variant
  outcomes.
- ``to_dict`` / ``from_dict`` round-trip for everything that gets
  persisted alongside checkpoints.
- Histogram + URL-coverage derivation from ``StepResult`` streams.

**Out of v1 scope (deliberate — see issue #248):**

- Concrete deviation strategies (try-alternative-click, alternative
  sub-goal phrasings, etc). Each strategy will land as a follow-up
  PR that emits :class:`ExperimentEvent` records from inside the
  runner's existing failure paths. v1 ships the substrate so each
  follow-up only has to add the emit-site, not the data type.
- Self-improving runner. The runtime *emits evidence* only; diff
  proposal + approval lives one layer up (a host-side refinement
  agent reads the outcome bundles and proposes recipe / plan diffs).
- Parallel-tab execution. v1 runs variants sequentially.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, ClassVar


# ── Experiment event kinds — closed enumeration ──────────────────────


EXPERIMENT_KINDS: tuple[str, ...] = (
    "action_alternative_tried",   # tried a different click/scroll/type
    "recipe_rejection_observed",  # captured the rejected payload + reason
    "dom_quirk_detected",         # right-click missed, modal blocked input, etc.
    "sub_goal_phrasing_variant",  # decomposer produced different micro-plan
    "navigation_drift",           # ended up on unintended URL
    "extraction_field_coverage",  # which fields the extractor could have read
)


@dataclass
class ExperimentEvent:
    """A recorded deviation / observation surfaced during an exploration
    run.

    The runtime never raises on an unknown ``kind`` (forward-compat: a
    refinement agent on a newer mantis might consume a kind this older
    schema doesn't enumerate). :data:`EXPERIMENT_KINDS` documents the
    canonical set callers should use.

    ``attempted`` / ``outcome`` are free-form dicts — the precise shape
    is per-kind. Refinement agents read them as opaque JSON.
    """

    kind: str
    intent: str
    page_url: str
    attempted: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    timestamp: float = field(default_factory=time.time)

    _PERSISTED: ClassVar[tuple[str, ...]] = (
        "kind", "intent", "page_url", "attempted", "outcome", "cost", "timestamp",
    )

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._PERSISTED}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentEvent:
        allowed = {"kind", "intent", "page_url", "attempted", "outcome", "cost", "timestamp"}
        return cls(**{k: v for k, v in payload.items() if k in allowed})


# ── Per-variant budget ───────────────────────────────────────────────


@dataclass
class ExplorationBudget:
    """Cost + wall-clock cap on a single exploration variant.

    Defaults are tuned for refinement-agent runs on a single
    listing/page: ~3 USD total spend, ~10 minutes wall-clock. The
    runtime checks both before each step and aborts the variant with
    ``terminal_status='budget_exceeded'`` when either is exceeded.
    """

    max_cost_usd: float = 3.0
    max_minutes: float = 10.0


# ── Per-variant outcome bundle ───────────────────────────────────────


@dataclass
class VariantOutcome:
    """Bundle of every signal a refinement agent needs to compare one
    exploration variant against another.

    ``terminal_status`` is one of:
      - ``"completed"``         — variant ran to plan completion
      - ``"halted"``            — variant aborted on a required-step failure
      - ``"budget_exceeded"``   — variant aborted on cost or time cap
      - ``"cancelled"``         — caller-side cancel event fired
    """

    variant_id: str
    terminal_status: str = "completed"
    step_results: list[Any] = field(default_factory=list)  # list[StepResult]
    experiments: list[ExperimentEvent] = field(default_factory=list)
    per_intent_alternative_count: dict[int, int] = field(default_factory=dict)
    recipe_rejection_histogram: dict[str, int] = field(default_factory=dict)
    dom_quirk_summary: list[dict[str, Any]] = field(default_factory=list)
    url_coverage: list[str] = field(default_factory=list)
    cost_total: float = 0.0
    wall_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form. ``step_results`` round-trip via
        ``StepResult.to_dict`` / ``from_dict`` to keep this module
        decoupled from the checkpoint dataclass (callers that pickle a
        VariantOutcome through JSON pre-flatten step_results to dicts
        first). ``experiments`` are flattened in-place."""
        return {
            "variant_id": self.variant_id,
            "terminal_status": self.terminal_status,
            "step_results": [
                s.to_dict() if hasattr(s, "to_dict") else dict(s)
                for s in self.step_results
            ],
            "experiments": [e.to_dict() for e in self.experiments],
            "per_intent_alternative_count": dict(self.per_intent_alternative_count),
            "recipe_rejection_histogram": dict(self.recipe_rejection_histogram),
            "dom_quirk_summary": list(self.dom_quirk_summary),
            "url_coverage": list(self.url_coverage),
            "cost_total": self.cost_total,
            "wall_time_s": self.wall_time_s,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VariantOutcome:
        """Inverse of :meth:`to_dict`. ``step_results`` arrive as raw
        dicts — caller rehydrates with :meth:`StepResult.from_dict`
        when typed access is required (matches the checkpoint pattern
        used by :class:`RunCheckpoint`)."""
        return cls(
            variant_id=payload["variant_id"],
            terminal_status=payload.get("terminal_status", "completed"),
            step_results=list(payload.get("step_results", [])),
            experiments=[
                ExperimentEvent.from_dict(e)
                for e in payload.get("experiments", [])
            ],
            per_intent_alternative_count={
                int(k): int(v)
                for k, v in (payload.get("per_intent_alternative_count") or {}).items()
            },
            recipe_rejection_histogram=dict(
                payload.get("recipe_rejection_histogram") or {}
            ),
            dom_quirk_summary=list(payload.get("dom_quirk_summary") or []),
            url_coverage=list(payload.get("url_coverage") or []),
            cost_total=float(payload.get("cost_total", 0.0)),
            wall_time_s=float(payload.get("wall_time_s", 0.0)),
        )


# ── Histogram + coverage helpers ─────────────────────────────────────


def rejection_histogram_from_steps(step_results: list[Any]) -> dict[str, int]:
    """Build a ``{reason: count}`` histogram from a list of StepResult.

    Counts every step whose ``data`` begins with the canonical recipe-
    rejection prefix (``REJECTED_DEALER``, ``REJECTED_INCOMPLETE``,
    …) OR whose ``skip=True`` envelope carries a ``skip_reason``.
    The histogram key is the short token (e.g. ``"dealer"``,
    ``"incomplete"``) extracted from either signal.

    Used by :class:`VariantOutcome` to expose rejection rates without
    forcing the refinement agent to re-parse step ``data`` strings.
    """
    hist: dict[str, int] = {}
    for s in step_results:
        # Prefer the structured skip envelope (issue #246) when present —
        # it's the recipe-author's canonical key.
        skip_reason = getattr(s, "skip_reason", None)
        if skip_reason:
            hist[skip_reason] = hist.get(skip_reason, 0) + 1
            continue
        # Fall back to parsing the ``data`` field's REJECTED_* prefix.
        data = str(getattr(s, "data", "") or "")
        if data.startswith("REJECTED_DEALER"):
            hist["dealer"] = hist.get("dealer", 0) + 1
        elif data.startswith("REJECTED_INCOMPLETE"):
            hist["incomplete"] = hist.get("incomplete", 0) + 1
    return hist


def url_coverage_from_steps(step_results: list[Any]) -> list[str]:
    """Distinct URLs surfaced in step ``data`` (preserves first-seen
    order). Reads the canonical ``URL:<url>`` prefix emitted by the
    ``extract_url`` handler plus any ``REJECTED_*|...`` records that
    embed a URL as their second pipe-delimited segment.

    Refinement agents use this to compare which pages a plan/recipe
    variant *actually visits* — a variant that loops on tile #3 has
    a thin coverage list versus one that exhausts page 1."""
    seen: set[str] = set()
    coverage: list[str] = []
    for s in step_results:
        data = str(getattr(s, "data", "") or "")
        if not data:
            continue
        url = ""
        if data.startswith("URL:"):
            url = data[4:]
        elif data.startswith("REJECTED_"):
            # REJECTED_DEALER|<reason>|<summary-or-url>
            parts = data.split("|", 3)
            if len(parts) >= 3 and parts[2].startswith(("http://", "https://")):
                url = parts[2]
        if url and url not in seen:
            seen.add(url)
            coverage.append(url)
    return coverage


__all__ = [
    "EXPERIMENT_KINDS",
    "ExperimentEvent",
    "ExplorationBudget",
    "VariantOutcome",
    "rejection_histogram_from_steps",
    "url_coverage_from_steps",
]
