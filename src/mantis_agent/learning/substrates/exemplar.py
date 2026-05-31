"""S1 — exemplar replay substrate (the session-durable rung).

One rung up from retrieval: instead of a single grounding anchor, replay
a whole *successful step* the agent took on this plan before — its intent,
the action that worked, the outcome it produced — so the brain can pattern-
match against a known-good example. It reads the trace files
:class:`~mantis_agent.gym.trace_exporter.TraceExporter` already writes and
labels them with the existing
:class:`~mantis_agent.gym.trace_labeller.TraceLabeller`, keeping only the
steps the labeller marks ``positive``.

Why this is S1 (PLAN §3):

* **Cheap** — labelling is pure heuristics over local JSON, no model call,
  so :meth:`cost_estimate` is ``0.0``. It sits above S0 only on the
  *durability* axis, not on cost.
* **Session-durable** — an exemplar persists for the life of the trace
  corpus (a session / deploy), longer than S0's single-task hint but not
  baked into a reusable macro (that's S2) or weights (S3). Hence
  :class:`Durability.SESSION`.

The retrieval is deliberately minimal: group positive steps by
``plan_signature`` and hand back the top-N for the plan about to run. The
trace exporter writes ``plan_signature`` at the top level but the labeller
drops it, so we read the raw JSON once and label it in-memory rather than
going through ``label_trace_file`` (which would re-read and lose the key).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ...gym.trace_labeller import TraceLabeller
from .base import Durability, SubstrateContext, SubstrateResult

logger = logging.getLogger(__name__)


class ExemplarSubstrate:
    """S1 — replay positive-labelled steps from prior runs of the same plan.

    ``trace_dir`` points at the corpus
    :class:`~mantis_agent.gym.trace_exporter.TraceExporter` writes
    (``$MANTIS_TRACE_EXPORT_DIR``); layout is ``<tenant>/<run_id>.json``,
    so the index globs ``**/*.json``. The index is built lazily on first
    :meth:`apply` and cached; call :meth:`refresh` after new traces land
    (e.g. between eval rounds) to rebuild it.
    """

    durability = Durability.SESSION

    def __init__(
        self,
        trace_dir: str | Path,
        *,
        labeller: TraceLabeller | None = None,
        name: str = "S1_exemplar",
        max_exemplars: int = 5,
    ) -> None:
        self.trace_dir = Path(trace_dir)
        self.labeller = labeller or TraceLabeller()
        self.name = name
        self.max_exemplars = max(1, int(max_exemplars))
        # plan_signature -> list of exemplar dicts. None ⇒ not built yet.
        self._index: dict[str, list[dict]] | None = None

    # ── index ───────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Rebuild the exemplar index from the trace corpus on disk."""
        self._index = self._build_index()

    def _ensure_index(self) -> dict[str, list[dict]]:
        if self._index is None:
            self._index = self._build_index()
        return self._index

    def _build_index(self) -> dict[str, list[dict]]:
        index: dict[str, list[dict]] = {}
        if not self.trace_dir.is_dir():
            return index
        for path in sorted(self.trace_dir.glob("**/*.json")):
            if not path.is_file():
                continue
            try:
                raw = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("exemplar: skipping %s (%s)", path, exc)
                continue
            plan_sig = str(raw.get("plan_signature", "") or "")
            if not plan_sig:
                continue
            labelled = self.labeller.label_trace(raw)
            for step in labelled.steps:
                if step.label != "positive":
                    continue
                index.setdefault(plan_sig, []).append({
                    "intent": step.intent,
                    "type": step.type,
                    "last_action": step.last_action,
                    "observed_outcome": step.observed_outcome,
                    "label_reason": step.label_reason,
                    "source_run": labelled.run_id,
                })
        return index

    # ── substrate protocol ──────────────────────────────────────────────

    def cost_estimate(self, context: SubstrateContext) -> float:  # noqa: ARG002
        """Always free — labelling is heuristic, no model call."""
        return 0.0

    def apply(self, context: SubstrateContext) -> SubstrateResult:
        plan_signature = str(context.extras.get("plan_signature", "") or "")
        if not plan_signature:
            return SubstrateResult(
                substrate=self.name,
                applied=False,
                dollars_spent=0.0,
                durability=self.durability,
                notes="no plan_signature in context.extras — cannot retrieve exemplars",
            )

        exemplars = self._ensure_index().get(plan_signature, [])[: self.max_exemplars]
        return SubstrateResult(
            substrate=self.name,
            applied=bool(exemplars),
            dollars_spent=0.0,
            durability=self.durability,
            delta_artifacts={
                "exemplars": exemplars,
                "plan_signature": plan_signature,
            },
            notes=(
                f"replaying {len(exemplars)} positive exemplar(s)"
                if exemplars
                else "no positive exemplars for this plan"
            ),
        )

    def observe(
        self, context: SubstrateContext, result: SubstrateResult, reward: float,
    ) -> None:
        """No-op — exemplar *capture* happens via trace export downstream;
        the index is rebuilt from disk by :meth:`refresh`, not from the
        allocator's reward signal."""
        return None


__all__ = ["ExemplarSubstrate"]
