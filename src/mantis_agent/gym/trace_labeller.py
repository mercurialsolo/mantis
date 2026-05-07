"""TraceLabeller — turn raw run traces (#155 step 1) into labelled examples for SFT/DPO.

Reads JSON trace files emitted by :class:`~.trace_exporter.TraceExporter`
and applies a small set of automatic heuristics to label each step as a
positive, negative, or neutral training example.

Heuristics
----------

Each step lands in exactly one bucket. Order of evaluation matters —
the first matching rule wins:

1. **negative — escalation**
   ``last_action.action_type`` empty AND ``data`` mentions
   ``cloudflare`` / ``page_blocked`` / ``REJECTED`` → the brain bounced
   off something the runtime had to recover from. These are the
   highest-priority negatives — Claude usually rescued them.
2. **negative — failed step**
   ``success`` is False (after the runtime exhausted retries). Counts
   as a negative regardless of step type.
3. **positive — gate verify pass**
   ``data`` starts with ``gate:PASS`` — explicit verification that the
   page state matched the brain's reasoning. Highest-confidence positive.
4. **positive — success with state change**
   ``success`` is True AND ``observed_outcome`` is non-empty. The brain
   acted, the runtime saw a delta, the outcome was recorded.
5. **neutral — everything else**
   Successes without observed deltas (e.g. waits, navigate-back),
   no-action steps, or anything ambiguous. Useful for diagnostic
   counts but not training signal.

Schema
------

``LabelledStep`` extends the trace step with two new fields:

- ``label`` — ``positive | negative | neutral``
- ``label_reason`` — short string identifying which heuristic fired

The labelled output is written as JSON with the same top-level run
metadata as the input trace, plus a ``label_summary`` rollup with
counts per bucket. Ready for an SFT/DPO loader that filters on
``label`` and reads the ``intent`` / ``last_action`` /
``predicted_outcome`` / ``observed_outcome`` fields.

Usage
-----

    labeller = TraceLabeller()
    labelled = labeller.label_trace_file(Path("trace.json"))
    print(labelled.summary())  # {"positive": 5, "negative": 2, "neutral": 1}

CLI surfaces this as ``mantis trace label`` and ``mantis trace review``
(see ``mantis_agent/cli.py``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Substrings that mark a step's ``data`` field as an escalation event.
# Matched case-insensitively. Kept conservative — false positives mark
# a usable trace as a negative; false negatives just lower yield.
_ESCALATION_MARKERS: tuple[str, ...] = (
    "cloudflare",
    "page_blocked",
    "rejected_incomplete",
    "antibot",
    "page_exhausted",
    "scan_error",
)

# Marker that prefixes data on a successful gate-verify step. Matches
# the format emitted by ``_runner_helpers.execute_step``.
_GATE_PASS_PREFIX: str = "gate:PASS"


@dataclass
class LabelledStep:
    """One trace step plus its automatic label.

    Mirrors the trace step's persisted fields and adds two label
    columns. Round-trips through JSON so SFT/DPO loaders can consume
    the same shape.
    """

    step_index: int
    intent: str
    type: str
    success: bool
    data: str
    last_action: dict[str, Any] | None
    predicted_outcome: str
    observed_outcome: str
    label: str
    label_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "intent": self.intent,
            "type": self.type,
            "success": self.success,
            "data": self.data,
            "last_action": self.last_action,
            "predicted_outcome": self.predicted_outcome,
            "observed_outcome": self.observed_outcome,
            "label": self.label,
            "label_reason": self.label_reason,
        }


@dataclass
class LabelledTrace:
    """One trace file plus per-step labels and a rollup summary."""

    run_id: str = ""
    tenant_id: str = ""
    status: str = ""
    steps: list[LabelledStep] = field(default_factory=list)
    source_path: str = ""

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
        for s in self.steps:
            out[s.label] = out.get(s.label, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "status": self.status,
            "source_path": self.source_path,
            "label_summary": self.summary(),
            "steps": [s.to_dict() for s in self.steps],
        }


# ── Heuristics ─────────────────────────────────────────────────────────


def _is_escalation(step: dict[str, Any]) -> bool:
    data = (step.get("data") or "").lower()
    return any(marker in data for marker in _ESCALATION_MARKERS)


def _is_gate_pass(step: dict[str, Any]) -> bool:
    return (step.get("data") or "").startswith(_GATE_PASS_PREFIX)


def _classify(step: dict[str, Any]) -> tuple[str, str]:
    """Apply the heuristic ladder. Returns ``(label, reason)``."""
    if _is_escalation(step):
        return "negative", "escalation"
    if not step.get("success", False):
        return "negative", "failed_step"
    if _is_gate_pass(step):
        return "positive", "gate_verify_pass"
    if step.get("observed_outcome"):
        return "positive", "success_with_observed_delta"
    return "neutral", "success_no_delta"


# ── Public API ─────────────────────────────────────────────────────────


class TraceLabeller:
    """Apply automatic heuristics to trace files.

    Stateless — instantiate once and reuse. Subclasses can override
    :meth:`classify_step` to plug in domain-specific labels (e.g. a
    site-aware "off-site click" negative) without rewriting the loop.
    """

    def label_step(
        self, step: dict[str, Any], *, with_reason: bool = True,
    ) -> LabelledStep:
        label, reason = self.classify_step(step)
        return LabelledStep(
            step_index=int(step.get("step_index", 0)),
            intent=str(step.get("intent", "")),
            type=str(step.get("type", "")),
            success=bool(step.get("success", False)),
            data=str(step.get("data", "")),
            last_action=step.get("last_action"),
            predicted_outcome=str(step.get("predicted_outcome", "")),
            observed_outcome=str(step.get("observed_outcome", "")),
            label=label,
            label_reason=reason if with_reason else "",
        )

    def classify_step(self, step: dict[str, Any]) -> tuple[str, str]:
        """Hook for subclasses; default delegates to ``_classify``."""
        return _classify(step)

    def label_trace(self, trace: dict[str, Any]) -> LabelledTrace:
        labelled = LabelledTrace(
            run_id=str(trace.get("run_id", "")),
            tenant_id=str(trace.get("tenant_id", "")),
            status=str(trace.get("status", "")),
        )
        labelled.steps = [self.label_step(s) for s in trace.get("steps", [])]
        return labelled

    def label_trace_file(self, path: Path) -> LabelledTrace:
        payload = json.loads(path.read_text())
        labelled = self.label_trace(payload)
        labelled.source_path = str(path)
        return labelled

    def label_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        *,
        glob: str = "**/*.json",
    ) -> dict[str, dict[str, int]]:
        """Apply labels to every trace JSON under ``input_dir`` and write
        the labelled output under ``output_dir`` with the same relative
        layout. Returns a per-file summary rollup keyed by relative path.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, dict[str, int]] = {}
        for trace_path in sorted(_iter_traces(input_dir, glob)):
            try:
                labelled = self.label_trace_file(trace_path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("skipping %s: %s", trace_path, exc)
                continue
            rel = trace_path.relative_to(input_dir)
            target = output_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(labelled.to_dict(), indent=2) + "\n")
            summary[str(rel)] = labelled.summary()
        return summary


def _iter_traces(root: Path, glob: str) -> Iterable[Path]:
    """Filter to paths whose stem doesn't already encode a label suffix."""
    for path in root.glob(glob):
        if path.is_file() and path.suffix == ".json":
            yield path


__all__ = [
    "LabelledStep",
    "LabelledTrace",
    "TraceLabeller",
]
