"""Structured per-step trace envelope (#783).

Extends what `TraceExporter` writes today (`StepResult` -> JSON) with
the fields the HN URL-collection user report (#785) called out as
missing for triage:

- before / after URL
- screenshot pre / post (paths, not bytes)
- vision-grounding target (description + bbox)
- actual click coordinates
- dispatch payload (xdotool argv on Computer Plane, structured verb
  on Browser-Use Plane)
- step verifier verdict (pass / fail / demoted)
- retry count + last retry reason
- emitted rows / outputs

The envelope is per-step. Runners attach one to each `StepResult`
they emit; `TraceExporter` serializes it inline when present. Handlers
that don't populate the envelope just leave the fields empty —
backwards-compatible with today's exports.

Storage layout (when `MANTIS_TRACE_EXPORT_DIR` is set):

    $MANTIS_TRACE_EXPORT_DIR/<tenant>/<run_id>.json
        — contains step_envelope blocks inline
    $MANTIS_TRACE_EXPORT_DIR/<tenant>/<run_id>/<step>_pre.png
    $MANTIS_TRACE_EXPORT_DIR/<tenant>/<run_id>/<step>_post.png
        — written when `MANTIS_TRACE_INCLUDE_SCREENSHOTS=1`
        — `_pre` only when the envelope actually carries a pre-action shot

Companion HTTP / CLI surface lands in PR 7 (`/v1/runs/{id}/trace`,
`mantis trace <run_id>`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GroundingTarget:
    """Vision-grounded click target — what the brain *aimed* at."""

    description: str = ""
    bbox: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2)
    confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "bbox": list(self.bbox) if self.bbox else None,
            "confidence": self.confidence,
        }


@dataclass
class DispatchRecord:
    """The actual action that was dispatched to the compute plane.

    `kind` discriminates: Computer Plane → `xdotool` with `argv`;
    Browser-Use Plane → `click` / `key` / `type` / `scroll` with
    structured `params`. CDP escape hatches use `cdp`.
    """

    kind: str = ""  # "xdotool" | "click" | "key" | "type" | "scroll" | "cdp"
    argv: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    coordinates: tuple[int, int] | None = None  # (x, y) when applicable

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "argv": list(self.argv),
            "params": dict(self.params),
            "coordinates": list(self.coordinates) if self.coordinates else None,
        }


@dataclass
class VerifierVerdict:
    """Outcome of the step's verification gate.

    `status` is one of:
      - `pass` — verifier confirmed expected outcome.
      - `fail` — verifier rejected; step counted as a failure.
      - `demoted` — verifier failed but `required=False`, so we
        continued (see `feedback_critic_gate_truthy_fail`).
      - `skipped` — no verifier configured.
    """

    status: str = "skipped"  # pass | fail | demoted | skipped
    reason: str = ""
    raw_verifier_output: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "raw_verifier_output": self.raw_verifier_output,
        }


@dataclass
class StepTraceEnvelope:
    """Per-step structured trace data.

    Attached to a `StepResult` via the `record_step_trace` helper.
    Empty envelopes (default-constructed) are semantically equivalent
    to "no envelope" — `TraceExporter` skips empty blocks to avoid
    bloating the JSON.
    """

    # URLs around the action.
    url_before: str = ""
    url_after: str = ""

    # Screenshot artifacts. Stored as paths in the export dir, not
    # bytes — keeps the JSON readable and the binary on disk.
    screenshot_pre_path: str = ""
    screenshot_post_path: str = ""

    # What the brain wanted vs what got dispatched.
    grounding: GroundingTarget | None = None
    dispatch: DispatchRecord | None = None

    # How verification + retry played out.
    verifier: VerifierVerdict | None = None
    retry_count: int = 0
    retry_reason: str = ""

    # What the step emitted to downstream (rows / leads / extracted
    # data). The runner's `step.data` is the source of truth; this
    # field carries a stable projection (e.g. row count, first row
    # sample) so a trace consumer doesn't have to parse plan-specific
    # blobs.
    emitted_count: int = 0
    emitted_sample: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return (
            not self.url_before
            and not self.url_after
            and not self.screenshot_pre_path
            and not self.screenshot_post_path
            and self.grounding is None
            and self.dispatch is None
            and self.verifier is None
            and self.retry_count == 0
            and not self.retry_reason
            and self.emitted_count == 0
            and not self.emitted_sample
        )

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "url_before": self.url_before,
            "url_after": self.url_after,
            "screenshot_pre_path": self.screenshot_pre_path,
            "screenshot_post_path": self.screenshot_post_path,
            "retry_count": self.retry_count,
            "retry_reason": self.retry_reason,
            "emitted_count": self.emitted_count,
            "emitted_sample": dict(self.emitted_sample),
        }
        if self.grounding is not None:
            d["grounding"] = self.grounding.as_dict()
        if self.dispatch is not None:
            d["dispatch"] = self.dispatch.as_dict()
        if self.verifier is not None:
            d["verifier"] = self.verifier.as_dict()
        return d


# ── In-process collector ────────────────────────────────────────────


@dataclass
class StepTraceCollector:
    """Accumulates per-step envelopes for one run.

    Runners hold a single instance and the step handlers `record()`
    into it. `envelopes_by_step_index` is the source of truth — the
    `TraceExporter._step_to_dict` patch reads from here to merge the
    extra fields into the exported JSON.
    """

    run_id: str = ""
    envelopes_by_step_index: dict[int, StepTraceEnvelope] = field(default_factory=dict)

    def record(self, step_index: int, envelope: StepTraceEnvelope) -> None:
        # Last write wins per step_index. Handlers may call `record`
        # multiple times during a step (e.g. once after grounding,
        # again after verifier) — caller should merge into a single
        # envelope per call, or call `update` (below).
        self.envelopes_by_step_index[step_index] = envelope

    def update(self, step_index: int, **fields: Any) -> StepTraceEnvelope:
        """Merge fields into the envelope at `step_index`, creating
        one if absent. Convenience for handlers that record bits as
        they go (URL on entry, dispatch on action, verifier on gate)."""
        env = self.envelopes_by_step_index.setdefault(
            step_index, StepTraceEnvelope()
        )
        for k, v in fields.items():
            if hasattr(env, k):
                setattr(env, k, v)
        return env

    def get(self, step_index: int) -> StepTraceEnvelope | None:
        return self.envelopes_by_step_index.get(step_index)

    def as_dict_by_index(self) -> dict[str, dict[str, Any]]:
        """Project to the export-friendly dict (string-keyed for JSON).
        Skips empty envelopes — `TraceExporter` does not need entries
        that carry no information beyond what `StepResult` already has.
        """
        out: dict[str, dict[str, Any]] = {}
        for idx, env in self.envelopes_by_step_index.items():
            if env.is_empty():
                continue
            out[str(idx)] = env.as_dict()
        return out


__all__ = [
    "DispatchRecord",
    "GroundingTarget",
    "StepTraceCollector",
    "StepTraceEnvelope",
    "VerifierVerdict",
]
