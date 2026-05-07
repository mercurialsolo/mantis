"""TraceExporter — persist production runs for offline analysis + SFT/DPO labelling (#155 step 1).

Each :class:`MicroPlanRunner` invocation that completes (or halts /
cancels / pauses) emits one structured trace JSON file. The schema is
versioned so consumers can downgrade gracefully when fields are added.

What's in a trace
-----------------

* ``schema_version`` — bumped on incompatible changes.
* Run metadata: ``run_id``, ``tenant_id``, ``session_name``,
  ``plan_signature``, ``status``, ``started_at`` / ``ended_at``,
  ``total_time_s``, costs.
* Step list: every executed :class:`StepResult` with its
  ``intent`` / ``type`` / ``success`` / ``data`` / ``duration`` /
  ``reversed`` / ``last_action`` / ``predicted_outcome`` /
  ``observed_outcome``.

Tenant isolation
----------------

Trace files land at ``$MANTIS_TRACE_EXPORT_DIR/<tenant_id>/<run_id>.json``.
Empty tenant ids land under ``__shared__/`` so legacy single-tenant runs
don't collide with multi-tenant ones. Operators wire tenant-bucket
sync (S3 / GCS / R2) by tenant directory.

Feature flag
------------

Off by default. Set ``MANTIS_TRACE_EXPORT_DIR=<path>`` to enable. When
unset, :meth:`TraceExporter.maybe_export` is a no-op so the runtime
pays no cost.

Optional ``MANTIS_TRACE_INCLUDE_SCREENSHOTS`` controls whether
``screenshot_png`` bytes are written alongside the JSON (one PNG per
step). Default off because screenshot bytes balloon the on-disk size
~100x and most triage workflows only need the action sequence + costs.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .checkpoint import StepResult

if TYPE_CHECKING:
    from .micro_runner import MicroPlanRunner

logger = logging.getLogger(__name__)


SCHEMA_VERSION: int = 2
"""``v1`` was the initial export shape (PR #194). ``v2`` (PR for #155
step 5) adds the optional top-level ``variant`` field for
shadow-deploy attribution. ``v1`` consumers can still read ``v2`` —
they just ignore the new key."""

ENV_DIR: str = "MANTIS_TRACE_EXPORT_DIR"
ENV_INCLUDE_SCREENSHOTS: str = "MANTIS_TRACE_INCLUDE_SCREENSHOTS"


def _action_to_dict(action: Any) -> dict[str, Any] | None:
    """Render an :class:`Action` (or ``None``) into a JSON-friendly dict."""
    if action is None:
        return None
    return {
        "action_type": getattr(getattr(action, "action_type", None), "value", ""),
        "params": dict(getattr(action, "params", {}) or {}),
    }


def _step_to_dict(step: StepResult) -> dict[str, Any]:
    """Serialize one step result for the trace.

    Drops binary screenshot bytes — those are written separately when
    ``MANTIS_TRACE_INCLUDE_SCREENSHOTS`` is set, and round-trip through
    the trace as filename references rather than embedded base64.
    """
    return {
        "step_index": step.step_index,
        "intent": step.intent,
        "type": getattr(step, "step_type", "") or "",
        "success": step.success,
        "data": step.data,
        "steps_used": step.steps_used,
        "duration": step.duration,
        "reversed": step.reversed,
        "last_action": _action_to_dict(step.last_action),
        "predicted_outcome": getattr(step, "predicted_outcome", "") or "",
        "observed_outcome": getattr(step, "observed_outcome", "") or "",
    }


@dataclass
class TraceExporter:
    """Writes per-run trace files when the export feature flag is set."""

    export_dir: str = ""
    include_screenshots: bool = False

    @classmethod
    def from_env(cls) -> "TraceExporter":
        """Build an exporter from the standard env vars. Returns a disabled
        exporter (``export_dir=""``) when ``MANTIS_TRACE_EXPORT_DIR`` is unset."""
        return cls(
            export_dir=os.environ.get(ENV_DIR, "").strip(),
            include_screenshots=_bool_env(ENV_INCLUDE_SCREENSHOTS),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.export_dir)

    def maybe_export(
        self,
        runner: "MicroPlanRunner",
        results: list[StepResult],
        *,
        status: str,
    ) -> str | None:
        """Write a trace file when the exporter is enabled. No-op otherwise.

        Returns the JSON path written, or ``None`` when disabled / on
        error. Telemetry-style: any IO failure is logged but never
        re-raised — trace export must not break a run.
        """
        if not self.enabled:
            return None
        try:
            return self._write(runner, results, status=status)
        except Exception as exc:  # noqa: BLE001 — telemetry never breaks runs
            logger.warning("trace export failed: %s", exc)
            return None

    def _write(
        self,
        runner: "MicroPlanRunner",
        results: list[StepResult],
        *,
        status: str,
    ) -> str:
        tenant_id = (getattr(runner, "tenant_id", "") or "").strip() or "__shared__"
        run_id = (
            getattr(runner, "run_key", "")
            or getattr(runner, "session_name", "")
            or "unknown"
        )
        base = Path(self.export_dir) / tenant_id
        base.mkdir(parents=True, exist_ok=True)

        if self.include_screenshots:
            shots_dir = base / f"{run_id}_screens"
            shots_dir.mkdir(exist_ok=True)
            self._write_screenshots(shots_dir, results)

        gpu, claude, proxy, total = runner._cost_totals()
        ended_at = time.time()
        started_at = getattr(runner, "_run_start", ended_at)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "tenant_id": tenant_id if tenant_id != "__shared__" else "",
            "session_name": getattr(runner, "session_name", ""),
            "plan_signature": getattr(runner, "plan_signature", ""),
            # #155 step 5: shadow-deploy variant tag. Empty string when
            # the runtime isn't running an A/B split; populated to
            # ``baseline`` or ``candidate`` (or any custom variant id)
            # when the ShadowRouter assigned this run.
            "variant": str(getattr(runner, "shadow_variant", "") or ""),
            "status": status or getattr(runner, "_final_status", "unknown"),
            "started_at": started_at,
            "ended_at": ended_at,
            "total_time_s": ended_at - started_at,
            "costs": {
                "gpu": round(gpu, 4),
                "claude": round(claude, 4),
                "proxy": round(proxy, 4),
                "total": round(total, 4),
            },
            "step_count": len(results),
            "steps": [_step_to_dict(s) for s in results],
        }

        out = base / f"{run_id}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        logger.debug("wrote trace: %s", out)
        return str(out)

    def _write_screenshots(
        self, shots_dir: Path, results: list[StepResult]
    ) -> None:
        """Persist post-step screenshot bytes one PNG per step.

        File naming: ``<step_index:04d>.png``. Skips steps without
        ``screenshot_png`` so the index → file mapping is direct (no
        gaps to interpret).
        """
        for step in results:
            png = step.screenshot_png
            if not png:
                continue
            (shots_dir / f"{step.step_index:04d}.png").write_bytes(png)


def _bool_env(name: str) -> bool:
    """Treat ``1``, ``true``, ``yes``, ``on`` (case-insensitive) as truthy."""
    raw = os.environ.get(name, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


__all__ = ["TraceExporter", "SCHEMA_VERSION", "ENV_DIR", "ENV_INCLUDE_SCREENSHOTS"]
