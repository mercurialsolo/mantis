"""Structured reasoning trace — every decision the runner makes,
streamed to disk so a viewer / debugger can scrub through them
alongside the MJPEG live feed.

Existing observability:

* :mod:`healing_events` — records plan mutations (rewrite, demote,
  handler_escalation, insert_step, replace_step). Append-only on
  ``runner._healing_events``. Surfaced at run-end via
  ``build_micro_result``.
* Modal log lines (``[critic-frontier]`` / ``[som-click]`` / etc.)
  — visible via ``modal app logs`` but require grep + correlate to
  reconstruct the timeline.

What's missing — and what this module adds — is a STRUCTURED LIVE
stream of the same decisions the log lines already make visible.
The runner appends to ``runner._reasoning_events`` and flushes
periodically to ``<run_dir>/reasoning.jsonl``; the API container
serves the file via ``action=reasoning_trace``. A future viewer
overlay polls the endpoint and renders a timeline beside the
MJPEG feed (separate PR).

Event shape (consistent across types)::

    {
        "ts": "2026-05-16T19:46:51Z",
        "step_index": 6,
        "layer": "critic-frontier" | "agentic-recovery"
                 | "som-click" | "gate-decision" | …,
        "kind": "fire" | "skip" | "decision" | "result",
        "summary": "edit_step → ReplaceStep",
        "detail": {...},
    }

The same ``runner._healing_events`` list backs both streams —
``record(...)`` here is a shim that appends to that list with a
``kind="reasoning"`` flag, so the existing ``healing_events.snapshot``
keeps working without changes.

Why no separate list: viewers and result envelopes want a single
ordered timeline. Splitting into two lists would force consumers
to merge-sort on every read; one list keeps the order natural.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record(
    runner: Any,
    *,
    layer: str,
    kind: str,
    summary: str,
    step_index: int | None = None,
    **detail: Any,
) -> None:
    """Append a reasoning event to the runner's event list.

    Defensive on every access — tests that hand the runner a
    ``MagicMock`` auto-create ``_healing_events`` as a Mock, not a
    list. A non-list silently drops the event so production never
    crashes on the trace path.

    The event is also written to the runner's reasoning_jsonl_path
    when the runner exposes one — the runtime sets this attribute
    so events stream to disk during the run.
    """
    log = getattr(runner, "_healing_events", None)
    if not isinstance(log, list):
        return
    event = {
        "ts": _now_iso(),
        "layer": str(layer),
        "kind": str(kind),
        "summary": str(summary)[:300],
        "step_index": int(step_index) if step_index is not None else -1,
        "detail": _sanitize_detail(detail),
    }
    # Tag as reasoning so ``healing_events.snapshot`` consumers can
    # filter if they only want the plan-mutation subset.
    event["category"] = "reasoning"
    log.append(event)

    # Stream to disk when the runner has a configured path. Errors
    # never propagate — the trace is observability; a broken FS
    # mustn't break the run.
    path = getattr(runner, "_reasoning_jsonl_path", None)
    if path:
        try:
            with Path(path).open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.debug("reasoning_trace flush failed: %s", exc)


def _sanitize_detail(detail: dict[str, Any]) -> dict[str, Any]:
    """JSON-serializable copy of the detail dict. Drops keys whose
    values can't be safely stringified."""
    out: dict[str, Any] = {}
    for key, val in detail.items():
        try:
            json.dumps(val)
            out[str(key)] = val
        except (TypeError, ValueError):
            try:
                out[str(key)] = str(val)[:200]
            except Exception:  # noqa: BLE001
                continue
    return out


def configure_disk_stream(runner: Any, jsonl_path: str | Path) -> None:
    """Tell the runner to mirror reasoning events to ``jsonl_path``
    in append mode. The runtime calls this at run start so the
    HTTP ``action=reasoning_trace`` endpoint sees fresh events
    DURING the run (not only after termination).

    Idempotent — calling twice with the same path just re-stamps
    the attribute. Calling with a different path swaps the
    destination for events emitted AFTER the call.
    """
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    runner._reasoning_jsonl_path = str(path)


def read_jsonl(path: str | Path, *, since_ts: str | None = None) -> list[dict]:
    """Read events from a JSONL file with optional ``since_ts``
    cursor. Returns events whose ``ts`` is strictly greater than
    ``since_ts`` — lexicographic compare on the ISO-8601 timestamp,
    which is order-preserving for UTC.

    Malformed lines are skipped silently — a partially-written
    final line (the runner crashed mid-flush) doesn't break the
    reader.
    """
    p = Path(path)
    if not p.exists():
        return []
    events: list[dict] = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(event, dict):
                    continue
                if since_ts and event.get("ts", "") <= since_ts:
                    continue
                events.append(event)
    except OSError as exc:
        logger.debug("reasoning_trace read failed: %s", exc)
    return events


__all__ = [
    "record",
    "configure_disk_stream",
    "read_jsonl",
]
