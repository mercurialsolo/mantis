"""Merge env-side events into the agent trace post-run.

After a plan finishes, the harness calls
:meth:`RuntimeBackend.fetch_events` to pull the env's structured event
log (timestamped HTTP requests + state mutations), then merges that
list into the run's output directory as ``env_events.jsonl`` plus a
unified ``merged_trace.jsonl`` that interleaves agent steps with env
events by timestamp.

The merge is opportunistic and additive:

* If the env returns zero events (or the fetch errors out), we still
  finish — the agent's own trace is untouched.
* If an agent trace file isn't present (single-run CLI invocation
  that bypasses ``TraceExporter``), we write only ``env_events.jsonl``.
* Output schema stays JSONL so downstream tools can read it line by
  line without buffering the whole file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .runtime import RuntimeBackend, RuntimeHandle

logger = logging.getLogger(__name__)


def merge_env_events(
    output_dir: Path,
    backend: RuntimeBackend,
    handle: RuntimeHandle,
    *,
    since: float | None = None,
) -> int:
    """Pull events from the env, write JSONL files, return event count.

    Returns the number of events fetched (zero if the env had none or
    the call errored). Caller can use the count for logging.
    """
    events = backend.fetch_events(handle, since=since)
    if not events:
        return 0

    out_path = output_dir / "env_events.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for evt in events:
            fh.write(json.dumps(evt) + "\n")

    # Best-effort interleave with the agent trace. The CLI writes
    # ``trace.jsonl`` only when the trace exporter is enabled
    # (MANTIS_TRACE_EXPORT_DIR); without it the merge is just the env
    # events.
    trace_jsonl = output_dir / "trace.jsonl"
    if trace_jsonl.exists():
        try:
            merged = _interleave(trace_jsonl, events)
        except Exception:  # noqa: BLE001
            logger.exception("trace merge: interleave failed; env_events.jsonl is authoritative")
            return len(events)
        (output_dir / "merged_trace.jsonl").write_text(
            "\n".join(json.dumps(item) for item in merged) + "\n",
            encoding="utf-8",
        )

    return len(events)


def _interleave(trace_jsonl: Path, env_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort agent steps and env events by timestamp into one stream."""
    items: list[tuple[float, dict[str, Any]]] = []
    for line in trace_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = float(obj.get("ts") or obj.get("started_at") or 0.0)
        items.append((ts, {"source": "agent", **obj}))
    for evt in env_events:
        ts = float(evt.get("ts") or 0.0)
        items.append((ts, {"source": "env", **evt}))
    items.sort(key=lambda pair: pair[0])
    return [obj for _, obj in items]
