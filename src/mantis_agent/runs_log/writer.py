"""Per-run JSONL writer — epic #362 Phase C.

One line per terminal run, atomic append, sharded by month. Pure
stdlib — no DuckDB dependency at write time (query side is also pure
Python; DuckDB can replace it later if row counts grow past comfort).

Atomicity: files opened with ``O_APPEND`` (Python's ``"a"`` mode) give
POSIX guarantees that ``write()`` of <= PIPE_BUF bytes is atomic
between concurrent writers on the same FS. A single JSONL line easily
fits under PIPE_BUF (4096+ on every modern OS); we write line +
newline in one call. ``fsync`` after each write makes durability
explicit so a crashing pod doesn't lose the last-completed runs.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1
RUNS_LOG_SUBDIR = "runs_log"


def _data_dir() -> Path:
    """Resolve the on-disk root from ``MANTIS_DATA_DIR``.

    Falls back to the same default as the rest of the server
    (``/workspace/mantis-data``) so the log lands alongside the
    existing per-tenant data.
    """
    return Path(os.environ.get("MANTIS_DATA_DIR", "/workspace/mantis-data"))


def _shard_path(now: datetime | None = None, data_dir: Path | None = None) -> Path:
    """Return the JSONL path for the current month: ``<dir>/runs_log/YYYY-MM.jsonl``."""
    now = now or datetime.now(timezone.utc)
    base = data_dir if data_dir is not None else _data_dir()
    return base / RUNS_LOG_SUBDIR / f"{now.strftime('%Y-%m')}.jsonl"


def row_from_result(
    *,
    run_id: str,
    tenant_id: str = "",
    profile_id: str = "",
    workflow_id: str = "",
    plan_signature: str = "",
    model: str = "",
    status: str,
    created_at: str = "",
    finished_at: str = "",
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Project the runtime's result envelope + status metadata into a
    schema-stable row.

    ``result`` is the dict returned by :func:`build_micro_result`
    (Phase B). When ``None`` (failed-before-result paths), the row
    still lands with ``total_time_s=0`` and empty breakdowns so
    aggregations can count failures correctly.
    """
    result = result or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "tenant_id": tenant_id,
        "profile_id": profile_id,
        "workflow_id": workflow_id,
        "plan_signature": plan_signature,
        "model": model,
        "status": status,
        "created_at": created_at,
        "finished_at": finished_at or _utc_now_iso(),
        "total_time_s": int(result.get("total_time_s", 0) or 0),
        "wall_time_breakdown": dict(result.get("wall_time_breakdown") or {}),
        "cost_breakdown": dict(result.get("cost_breakdown") or {}),
        "steps_executed": int(result.get("steps_executed", 0) or 0),
        "viable": int(result.get("viable", 0) or 0),
        "error": error,
    }


def append_run(
    row: dict[str, Any],
    *,
    now: datetime | None = None,
    data_dir: Path | None = None,
) -> Path:
    """Append one JSONL row to this month's shard. Returns the path
    written to. Best-effort: a write failure is logged and re-raised
    only to the caller — never break the parent run on observability
    I/O.

    The caller is expected to have already validated the row shape via
    :func:`row_from_result`; arbitrary dicts will still serialize, but
    cross-run aggregation assumes the canonical keys.
    """
    path = _shard_path(now=now, data_dir=data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
    # Single ``write`` of a small line is atomic w.r.t. concurrent
    # appenders on POSIX (PIPE_BUF >= 4096 on Linux/macOS). ``fsync``
    # makes the write durable past a pod crash.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError as exc:  # noqa: BLE001 — some FSes (tmpfs) can't fsync
            logger.debug("runs_log: fsync failed (path=%s): %s", path, exc)
    return path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "RUNS_LOG_SUBDIR",
    "SCHEMA_VERSION",
    "append_run",
    "row_from_result",
]
