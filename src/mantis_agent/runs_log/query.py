"""Pure-Python aggregation over the JSONL run-log (epic #362 Phase C).

Streams rows from ``$MANTIS_DATA_DIR/runs_log/*.jsonl``, filters by
workflow / status / time window, and computes p50/p95/p99 per
wall-time bucket. No DuckDB dependency — fast enough up to ~100k
rows. The DuckDB-backed query path is reserved for follow-up if /
when row counts demand it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

from .writer import RUNS_LOG_SUBDIR, _data_dir

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    """Aggregated stats for a workflow across N runs.

    ``percentiles`` is keyed by bucket (``perceive`` / ``think`` / etc.
    plus ``total_time_s``) and each value is ``(p50, p95, p99)``.
    ``run_count`` reflects the rows that passed all filters.
    """

    workflow_id: str
    run_count: int
    percentiles: dict[str, tuple[float, float, float]] = field(default_factory=dict)


def _iter_jsonl(paths: Iterable[Path], *, reverse: bool = False) -> Iterator[dict]:
    """Yield parsed rows from a set of JSONL shards. Malformed lines
    are logged and skipped — one bad line shouldn't poison a query.

    When ``reverse=True``, each shard's lines are yielded
    newest-line-first (load + reverse the line list); combined with
    a newest-shard-first iteration this gives a global newest-first
    walk that ``last_n`` can truncate cheaply.
    """
    for path in paths:
        try:
            with open(path, encoding="utf-8") as fh:
                lines = list(enumerate(fh, start=1))
        except FileNotFoundError:
            continue
        if reverse:
            lines = list(reversed(lines))
        for lineno, raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "runs_log: skip malformed line %s:%d: %s",
                    path, lineno, exc,
                )


def _shard_paths(data_dir: Path | None = None) -> list[Path]:
    base = (data_dir if data_dir is not None else _data_dir()) / RUNS_LOG_SUBDIR
    if not base.exists():
        return []
    return sorted(base.glob("*.jsonl"))


def _percentile(samples: list[float], q: float) -> float:
    """``q`` is a fraction in [0, 1]. Linear interpolation between
    closest ranks — matches numpy's default 'linear' method."""
    if not samples:
        return 0.0
    s = sorted(samples)
    if len(s) == 1:
        return float(s[0])
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def stats(
    workflow_id: str,
    *,
    last_n: int = 100,
    since: timedelta | None = None,
    buckets: list[str] | None = None,
    status_filter: str = "succeeded",
    data_dir: Path | None = None,
    _now: datetime | None = None,
) -> RunStats:
    """Return p50/p95/p99 per bucket for the last ``last_n`` runs of
    ``workflow_id``.

    Filtering:

    * ``status_filter`` — defaults to ``"succeeded"`` so cost/time
      stats reflect only completed runs. Pass ``""`` to include every
      terminal status (useful for failure-mode analysis).
    * ``since`` — if set, drops rows whose ``finished_at`` is older
      than ``now - since``. Combine with ``last_n`` for a sliding
      window.
    * ``buckets`` — restrict to a subset of wall-time buckets. Default
      is all buckets present in the rows + the ``total_time_s`` sum.

    Rows are read newest-shard-first so ``last_n`` is the most-recent
    runs, not a random tail.
    """
    cutoff = None
    if since is not None:
        now = _now or datetime.now(timezone.utc)
        cutoff = now - since

    # Walk shards newest-first AND lines within each shard newest-first
    # so ``last_n`` truly truncates the most recent runs.
    paths = list(reversed(_shard_paths(data_dir=data_dir)))
    matched: list[dict] = []
    for row in _iter_jsonl(paths, reverse=True):
        if row.get("workflow_id") != workflow_id:
            continue
        if status_filter and row.get("status") != status_filter:
            continue
        if cutoff is not None:
            finished = _parse_iso(row.get("finished_at"))
            if finished is None or finished < cutoff:
                continue
        matched.append(row)
        if len(matched) >= last_n:
            break

    if not matched:
        return RunStats(workflow_id=workflow_id, run_count=0)

    # Collect samples per bucket.
    bucket_samples: dict[str, list[float]] = {}
    totals: list[float] = []
    for row in matched:
        bd = row.get("wall_time_breakdown") or {}
        for k, v in bd.items():
            try:
                bucket_samples.setdefault(k, []).append(float(v))
            except (TypeError, ValueError):
                continue
        try:
            totals.append(float(row.get("total_time_s") or 0))
        except (TypeError, ValueError):
            continue

    if buckets:
        bucket_samples = {k: v for k, v in bucket_samples.items() if k in buckets}

    out: dict[str, tuple[float, float, float]] = {}
    for k, samples in bucket_samples.items():
        out[k] = (
            _percentile(samples, 0.50),
            _percentile(samples, 0.95),
            _percentile(samples, 0.99),
        )
    out["total_time_s"] = (
        _percentile(totals, 0.50),
        _percentile(totals, 0.95),
        _percentile(totals, 0.99),
    )

    return RunStats(
        workflow_id=workflow_id,
        run_count=len(matched),
        percentiles=out,
    )


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # Tolerate both ``...Z`` and ``...+00:00`` shapes.
        v = value.replace("Z", "+00:00") if value.endswith("Z") else value
        return datetime.fromisoformat(v)
    except (ValueError, TypeError):
        return None


__all__ = ["RunStats", "stats"]
