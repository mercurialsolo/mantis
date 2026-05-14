"""Durable per-run JSONL log + cross-run aggregation (epic #362 Phase C).

Writes one structured row per terminal run to::

    $MANTIS_DATA_DIR/runs_log/<YYYY-MM>.jsonl

Append-only, sharded by month so a single file never grows unbounded
and old months can be archived independently. Atomic append via
``open('a') + fsync`` — concurrent writers on one pod are serialized
by the OS append guarantee for files opened with ``O_APPEND``.

Public surface:

* :func:`writer.append_run` — append one row from a terminal result.
* :func:`query.stats` — load rows and compute p50/p95/p99 per bucket
  for a given workflow.

The schema lives in :data:`writer.SCHEMA_VERSION` + the helper
:func:`writer.row_from_result`. Permissive (extra keys allowed) so
future fields don't break readers.
"""

from __future__ import annotations

from .query import RunStats, stats
from .writer import RUNS_LOG_SUBDIR, SCHEMA_VERSION, append_run, row_from_result

__all__ = [
    "RUNS_LOG_SUBDIR",
    "RunStats",
    "SCHEMA_VERSION",
    "append_run",
    "row_from_result",
    "stats",
]
