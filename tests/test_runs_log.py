"""Durable JSONL runs-log (epic #362 Phase C).

Pins the writer's atomic-append semantics, the row schema, and the
percentile aggregation that the ``mantis runs stats`` CLI consumes.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mantis_agent.runs_log import (
    RUNS_LOG_SUBDIR,
    SCHEMA_VERSION,
    append_run,
    row_from_result,
    stats,
)


# ── row_from_result projection ──────────────────────────────────────────


def test_row_from_result_lifts_breakdowns_from_result_envelope() -> None:
    result = {
        "total_time_s": 247,
        "wall_time_breakdown": {"think": 88.1, "act": 6.7, "overhead": 1.3},
        "cost_breakdown": {"gpu": 0.12, "claude": 0.30},
        "steps_executed": 17,
        "viable": 3,
        "plan_signature": "abc123",
        "model": "Hcompany/Holo3-35B-A3B",
    }
    row = row_from_result(
        run_id="r1", tenant_id="t1", profile_id="p1", workflow_id="w1",
        status="succeeded", finished_at="2026-05-13T18:00:00Z",
        result=result,
    )
    assert row["schema_version"] == SCHEMA_VERSION
    assert row["run_id"] == "r1"
    assert row["workflow_id"] == "w1"
    assert row["status"] == "succeeded"
    assert row["wall_time_breakdown"] == {
        "think": 88.1, "act": 6.7, "overhead": 1.3,
    }
    assert row["cost_breakdown"] == {"gpu": 0.12, "claude": 0.30}
    assert row["total_time_s"] == 247
    assert row["steps_executed"] == 17


def test_row_from_result_handles_missing_result() -> None:
    """Failed-before-build paths still produce a valid row so failures
    don't disappear from the log."""
    row = row_from_result(
        run_id="r1", workflow_id="w1", status="failed",
        finished_at="2026-05-13T18:00:00Z",
        result=None, error="page_blocked",
    )
    assert row["status"] == "failed"
    assert row["error"] == "page_blocked"
    assert row["total_time_s"] == 0
    assert row["wall_time_breakdown"] == {}
    assert row["cost_breakdown"] == {}


# ── append_run atomic-append + schema round-trip ────────────────────────


def test_append_run_creates_shard_and_writes_jsonl(tmp_path: Path) -> None:
    row = row_from_result(
        run_id="r1", workflow_id="w1", status="succeeded",
        finished_at="2026-05-13T18:00:00Z",
        result={"total_time_s": 100},
    )
    path = append_run(row, data_dir=tmp_path)
    assert path.exists()
    assert path.parent.name == RUNS_LOG_SUBDIR

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["run_id"] == "r1"
    assert parsed["workflow_id"] == "w1"
    assert parsed["total_time_s"] == 100


def test_append_run_appends_to_existing_shard(tmp_path: Path) -> None:
    for run_id in ("r1", "r2", "r3"):
        append_run(
            row_from_result(
                run_id=run_id, workflow_id="w1", status="succeeded",
                finished_at="2026-05-13T18:00:00Z",
                result={"total_time_s": 100},
            ),
            data_dir=tmp_path,
        )
    paths = list((tmp_path / RUNS_LOG_SUBDIR).glob("*.jsonl"))
    assert len(paths) == 1
    lines = paths[0].read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["run_id"] for line in lines] == ["r1", "r2", "r3"]


def test_append_run_shards_by_month(tmp_path: Path) -> None:
    """Two runs in different months must land in different shards so a
    single file stays bounded over time."""
    apr = datetime(2026, 4, 30, tzinfo=timezone.utc)
    may = datetime(2026, 5, 1, tzinfo=timezone.utc)
    row = row_from_result(
        run_id="r1", workflow_id="w1", status="succeeded",
        finished_at="...", result={},
    )
    p_apr = append_run(row, now=apr, data_dir=tmp_path)
    p_may = append_run(row, now=may, data_dir=tmp_path)
    assert p_apr != p_may
    assert "2026-04.jsonl" in str(p_apr)
    assert "2026-05.jsonl" in str(p_may)


def test_append_run_is_safe_under_concurrent_writers(tmp_path: Path) -> None:
    """20 threads appending in parallel — every line must land intact.

    Tests POSIX O_APPEND atomicity (Python's ``"a"`` mode opens with
    O_APPEND) which guarantees small ``write()`` calls don't interleave.
    A regression that buffers across writes or seeks before appending
    would fail this with truncated or mixed-up lines.
    """
    def worker(i: int) -> None:
        for j in range(5):
            append_run(
                row_from_result(
                    run_id=f"t{i}-r{j}", workflow_id="w1",
                    status="succeeded", finished_at="x",
                    result={"total_time_s": 10},
                ),
                data_dir=tmp_path,
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    paths = list((tmp_path / RUNS_LOG_SUBDIR).glob("*.jsonl"))
    assert len(paths) == 1
    lines = paths[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100
    # All lines parse — no truncation / mid-line interleaving.
    parsed = [json.loads(line) for line in lines]
    run_ids = sorted(p["run_id"] for p in parsed)
    expected = sorted(f"t{i}-r{j}" for i in range(20) for j in range(5))
    assert run_ids == expected


# ── stats() aggregation ─────────────────────────────────────────────────


def _seed_runs(
    tmp_path: Path, workflow_id: str, samples: list[dict],
    status: str = "succeeded",
) -> None:
    """Append a batch of fake-row samples. Each sample is a dict of
    bucket → seconds; total_time_s is summed automatically."""
    for i, sample in enumerate(samples):
        result = {
            "total_time_s": sum(sample.values()),
            "wall_time_breakdown": sample,
        }
        row = row_from_result(
            run_id=f"r{i}", workflow_id=workflow_id, status=status,
            finished_at=f"2026-05-13T18:{i:02d}:00Z",
            result=result,
        )
        append_run(row, data_dir=tmp_path)


def test_stats_returns_zero_run_count_when_log_missing(tmp_path: Path) -> None:
    result = stats("w1", data_dir=tmp_path)
    assert result.run_count == 0
    assert result.percentiles == {}


def test_stats_filters_by_workflow_id(tmp_path: Path) -> None:
    _seed_runs(tmp_path, "w1", [{"think": 10}, {"think": 20}])
    _seed_runs(tmp_path, "w2", [{"think": 999}])
    result = stats("w1", data_dir=tmp_path)
    assert result.run_count == 2
    # p50 of [10, 20] = 15 (linear interpolation).
    assert result.percentiles["think"][0] == 15.0


def test_stats_computes_percentiles_per_bucket(tmp_path: Path) -> None:
    _seed_runs(tmp_path, "w1", [
        {"think": 10, "act": 1},
        {"think": 20, "act": 2},
        {"think": 30, "act": 3},
        {"think": 40, "act": 4},
        {"think": 50, "act": 5},
    ])
    result = stats("w1", data_dir=tmp_path)
    assert result.run_count == 5
    # think samples: 10,20,30,40,50 → p50=30, p95≈48, p99≈49.6
    p50, p95, p99 = result.percentiles["think"]
    assert p50 == 30.0
    assert 47.5 <= p95 <= 49.0
    assert 49.0 <= p99 <= 50.0
    assert result.percentiles["total_time_s"][0] == 33.0  # p50 of 11..55


def test_stats_respects_last_n_cap_newest_first(tmp_path: Path) -> None:
    """Older runs beyond last_n must be ignored — last_n=2 over five
    runs should aggregate only the last two appended."""
    _seed_runs(tmp_path, "w1", [
        {"think": 100}, {"think": 200}, {"think": 300},
        {"think": 1}, {"think": 2},  # most recent
    ])
    result = stats("w1", last_n=2, data_dir=tmp_path)
    assert result.run_count == 2
    # p50 of [1, 2] = 1.5
    assert result.percentiles["think"][0] == 1.5


def test_stats_status_filter_default_excludes_failures(tmp_path: Path) -> None:
    _seed_runs(tmp_path, "w1", [{"think": 10}])
    _seed_runs(tmp_path, "w1", [{"think": 999}], status="failed")
    result = stats("w1", data_dir=tmp_path)
    assert result.run_count == 1
    assert result.percentiles["think"][0] == 10.0


def test_stats_status_filter_empty_includes_all_terminals(tmp_path: Path) -> None:
    _seed_runs(tmp_path, "w1", [{"think": 10}])
    _seed_runs(tmp_path, "w1", [{"think": 50}], status="failed")
    result = stats("w1", status_filter="", data_dir=tmp_path)
    assert result.run_count == 2


def test_stats_since_drops_old_rows(tmp_path: Path) -> None:
    now = datetime(2026, 5, 13, 18, 0, 0, tzinfo=timezone.utc)
    # Manually write rows with explicit finished_at so we can test the
    # cutoff. One an hour ago, one a week ago.
    for run_id, hours in (("recent", 1), ("ancient", 168)):
        row = row_from_result(
            run_id=run_id, workflow_id="w1", status="succeeded",
            finished_at=(now - timedelta(hours=hours)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            result={"total_time_s": 100, "wall_time_breakdown": {"think": 10}},
        )
        append_run(row, data_dir=tmp_path, now=now)
    # since=24h → only the recent row should land.
    result = stats(
        "w1", since=timedelta(hours=24), data_dir=tmp_path, _now=now,
    )
    assert result.run_count == 1


def test_stats_buckets_filter_restricts_output(tmp_path: Path) -> None:
    _seed_runs(tmp_path, "w1", [
        {"think": 10, "act": 1, "settle": 5},
        {"think": 20, "act": 2, "settle": 6},
    ])
    result = stats("w1", buckets=["think"], data_dir=tmp_path)
    assert "think" in result.percentiles
    assert "act" not in result.percentiles
    assert "settle" not in result.percentiles
    # total_time_s is always included as the headline aggregate.
    assert "total_time_s" in result.percentiles


def test_stats_skips_malformed_lines(tmp_path: Path) -> None:
    """A truncated line from a crashed write must not poison the query."""
    # Seed one valid run.
    _seed_runs(tmp_path, "w1", [{"think": 10}])
    # Append a malformed line.
    path = (tmp_path / RUNS_LOG_SUBDIR).glob("*.jsonl").__next__()
    with open(path, "a") as fh:
        fh.write("{not valid json\n")
    # Seed another valid run.
    _seed_runs(tmp_path, "w1", [{"think": 20}])

    result = stats("w1", data_dir=tmp_path)
    assert result.run_count == 2
