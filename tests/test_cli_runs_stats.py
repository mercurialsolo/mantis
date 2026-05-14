"""``mantis runs stats <workflow_id>`` CLI surface (epic #362 Phase C).

Pins the argument parsing, the table / JSON rendering, and the
``--since`` duration parser. End-to-end uses a tmp data dir seeded
with fake rows so no actual server is required.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from mantis_agent.cli import EXIT_OK, _parse_since, main
from mantis_agent.runs_log import append_run, row_from_result


# ── _parse_since duration parser ────────────────────────────────────────


def test_parse_since_days() -> None:
    assert _parse_since("7d") == timedelta(days=7)


def test_parse_since_hours() -> None:
    assert _parse_since("24h") == timedelta(hours=24)


def test_parse_since_minutes() -> None:
    assert _parse_since("30m") == timedelta(minutes=30)


def test_parse_since_rejects_unknown_suffix() -> None:
    assert _parse_since("7w") is None


def test_parse_since_rejects_non_numeric() -> None:
    assert _parse_since("xd") is None


def test_parse_since_rejects_negative() -> None:
    assert _parse_since("-1d") is None


def test_parse_since_empty_or_short() -> None:
    assert _parse_since("") is None
    assert _parse_since("d") is None


# ── stats CLI end-to-end ────────────────────────────────────────────────


def _seed(tmp_path: Path, samples: list[dict], workflow_id: str = "w1") -> None:
    for i, sample in enumerate(samples):
        row = row_from_result(
            run_id=f"r{i}", workflow_id=workflow_id, status="succeeded",
            finished_at=f"2026-05-13T18:{i:02d}:00Z",
            result={
                "total_time_s": sum(sample.values()),
                "wall_time_breakdown": sample,
            },
        )
        append_run(row, data_dir=tmp_path)


def test_stats_no_runs_returns_ok(tmp_path: Path, monkeypatch, capsys) -> None:
    """Empty store should not crash — return OK with a friendly note."""
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    code = main(["runs", "stats", "w1"])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "No runs matched" in out
    assert "w1" in out


def test_stats_table_render_orders_buckets_by_p50(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    _seed(tmp_path, [
        {"think": 40, "act": 5, "settle": 15},
        {"think": 50, "act": 6, "settle": 16},
    ])
    code = main(["runs", "stats", "w1"])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    # Header should mention the matched count + workflow.
    assert "Across last 2" in out
    assert "w1" in out
    # think (largest p50) lands before settle which lands before act.
    think_idx = out.find("think")
    settle_idx = out.find("settle")
    act_idx = out.find("act")
    assert 0 < think_idx < settle_idx < act_idx
    # total_time_s is always last.
    assert "total_time_s" in out
    assert out.find("total_time_s") > act_idx


def test_stats_json_output(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    _seed(tmp_path, [
        {"think": 10}, {"think": 20}, {"think": 30},
    ])
    code = main(["runs", "stats", "w1", "--json"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["workflow_id"] == "w1"
    assert payload["run_count"] == 3
    assert payload["percentiles"]["think"]["p50"] == 20.0
    assert "total_time_s" in payload["percentiles"]


def test_stats_bucket_filter_restricts_table(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    _seed(tmp_path, [
        {"think": 10, "act": 1, "settle": 5},
        {"think": 20, "act": 2, "settle": 6},
    ])
    code = main(["runs", "stats", "w1", "--bucket", "think"])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "think" in out
    # act / settle filtered out of the breakdown rows.
    # (we just check they don't appear as a row prefix)
    for line in out.splitlines():
        if line.lstrip().startswith(("act ", "settle ")):
            pytest.fail(f"unexpected bucket row: {line!r}")


def test_stats_last_n_caps_input(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    _seed(tmp_path, [
        {"think": 100}, {"think": 200}, {"think": 300},
        {"think": 1}, {"think": 2},
    ])
    code = main(["runs", "stats", "w1", "--last", "2", "--json"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_count"] == 2
    # Most recent two are [1, 2] → p50 = 1.5
    assert payload["percentiles"]["think"]["p50"] == 1.5


def test_stats_invalid_since_returns_error(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    code = main(["runs", "stats", "w1", "--since", "garbage"])
    assert code != EXIT_OK
    err = capsys.readouterr().err
    assert "--since" in err
    assert "7d" in err or "24h" in err
