"""Tests for the Phase-2 Table 1 consolidator."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.learning_allocator import consolidate

_BT02_TABLE = (
    "# SOURCE=live-modal-daytona — REAL agent runs\n"
    "policy\tsealed_score\tvisible_score\tvisible_minus_sealed\t"
    "score_per_dollar\ttotal_dollars\tn_runs\n"
    "frozen\t0.0\t0.0\t0.0\t0.0\t0.4355\t8\n"
    "S0_only\t1.0\t1.0\t0.0\t21.8579\t0.366\t8\n"
    "oracle_allocator\t1.0\t1.0\t0.0\t22.4972\t0.0889\t2\n"
)

_BT03_TABLE = (
    "# SOURCE=live-modal-daytona — REAL agent runs\n"
    "policy\tsealed_score\tvisible_score\tvisible_minus_sealed\t"
    "score_per_dollar\ttotal_dollars\tn_runs\n"
    "frozen\t0.0\t0.0\t0.0\t0.0\t0.20\t4\n"
    "S1_only\t1.0\t1.0\t0.0\t8.0\t0.25\t4\n"
)


def test_collect_marks_bt03_pending_when_matrix_missing(tmp_path: Path) -> None:
    (tmp_path / "bt02_matrix_table1.tsv").write_text(_BT02_TABLE)
    rows, pending = consolidate.collect(tmp_path)
    assert {c.cluster for c in pending} == {"policy"}
    knowledge = [r for r in rows if r.cluster.cluster == "knowledge"]
    assert {r.policy for r in knowledge} == {"frozen", "S0_only", "oracle_allocator"}
    policy = [r for r in rows if r.cluster.cluster == "policy"]
    assert all(r.sealed_score is None for r in policy)
    assert {r.policy for r in policy} == {"frozen", "S0_only", "S1_only", "allocator"}


def test_collect_skips_offline_only_cluster(tmp_path: Path) -> None:
    # BT01 has matrix_stem=None (offline-only). Even with no files at all, it
    # must NOT show up in the consolidated table — it's intentionally not a
    # live row, not a missing-file gap.
    rows, pending = consolidate.collect(tmp_path)
    assert all(r.cluster.cluster != "capability" for r in rows)
    assert all(c.cluster != "capability" for c in pending)


def test_collect_reads_both_clusters_when_present(tmp_path: Path) -> None:
    (tmp_path / "bt02_matrix_table1.tsv").write_text(_BT02_TABLE)
    (tmp_path / "bt03_gated_matrix_table1.tsv").write_text(_BT03_TABLE)
    rows, pending = consolidate.collect(tmp_path)
    assert pending == []
    policy = [r for r in rows if r.cluster.cluster == "policy"]
    assert {r.policy for r in policy} == {"frozen", "S1_only"}
    s1 = next(r for r in policy if r.policy == "S1_only")
    assert s1.sealed_score == 1.0 and s1.n_runs == 4


def test_format_table_prints_pending_live_literally(tmp_path: Path) -> None:
    rows, _ = consolidate.collect(tmp_path)
    out = consolidate.format_table(rows)
    assert "PENDING-LIVE" in out


def test_write_tsv_carries_provenance_banner(tmp_path: Path) -> None:
    (tmp_path / "bt02_matrix_table1.tsv").write_text(_BT02_TABLE)
    rows, _ = consolidate.collect(tmp_path)
    out_path = tmp_path / "phase2_table1.tsv"
    consolidate.write_tsv(rows, out_path)
    text = out_path.read_text()
    assert text.startswith("# SOURCE=phase2-consolidator")
    # Header column ordering — locks the wire shape for downstream readers.
    header_line = [ln for ln in text.splitlines() if not ln.startswith("#")][0]
    cols = header_line.split("\t")
    assert cols[:3] == ["cluster", "expects_substrate", "policy"]


def test_read_table1_rejects_missing_columns(tmp_path: Path) -> None:
    # A truncated TSV (someone hand-edited it down) must surface loudly —
    # otherwise the consolidator silently shows a partial cluster row.
    bad = tmp_path / "bt02_matrix_table1.tsv"
    bad.write_text("policy\tsealed_score\nfrozen\t0.0\n")
    with pytest.raises(ValueError, match="missing expected columns"):
        consolidate.collect(tmp_path)


def test_main_strict_returns_2_when_pending(tmp_path: Path) -> None:
    rc = consolidate.main(
        ["--eval-dir", str(tmp_path), "--out", str(tmp_path / "out.tsv"), "--strict"],
    )
    assert rc == 2


def test_main_strict_returns_0_when_all_clusters_have_matrix(tmp_path: Path) -> None:
    (tmp_path / "bt02_matrix_table1.tsv").write_text(_BT02_TABLE)
    (tmp_path / "bt03_gated_matrix_table1.tsv").write_text(_BT03_TABLE)
    rc = consolidate.main(
        ["--eval-dir", str(tmp_path), "--out", str(tmp_path / "out.tsv"), "--strict"],
    )
    assert rc == 0
