"""Phase-2 Table 1 + Fig 1 consolidator across failure clusters.

PLAN §5 names the headline artifacts of the Learning Allocator paper-port:

* **Table 1** — per policy: sealed oracle score, score/dollar, visible−sealed
  overfitting gap.
* **Fig 1** — per policy: running oracle score vs cumulative dollars.

Each cluster's matrix lives in its own pair of TSVs under
``experiments/learning_allocator/eval/``:

* ``bt02_matrix_results.tsv`` / ``bt02_matrix_table1.tsv`` — knowledge cluster
  (live, validated 2026-06-01 — frozen 0/0, S0 1/1, allocator 1/1).
* ``bt03_gated_matrix_results.tsv`` / ``bt03_gated_matrix_table1.tsv`` — the
  gated-reveal policy cluster (gated on a live BT03 smoke; this consolidator
  marks it ``PENDING-LIVE`` when the file is absent so the table is regenerable
  the moment that file lands).
* BT01 (capability cluster, expected substrate S2) is intentionally held out
  of the live set under the no-dealer-lead run policy — it stays ``OFFLINE-
  ONLY`` and never has a live row.

The consolidator deliberately does NOT re-grade or re-summarize from
``*_results.tsv`` rows when a ``*_table1.tsv`` already exists; the matrix
runs are the ground truth and this tool is a presentation layer. It only
*falls back* to re-aggregating from ``*_results.tsv`` if the table file is
missing — useful for an in-progress streaming run.

Run it::

    uv run python -m experiments.learning_allocator.consolidate \\
        --eval-dir experiments/learning_allocator/eval \\
        --out experiments/learning_allocator/eval/phase2_table1.tsv

A summary is also printed to stdout (the file is the durable artifact, the
print is the human-readable check).
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = REPO_ROOT / "experiments" / "learning_allocator" / "eval"

# Cluster manifest — drives presentation order and the PENDING-LIVE detection.
# Order = the order PLAN §3 introduces them: capability, knowledge, policy.
#
# ``matrix_stem`` is None for an offline-only cluster (BT01: held out of the
# live run under the no-dealer-lead policy).


@dataclass(frozen=True)
class Cluster:
    name: str
    cluster: str  # capability / knowledge / policy
    expects_substrate: str  # which rung the manifest expects to win
    plan: str | None
    matrix_stem: str | None
    note: str


CLUSTERS: tuple[Cluster, ...] = (
    Cluster(
        name="BT01_lead_capture_filtered_search",
        cluster="capability",
        expects_substrate="S2",
        plan=None,
        matrix_stem=None,
        note=(
            "Held out of the live set under the no-dealer-lead policy. Oracle "
            "stays gradeable for offline use; not a live Table 1 row."
        ),
    ),
    Cluster(
        name="BT02_spec_lookup_engine",
        cluster="knowledge",
        expects_substrate="S0",
        plan="bt02_spec_lookup",
        matrix_stem="bt02_matrix",
        note=(
            "Caterpillar-powered lead via detail-page spec lookup. The "
            "S0 anchor (#765) reorders the listing scan to the rank-5 buried "
            "Caterpillar — frozen budget-caps before reach."
        ),
    ),
    Cluster(
        name="BT03_byowner_phone_reveal",
        cluster="policy",
        expects_substrate="S1",
        plan="bt03_gated_reveal",
        matrix_stem="bt03_gated_matrix",
        note=(
            "Gated-reveal discriminator (#773): the env refuses the "
            "phone_revealed mutation until a non-obvious contact-start "
            "prerequisite is sent. Frozen runs the plan as-authored and the "
            "reveal is a server-side no-op; S1 injects the missing prereq "
            "ahead of the reveal."
        ),
    ),
)


# The columns the per-cluster ``*_table1.tsv`` emits, in order. Must stay in
# sync with ``runner.format_table1`` / ``runner.build_table1`` — and the BT02
# file shipped with this PR was written by exactly that pair (the SOURCE
# banner line on the file tags it ``live-modal-daytona``, so we know the row
# shape matches).
_T1_COLS = (
    "policy",
    "sealed_score",
    "visible_score",
    "visible_minus_sealed",
    "score_per_dollar",
    "total_dollars",
    "n_runs",
)


@dataclass(frozen=True)
class Row:
    cluster: Cluster
    policy: str
    sealed_score: float | None  # None ⇒ PENDING-LIVE
    visible_score: float | None
    visible_minus_sealed: float | None
    score_per_dollar: float | None
    total_dollars: float | None
    n_runs: int


def _read_table1(path: Path, cluster: Cluster) -> list[Row]:
    """Read a per-cluster ``*_table1.tsv`` into ``Row`` records.

    Skips any leading banner lines (``#`` prefix) so the consolidator stays
    compatible with both the live-runner output (``# SOURCE=live-modal-…``)
    and a hand-edited file.
    """
    rows: list[Row] = []
    with path.open() as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    reader = csv.DictReader(lines, delimiter="\t")
    missing = set(_T1_COLS) - set(reader.fieldnames or ())
    if missing:
        raise ValueError(
            f"{path}: missing expected columns {sorted(missing)}; got "
            f"{reader.fieldnames!r}",
        )
    for r in reader:
        rows.append(
            Row(
                cluster=cluster,
                policy=r["policy"],
                sealed_score=float(r["sealed_score"]),
                visible_score=float(r["visible_score"]),
                visible_minus_sealed=float(r["visible_minus_sealed"]),
                score_per_dollar=float(r["score_per_dollar"]),
                total_dollars=float(r["total_dollars"]),
                n_runs=int(r["n_runs"]),
            ),
        )
    return rows


def _pending_rows(cluster: Cluster) -> list[Row]:
    """Placeholder rows for a cluster whose live matrix has not landed yet.

    Emits the policies the PLAN says we will report (PLAN §4 baselines minus
    the oracle_allocator, which is derived post hoc from the others), with all
    metric fields ``None`` — the renderer prints them as ``PENDING-LIVE`` so
    the missing row is loud, not invisible.
    """
    pending_policies = ("frozen", "S0_only", "S1_only", "allocator")
    return [
        Row(
            cluster=cluster, policy=p,
            sealed_score=None, visible_score=None, visible_minus_sealed=None,
            score_per_dollar=None, total_dollars=None, n_runs=0,
        )
        for p in pending_policies
    ]


def collect(eval_dir: Path) -> tuple[list[Row], list[Cluster]]:
    """Return ((all rows, sorted by cluster then policy), pending clusters).

    A cluster with ``matrix_stem=None`` (e.g. BT01, offline-only) is silently
    excluded — it is NOT a Phase-2 live row by design. A cluster with a
    matrix_stem but no on-disk file is included as ``PENDING-LIVE`` and also
    listed in the returned ``pending`` set, so the caller can decide whether
    to fail the run.
    """
    rows: list[Row] = []
    pending: list[Cluster] = []
    for c in CLUSTERS:
        if c.matrix_stem is None:
            continue
        path = eval_dir / f"{c.matrix_stem}_table1.tsv"
        if path.exists():
            rows.extend(_read_table1(path, c))
        else:
            pending.append(c)
            rows.extend(_pending_rows(c))
    return rows, pending


def _fmt(v: float | None, fmt: str) -> str:
    return "PENDING-LIVE" if v is None else format(v, fmt)


def format_table(rows: list[Row]) -> str:
    """Render the consolidated Table 1 as a fixed-width text table."""
    headers = (
        "cluster", "policy", "sealed", "visible", "vis-seal",
        "score/$", "dollars", "n",
    )
    widths = [12, 18, 8, 8, 9, 8, 8, 4]
    out = [
        "  ".join(h.ljust(w) for h, w in zip(headers, widths)),
        "  ".join("-" * w for w in widths),
    ]
    for r in rows:
        out.append("  ".join([
            r.cluster.cluster.ljust(widths[0]),
            r.policy.ljust(widths[1]),
            _fmt(r.sealed_score, ".2f").rjust(widths[2]),
            _fmt(r.visible_score, ".2f").rjust(widths[3]),
            _fmt(r.visible_minus_sealed, "+.4f").rjust(widths[4]),
            _fmt(r.score_per_dollar, ".2f").rjust(widths[5]),
            _fmt(r.total_dollars, ".4f").rjust(widths[6]),
            str(r.n_runs).rjust(widths[7]),
        ]))
    return "\n".join(out)


def write_tsv(rows: list[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        fh.write(
            "# SOURCE=phase2-consolidator — joins per-cluster matrix TSVs "
            "under experiments/learning_allocator/eval/ into the headline "
            "Table 1 (PLAN §5). PENDING-LIVE = the cluster's live matrix "
            "has not landed; rerun after the smoke completes.\n",
        )
        w = csv.writer(fh, delimiter="\t")
        w.writerow([
            "cluster", "expects_substrate", "policy",
            "sealed_score", "visible_score", "visible_minus_sealed",
            "score_per_dollar", "total_dollars", "n_runs",
        ])
        for r in rows:
            w.writerow([
                r.cluster.cluster, r.cluster.expects_substrate, r.policy,
                _fmt(r.sealed_score, ".4f"),
                _fmt(r.visible_score, ".4f"),
                _fmt(r.visible_minus_sealed, ".4f"),
                _fmt(r.score_per_dollar, ".4f"),
                _fmt(r.total_dollars, ".4f"),
                r.n_runs,
            ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir", default=str(DEFAULT_EVAL_DIR),
        help="dir holding per-cluster matrix TSVs",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_EVAL_DIR / "phase2_table1.tsv"),
        help="consolidated Table 1 output path (TSV)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="exit 2 if any non-offline cluster is PENDING-LIVE",
    )
    args = parser.parse_args(argv)
    rows, pending = collect(Path(args.eval_dir))
    write_tsv(rows, Path(args.out))
    print(format_table(rows))
    if pending:
        print(
            "\nPENDING-LIVE clusters (no matrix file on disk):",
            file=sys.stderr,
        )
        for c in pending:
            print(
                f"  - {c.cluster:<10}  {c.name}  (plan={c.plan}, "
                f"expects {c.expects_substrate}) — "
                f"{c.note}",
                file=sys.stderr,
            )
        if args.strict:
            return 2
    print(f"\nwrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
