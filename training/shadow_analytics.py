#!/usr/bin/env python3
"""Shadow-deploy analytics — compare escalation rates per variant (#155 step 5).

Reads labelled trace files (the output of ``mantis trace label``) and
computes, per ``variant``:

  * ``run_count`` — number of runs assigned to this variant
  * ``step_count`` — total executed steps
  * ``escalation_count`` — steps labelled ``negative`` with reason ``escalation``
  * ``escalation_rate`` — escalations / step_count
  * ``failure_count`` — steps labelled ``negative`` (any reason)
  * ``failure_rate`` — failures / step_count

Then reports the gap between ``baseline`` and ``candidate`` so operators
can decide whether the candidate weights are safe to promote to 100%.

Usage::

    python -m training.shadow_analytics \\
        --labelled /data/labelled \\
        --output reports/shadow_summary.json

Exit code is 0 when the candidate's escalation rate is ≤ baseline's
(within the configurable tolerance), 1 otherwise. Useful in CI.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Data shapes ────────────────────────────────────────────────────────


@dataclass
class VariantStats:
    """Aggregated escalation + failure metrics for one variant."""

    variant: str
    run_count: int = 0
    step_count: int = 0
    escalation_count: int = 0
    failure_count: int = 0

    @property
    def escalation_rate(self) -> float:
        return self.escalation_count / self.step_count if self.step_count else 0.0

    @property
    def failure_rate(self) -> float:
        return self.failure_count / self.step_count if self.step_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "run_count": self.run_count,
            "step_count": self.step_count,
            "escalation_count": self.escalation_count,
            "failure_count": self.failure_count,
            "escalation_rate": round(self.escalation_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
        }


@dataclass
class ShadowReport:
    """Full per-variant rollup plus baseline-vs-candidate comparison."""

    variants: dict[str, VariantStats] = field(default_factory=dict)

    def baseline(self) -> VariantStats | None:
        return self.variants.get("baseline")

    def candidate(self) -> VariantStats | None:
        return self.variants.get("candidate")

    def comparison(self) -> dict[str, Any]:
        """Empty dict when one side is missing — promotion gating
        requires both variants to have run."""
        b = self.baseline()
        c = self.candidate()
        if b is None or c is None:
            return {}
        return {
            "escalation_rate_delta": round(
                c.escalation_rate - b.escalation_rate, 4,
            ),
            "failure_rate_delta": round(
                c.failure_rate - b.failure_rate, 4,
            ),
            "baseline_runs": b.run_count,
            "candidate_runs": c.run_count,
            "candidate_escalation_rate_lower": (
                c.escalation_rate <= b.escalation_rate
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "variants": {k: v.to_dict() for k, v in sorted(self.variants.items())},
            "comparison": self.comparison(),
        }


# ── Aggregation ────────────────────────────────────────────────────────


def _iter_labelled_traces(root: Path) -> Iterable[Path]:
    """Walk ``root`` for labelled JSON trace files."""
    if root.is_file():
        yield root
        return
    for path in sorted(root.glob("**/*.json")):
        if path.is_file():
            yield path


def aggregate(input_dir: Path) -> ShadowReport:
    """Walk ``input_dir`` for labelled trace files and aggregate per-variant
    stats. Traces with no ``variant`` field land under the bucket
    ``__unassigned__`` so legacy traces don't get silently dropped.
    """
    report = ShadowReport()
    for path in _iter_labelled_traces(input_dir):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("skipping %s: %s", path, exc)
            continue
        variant = (payload.get("variant") or "").strip() or "__unassigned__"
        stats = report.variants.setdefault(variant, VariantStats(variant=variant))
        stats.run_count += 1
        for step in payload.get("steps") or []:
            stats.step_count += 1
            label = step.get("label", "")
            reason = step.get("label_reason", "")
            if label == "negative":
                stats.failure_count += 1
                if reason == "escalation":
                    stats.escalation_count += 1
    return report


# ── CLI ────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labelled", required=True,
        help="Labelled trace JSON file or directory (output of mantis trace label)",
    )
    parser.add_argument(
        "--output", default="",
        help="Optional output JSON path",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.0,
        help="Acceptable escalation_rate increase (candidate - baseline). "
        "Exit 1 when delta exceeds this. Default 0.0 (no regression allowed).",
    )
    args = parser.parse_args(argv)

    report = aggregate(Path(args.labelled).expanduser())
    payload = report.to_dict()
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  wrote {out}")

    print(json.dumps(payload, indent=2))

    cmp = report.comparison()
    if not cmp:
        # Single-variant or unassigned-only — nothing to gate on.
        return 0
    delta = cmp["escalation_rate_delta"]
    if delta > args.tolerance:
        print(
            f"\n  REGRESSION: candidate escalation_rate is {delta:+.4f} above "
            f"baseline (tolerance {args.tolerance:.4f}).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
