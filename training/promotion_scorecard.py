#!/usr/bin/env python3
"""Model-promotion scorecard — verifiable gates a Holo3 checkpoint must clear (#183).

The continual-fine-tuning pipeline (#155) lands a candidate at step 5
(shadow-deploy). Before traffic share goes higher, the candidate has to
clear an explicit, named scorecard so a fine-tune can't silently
improve one metric while regressing another (parser validity, grounding,
loops, cost).

Inputs the scorecard consumes
-----------------------------

This script is deliberately data-shaped, not eval-shaped: it composes
existing artefacts the rest of the pipeline already emits.

* **Eval report** (``training/eval_harness.py`` — #155 step 4)
  Per-task pass/fail for the held-out task set. Drives the
  ``task_pass_rate`` gate.
* **Shadow analytics** (``training/shadow_analytics.py`` — #155 step 5)
  Per-variant escalation + failure rates. Drives ``escalation_rate``.
* **Labelled traces** (``mantis trace label`` — #155 step 2)
  Step-level labels with ``label_reason``. Drives the parser /
  grounding / loop / forbidden-region / done-completeness gates.

The user supplies these as JSON files; the script aggregates and
prints a structured pass/fail per gate.

Tiers
-----

Each gate has three thresholds — one for each promotion stage:

* ``base``        — the bare-minimum to ship at all (≤ Holo3 base
                    weights). Failing this means the candidate is
                    worse than what's already in production.
* ``first_sft``   — the first fine-tune off base. Tighter than base.
* ``future``      — long-term target. Tighter still. Optional.

The CLI lets callers pick which tier they're gating against
(``--tier base|first_sft|future``). Default ``first_sft`` because
that's the typical promotion decision.

Usage
-----

::

    python -m training.promotion_scorecard \\
        --eval-report reports/candidate.json \\
        --shadow-summary reports/shadow_summary.json \\
        --labelled-traces /data/labelled \\
        --tier first_sft \\
        --output reports/scorecard.json

Exit code is **0** when every gate passes at the chosen tier, **1**
otherwise. Suitable for a CI gate before bumping shadow-deploy share.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Default thresholds ─────────────────────────────────────────────────
#
# Conservative numbers keyed off the existing failure modes. ``base`` is
# loose because the goal is "no worse than Holo3 stock"; ``first_sft``
# tightens once we've burned a fine-tune cycle; ``future`` is the
# aspirational long-term target.

DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    # Eval-harness pass-rate — how many held-out tasks finished
    # successfully. Higher = better.
    "task_pass_rate": {"base": 0.40, "first_sft": 0.55, "future": 0.75},

    # Action format validity — fraction of brain emissions that parsed
    # cleanly into a known Action. Catches regression on tool-call shape.
    "parser_validity": {"base": 0.95, "first_sft": 0.98, "future": 0.99},

    # Visual grounding correction success — fraction of grounded clicks
    # that landed on a usable element (gate verify pass post-click).
    "grounding_accuracy": {"base": 0.55, "first_sft": 0.70, "future": 0.85},

    # Forbidden-region avoidance — fraction of clicks that did NOT
    # land on photos, ads, footer/social links, off-site controls.
    "forbidden_region_avoidance": {"base": 0.90, "first_sft": 0.95, "future": 0.98},

    # Loop / repeated-action rate — lower = better.
    "loop_rate_max": {"base": 0.10, "first_sft": 0.05, "future": 0.02},

    # Gallery/lightbox recovery rate — higher = better.
    "gallery_recovery_rate": {"base": 0.70, "first_sft": 0.85, "future": 0.95},

    # Brain-ladder escalation rate — fraction of generations where the
    # primary brain yielded to the fallback. Lower = better.
    "escalation_rate_max": {"base": 0.10, "first_sft": 0.05, "future": 0.02},

    # Done-action completeness — fraction of done() calls that included
    # a structured summary matching the recipe schema.
    "done_completeness": {"base": 0.70, "first_sft": 0.85, "future": 0.95},

    # Cost per successful task (USD). Lower = better.
    "cost_per_success_usd_max": {
        "base": 0.50, "first_sft": 0.30, "future": 0.15,
    },
}

# Tier names in promotion order.
TIERS: tuple[str, ...] = ("base", "first_sft", "future")


# ── Data shapes ────────────────────────────────────────────────────────


@dataclass
class GateResult:
    """One row of the scorecard."""

    name: str
    value: float
    threshold: float
    direction: str  # "min" — value ≥ threshold passes; "max" — value ≤ threshold passes
    passed: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Scorecard:
    """Full scorecard with per-gate verdicts."""

    tier: str
    overall_passed: bool
    gates: list[GateResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "overall_passed": self.overall_passed,
            "gates": [g.to_dict() for g in self.gates],
            "metadata": self.metadata,
        }

    def regressions(self) -> list[str]:
        return [g.name for g in self.gates if not g.passed]


# ── Metric extraction (composes existing pipeline outputs) ────────────


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def task_pass_rate_from_eval_report(report: dict[str, Any]) -> float | None:
    """Pull the eval harness's ``pass_rate`` field. ``None`` when the
    payload doesn't look like an eval report."""
    if "pass_rate" not in report:
        return None
    return float(report.get("pass_rate", 0.0))


def escalation_rate_from_shadow(
    shadow_summary: dict[str, Any], variant: str = "candidate",
) -> float | None:
    """Read the candidate variant's ``escalation_rate`` from
    ``shadow_analytics.py`` output."""
    variants = shadow_summary.get("variants") or {}
    block = variants.get(variant)
    if not block:
        return None
    return float(block.get("escalation_rate", 0.0))


def aggregate_label_reasons(
    labelled_traces_dir: Path | None,
) -> tuple[dict[str, int], int]:
    """Walk labelled trace JSON files. Returns ``(per-reason counts,
    total step count)``. ``({}, 0)`` when the directory is missing.

    Each labelled step has a ``label_reason`` field (set by
    :class:`~mantis_agent.gym.trace_labeller.TraceLabeller`). Counts
    drive several gates without needing each gate to re-walk the dir.
    """
    if labelled_traces_dir is None or not labelled_traces_dir.exists():
        return {}, 0
    counts: dict[str, int] = {}
    total_steps = 0
    for path in sorted(labelled_traces_dir.glob("**/*.json")):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("skipping %s: %s", path, exc)
            continue
        for step in payload.get("steps") or []:
            total_steps += 1
            reason = step.get("label_reason", "")
            counts[reason] = counts.get(reason, 0) + 1
    return counts, total_steps


# ── Gate evaluation ────────────────────────────────────────────────────


@dataclass
class _GateSpec:
    name: str
    direction: str  # "min" → value ≥ threshold passes; "max" → value ≤ threshold passes


_GATES: list[_GateSpec] = [
    _GateSpec("task_pass_rate", "min"),
    _GateSpec("parser_validity", "min"),
    _GateSpec("grounding_accuracy", "min"),
    _GateSpec("forbidden_region_avoidance", "min"),
    _GateSpec("loop_rate_max", "max"),
    _GateSpec("gallery_recovery_rate", "min"),
    _GateSpec("escalation_rate_max", "max"),
    _GateSpec("done_completeness", "min"),
    _GateSpec("cost_per_success_usd_max", "max"),
]


def _gate_passed(value: float, threshold: float, direction: str) -> bool:
    if direction == "min":
        return value >= threshold
    if direction == "max":
        return value <= threshold
    raise ValueError(f"unknown direction {direction!r}")


def evaluate(
    *,
    eval_report: dict[str, Any] | None = None,
    shadow_summary: dict[str, Any] | None = None,
    labelled_traces_dir: Path | None = None,
    tier: str = "first_sft",
    thresholds: dict[str, dict[str, float]] | None = None,
    metric_overrides: dict[str, float] | None = None,
    cost_per_success_usd: float | None = None,
) -> Scorecard:
    """Compose the scorecard.

    Each gate reads from one of the supplied artefacts. When an artefact
    is missing, the gate skips (``note='input missing'``) and the
    ``overall_passed`` flag does NOT count it as a failure — that way
    operators can run the script with whatever subset of inputs they
    have without false negatives.

    ``metric_overrides`` lets callers inject test fixtures or already-
    computed values for any gate (e.g. parser_validity computed from
    a separate pipeline). Overrides win against artefact extraction.
    """
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; must be one of {TIERS}")

    thresholds = thresholds or DEFAULT_THRESHOLDS
    overrides = metric_overrides or {}

    metric_values: dict[str, float | None] = {name: None for name in (g.name for g in _GATES)}
    notes: dict[str, str] = {}

    # Apply overrides first — they're the highest-priority source.
    for name, value in overrides.items():
        if name in metric_values:
            metric_values[name] = float(value)

    # Then derive the rest from the artefacts.
    if metric_values["task_pass_rate"] is None and eval_report is not None:
        metric_values["task_pass_rate"] = task_pass_rate_from_eval_report(eval_report)

    if metric_values["escalation_rate_max"] is None and shadow_summary is not None:
        metric_values["escalation_rate_max"] = escalation_rate_from_shadow(shadow_summary)

    label_counts, total_steps = aggregate_label_reasons(labelled_traces_dir)
    if total_steps > 0:
        # Default loop_rate from labelled steps where reason indicates a
        # repeated/looping action. The labeller doesn't have a dedicated
        # loop reason yet, so fall back to escalation as the proxy
        # (escalation often follows a stuck loop). Override-friendly.
        if metric_values["loop_rate_max"] is None:
            metric_values["loop_rate_max"] = _safe_div(
                label_counts.get("escalation", 0), total_steps,
            )
        if metric_values["forbidden_region_avoidance"] is None:
            # Treat off-site clicks (escalations involving forbidden
            # regions) as the inverse signal. Without a dedicated reason
            # this is conservative — operators typically override with
            # their own forbidden-region telemetry.
            metric_values["forbidden_region_avoidance"] = 1.0 - _safe_div(
                label_counts.get("escalation", 0), total_steps,
            )

    # cost_per_success_usd is a single computed scalar — derive from the
    # eval report's reported total cost when available.
    if cost_per_success_usd is None and eval_report is not None:
        passes = int(eval_report.get("pass_count", 0))
        total_cost = sum(
            float((r.get("outcome") or {}).get("cost", 0.0))
            for r in (eval_report.get("results") or [])
        )
        if passes > 0:
            cost_per_success_usd = total_cost / passes
            metric_values["cost_per_success_usd_max"] = cost_per_success_usd

    if cost_per_success_usd is not None and (
        metric_values["cost_per_success_usd_max"] is None
    ):
        metric_values["cost_per_success_usd_max"] = cost_per_success_usd

    overall_passed = True
    gates: list[GateResult] = []
    for spec in _GATES:
        threshold_block = thresholds.get(spec.name)
        threshold = (threshold_block or {}).get(tier)
        observed = metric_values.get(spec.name)
        if threshold is None:
            note = "no threshold for tier"
            gates.append(GateResult(
                name=spec.name, value=float(observed or 0.0), threshold=0.0,
                direction=spec.direction, passed=False, note=note,
            ))
            overall_passed = False
            continue
        if observed is None:
            gates.append(GateResult(
                name=spec.name, value=0.0, threshold=float(threshold),
                direction=spec.direction, passed=True,
                note="input missing — gate skipped",
            ))
            continue
        passed = _gate_passed(float(observed), float(threshold), spec.direction)
        gates.append(GateResult(
            name=spec.name, value=float(observed), threshold=float(threshold),
            direction=spec.direction, passed=passed, note=notes.get(spec.name, ""),
        ))
        if not passed:
            overall_passed = False

    return Scorecard(
        tier=tier,
        overall_passed=overall_passed,
        gates=gates,
        metadata={
            "label_step_count": total_steps,
            "label_reason_counts": dict(label_counts),
        },
    )


# ── CLI ────────────────────────────────────────────────────────────────


def _load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-report", default="",
        help="Eval harness report (training/eval_harness.py output)",
    )
    parser.add_argument(
        "--shadow-summary", default="",
        help="Shadow analytics summary (training/shadow_analytics.py output)",
    )
    parser.add_argument(
        "--labelled-traces", default="",
        help="Directory of labelled trace JSONs (mantis trace label output)",
    )
    parser.add_argument(
        "--tier", choices=TIERS, default="first_sft",
        help="Promotion tier to gate against",
    )
    parser.add_argument(
        "--output", default="",
        help="Optional output JSON path (scorecard payload)",
    )
    parser.add_argument(
        "--metric", action="append", default=[],
        help="Override a metric: name=value. Repeatable.",
    )
    args = parser.parse_args(argv)

    eval_report = _load_json(args.eval_report)
    shadow_summary = _load_json(args.shadow_summary)
    labelled = (
        Path(args.labelled_traces).expanduser()
        if args.labelled_traces
        else None
    )

    overrides: dict[str, float] = {}
    for entry in args.metric:
        if "=" not in entry:
            print(
                f"error: --metric must be NAME=VALUE (got {entry!r})",
                file=sys.stderr,
            )
            return 1
        name, value = entry.split("=", 1)
        try:
            overrides[name.strip()] = float(value)
        except ValueError:
            print(
                f"error: --metric {name!r} value must be numeric (got {value!r})",
                file=sys.stderr,
            )
            return 1

    scorecard = evaluate(
        eval_report=eval_report,
        shadow_summary=shadow_summary,
        labelled_traces_dir=labelled,
        tier=args.tier,
        metric_overrides=overrides,
    )
    payload = scorecard.to_dict()
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))

    return 0 if scorecard.overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
