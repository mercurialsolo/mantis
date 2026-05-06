"""``mantis`` command-line interface — first-class plan-authoring product surface (#154).

Subcommands:
    plan validate <path>    Run PlanValidator on a JSON micro-plan.
                            Exits 0 on clean, 1 on errors, 2 on warnings only.
    plan dry-run <path>     Walk the plan graph and print the step sequence
                            the runner would attempt — no browser, no API
                            calls, no model load. Annotates gates, loops,
                            required steps, sections.

Planned follow-up (#154):
    init <url> [--task ...] Scaffold a starter plan via PlanDecomposer.

The CLI is wired through ``mantis_agent/main.py``: ``mantis plan ...``
invocations dispatch to this module BEFORE any heavy import (no
transformers / torch / mss / pyautogui), so the plan-authoring surface
stays fast.

    mantis plan validate examples/extract_listings.json
    mantis plan dry-run examples/extract_jobs.json
    mantis plan validate path.json --json   # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


# Exit codes documented in the module docstring.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_WARNING = 2


def _load_plan(source: str) -> tuple[Any, str]:
    """Load a plan JSON from a file path or ``-`` (stdin).

    Returns ``(plan_obj, label)`` where ``label`` is what to print in error
    messages (file path or ``<stdin>``).
    """
    if source == "-":
        return json.load(sys.stdin), "<stdin>"
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"plan file not found: {source}")
    with path.open() as fh:
        return json.load(fh), str(path)


def _format_issue(issue: Any, label: str) -> str:
    """Render a :class:`PlanIssue` for terminal output."""
    sev = issue.severity.upper()
    where = f"step[{issue.step_index}]" if issue.step_index >= 0 else "plan"
    line = f"  {sev:7s} {where:8s} {issue.code}: {issue.message}"
    if issue.auto_fix:
        line += f"\n           auto-fix: {issue.auto_fix}"
    return line


def cmd_plan_validate(args: argparse.Namespace) -> int:
    """``mantis plan validate <path>`` — run :class:`PlanValidator` on a JSON plan.

    Exits 0 on a clean plan (no issues), 2 if all issues are warnings,
    1 if at least one error is found. ``--json`` emits machine-readable
    output instead of human-formatted lines.
    """
    from .graph.plan_validator import PlanValidator
    from .plan_decomposer import MicroPlan

    try:
        payload, label = _load_plan(args.path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in {args.path}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        plan = MicroPlan.from_dict(payload)
    except Exception as exc:  # noqa: BLE001 — anything not JSON-shaped
        print(f"error: cannot parse plan from {label}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    validator = PlanValidator()
    issues = validator.validate(plan)

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    if args.json:
        print(json.dumps({
            "path": label,
            "step_count": len(plan.steps),
            "errors": [_issue_to_dict(i) for i in errors],
            "warnings": [_issue_to_dict(i) for i in warnings],
        }, indent=2))
    else:
        print(f"{label}: {len(plan.steps)} steps")
        if not issues:
            print("  ✓ clean — no issues")
        else:
            for issue in issues:
                print(_format_issue(issue, label))
            print(
                f"\nresult: {len(errors)} error(s), {len(warnings)} warning(s)"
            )

    if errors:
        return EXIT_ERROR
    if warnings:
        return EXIT_WARNING
    return EXIT_OK


def _issue_to_dict(issue: Any) -> dict[str, Any]:
    return {
        "severity": issue.severity,
        "code": issue.code,
        "message": issue.message,
        "step_index": issue.step_index,
        "auto_fix": issue.auto_fix,
    }


# ── plan dry-run ──────────────────────────────────────────────────────


def _annotate_step(idx: int, step: Any) -> str:
    """Render one step as a single-line human-readable summary.

    Format::

        [00] navigate           !req           "Navigate to https://example.com..."
        [01] extract_data       gate  setup    "Verify page has loaded..."
        [05] loop                     extract  → step [02] (count=10)

    The flag column shows ``!req`` for required, ``gate`` for verification
    gates, ``cl`` for ``claude_only`` extract steps. The section column
    shows the step's ``section`` field. Loop steps show their target index
    and configured count.
    """
    flags: list[str] = []
    if step.required:
        flags.append("!req")
    if step.gate:
        flags.append("gate")
    if step.claude_only:
        flags.append("cl")
    flag_str = " ".join(flags) if flags else "—"
    section_str = step.section or "—"

    intent_or_target = step.intent.replace("\n", " ").strip()[:60]
    if step.type == "loop":
        loop_target = step.loop_target if step.loop_target >= 0 else "?"
        loop_count = step.loop_count if step.loop_count > 0 else "∞"
        target_repr = f"→ step [{loop_target:02d}]" if isinstance(loop_target, int) else f"→ step [{loop_target}]"
        intent_or_target = f"{target_repr} (count={loop_count})"

    return (
        f"  [{idx:02d}] {step.type:18s} {flag_str:14s} "
        f"{section_str:11s} \"{intent_or_target}\""
    )


def _section_summary(steps: list[Any]) -> dict[str, int]:
    """Count steps per section for the header rollup."""
    counts: dict[str, int] = {}
    for s in steps:
        key = s.section or "—"
        counts[key] = counts.get(key, 0) + 1
    return counts


def cmd_plan_dry_run(args: argparse.Namespace) -> int:
    """``mantis plan dry-run <path>`` — print the step sequence the runner would attempt.

    No browser, no API calls, no model load. Pure structural walk. Useful
    as the inner authoring loop before paying the 9-13 min Modal/Baseten
    roundtrip.

    Returns 0 on a parseable plan, 1 on file or JSON error.
    """
    from .plan_decomposer import MicroPlan

    try:
        payload, label = _load_plan(args.path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in {args.path}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        plan = MicroPlan.from_dict(payload)
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot parse plan from {label}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.json:
        print(json.dumps({
            "path": label,
            "domain": plan.domain,
            "step_count": len(plan.steps),
            "sections": _section_summary(plan.steps),
            "steps": [
                {
                    "index": i,
                    "type": s.type,
                    "intent": s.intent,
                    "section": s.section,
                    "required": s.required,
                    "gate": s.gate,
                    "claude_only": s.claude_only,
                    "loop_target": s.loop_target,
                    "loop_count": s.loop_count,
                    "verify": s.verify,
                    "params": s.params,
                    "hints": s.hints,
                }
                for i, s in enumerate(plan.steps)
            ],
        }, indent=2))
        return EXIT_OK

    sections = _section_summary(plan.steps)
    section_str = ", ".join(f"{k}={v}" for k, v in sections.items())
    print(
        f"{label}: {len(plan.steps)} steps"
        + (f" — {plan.domain}" if plan.domain else "")
    )
    if section_str:
        print(f"  sections: {section_str}")
    print()
    print(f"  {'idx':4s} {'type':18s} {'flags':14s} {'section':11s} intent / target")
    print(f"  {'-'*4} {'-'*18} {'-'*14} {'-'*11} {'-'*40}")
    for i, step in enumerate(plan.steps):
        print(_annotate_step(i, step))

    # Check for non-loop steps that reference an out-of-range loop_target
    # — surfaces broken plans the same way the runner will at execution
    # time, so authors see the issue at dry-run.
    bad_targets: list[tuple[int, int]] = []
    for i, step in enumerate(plan.steps):
        if step.type == "loop" and step.loop_target >= 0:
            if step.loop_target >= len(plan.steps):
                bad_targets.append((i, step.loop_target))
    if bad_targets:
        print()
        for idx, target in bad_targets:
            print(
                f"  WARNING step[{idx}] loop_target={target} is out of range "
                f"(plan has {len(plan.steps)} steps)"
            )

    return EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mantis",
        description="Mantis CUA — plan authoring + ops CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Plan authoring subcommands")
    plan_sub = plan.add_subparsers(dest="plan_command", required=True)

    validate = plan_sub.add_parser(
        "validate",
        help="Run PlanValidator on a JSON micro-plan and report issues",
    )
    validate.add_argument(
        "path",
        help="Path to JSON plan file, or '-' to read stdin",
    )
    validate.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-formatted lines",
    )
    validate.set_defaults(func=cmd_plan_validate)

    dry_run = plan_sub.add_parser(
        "dry-run",
        help="Walk the plan graph and print the step sequence — no browser",
    )
    dry_run.add_argument(
        "path",
        help="Path to JSON plan file, or '-' to read stdin",
    )
    dry_run.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-formatted lines",
    )
    dry_run.set_defaults(func=cmd_plan_dry_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. ``argv`` defaults to ``sys.argv[1:]``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return EXIT_ERROR
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
