"""``mantis`` command-line interface — first-class plan-authoring product surface (#154).

Subcommands (this PR):
    plan validate <path>    Run PlanValidator on a JSON micro-plan.
                            Exits 0 on clean, 1 on errors, 2 on warnings only.

Planned follow-ups (#154):
    plan dry-run <path>     Walk the plan graph without browser/Xvfb.
    init <url> [--task ...] Scaffold a starter plan via PlanDecomposer.

The CLI is registered as ``mantis = mantis_agent.cli:main`` in
``pyproject.toml::project.scripts``. Invocation:

    mantis plan validate examples/extract_listings.json
    mantis plan validate -            # read stdin
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
