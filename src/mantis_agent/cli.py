"""``mantis`` command-line interface — plan authoring + trace tooling + run.

Subcommands:
    plan validate <path>    Run PlanValidator on a JSON micro-plan.
                            Exits 0 on clean, 1 on errors, 2 on warnings only.
    plan dry-run <path>     Walk the plan graph and print the step sequence
                            the runner would attempt — no browser, no API
                            calls, no model load.
    plan init <url>         Scaffold a starter plan via PlanDecomposer
        --task "<desc>"     (Claude API call; ~\$0.005 per scaffold).
                            Validates and dry-runs before writing.
    plan run <path>         End-to-end execution against a remote Mantis brain
        --platform ...      (Baseten / Modal / custom) and a local browser.
        --endpoint ...      Loads the plan (text → decompose, JSON → direct),
                            wires Holo3Brain + ClaudeGrounding + ClaudeExtractor +
                            PlaywrightGymEnv, runs MicroPlanRunner, and writes
                            ``plan.json`` + ``result.json`` to ``--output-dir``.
    trace label <input>     Batch-label exported run traces (#155 step 2).
        --output <dir>      Emits one labelled JSON per input, with each
                            step tagged positive / negative / neutral.
    trace review <path>     Print a labelled summary of one trace JSON
                            for human inspection.

The CLI is wired through ``mantis_agent/main.py``: ``mantis plan ...``
and ``mantis trace ...`` invocations dispatch to this module BEFORE any
heavy import (no transformers / torch / mss / pyautogui), so the
plan-authoring + trace tooling surfaces stay fast. ``plan run`` does
import the heavy stack — it's the only subcommand that needs the
browser + brain client, so the import is gated inside the handler.

    mantis plan validate examples/extract_listings.json
    mantis plan dry-run examples/extract_jobs.json
    mantis plan init https://news.ycombinator.com --task "Extract top 10 stories"
    mantis plan run plans/staff-crm.txt \\
        --platform modal --endpoint https://workspace--app-fn.modal.run/v1 \\
        --output-dir outputs/staff-crm-validation
    mantis trace label /data/traces --output /data/labelled
    mantis trace review /data/traces/__shared__/run123.json
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


# ── plan init ──────────────────────────────────────────────────────────


def _default_output_path(url: str) -> str:
    """Derive a sensible filename from a URL, e.g. https://news.ycombinator.com/
    → news_ycombinator_com_plan.json. Hostname-only — no path / query salt
    so authors writing one plan per site get predictable names.
    """
    from urllib.parse import urlparse

    host = urlparse(url).netloc or url
    # Strip leading www., replace dots / dashes with underscores so the
    # filename is identifier-shaped (works as a Python module name later
    # if anyone uses it for code-gen).
    host = host.lower().lstrip(".")
    if host.startswith("www."):
        host = host[len("www."):]
    safe = "".join(c if c.isalnum() else "_" for c in host).strip("_")
    return f"{safe or 'plan'}_plan.json"


def _build_seed_plan_text(url: str, task: str) -> str:
    """Compose the prompt PlanDecomposer ingests.

    The decomposer expects free-form text; ``"{task} at {url}"`` is the
    minimal viable shape. Authors who want richer context (auth
    requirements, filters, output schema) can post-process the resulting
    JSON or hand-author from there.
    """
    if not task.strip():
        return f"Open {url} and describe what you see."
    return f"{task.strip()}\n\nTarget URL: {url}"


def cmd_plan_init(args: argparse.Namespace) -> int:
    """``mantis plan init <url> --task "..."`` — scaffold a starter plan.

    Calls :class:`PlanDecomposer` (one Claude API call, ~\$0.005), writes
    the resulting plan JSON to ``--output`` (or a derived path),
    optionally validates and dry-runs the result inline so authors see
    structural feedback at scaffold-time.

    Exits 0 on a written-and-validated plan, 1 on any error, 2 if the
    written plan has validation warnings only.
    """
    from .plan_decomposer import PlanDecomposer

    if not args.task:
        print("error: --task is required", file=sys.stderr)
        return EXIT_ERROR

    output_path = Path(args.output or _default_output_path(args.url))
    if output_path.exists() and not args.overwrite:
        print(
            f"error: {output_path} already exists — pass --overwrite to replace",
            file=sys.stderr,
        )
        return EXIT_ERROR

    plan_text = _build_seed_plan_text(args.url, args.task)
    print(f"Decomposing via Claude (api={args.model})…")

    decomposer = PlanDecomposer(model=args.model)
    try:
        plan = decomposer.decompose_text(plan_text)
    except RuntimeError as exc:
        # decompose_text raises RuntimeError when ANTHROPIC_API_KEY is unset.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 — surface decomposer errors verbatim
        print(f"error: decomposer failed: {exc}", file=sys.stderr)
        return EXIT_ERROR

    payload = plan.to_dict()
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote {output_path}  ({len(plan.steps)} steps)")

    if not args.no_validate:
        from .graph.plan_validator import PlanValidator
        issues = PlanValidator().validate(plan)
        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]
        if issues:
            print("\nValidator findings:")
            for issue in issues:
                print(_format_issue(issue, str(output_path)))
            print(
                f"  result: {len(errors)} error(s), {len(warnings)} warning(s)"
            )
            if errors:
                return EXIT_ERROR
            if warnings:
                return EXIT_WARNING
        else:
            print("  ✓ validator clean")

    if not args.no_dry_run:
        print("\nDry-run preview:")
        # Reuse the same renderer the dry-run subcommand uses so authors see
        # the identical row format they'll see if they re-run later.
        for i, step in enumerate(plan.steps):
            print(_annotate_step(i, step))

    return EXIT_OK


# ── trace label / review ──────────────────────────────────────────────


def cmd_trace_label(args: argparse.Namespace) -> int:
    """``mantis trace label <input> --output <dir>`` — batch-label traces.

    Walks ``input`` for ``*.json`` trace files and writes labelled
    versions to ``output`` mirroring the input subtree (so tenant
    directories survive the round-trip). Returns 0 on at least one
    successful label, 1 on any error.
    """
    from .gym.trace_labeller import TraceLabeller

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return EXIT_ERROR

    labeller = TraceLabeller()
    if input_path.is_file():
        labelled = labeller.label_trace_file(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        target = (
            output_path
            if output_path.suffix == ".json"
            else output_path / input_path.name
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(labelled.to_dict(), indent=2) + "\n")
        if args.json:
            print(json.dumps({str(input_path): labelled.summary()}, indent=2))
        else:
            print(f"  {input_path} → {target}  {labelled.summary()}")
        return EXIT_OK

    summary = labeller.label_directory(input_path, output_path)
    if not summary:
        print(f"warning: no traces found under {args.input}", file=sys.stderr)
        return EXIT_OK
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for rel, counts in summary.items():
            total = sum(counts.values())
            print(
                f"  {rel}  total={total}  pos={counts.get('positive', 0)}  "
                f"neg={counts.get('negative', 0)}  neu={counts.get('neutral', 0)}"
            )
        print(f"\n  labelled {len(summary)} traces → {output_path}")
    return EXIT_OK


def cmd_trace_review(args: argparse.Namespace) -> int:
    """``mantis trace review <path>`` — human-readable summary of one labelled trace.

    Read-only: applies the same heuristics as ``trace label`` but prints
    a per-step table to stdout instead of writing files. Useful for
    spot-checking a single run before committing the batch labels to a
    training set.
    """
    from .gym.trace_labeller import TraceLabeller

    path = Path(args.path)
    if not path.exists():
        print(f"error: trace not found: {args.path}", file=sys.stderr)
        return EXIT_ERROR
    try:
        labelled = TraceLabeller().label_trace_file(path)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in {args.path}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.json:
        print(json.dumps(labelled.to_dict(), indent=2))
        return EXIT_OK

    rollup = labelled.summary()
    print(
        f"{path}: run_id={labelled.run_id or '—'}  "
        f"tenant={labelled.tenant_id or '—'}  status={labelled.status or '—'}"
    )
    print(
        f"  totals: pos={rollup.get('positive', 0)}  "
        f"neg={rollup.get('negative', 0)}  neu={rollup.get('neutral', 0)}"
    )
    print()
    print(f"  {'idx':4s} {'label':9s} {'reason':24s} {'type':12s} intent / data")
    print(f"  {'-'*4} {'-'*9} {'-'*24} {'-'*12} {'-'*40}")
    for s in labelled.steps:
        intent = (s.intent or s.data or "").replace("\n", " ").strip()[:60]
        print(
            f"  [{s.step_index:02d}] {s.label:9s} {s.label_reason:24s} "
            f"{(s.type or '—'):12s} {intent}"
        )
    return EXIT_OK


# ── plan run ──────────────────────────────────────────────────────────


# Auth-header style per platform. Modal and Baseten both accept a Bearer
# token at the OpenAI-compatible endpoint, so the structural difference
# is mostly the URL shape (which the caller already supplies via
# --endpoint). ``custom`` exists for self-hosted gateways that need a
# different header — pass headers via ``--header`` repeats.
_PLATFORMS: tuple[str, ...] = ("modal", "baseten", "custom")


def _platform_default_model(platform: str) -> str:
    """Default brain model name per platform.

    The model name is informational on the Mantis side — vLLM / llama.cpp
    proxies route based on the served-model registration, not the value
    in the request body. Defaults match what the canonical deployments
    serve so a typical ``mantis plan run`` works without spelling it out.
    """
    if platform == "modal":
        return "Hcompany/Holo3-35B-A3B"
    if platform == "baseten":
        return "holo3-35b-a3b"
    return "holo3-35b-a3b"


def _parse_extra_headers(items: list[str] | None) -> dict[str, str]:
    """Parse ``--header KEY=VALUE`` repeats into a dict for the brain client."""
    out: dict[str, str] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"--header expects KEY=VALUE; got: {raw!r}")
        k, v = raw.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"--header has empty key: {raw!r}")
        out[k] = v
    return out


def _looks_like_text_plan(path: Path) -> bool:
    """Heuristic for the plan-format dispatch.

    A path with ``.json`` extension is treated as an already-decomposed
    micro-plan; anything else is a natural-language plan that needs a
    decomposer pass. Reading the file content as a sanity check is
    deliberately avoided so an empty / not-yet-written ``.txt`` still
    routes through the decomposer with a helpful error rather than a
    silent JSON parse fail.
    """
    return path.suffix.lower() not in {".json"}


def _resolve_first_navigate_url(plan_obj: Any) -> str:
    """Pull the first ``navigate`` step's URL out of a MicroPlan.

    Used as the default ``start_url`` for the browser env when the caller
    didn't pass ``--start-url``. Returns ``""`` if no navigate step has a
    URL — the runner will surface that as a no-URL navigate failure with
    a clear log line.
    """
    import re as _re

    for step in getattr(plan_obj, "steps", []):
        if step.type != "navigate":
            continue
        m = _re.search(r'https?://[^\s"]+', step.intent or "")
        if m:
            return m.group()
        params = step.params if isinstance(step.params, dict) else {}
        params_url = params.get("url")
        if isinstance(params_url, str):
            m2 = _re.search(r'https?://[^\s"]+', params_url)
            if m2:
                return m2.group()
    return ""


def cmd_plan_run(args: argparse.Namespace) -> int:
    """``mantis plan run <plan>`` — execute a plan against a remote Mantis brain.

    End-to-end smoke / validation runner: takes a natural-language plan
    (or pre-decomposed JSON), decomposes via PlanDecomposer if needed,
    wires :class:`Holo3Brain` (HTTP → ``--endpoint``) +
    :class:`ClaudeGrounding` + :class:`ClaudeExtractor` +
    :class:`PlaywrightGymEnv` into :class:`MicroPlanRunner`, runs the
    plan, and writes ``plan.json`` + ``result.json`` to ``--output-dir``.

    The handler keeps to the project's stay-generic discipline: the
    default :class:`SiteConfig` is the neutral one (relies on the #211
    path-extension heuristic for detail-page detection); a per-plan
    pattern can be injected via ``--detail-page-pattern`` rather than
    baked into the framework.

    Heavy imports — playwright, brain, runner — are gated inside this
    function so ``mantis plan validate`` / ``dry-run`` / ``init`` stay
    fast.

    Returns 0 if every step succeeded, 1 if any failed or the runner
    raised.
    """
    import os
    import time as _time

    # Plan source — text → decompose, JSON → load directly.
    plan_path = Path(args.path)
    if not plan_path.exists():
        print(f"error: plan file not found: {args.path}", file=sys.stderr)
        return EXIT_ERROR

    from .plan_decomposer import MicroPlan, PlanDecomposer

    # Raw payload retained so the env-harness path can read top-level
    # ``task_id`` metadata that ``MicroPlan.to_dict()`` strips on
    # round-trip. ``None`` when the plan came from a text source.
    raw_plan_payload: dict[str, Any] | None = None

    if _looks_like_text_plan(plan_path):
        plan_text = plan_path.read_text(encoding="utf-8")
        if not plan_text.strip():
            print(f"error: plan file is empty: {plan_path}", file=sys.stderr)
            return EXIT_ERROR
        anthropic_key = (
            args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        if not anthropic_key:
            print(
                "error: text plan requires Claude for decomposition; pass "
                "--anthropic-api-key or set ANTHROPIC_API_KEY",
                file=sys.stderr,
            )
            return EXIT_ERROR
        # PlanDecomposer reads ANTHROPIC_API_KEY from env; export it for the call.
        if anthropic_key and not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        print(f"Decomposing {plan_path.name} via Claude…")
        try:
            plan = PlanDecomposer().decompose_text(plan_text)
        except Exception as exc:  # noqa: BLE001
            print(f"error: decomposer failed: {exc}", file=sys.stderr)
            return EXIT_ERROR
    else:
        try:
            with plan_path.open() as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"error: invalid JSON in {plan_path}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        try:
            plan = MicroPlan.from_dict(payload)
        except Exception as exc:  # noqa: BLE001
            print(f"error: cannot parse plan from {plan_path}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        raw_plan_payload = payload if isinstance(payload, dict) else None

    if not plan.steps:
        print("error: plan has no steps", file=sys.stderr)
        return EXIT_ERROR

    # ── env harness (#336) ─────────────────────────────────────────────
    # When ``--env <name>`` is set, boot a simulated env, template
    # ``{{ENV_URL}}`` into the plan, and prepare grading at end of run.
    # Skipping when ``--env`` is absent keeps the existing path bit-
    # identical for every plan that targets a live URL.
    env_session = None
    env_task_id = ""
    if args.env:
        from .sim_envs.cli_integration import EnvSession, detect_default_runtime
        from .sim_envs.templating import substitute_env_url

        runtime = args.runtime or detect_default_runtime()
        try:
            env_session = EnvSession(
                args.env,
                runtime=runtime,
                seed=int(args.seed),
                now=args.now,
                keep=bool(args.keep),
            ).__enter__()
        except Exception as exc:  # noqa: BLE001 — surface boot errors
            print(f"error: env boot failed ({args.env}/{runtime}): {exc}", file=sys.stderr)
            return EXIT_ERROR

        env_url = env_session.url
        # Substitute on the plan dict (covers params.url and intent body),
        # then re-build the MicroPlan from the substituted payload.
        plan_dict = plan.to_dict()
        plan_dict = substitute_env_url(plan_dict, env_url)
        plan = MicroPlan.from_dict(plan_dict)

        # Pull task_id off the raw JSON payload (text plans land none).
        env_task_id = ""
        if raw_plan_payload is not None:
            env_task_id = str(raw_plan_payload.get("task_id", "")).strip()
        if not env_task_id:
            env_task_id = plan_path.stem  # fall back to filename
        print(
            f"  env:     {args.env}  (runtime={runtime}, url={env_url}, "
            f"task_id={env_task_id})"
        )

    # Output dir.
    output_dir = Path(args.output_dir or f"outputs/run-{int(_time.time())}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8",
    )
    print(f"  plan: {len(plan.steps)} steps → {output_dir / 'plan.json'}")

    # Endpoint + auth.
    endpoint = args.endpoint
    if not endpoint:
        print(
            f"error: --endpoint is required (platform={args.platform}). "
            "Pass the OpenAI-compatible v1 base URL, e.g. "
            "https://workspace--app-fn.modal.run/v1",
            file=sys.stderr,
        )
        return EXIT_ERROR

    api_key = (
        args.api_key
        or os.environ.get("MANTIS_API_KEY", "")
        or os.environ.get("HAI_API_KEY", "")
    )
    anthropic_key = (
        args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    )
    if not anthropic_key:
        print(
            "error: ClaudeGrounding + ClaudeExtractor need ANTHROPIC_API_KEY "
            "(pass --anthropic-api-key or export it)",
            file=sys.stderr,
        )
        return EXIT_ERROR

    try:
        extra_headers = _parse_extra_headers(args.header)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    # Build brain — Holo3Brain is the canonical OpenAI-compatible client.
    from .brain_holo3 import Holo3Brain

    brain = Holo3Brain(
        base_url=endpoint,
        model=args.model or _platform_default_model(args.platform),
        api_key=api_key,
        extra_headers=extra_headers or None,
    )

    # Build env — playwright is the lighter default; xdotool needs Xvfb.
    start_url = args.start_url or _resolve_first_navigate_url(plan)
    if not start_url:
        print(
            "error: no --start-url given and no navigate step in the plan "
            "carries an http(s):// URL — the browser has nowhere to start",
            file=sys.stderr,
        )
        return EXIT_ERROR

    # Build the optional proxy dict before constructing the env. Sites
    # behind anti-bot challenges (Cloudflare, DataDome, etc.) often
    # serve a 403 to a vanilla headless Chromium IP. The host
    # integration uses an Oxylabs residential proxy for the same
    # reason; the CLI mirrors that contract with --proxy-from-env.
    #
    # Oxylabs residential auth format: the username sent to the
    # proxy gateway must be ``customer-<USERNAME>`` (not the raw
    # USERNAME), with optional dash-delimited geo segments for
    # country / state / city pinning:
    #
    #   customer-{USERNAME}-cc-{COUNTRY}-st-{STATE}-city-{CITY}
    #   customer-{USERNAME}-cc-{COUNTRY}                          # country only
    #   customer-{USERNAME}                                        # rotating, no pin
    #
    # The base USERNAME-only form returns a sticky single IP that
    # often gets blocklisted; the geo-pinned form gives a fresh IP
    # per session token. ``--proxy-session`` (default ``mantis``)
    # is appended as a sessid suffix so consecutive runs can either
    # share an IP (same session) or rotate (different session).
    proxy_dict: dict | None = None
    if args.proxy_from_env:
        proxy_url = os.environ.get("OXYLABS_ENTRYPOINT", "").strip()
        raw_user = os.environ.get("OXYLABS_USERNAME", "").strip()
        proxy_pass = os.environ.get("OXYLABS_PASSWORD", "").strip()
        if not proxy_url or not raw_user or not proxy_pass:
            print(
                "error: --proxy-from-env set but OXYLABS_ENTRYPOINT / "
                "OXYLABS_USERNAME / OXYLABS_PASSWORD are not all populated. "
                "Export all three for the Oxylabs residential proxy.",
                file=sys.stderr,
            )
            return EXIT_ERROR

        # Playwright accepts a bare host:port; if the user supplied a
        # scheme-less host:port, prepend http:// so launchers downstream
        # don't reject the malformed URL.
        if "://" not in proxy_url:
            proxy_url = f"http://{proxy_url}"

        # Build the customer-prefixed username with geo + session
        # segments. Honour OXYLABS_COUNTRY / OXYLABS_STATE /
        # OXYLABS_CITY when any are present; skip silently if absent.
        username_parts = [f"customer-{raw_user}"]
        country = os.environ.get("OXYLABS_COUNTRY", "").strip()
        state = os.environ.get("OXYLABS_STATE", "").strip()
        city = os.environ.get("OXYLABS_CITY", "").strip()
        if country:
            username_parts += ["cc", country]
        if state:
            username_parts += ["st", state]
        if city:
            username_parts += ["city", city]
        if args.proxy_session:
            username_parts += ["sessid", args.proxy_session]
        proxy_user = "-".join(username_parts)

        proxy_dict = {
            "server": proxy_url,
            "username": proxy_user,
            "password": proxy_pass,
        }
        # Diagnostic — operators commonly need to see the username
        # actually sent (the customer- prefix and geo/sessid segments
        # are silently constructed) when debugging "why does this IP
        # keep getting Cloudflare-blocked?".
        print(f"  proxy:   {proxy_url}  (username={proxy_user})")

    env: Any
    if args.browser == "xdotool":
        try:
            from .gym.xdotool_env import XdotoolGymEnv
        except ImportError as exc:
            print(
                f"error: xdotool env requires the local-cua extras: {exc}. "
                "Reinstall with `pip install mantis-agent[local-cua]` or "
                "use --browser playwright.",
                file=sys.stderr,
            )
            return EXIT_ERROR
        env = XdotoolGymEnv()
    else:
        try:
            from .gym.playwright_env import PlaywrightGymEnv
        except ImportError as exc:
            print(
                f"error: playwright env not available: {exc}. "
                "Install with `pip install playwright && playwright install chromium`.",
                file=sys.stderr,
            )
            return EXIT_ERROR
        env = PlaywrightGymEnv(
            start_url=start_url,
            headless=bool(args.headless),
            proxy=proxy_dict,
        )

    # Build grounding + extractor — both Claude-backed, share the key.
    from .extraction import ClaudeExtractor, ExtractionSchema  # noqa: F401
    from .grounding import ClaudeGrounding

    grounding = ClaudeGrounding(api_key=anthropic_key)
    extractor = ClaudeExtractor(api_key=anthropic_key)

    # Site config — neutral by default. ``--detail-page-pattern`` is the
    # per-plan override hook; without it we rely on the path-extension
    # heuristic from #211. When ``--env <name>`` is set, we auto-load
    # ``SiteConfig.default_<env>()`` if one exists (#336 §8).
    from .site_config import SiteConfig

    site_config = SiteConfig(
        detail_page_pattern=args.detail_page_pattern or "",
    )
    if args.env:
        method_name = f"default_{args.env.replace('-', '_')}"
        factory = getattr(SiteConfig, method_name, None)
        if factory is not None:
            env_site_config = factory()
            # ``--detail-page-pattern`` still wins if the user passed one.
            if args.detail_page_pattern:
                env_site_config.detail_page_pattern = args.detail_page_pattern
            site_config = env_site_config
            print(f"  site:    SiteConfig.{method_name}() auto-loaded for --env {args.env}")

    # Wire the runner. ``session_name`` flows into checkpoint paths and
    # the dynamic verifier; derive a stable label from the plan filename
    # so resumed runs find their checkpoint.
    from .gym.micro_runner import MicroPlanRunner

    session_name = plan_path.stem
    checkpoint_path = str(output_dir / "checkpoint.json")
    runner = MicroPlanRunner(
        brain=brain,
        env=env,
        grounding=grounding,
        extractor=extractor,
        site_config=site_config,
        checkpoint_path=checkpoint_path,
        run_key=session_name,
        session_name=session_name,
        max_cost=float(args.max_cost),
        max_time_minutes=int(args.max_time_minutes),
        seed=int(args.seed),
    )

    print(
        f"  brain:   {endpoint}  (platform={args.platform}, "
        f"model={brain.model}, headers={'+'.join(extra_headers) or '—'})"
    )
    print(f"  browser: {args.browser} (start_url={start_url})")
    print(f"  output:  {output_dir}")
    print()

    # Pre-warm the env at ``start_url`` before handing it to the runner.
    # Plans that begin with a ``navigate`` step will reset again on their
    # first action — harmless. Plans that omit ``navigate`` (e.g. bench
    # variants that assume the runtime already opened the browser, like
    # ``boattrader_scrape_bench``) need this prewarm: without it, the
    # first handler that calls ``env.screenshot()`` hits a clear-but-
    # blocking RuntimeError because the Playwright page hasn't been
    # constructed yet. Equivalent to what the host-integration runtime
    # does before invoking ``MicroPlanRunner.run``.
    try:
        env.reset(task="cli_plan_run", start_url=start_url)
    except Exception as exc:  # noqa: BLE001
        print(f"error: env.reset failed before runner.run: {exc}", file=sys.stderr)
        return EXIT_ERROR

    t0 = _time.time()
    runner_exc: BaseException | None = None
    step_results: list[Any] = []
    try:
        try:
            step_results = runner.run(plan, resume=bool(args.resume))
        except Exception as exc:  # noqa: BLE001
            runner_exc = exc
            print(f"error: runner raised: {exc}", file=sys.stderr)
    finally:
        elapsed = _time.time() - t0

        # ── grading + env teardown (#336) ──────────────────────────────
        grading_dict: dict[str, Any] | None = None
        if env_session is not None:
            from .gym.grading import grade_run
            try:
                grading = grade_run(
                    env_session.url,
                    env_session.admin_token,
                    env_task_id,
                )
                grading_dict = grading.to_dict()
                (output_dir / "oracle.json").write_text(
                    json.dumps(grading_dict, indent=2) + "\n", encoding="utf-8",
                )
                print(
                    f"  oracle:  passed={grading.passed} score={grading.score} "
                    f"task_id={env_task_id}  → {output_dir / 'oracle.json'}"
                )
            except Exception as exc:  # noqa: BLE001 — never crash teardown
                print(f"  oracle:  call failed: {exc}", file=sys.stderr)

            # Merge env-side events into the agent trace (additive).
            try:
                from .sim_envs.trace_merge import merge_env_events

                merge_env_events(
                    output_dir, env_session.backend, env_session.handle,
                    since=t0,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  trace-merge: failed: {exc}", file=sys.stderr)

            # Tear down (unless --keep). EnvSession.__exit__ is idempotent.
            try:
                env_session.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

    if runner_exc is not None:
        return EXIT_ERROR

    # Result summary — write the structured payload then print a
    # human-readable rollup.
    successes = sum(1 for r in step_results if r.success)
    failures = len(step_results) - successes
    final_url = getattr(runner, "_last_known_url", "")
    from .gym.result_payload import pack_step
    time_meter = getattr(runner, "time_meter", None)
    wall_time_breakdown = (
        {k: round(v, 3) for k, v in time_meter.breakdown().items()}
        if time_meter is not None else {}
    )
    # Epic #377 Phase C: self-healing audit log.
    from .gym import healing_events as _healing
    healing_log = _healing.snapshot(runner)
    result_payload = {
        "plan_signature": runner.plan_signature,
        "session": session_name,
        "platform": args.platform,
        "endpoint": endpoint,
        "step_count": len(step_results),
        "successes": successes,
        "failures": failures,
        # Schema-aligned with build_micro_result (HTTP API path).
        # elapsed_seconds is the legacy float; total_time_s matches
        # the HTTP API shape so consumers can use one parser.
        "total_time_s": round(elapsed),
        "elapsed_seconds": round(elapsed, 2),
        "wall_time_breakdown": wall_time_breakdown,
        "healing_events": healing_log,
        "final_url": final_url,
        "costs": dict(getattr(runner, "costs", {})),
        "steps": [
            pack_step(
                r,
                time_breakdown=(
                    time_meter.step_breakdown(r.step_index)
                    if time_meter is not None else None
                ),
            )
            for r in step_results
        ],
    }
    if grading_dict is not None:
        # Additive — RunReport surface in result.json gains the oracle
        # verdict. Consumers that don't know about ``grading`` ignore it.
        result_payload["grading"] = grading_dict
    (output_dir / "result.json").write_text(
        json.dumps(result_payload, indent=2) + "\n", encoding="utf-8",
    )
    print(
        f"\n  result: {successes}/{len(step_results)} succeeded "
        f"({elapsed:.1f}s) — {output_dir / 'result.json'}"
    )
    if final_url:
        print(f"  final URL: {final_url}")
    return EXIT_OK if failures == 0 else EXIT_ERROR


def cmd_runs_stats(args: argparse.Namespace) -> int:
    """``mantis runs stats <workflow_id>`` — render p50/p95/p99 per
    wall-time bucket across the last N completed runs of a workflow.

    Reads the durable JSONL log written by the runtime on terminal
    status (epic #362 Phase C). Pure local query — no network, no
    server roundtrip. Renders a fixed-width table by default; pass
    ``--json`` to feed downstream tooling.
    """
    from .runs_log import stats

    since = _parse_since(args.since) if args.since else None
    if args.since and since is None:
        print(
            f"error: --since {args.since!r} must look like '7d' / '24h' / '30m'",
            file=sys.stderr,
        )
        return EXIT_ERROR

    result = stats(
        args.workflow_id,
        last_n=int(args.last),
        since=since,
        buckets=list(args.bucket) if args.bucket else None,
        status_filter=args.status,
    )

    if args.json:
        payload = {
            "workflow_id": result.workflow_id,
            "run_count": result.run_count,
            "percentiles": {
                k: {"p50": p50, "p95": p95, "p99": p99}
                for k, (p50, p95, p99) in result.percentiles.items()
            },
        }
        print(json.dumps(payload, indent=2))
        return EXIT_OK

    if result.run_count == 0:
        print(
            f"No runs matched workflow_id={args.workflow_id!r} "
            f"(status_filter={args.status!r}, last_n={args.last})."
        )
        return EXIT_OK

    # Render the fixed-width table: bucket | p50 | p95 | p99.
    header = (
        f"\nAcross last {result.run_count} {args.status or 'terminal'} "
        f"runs of {args.workflow_id}:\n"
    )
    print(header)
    # Order buckets by p50 descending so the worst offender lands first.
    rows = sorted(
        ((k, v) for k, v in result.percentiles.items() if k != "total_time_s"),
        key=lambda kv: kv[1][0],
        reverse=True,
    )
    total = result.percentiles.get("total_time_s")
    print(f"{'bucket':<16} {'p50':>10} {'p95':>10} {'p99':>10}")
    print("─" * 50)
    for k, (p50, p95, p99) in rows:
        print(f"{k:<16} {p50:>9.1f}s {p95:>9.1f}s {p99:>9.1f}s")
    print("─" * 50)
    if total is not None:
        p50, p95, p99 = total
        print(f"{'total_time_s':<16} {p50:>9.1f}s {p95:>9.1f}s {p99:>9.1f}s")
    return EXIT_OK


def _parse_since(value: str):
    """Parse a duration like '7d' / '24h' / '30m' into a ``timedelta``.

    Returns ``None`` for unparseable inputs so the caller can surface a
    friendly error message.
    """
    from datetime import timedelta

    value = (value or "").strip().lower()
    if not value or len(value) < 2:
        return None
    suffix = value[-1]
    try:
        n = int(value[:-1])
    except ValueError:
        return None
    if n < 0:
        return None
    if suffix == "d":
        return timedelta(days=n)
    if suffix == "h":
        return timedelta(hours=n)
    if suffix == "m":
        return timedelta(minutes=n)
    return None


def cmd_plan_run_modal(args: argparse.Namespace) -> int:
    """``mantis plan run-modal <plan>`` — execute a plan inside Modal.

    Thin remote driver: looks up the deployed ``run_plan`` function on
    ``--app-name`` (default ``mantis-plan-runner``, see
    ``deploy/modal/modal_plan_runner.py``), submits the plan + brain
    config, and writes the returned result-payload to
    ``<output-dir>/result.json``. The browser, decomposer (if needed),
    grounding, and extractor all run on Modal — so Cloudflare-protected
    sites that block local headless Chromium pass through normally
    because the Modal image runs full Chromium under Xvfb.

    The remote function reads ``ANTHROPIC_API_KEY`` and the optional
    Oxylabs credentials from the ``mantis-plan-runner-secrets`` Modal
    Secret — pass-through args here are limited to the brain endpoint,
    plan source, and per-run knobs (start URL, detail-page pattern,
    cost / time caps, proxy toggle).
    """
    import time as _time

    plan_path = Path(args.path)
    if not plan_path.exists():
        print(f"error: plan file not found: {args.path}", file=sys.stderr)
        return EXIT_ERROR

    plan_text: str | None = None
    plan_json: dict | None = None
    if _looks_like_text_plan(plan_path):
        plan_text = plan_path.read_text(encoding="utf-8")
        if not plan_text.strip():
            print(f"error: plan file is empty: {plan_path}", file=sys.stderr)
            return EXIT_ERROR
    else:
        try:
            with plan_path.open() as fh:
                plan_json = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"error: invalid JSON in {plan_path}: {exc}", file=sys.stderr)
            return EXIT_ERROR

    endpoint = args.endpoint
    if not endpoint:
        print(
            "error: --endpoint is required (the brain URL Modal will hit)",
            file=sys.stderr,
        )
        return EXIT_ERROR

    try:
        extra_headers = _parse_extra_headers(args.header)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        import modal  # type: ignore[import-not-found]
    except ImportError:
        print(
            "error: modal SDK not installed. `pip install modal` or "
            "use the [modal] extras to enable run-modal.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    try:
        run_plan_fn = modal.Function.from_name(args.app_name, "run_plan")
    except Exception as exc:  # noqa: BLE001 — modal raises various types
        print(
            f"error: cannot resolve Modal function "
            f"{args.app_name}/run_plan: {exc}. Deploy first via "
            "`uv run modal deploy deploy/modal/modal_plan_runner.py`.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    # Output dir + start_url derivation. We need a MicroPlan object to
    # find the first navigate URL when --start-url is omitted, but only
    # for json plans — text plans get decomposed inside Modal so we
    # require an explicit --start-url in that case.
    start_url = args.start_url or ""
    if not start_url and plan_json is not None:
        from .plan_decomposer import MicroPlan
        try:
            mp = MicroPlan.from_dict(plan_json)
        except Exception as exc:  # noqa: BLE001
            print(f"error: cannot parse plan from {plan_path}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        start_url = _resolve_first_navigate_url(mp)
    if not start_url:
        print(
            "error: --start-url is required for text plans (decomposer "
            "runs in Modal so we can't introspect plan steps locally)",
            file=sys.stderr,
        )
        return EXIT_ERROR

    output_dir = Path(args.output_dir or f"outputs/run-modal-{int(_time.time())}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if plan_json is not None:
        (output_dir / "plan.json").write_text(
            json.dumps(plan_json, indent=2) + "\n", encoding="utf-8",
        )
    else:
        (output_dir / "plan.txt").write_text(plan_text or "", encoding="utf-8")

    session_name = plan_path.stem

    print(f"  app:     {args.app_name} (modal)")
    print(f"  brain:   {endpoint}")
    print(f"  plan:    {plan_path.name} → {output_dir}")
    print(f"  start:   {start_url}")
    print()

    t0 = _time.time()
    try:
        result = run_plan_fn.remote(
            plan_text=plan_text,
            plan_json=plan_json,
            brain_endpoint=endpoint,
            brain_extra_headers=extra_headers or None,
            brain_model=args.model or _platform_default_model("modal"),
            start_url=start_url,
            detail_page_pattern=args.detail_page_pattern or "",
            max_cost_usd=float(args.max_cost),
            max_time_minutes=int(args.max_time_minutes),
            use_proxy=bool(args.use_proxy),
            proxy_session=args.proxy_session,
            session_name=session_name,
            seed=int(args.seed),
        )
    except Exception as exc:  # noqa: BLE001 — modal invocation errors
        print(f"error: Modal run_plan raised: {exc}", file=sys.stderr)
        return EXIT_ERROR
    elapsed = _time.time() - t0

    if not isinstance(result, dict):
        print(
            f"error: unexpected result shape from Modal: {type(result).__name__}",
            file=sys.stderr,
        )
        return EXIT_ERROR

    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )

    successes = int(result.get("successes", 0))
    step_count = int(result.get("step_count", 0))
    failures = int(result.get("failures", step_count - successes))
    final_url = result.get("final_url", "")
    print(
        f"  result: {successes}/{step_count} succeeded "
        f"(remote {result.get('elapsed_seconds', 0)}s, wall {elapsed:.1f}s) — "
        f"{output_dir / 'result.json'}"
    )
    if final_url:
        print(f"  final URL: {final_url}")
    return EXIT_OK if failures == 0 else EXIT_ERROR


def cmd_cua_run(args: argparse.Namespace) -> int:
    """``mantis cua run`` — pure CUA pass-through to ``POST /v1/cua``.

    No decomposition, no Claude grounding, no Claude extraction. The
    instruction is handed verbatim to the brain (Holo3) running on the
    Mantis deployment, which drives ``XdotoolGymEnv`` directly via the
    Holo3 action surface (click / type_text / scroll / drag / key_press /
    wait / done).

    Unlike ``mantis plan run``, this subcommand does NOT pull in
    Playwright, Anthropic, or any local brain — it's a thin HTTP shim.
    The server-side ``/v1/cua`` route owns the env + brain + runner.

    Exits 0 on ``success=true`` in the response, 1 otherwise.
    """
    import os

    import requests

    endpoint = (args.endpoint or os.environ.get("MANTIS_ENDPOINT", "")).rstrip("/")
    if not endpoint:
        print(
            "error: --endpoint is required (or set MANTIS_ENDPOINT). "
            "Pass the deployment base URL, e.g. "
            "https://workspace--app-fn.modal.run",
            file=sys.stderr,
        )
        return EXIT_ERROR
    # Strip a trailing ``/v1`` if a user pasted the OpenAI-style base.
    if endpoint.endswith("/v1"):
        endpoint = endpoint[: -len("/v1")]
    url = f"{endpoint}/v1/cua"

    token = args.token or os.environ.get("MANTIS_TOKEN", "")
    if not token:
        print(
            "error: --token is required (or set MANTIS_TOKEN). "
            "This is the per-tenant Mantis token (X-Mantis-Token).",
            file=sys.stderr,
        )
        return EXIT_ERROR

    try:
        extra_headers = _parse_extra_headers(args.header)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    body: dict[str, Any] = {
        "instruction": args.instruction,
        "start_url": args.start_url or "",
        "max_steps": args.max_steps,
        "frames_per_inference": args.frames_per_inference,
        "settle_time": args.settle_time,
        "detached": args.detached,
        "proxy_disabled": args.proxy_disabled,
        "record_video": args.record_video,
    }
    if args.state_key:
        body["state_key"] = args.state_key
    if args.profile_id:
        body["profile_id"] = args.profile_id
    if args.workflow_id:
        body["workflow_id"] = args.workflow_id
    if args.proxy_city:
        body["proxy_city"] = args.proxy_city
    if args.proxy_state:
        body["proxy_state"] = args.proxy_state

    headers = {
        "Content-Type": "application/json",
        "X-Mantis-Token": token,
        **extra_headers,
    }

    print(f"POST {url}")
    print(f"  instruction: {args.instruction[:80]}{'…' if len(args.instruction) > 80 else ''}")
    if args.start_url:
        print(f"  start_url:   {args.start_url}")
    print(f"  max_steps:   {args.max_steps}  detached: {args.detached}")

    try:
        resp = requests.post(url, json=body, headers=headers, timeout=args.timeout)
    except requests.RequestException as exc:
        print(f"error: request failed: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if resp.status_code >= 400:
        print(f"error: HTTP {resp.status_code}: {resp.text[:500]}", file=sys.stderr)
        return EXIT_ERROR

    try:
        result = resp.json()
    except ValueError:
        print(f"error: non-JSON response: {resp.text[:500]}", file=sys.stderr)
        return EXIT_ERROR

    if args.json:
        print(json.dumps(result, indent=2))
        return EXIT_OK if result.get("success") else EXIT_ERROR

    # Detached: surface the run handle the same way /v1/predict does.
    if result.get("mode") == "detached" or result.get("status") == "queued":
        print(f"  run_id:      {result.get('run_id', '?')}")
        print(f"  status:      {result.get('status', 'queued')}")
        print(f"  status_path: {result.get('status_path', '')}")
        return EXIT_OK

    success = bool(result.get("success"))
    print()
    print("═" * 60)
    print(f"  {'SUCCESS' if success else 'FAILED'}  steps={result.get('steps', '?')}  "
          f"duration={result.get('duration_s', '?')}s  "
          f"termination={result.get('termination_reason', '?')}")
    print(f"  run_id: {result.get('run_id', '?')}")
    print("═" * 60)
    return EXIT_OK if success else EXIT_ERROR


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

    init = plan_sub.add_parser(
        "init",
        help="Scaffold a starter plan via PlanDecomposer (Claude API call)",
    )
    init.add_argument(
        "url",
        help="Target URL for the new plan (used as the navigate destination)",
    )
    init.add_argument(
        "--task",
        required=True,
        help="One-sentence description of what the plan should accomplish",
    )
    init.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON path (default: derived from URL hostname)",
    )
    init.add_argument(
        "--model",
        default="claude-opus-4-7",
        help="Claude model to use for decomposition",
    )
    init.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the post-decompose PlanValidator run",
    )
    init.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Skip the post-decompose dry-run preview",
    )
    init.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    init.set_defaults(func=cmd_plan_init)

    run = plan_sub.add_parser(
        "run",
        help="Execute a plan against a remote Mantis brain (Modal/Baseten/custom)",
    )
    run.add_argument(
        "path",
        help="Path to plan file. .json → pre-decomposed micro-plan; "
             "any other extension → natural-language plan, decomposed "
             "via Claude before running.",
    )
    run.add_argument(
        "--platform",
        choices=_PLATFORMS,
        default="modal",
        help="Inference platform tag (informational; auth header style). "
             "Pass --endpoint for the actual base URL.",
    )
    run.add_argument(
        "--endpoint",
        default=None,
        help="OpenAI-compatible v1 base URL of the brain "
             "(e.g. https://workspace--app-fn.modal.run/v1 or "
             "https://model-x.api.baseten.co/production/sync/v1)",
    )
    run.add_argument(
        "--api-key",
        default=None,
        help="Brain endpoint key (env: MANTIS_API_KEY or HAI_API_KEY)",
    )
    run.add_argument(
        "--anthropic-api-key",
        default=None,
        help="Anthropic key for ClaudeGrounding + ClaudeExtractor + "
             "decompose pass on text plans (env: ANTHROPIC_API_KEY)",
    )
    run.add_argument(
        "--model",
        default=None,
        help="Brain model name; defaults derived from --platform "
             "(modal: Hcompany/Holo3-35B-A3B, baseten: holo3-35b-a3b)",
    )
    run.add_argument(
        "--header",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Additional HTTP header sent with every brain request. "
             "Repeat for multiple. Useful for tenant tokens or gateway "
             "auth schemes that override Bearer.",
    )
    run.add_argument(
        "--browser",
        choices=("playwright", "xdotool"),
        default="playwright",
        help="Browser env. playwright (default) is light + headless-friendly; "
             "xdotool needs Xvfb + Chromium running on the host.",
    )
    run.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Headless browser (default). --no-headless opens a visible window "
             "(playwright only).",
    )
    run.add_argument(
        "--start-url",
        default=None,
        help="Initial URL for the browser. Defaults to the first navigate "
             "step's URL in the plan.",
    )
    run.add_argument(
        "--proxy-from-env",
        action="store_true",
        default=False,
        help="Route the browser through OXYLABS_ENTRYPOINT / OXYLABS_USERNAME "
             "/ OXYLABS_PASSWORD (the host integration's residential proxy). "
             "The username sent to Oxylabs is built as "
             "``customer-{USERNAME}-cc-{COUNTRY}-st-{STATE}-city-{CITY}-sessid-{SESSION}``"
             " — geo segments come from OXYLABS_COUNTRY / OXYLABS_STATE / "
             "OXYLABS_CITY when present; SESSION from --proxy-session. "
             "Required for sites behind Cloudflare / DataDome that block "
             "headless Chromium IPs.",
    )
    run.add_argument(
        "--proxy-session",
        default="mantis",
        help="Sessid suffix appended to the Oxylabs proxy username. Same "
             "session reuses the same sticky IP; a different session forces "
             "a fresh IP. Useful when a specific IP got blocklisted and you "
             "want to rotate without restarting the proxy. Default: 'mantis'.",
    )
    run.add_argument(
        "--detail-page-pattern",
        default=None,
        help="Optional regex injected into SiteConfig.detail_page_pattern. "
             "When unset the framework falls back to the generic "
             "path-extension heuristic (#211).",
    )
    run.add_argument(
        "--max-cost",
        type=float,
        default=10.0,
        help="Hard cap on USD spend (Anthropic + brain). Runner halts when "
             "exceeded (default: 10.0).",
    )
    run.add_argument(
        "--max-time-minutes",
        type=int,
        default=30,
        help="Hard wall-clock cap (default: 30 min).",
    )
    run.add_argument(
        "--output-dir",
        default=None,
        help="Where to write plan.json + result.json + checkpoint.json. "
             "Defaults to outputs/run-<unix-timestamp>.",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint at --output-dir/checkpoint.json if present.",
    )
    # ── env harness (#336) ────────────────────────────────────────────
    run.add_argument(
        "--env",
        default=None,
        help="Simulated env to boot for this run (e.g. 'stub', 'mantis-crm'). "
             "When set, the harness boots the env via --runtime, templates "
             "{{ENV_URL}} into the plan, calls grade_run after, and writes "
             "oracle.json + env_events.jsonl into --output-dir.",
    )
    run.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for the runner (default: 42). Seeds "
             "Python's random module so human_speed action delays are "
             "reproducible across reruns of the same plan. Also passed "
             "via SEED= to the sim-env when --env is set.",
    )
    run.add_argument(
        "--now",
        default="2026-01-15T09:00:00Z",
        help="Frozen wall-clock for the env via FAKE_NOW= (default: "
             "2026-01-15T09:00:00Z). Only consulted when --env is set.",
    )
    run.add_argument(
        "--runtime",
        choices=("local", "modal", "e2b"),
        default=None,
        help="Where to boot the env: local (Docker / subprocess stub), modal "
             "(one app per env), or e2b (reserved, not implemented in v1). "
             "Defaults to modal when running on Modal, local otherwise.",
    )
    run.add_argument(
        "--keep",
        action="store_true",
        help="Leave the env running after the plan finishes (for debugging). "
             "Default: env is torn down on exit.",
    )
    run.set_defaults(func=cmd_plan_run)

    run_modal = plan_sub.add_parser(
        "run-modal",
        help="Execute a plan inside Modal (browser + extractor + grounding "
             "all run remotely under Xvfb — bypasses local Cloudflare blocks)",
    )
    run_modal.add_argument(
        "path",
        help="Path to plan file. .json → pre-decomposed micro-plan; "
             "any other extension → natural-language plan, decomposed "
             "by Claude inside Modal.",
    )
    run_modal.add_argument(
        "--app-name",
        default="mantis-plan-runner",
        help="Modal app name. Must match the deployed "
             "deploy/modal/modal_plan_runner.py app (default: "
             "mantis-plan-runner).",
    )
    run_modal.add_argument(
        "--endpoint",
        default=None,
        required=True,
        help="OpenAI-compatible v1 base URL of the brain. Modal calls "
             "this endpoint over HTTP, so it must be reachable from "
             "Modal's egress.",
    )
    run_modal.add_argument(
        "--model",
        default=None,
        help="Brain model name (default: Hcompany/Holo3-35B-A3B for modal).",
    )
    run_modal.add_argument(
        "--header",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Additional HTTP header sent with every brain request "
             "(e.g. X-Mantis-Token=…). Repeat for multiple.",
    )
    run_modal.add_argument(
        "--start-url",
        default=None,
        help="Initial URL for the browser. Required for text plans "
             "(decomposer runs in Modal, so the CLI can't introspect "
             "the navigate step locally). Optional for JSON plans.",
    )
    run_modal.add_argument(
        "--detail-page-pattern",
        default=None,
        help="Optional regex injected into SiteConfig.detail_page_pattern. "
             "When unset the framework falls back to the generic "
             "path-extension heuristic (#211).",
    )
    run_modal.add_argument(
        "--max-cost",
        type=float,
        default=10.0,
        help="Hard cap on USD spend (Anthropic + brain) inside Modal "
             "(default: 10.0).",
    )
    run_modal.add_argument(
        "--max-time-minutes",
        type=int,
        default=30,
        help="Hard wall-clock cap inside Modal (default: 30 min).",
    )
    run_modal.add_argument(
        "--use-proxy",
        action="store_true",
        help="Route the Modal-side browser through Oxylabs (creds from "
             "the mantis-plan-runner-secrets Modal Secret).",
    )
    run_modal.add_argument(
        "--proxy-session",
        default="mantis",
        help="Oxylabs session ID for sticky-IP behavior (default: mantis).",
    )
    run_modal.add_argument(
        "--output-dir",
        default=None,
        help="Where to write plan.json / plan.txt + result.json. "
             "Defaults to outputs/run-modal-<unix-timestamp>.",
    )
    run_modal.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for the runner (default: 42). Forwarded "
             "to MicroPlanRunner inside Modal so human_speed delays are "
             "reproducible across reruns of the same plan.",
    )
    run_modal.set_defaults(func=cmd_plan_run_modal)

    # ── cua: pure Holo3 pass-through (no decomposition, no Claude) ──
    cua = sub.add_parser(
        "cua",
        help="Pure CUA pass-through: forward an instruction to /v1/cua "
             "(brain ↔ XdotoolGymEnv, no Claude).",
    )
    cua_sub = cua.add_subparsers(dest="cua_command", required=True)

    cua_run = cua_sub.add_parser(
        "run",
        help="POST /v1/cua with a single instruction — Mantis becomes "
             "a thin pass-through for Holo3.",
    )
    cua_run.add_argument(
        "instruction",
        help="Natural-language instruction handed verbatim to the brain.",
    )
    cua_run.add_argument(
        "--endpoint",
        default=None,
        help="Mantis deployment base URL (e.g. "
             "https://workspace--app-fn.modal.run). Trailing /v1 is stripped. "
             "Env: MANTIS_ENDPOINT.",
    )
    cua_run.add_argument(
        "--token",
        default=None,
        help="Per-tenant Mantis token (sent as X-Mantis-Token). "
             "Env: MANTIS_TOKEN.",
    )
    cua_run.add_argument(
        "--header",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Additional HTTP header. Repeat for multiple.",
    )
    cua_run.add_argument(
        "--start-url",
        default="",
        help="Initial URL for the browser inside the deployment.",
    )
    cua_run.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum CUA loop steps before forced termination "
             "(default: 30, server cap: MANTIS_MAX_STEPS_PER_PLAN).",
    )
    cua_run.add_argument(
        "--frames-per-inference",
        type=int,
        default=1,
        help="Frames the brain sees per inference (default: 1 — Holo3 "
             "uses one screenshot per step).",
    )
    cua_run.add_argument(
        "--settle-time",
        type=float,
        default=2.0,
        help="Seconds to wait after each action before re-screenshotting "
             "(default: 2.0; bump for slow sites).",
    )
    cua_run.add_argument(
        "--state-key",
        default=None,
        help="Legacy single-field identity (#341). When set alone, the "
             "server routes it to both --profile-id and --workflow-id. "
             "Prefer the split flags below for new code.",
    )
    cua_run.add_argument(
        "--profile-id",
        default=None,
        help="Chrome user-data-dir identity (#341). Sticky across plan "
             "revisions — same id => same cookies / logged-in sessions.",
    )
    cua_run.add_argument(
        "--workflow-id",
        default=None,
        help="Checkpoint identity (#341). Rotate when the plan definition "
             "changes; pair with --resume-state to pick up where the last "
             "run with this id left off.",
    )
    cua_run.add_argument(
        "--detached",
        action="store_true",
        help="Queue as a background run and return a run_id immediately. "
             "Poll via /v1/predict with action=status.",
    )
    cua_run.add_argument(
        "--proxy-city",
        default=None,
        help="Override the deployment's default proxy city (allowlist-gated).",
    )
    cua_run.add_argument(
        "--proxy-state",
        default=None,
        help="Override the deployment's default proxy state.",
    )
    cua_run.add_argument(
        "--proxy-disabled",
        action="store_true",
        help="Skip the upstream proxy entirely (use for test sites that "
             "don't need bot-detection bypass).",
    )
    cua_run.add_argument(
        "--record-video",
        action="store_true",
        help="Record a screencast of the run; fetch via "
             "GET /v1/runs/{run_id}/video.",
    )
    cua_run.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="HTTP timeout for the sync call in seconds (default: 900). "
             "Ignored when --detached is set.",
    )
    cua_run.add_argument(
        "--json",
        action="store_true",
        help="Emit the raw response JSON instead of a human summary.",
    )
    cua_run.set_defaults(func=cmd_cua_run)

    trace = sub.add_parser("trace", help="Trace tooling subcommands (#155)")
    trace_sub = trace.add_subparsers(dest="trace_command", required=True)

    label = trace_sub.add_parser(
        "label",
        help="Batch-label exported run traces using automatic heuristics",
    )
    label.add_argument(
        "input",
        help="Trace file or directory (recurses for *.json under directories)",
    )
    label.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory (subtree mirrors the input)",
    )
    label.add_argument(
        "--json",
        action="store_true",
        help="Emit summary as machine-readable JSON",
    )
    label.set_defaults(func=cmd_trace_label)

    review = trace_sub.add_parser(
        "review",
        help="Print a labelled summary of one trace JSON for human inspection",
    )
    review.add_argument("path", help="Path to a single trace JSON file")
    review.add_argument(
        "--json",
        action="store_true",
        help="Emit the labelled trace as JSON",
    )
    review.set_defaults(func=cmd_trace_review)

    # ── runs: cross-run aggregation over the JSONL log (epic #362 C) ──
    runs = sub.add_parser(
        "runs",
        help="Cross-run aggregation over the durable JSONL log written "
             "on terminal status by the mantis-server runtime.",
    )
    runs_sub = runs.add_subparsers(dest="runs_command", required=True)

    runs_stats = runs_sub.add_parser(
        "stats",
        help="Aggregate p50/p95/p99 per wall-time bucket across the "
             "last N runs of a workflow.",
    )
    runs_stats.add_argument(
        "workflow_id",
        help="Workflow ID to aggregate over (matches summary.workflow_id "
             "and the JSONL row's workflow_id field).",
    )
    runs_stats.add_argument(
        "--last", type=int, default=100,
        help="Newest-first cap on rows considered (default: 100).",
    )
    runs_stats.add_argument(
        "--bucket", action="append", default=None,
        metavar="NAME",
        help="Restrict to one or more wall-time buckets (e.g. think, "
             "claude_extract). Repeat to select multiple. Default: all.",
    )
    runs_stats.add_argument(
        "--since",
        default=None,
        help="Restrict to rows finished within the last DURATION "
             "(formats: '7d', '24h', '30m'). Default: no time bound.",
    )
    runs_stats.add_argument(
        "--status", default="succeeded",
        help="Filter rows by status (default: succeeded). Pass '' to "
             "include every terminal status.",
    )
    runs_stats.add_argument(
        "--json", action="store_true",
        help="Emit results as JSON instead of a table.",
    )
    runs_stats.set_defaults(func=cmd_runs_stats)

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
