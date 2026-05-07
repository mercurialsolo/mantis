#!/usr/bin/env python3
"""Analyze a Mantis Baseten run against a text plan.

Pulls events.log + status from the deployed runtime, then produces a Markdown
report covering: which plan step the model reached, what action types it used,
diversity / repetition stats, and the termination cause. Designed for the
"long English plan" benchmark — supports both `task_suite` and `plan_text`
runs.

Usage:
    python scripts/analyze_text_plan_run.py \\
        --run-id 20260506_230028_f27c7e46 \\
        --plan-file plans/fb_plan_v5_bench
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from baseten_workload import (  # noqa: E402
    DEFAULT_ENVIRONMENT,
    DEFAULT_MODEL_ID,
    load_dotenv,
    post_json,
)


# ── Plan parsing ──────────────────────────────────────────────────────────────

STEP_HEADER = re.compile(r"^##\s+Step\s+(\d+(?:\.\d+)?)\s*[-—–:]\s*(.+?)\s*$", re.M)
# Plain-prose step headers: "Step N — Title" with no markdown prefix
STEP_PLAIN = re.compile(r"^Step\s+(\d+(?:\.\d+)?)\s*[-—–:]\s*(.+?)\s*$", re.M)
NUMBERED_LIST = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$", re.M)
URL_RE = re.compile(r"https?://[\w.\-/?=&%#]+")


@dataclass
class PlanStep:
    number: str
    title: str
    body: str

    @property
    def urls(self) -> list[str]:
        return URL_RE.findall(self.body)

    @property
    def keywords(self) -> set[str]:
        # Cheap signal: distinctive 5+ char words in title + first paragraph
        head = self.title + " " + self.body.split("\n\n", 1)[0]
        return {w.lower() for w in re.findall(r"\b[a-zA-Z]{5,}\b", head)}


def parse_plan(text: str) -> list[PlanStep]:
    # 1. Markdown-style: "## Step N — Title"
    matches = list(STEP_HEADER.finditer(text))
    if matches:
        return _matches_to_steps(text, matches)
    # 2. Plain-prose: "Step N — Title" with no markdown prefix
    matches = list(STEP_PLAIN.finditer(text))
    if matches:
        return _matches_to_steps(text, matches)
    # 3. Fallback: numbered list (e.g. "## Steps\n1. Go to ...\n2. Log in ...")
    list_matches = list(NUMBERED_LIST.finditer(text))
    return _matches_to_steps(text, list_matches)


def _matches_to_steps(text: str, matches: list[re.Match[str]]) -> list[PlanStep]:
    steps: list[PlanStep] = []
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = m.group(2).strip()
        steps.append(PlanStep(number=m.group(1), title=title, body=text[body_start:body_end].strip() or title))
    return steps


# ── Event parsing ─────────────────────────────────────────────────────────────

ACTION_RE = re.compile(r"Action:\s+(\w+)\((.*?)\)\s*\(([\d.]+)s\)")
RUNNER_STEP_RE = re.compile(r"---\s*Step\s+(\d+)/(\d+)\s*---")
TASK_FINISHED_RE = re.compile(r"Task finished:\s+(\w+),\s+(\d+)\s+steps,\s+([\d.]+)s")
NAV_URL_RE = re.compile(r"Browser started:.*?→\s*(\S+)|navigated?\s+to\s+(\S+)|navigate\s*\(.*?url['\"]?\s*[:=]\s*['\"]?(https?://\S+)")
# Decompose-path (MicroPlanRunner) markers
MICRO_STEP_RE = re.compile(r"micro_runner:\s+\[\s*(\d+)\]\s+(\w+)\s+(.+)$")
MICRO_DIFF_RE = re.compile(r"micro_runner:\s+\[diff\]\s+(\w+)/(ok|fail|err)\s+step=(\d+)")
MICRO_DONE_RE = re.compile(r"MicroPlan complete:\s+(\d+)\s+steps executed")
DECOMPOSE_COUNT_RE = re.compile(r"Decomposed into\s+(\d+)\s+micro-intents")


@dataclass
class RunnerStep:
    index: int
    action_type: str | None = None
    action_args: str | None = None
    feedback: str | None = None
    duration_s: float | None = None


@dataclass
class ParsedRun:
    runner_steps: list[RunnerStep] = field(default_factory=list)
    nav_urls: list[str] = field(default_factory=list)
    task_finished_reason: str | None = None
    task_finished_steps: int | None = None
    task_finished_seconds: float | None = None
    starting_task: str | None = None
    raw_events: list[str] = field(default_factory=list)
    # Decompose-path fields. Populated only on `--decompose` runs where
    # MicroPlanRunner walks a structured plan instead of GymRunner.
    decompose_intents_total: int = 0
    decompose_steps_executed: int = 0
    decompose_step_log: list[str] = field(default_factory=list)
    is_decompose_run: bool = False


def parse_events(events: list[str]) -> ParsedRun:
    out = ParsedRun()
    current: RunnerStep | None = None
    for raw in events:
        try:
            ev = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            continue
        msg = ev.get("message", "") if isinstance(ev, dict) else str(ev)
        out.raw_events.append(msg)

        if "Starting task:" in msg and out.starting_task is None:
            out.starting_task = msg.split("Starting task:", 1)[1].strip()[:200]

        # Only count URLs the model itself navigates to. The runtime's initial
        # `Browser started: → start_url` line is environment setup, not a
        # plan step the model executed.
        if "Browser started" not in msg:
            m = NAV_URL_RE.search(msg)
            if m:
                url = next((g for g in m.groups() if g), None)
                if url:
                    out.nav_urls.append(url)

        m = RUNNER_STEP_RE.search(msg)
        if m:
            if current is not None:
                out.runner_steps.append(current)
            current = RunnerStep(index=int(m.group(1)))
            continue

        if current is not None:
            m = ACTION_RE.search(msg)
            if m:
                current.action_type = m.group(1)
                current.action_args = m.group(2)
                current.duration_s = float(m.group(3))
            elif "Feedback:" in msg:
                current.feedback = msg.split("Feedback:", 1)[1].strip()[:200]

        m = TASK_FINISHED_RE.search(msg)
        if m:
            out.task_finished_reason = m.group(1)
            out.task_finished_steps = int(m.group(2))
            out.task_finished_seconds = float(m.group(3))

        # Decompose-path parsing — runs through MicroPlanRunner, which
        # emits a different log shape than GymRunner. Coexists with the
        # GymRunner parsing above (a run uses one path or the other).
        m = DECOMPOSE_COUNT_RE.search(msg)
        if m:
            out.decompose_intents_total = int(m.group(1))
            out.is_decompose_run = True
        m = MICRO_STEP_RE.search(msg)
        if m:
            idx, kind, intent = m.group(1), m.group(2), m.group(3).strip()
            out.decompose_step_log.append(f"[{idx}] {kind}: {intent[:80]}")
            out.is_decompose_run = True
        m = MICRO_DONE_RE.search(msg)
        if m:
            out.decompose_steps_executed = int(m.group(1))
            out.task_finished_reason = "micro_plan_complete"
            out.task_finished_steps = out.decompose_steps_executed

    if current is not None:
        out.runner_steps.append(current)
    return out


# ── Plan-step inference ───────────────────────────────────────────────────────


def infer_reached_steps(plan: list[PlanStep], parsed: ParsedRun) -> dict[str, str]:
    """Return {step_number: evidence} for plan steps the run probably reached."""
    reached: dict[str, str] = {}
    if not plan:
        return reached

    # Always count Step 1 as "reached" if any runner step ran (the model saw the plan)
    if parsed.runner_steps:
        reached[plan[0].number] = "model began executing"

    # Match navigation URLs to plan-step URL mentions
    for url in parsed.nav_urls:
        for step in plan:
            for plan_url in step.urls:
                if _urls_align(url, plan_url):
                    reached.setdefault(step.number, f"navigated to {url}")

    # Heuristic: action-type → keyword presence in plan step
    action_keywords = {
        "type": {"type", "typing", "input", "field", "credentials", "password"},
        "scroll": {"scroll", "below"},
        "navigate": {"navigate", "url", "address"},
    }
    for rs in parsed.runner_steps:
        if not rs.action_type:
            continue
        toks = action_keywords.get(rs.action_type, set())
        if not toks:
            continue
        for step in plan:
            if step.keywords & toks and step.number not in reached:
                reached[step.number] = f"action={rs.action_type} matches keywords"

    return reached


def _urls_align(a: str, b: str) -> bool:
    def host(u: str) -> str:
        m = re.search(r"https?://([^/]+)", u)
        return (m.group(1) if m else "").lower()
    ha, hb = host(a), host(b)
    if not ha or not hb:
        return False
    return ha == hb or ha.endswith("." + hb) or hb.endswith("." + ha)


# ── Failure-mode classification ──────────────────────────────────────────────


def classify_failure(parsed: ParsedRun, status: dict[str, Any]) -> tuple[str, str]:
    final_status = status.get("status", "unknown")
    reason = parsed.task_finished_reason or "no_terminator"

    if reason == "loop":
        # Diagnose the loop signature
        action_counts = Counter(rs.action_type for rs in parsed.runner_steps if rs.action_type)
        if len(action_counts) == 1:
            return ("loop_single_action", f"model fired {next(iter(action_counts))!r} {sum(action_counts.values())}x — no recovery on no-visible-change")
        coord_counts = Counter(rs.action_args for rs in parsed.runner_steps if rs.action_args)
        most_common = coord_counts.most_common(1)
        if most_common and most_common[0][1] >= 3:
            return ("loop_repeated_coords", f"same args {most_common[0][1]}x: {most_common[0][0][:80]}")
        return ("loop", "loop detector tripped, no single dominant action")

    if reason == "done":
        # Look in the last action for success boolean
        last = next((rs for rs in reversed(parsed.runner_steps) if rs.action_type == "done"), None)
        if last and last.action_args:
            success = "True" if "'success': True" in last.action_args or '"success": true' in last.action_args else "False"
            return (f"model_done_success={success}", last.action_args[:200])
        return ("model_done", "model emitted done with unknown success")

    if final_status == "failed":
        return ("server_error", status.get("error", "(no error string)"))

    if reason == "max_steps":
        return ("max_steps", "ran out of step budget without emitting done")

    return (reason, f"final status={final_status}")


# ── Report ────────────────────────────────────────────────────────────────────


def render_report(
    *,
    run_id: str,
    plan_path: Path,
    plan: list[PlanStep],
    parsed: ParsedRun,
    status: dict[str, Any],
) -> str:
    reached = infer_reached_steps(plan, parsed)
    failure_kind, failure_detail = classify_failure(parsed, status)

    action_counter = Counter(rs.action_type for rs in parsed.runner_steps if rs.action_type)
    args_counter = Counter(rs.action_args for rs in parsed.runner_steps if rs.action_args)
    no_change_count = sum(1 for rs in parsed.runner_steps if rs.feedback and "no visible change" in rs.feedback.lower())
    payload = status.get("payload", {})

    lines: list[str] = []
    lines.append(f"# Text-Plan Run Report — `{run_id}`")
    lines.append("")
    lines.append(f"- **Plan**: `{plan_path}` ({len(plan)} numbered steps)")
    lines.append(f"- **Final status**: `{status.get('status','?')}`")
    lines.append(f"- **Started**: {status.get('started_at','?')}  →  **Finished**: {status.get('finished_at') or status.get('updated_at','?')}")
    lines.append(f"- **Caps applied**: max_steps={payload.get('max_steps','?')}, max_cost=${payload.get('max_cost','?')}, max_time={payload.get('max_time_minutes','?')}min")
    lines.append("")

    lines.append("## Plan-step coverage")
    lines.append("")
    lines.append("| # | Title | Reached? | Evidence |")
    lines.append("|---|---|---|---|")
    for step in plan:
        hit = "✅" if step.number in reached else " "
        ev = reached.get(step.number, "")
        title = step.title if len(step.title) < 60 else step.title[:57] + "..."
        lines.append(f"| {step.number} | {title} | {hit} | {ev} |")
    lines.append("")
    lines.append(f"**Plan steps reached: {len(reached)} / {len(plan)}**")
    lines.append("")

    if parsed.is_decompose_run:
        lines.append("## Decompose path (MicroPlanRunner)")
        lines.append("")
        lines.append(f"- **Intents decomposed**: {parsed.decompose_intents_total}")
        lines.append(f"- **Steps executed**: {parsed.decompose_steps_executed}")
        lines.append("")
        if parsed.decompose_step_log:
            lines.append("Per-intent log:")
            lines.append("")
            for entry in parsed.decompose_step_log[:30]:
                lines.append(f"- `{entry}`")
            lines.append("")

    lines.append("## Termination")
    lines.append("")
    lines.append(f"- **Kind**: `{failure_kind}`")
    lines.append(f"- **Detail**: {failure_detail}")
    if parsed.task_finished_reason:
        secs = (
            f"{parsed.task_finished_seconds:.1f}s"
            if parsed.task_finished_seconds is not None else "?"
        )
        lines.append(
            f"- **Runner finish**: reason=`{parsed.task_finished_reason}`, "
            f"steps={parsed.task_finished_steps}, time={secs}"
        )
    lines.append("")

    lines.append("## Action profile")
    lines.append("")
    lines.append(f"- **Runner steps executed**: {len(parsed.runner_steps)}")
    lines.append(f"- **No-visible-change feedbacks**: {no_change_count} / {len(parsed.runner_steps)}")
    lines.append("")
    lines.append("Action-type histogram:")
    lines.append("")
    if action_counter:
        for kind, n in action_counter.most_common():
            lines.append(f"- `{kind}`: {n}")
    else:
        lines.append("_no actions captured_")
    lines.append("")
    if args_counter:
        top = args_counter.most_common(3)
        if top[0][1] >= 3:
            lines.append("Top repeated action-args:")
            lines.append("")
            for args, n in top:
                if n < 3:
                    break
                lines.append(f"- `{args[:90]}` × {n}")
            lines.append("")

    lines.append("## Step-by-step trace")
    lines.append("")
    lines.append("| # | Action | Args | Dur (s) | Feedback |")
    lines.append("|---|---|---|---|---|")
    for rs in parsed.runner_steps:
        args = (rs.action_args or "")[:60]
        fb = (rs.feedback or "")[:60]
        dur = f"{rs.duration_s:.1f}" if rs.duration_s is not None else "-"
        lines.append(f"| {rs.index} | {rs.action_type or '-'} | `{args}` | {dur} | {fb} |")
    lines.append("")

    if parsed.nav_urls:
        lines.append("## Navigation URLs observed")
        lines.append("")
        for url in parsed.nav_urls:
            lines.append(f"- {url}")
        lines.append("")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--tail", type=int, default=2000, help="event-log lines to fetch")
    parser.add_argument("--out", help="write the report to this file (also printed to stdout)")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--endpoint", help="full Baseten predict endpoint")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    load_dotenv(Path(args.env_file))

    status = post_json(args, {"action": "status", "run_id": args.run_id})
    logs = post_json(args, {"action": "logs", "run_id": args.run_id, "tail": args.tail})
    events = logs.get("events", []) if isinstance(logs, dict) else []

    plan_path = Path(args.plan_file)
    plan = parse_plan(plan_path.read_text())
    parsed = parse_events(events)

    report = render_report(run_id=args.run_id, plan_path=plan_path, plan=plan, parsed=parsed, status=status)
    print(report)
    if args.out:
        Path(args.out).write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
