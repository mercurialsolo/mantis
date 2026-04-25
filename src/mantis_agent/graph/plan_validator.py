"""PlanValidator — structural checks on compiled MicroPlans before execution.

Catches common issues that cause silent failures at runtime:
  - Missing navigate step (browser starts on about:blank)
  - Filters in objective but no filter steps in plan
  - No gate after filter steps (extraction runs on wrong page)
  - Extraction loop with no navigate_back (stuck on detail page)
  - Loop targets pointing to wrong step index
  - Missing claude_only on extract_url/extract_data steps
  - Pagination loop without extraction loop

Runs after GraphCompiler or PlanDecomposer, before execution.
No API calls — pure structural analysis.

Usage:
    from mantis_agent.graph.plan_validator import PlanValidator

    validator = PlanValidator()
    issues = validator.validate(micro_plan, objective=spec)
    if issues:
        enhanced_plan = validator.enhance(micro_plan, objective=spec)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..plan_decomposer import MicroIntent, MicroPlan

logger = logging.getLogger(__name__)


@dataclass
class PlanIssue:
    """A structural issue found in a MicroPlan."""

    severity: str  # "error" (blocks execution), "warning" (may cause failures)
    code: str  # machine-readable: "missing_navigate", "no_gate", etc.
    message: str  # human-readable description
    step_index: int = -1  # which step, or -1 for plan-level issues
    auto_fix: str = ""  # what enhance() would do, or "" if not auto-fixable


class PlanValidator:
    """Validate and enhance MicroPlans before execution."""

    def validate(
        self,
        plan: MicroPlan,
        objective: Any = None,
    ) -> list[PlanIssue]:
        """Check plan for structural issues. Returns list of issues (empty = clean)."""
        issues: list[PlanIssue] = []
        steps = plan.steps

        if not steps:
            issues.append(PlanIssue(
                severity="error", code="empty_plan",
                message="Plan has no steps",
            ))
            return issues

        issues.extend(self._check_navigate(steps))
        issues.extend(self._check_filters(steps, objective))
        issues.extend(self._check_gate(steps))
        issues.extend(self._check_extraction_loop(steps))
        issues.extend(self._check_loop_targets(steps))
        issues.extend(self._check_claude_only(steps))
        issues.extend(self._check_pagination(steps))
        issues.extend(self._check_sections(steps))

        return issues

    def enhance(
        self,
        plan: MicroPlan,
        objective: Any = None,
    ) -> MicroPlan:
        """Fix auto-fixable issues in the plan. Returns a new MicroPlan."""
        issues = self.validate(plan, objective)
        if not issues:
            return plan

        steps = list(plan.steps)
        applied: list[str] = []

        # Fix missing navigate
        if any(i.code == "missing_navigate" for i in issues):
            url = ""
            if objective:
                url = getattr(objective, "start_url", "") or ""
            if not url and objective:
                domains = getattr(objective, "domains", [])
                if domains:
                    url = f"https://www.{domains[0]}/"
            if url:
                steps.insert(0, MicroIntent(
                    intent=f"Navigate to {url}",
                    type="navigate",
                    budget=3,
                    section="setup",
                    required=True,
                ))
                applied.append(f"inserted navigate to {url}")

        # Fix missing gate after filters
        if any(i.code == "no_gate_after_filters" for i in issues):
            # Find last filter step
            last_filter = -1
            for idx, s in enumerate(steps):
                if s.type == "filter" or (s.section == "setup" and s.required and s.type != "navigate"):
                    last_filter = idx
            if last_filter >= 0:
                entity = ""
                if objective:
                    entity = getattr(objective, "target_entity", "") or "listing"
                    filters = getattr(objective, "required_filters", [])
                    filter_summary = ", ".join(filters) if filters else "required filters"
                else:
                    entity = "listing"
                    filter_summary = "required filters"
                gate_step = MicroIntent(
                    intent=f"Verify page shows {entity} results with {filter_summary} applied",
                    type="extract_data",
                    claude_only=True,
                    budget=0,
                    section="setup",
                    gate=True,
                    verify=f"Page shows filtered {entity} results with {filter_summary} active",
                )
                steps.insert(last_filter + 1, gate_step)
                applied.append(f"inserted gate after step {last_filter}")

        # Fix claude_only on extract steps
        for idx, s in enumerate(steps):
            if s.type in ("extract_url", "extract_data") and not s.claude_only:
                s.claude_only = True
                s.budget = 0
                applied.append(f"set claude_only on step {idx} ({s.type})")

        # Fix loop targets (re-validate after inserts)
        steps = self._fix_loop_targets(steps)

        if applied:
            logger.info("PlanValidator enhanced plan: %s", "; ".join(applied))

        enhanced = MicroPlan(steps=steps, source_plan=plan.source_plan, domain=plan.domain)
        return enhanced

    # ── Individual checks ──

    def _check_navigate(self, steps: list[MicroIntent]) -> list[PlanIssue]:
        """First step should be navigate with a URL."""
        issues = []
        has_navigate = any(s.type == "navigate" for s in steps)
        if not has_navigate:
            # Check if first step intent contains a URL
            first = steps[0]
            has_url = bool(re.search(r"https?://", first.intent))
            if not has_url:
                issues.append(PlanIssue(
                    severity="error", code="missing_navigate",
                    message="No navigate step — browser will start on about:blank",
                    auto_fix="Insert navigate step with objective.start_url",
                ))
        return issues

    def _check_filters(self, steps: list[MicroIntent], objective: Any) -> list[PlanIssue]:
        """If objective has required_filters, plan should have filter steps."""
        issues = []
        if not objective:
            return issues
        required_filters = getattr(objective, "required_filters", [])
        if not required_filters:
            return issues
        filter_steps = [s for s in steps if s.type == "filter"]
        if not filter_steps:
            issues.append(PlanIssue(
                severity="warning", code="no_filter_steps",
                message=f"Objective requires {len(required_filters)} filters but plan has no filter steps: {required_filters}",
            ))
        return issues

    def _check_gate(self, steps: list[MicroIntent]) -> list[PlanIssue]:
        """Should have a gate step after setup filters, before extraction."""
        issues = []
        has_filters = any(s.type == "filter" or (s.section == "setup" and s.required and s.type != "navigate") for s in steps)
        has_gate = any(s.gate for s in steps)
        if has_filters and not has_gate:
            issues.append(PlanIssue(
                severity="error", code="no_gate_after_filters",
                message="Filter steps exist but no gate verification — extraction may run on wrong page",
                auto_fix="Insert gate step after last filter",
            ))
        return issues

    def _check_extraction_loop(self, steps: list[MicroIntent]) -> list[PlanIssue]:
        """Extraction section should have navigate_back before loop."""
        issues = []
        extraction_steps = [s for s in steps if s.section == "extraction"]
        if not extraction_steps:
            return issues
        has_click = any(s.type == "click" for s in extraction_steps)
        has_back = any(s.type == "navigate_back" for s in extraction_steps)
        has_loop = any(s.type == "loop" for s in extraction_steps)
        if has_click and has_loop and not has_back:
            issues.append(PlanIssue(
                severity="error", code="no_navigate_back_in_loop",
                message="Extraction loop has click but no navigate_back — will get stuck on detail page",
            ))
        return issues

    def _check_loop_targets(self, steps: list[MicroIntent]) -> list[PlanIssue]:
        """Loop targets should point to valid step indices."""
        issues = []
        for idx, s in enumerate(steps):
            if s.type == "loop" and s.loop_target >= 0:
                if s.loop_target >= len(steps):
                    issues.append(PlanIssue(
                        severity="error", code="loop_target_out_of_range",
                        message=f"Step {idx} loop_target={s.loop_target} but plan only has {len(steps)} steps",
                        step_index=idx,
                    ))
                elif s.loop_target >= idx:
                    issues.append(PlanIssue(
                        severity="error", code="loop_target_forward",
                        message=f"Step {idx} loop_target={s.loop_target} points forward (should loop back)",
                        step_index=idx,
                    ))
                if s.loop_count <= 0:
                    issues.append(PlanIssue(
                        severity="warning", code="loop_count_zero",
                        message=f"Step {idx} has loop_count={s.loop_count} — loop will never execute",
                        step_index=idx,
                    ))
        return issues

    def _check_claude_only(self, steps: list[MicroIntent]) -> list[PlanIssue]:
        """extract_url and extract_data should be claude_only."""
        issues = []
        for idx, s in enumerate(steps):
            if s.type in ("extract_url", "extract_data") and not s.claude_only:
                issues.append(PlanIssue(
                    severity="warning", code="extract_not_claude_only",
                    message=f"Step {idx} is {s.type} but claude_only=False — will waste GPU steps",
                    step_index=idx,
                    auto_fix="Set claude_only=True and budget=0",
                ))
        return issues

    def _check_pagination(self, steps: list[MicroIntent]) -> list[PlanIssue]:
        """Pagination loop should exist if extraction loop exists."""
        issues = []
        extraction_loops = [s for s in steps if s.type == "loop" and s.section == "extraction"]
        pagination_loops = [s for s in steps if s.type == "loop" and s.section == "pagination"]
        has_paginate = any(s.type == "paginate" for s in steps)
        if extraction_loops and not has_paginate:
            issues.append(PlanIssue(
                severity="warning", code="no_pagination",
                message="Extraction loop exists but no paginate step — only first page will be processed",
            ))
        if has_paginate and not pagination_loops:
            issues.append(PlanIssue(
                severity="warning", code="paginate_without_loop",
                message="Paginate step exists but no pagination loop — only one page transition",
            ))
        return issues

    def _check_sections(self, steps: list[MicroIntent]) -> list[PlanIssue]:
        """Section ordering should be setup → extraction → pagination."""
        issues = []
        seen_extraction = False
        seen_pagination = False
        for idx, s in enumerate(steps):
            if s.section == "extraction":
                seen_extraction = True
            elif s.section == "pagination":
                seen_pagination = True
            elif s.section == "setup" and seen_extraction:
                issues.append(PlanIssue(
                    severity="warning", code="setup_after_extraction",
                    message=f"Step {idx} is setup but comes after extraction steps",
                    step_index=idx,
                ))
        return issues

    def _fix_loop_targets(self, steps: list[MicroIntent]) -> list[MicroIntent]:
        """Ensure extraction loops target the first extraction click step."""
        click_idx = None
        for i, s in enumerate(steps):
            if s.type == "click" and s.section == "extraction":
                click_idx = i
                break
        if click_idx is None:
            return steps
        for s in steps:
            if s.type == "loop" and s.loop_target >= 0:
                if s.section == "extraction" and s.loop_target != click_idx:
                    if abs(s.loop_target - click_idx) <= 3:
                        s.loop_target = click_idx
                elif s.section == "pagination" and s.loop_target != click_idx:
                    if abs(s.loop_target - click_idx) <= 3:
                        s.loop_target = click_idx
        return steps
