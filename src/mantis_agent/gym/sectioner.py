"""Auto-sectioner — breaks a big plan into manageable executable sections.

Takes a workflow specification (plain text or structured) and automatically
splits it into bounded sections, each with:
- Clear entry/exit conditions
- Independent retry budget
- State passing between sections (via session persistence)
- Appropriate step limits based on section complexity

The key insight: a 500-step monolithic task fails because the model loses
context and the loop detector fires. But 6 sections of 60 steps each work
because each section has a focused goal.

Usage:
    from mantis_agent.gym.sectioner import section_workflow

    sections = section_workflow(
        plan_text=open("plans/boattrader/spec.md").read(),
        inputs={"zip_code": "33101", "search_radius": "35"},
    )
    # Returns a task suite JSON (same format as sectioned_workflow.json)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Section:
    """A bounded executable section of a workflow."""
    task_id: str
    intent: str
    max_steps: int = 60
    max_retries: int = 3
    save_session: bool = False
    require_session: bool = False
    start_url: str = ""
    verify: dict = field(default_factory=dict)


def section_workflow(
    plan_text: str,
    inputs: dict[str, str] | None = None,
    session_name: str = "workflow",
    base_url: str = "",
    max_steps_per_section: int = 60,
) -> dict:
    """Break a workflow plan into executable sections.

    Analyzes the plan text for natural section boundaries:
    - "Step N" markers
    - Login/authentication blocks
    - Iteration patterns ("for each", "repeat")
    - Navigation between sites/tabs

    Args:
        plan_text: The full workflow specification text.
        inputs: Input variables to resolve in the plan.
        session_name: Name for session persistence.
        base_url: Default starting URL.
        max_steps_per_section: Max steps per section.

    Returns:
        Task suite JSON dict (compatible with modal_web_tasks_opencua.py).
    """
    inputs = inputs or {}

    # Resolve input variables in the plan text
    resolved = plan_text
    for key, value in inputs.items():
        resolved = resolved.replace(f"{{{key}}}", value)
        resolved = resolved.replace(f"{{{{{key}}}}}", value)

    # Detect section boundaries
    sections = _detect_sections(resolved, base_url, max_steps_per_section)

    # Build task suite
    tasks = []
    for i, section in enumerate(sections):
        task = {
            "task_id": section.task_id,
            "intent": section.intent,
            "start_url": section.start_url or base_url,
        }
        if section.save_session:
            task["save_session"] = True
        if section.require_session:
            task["require_session"] = True
        if section.verify:
            task["verify"] = section.verify
        tasks.append(task)

    return {
        "session_name": session_name,
        "base_url": base_url,
        "tasks": tasks,
    }


def _detect_sections(text: str, base_url: str, max_steps: int) -> list[Section]:
    """Detect natural section boundaries in workflow text."""
    sections: list[Section] = []

    # Strategy 1: Look for explicit "Step N" or "=== Section ===" markers
    step_pattern = re.compile(
        r'(?:^|\n)(?:=+\s*)?(?:Step|STEP|Phase|PHASE|Section|SECTION)\s*(\d+)[:\s—\-]+(.+?)(?=\n(?:=+\s*)?(?:Step|STEP|Phase|PHASE|Section|SECTION)\s*\d+|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    matches = list(step_pattern.finditer(text))
    if matches:
        for i, match in enumerate(matches):
            step_num = match.group(1)
            content = match.group(2).strip()
            title = content.split('\n')[0][:60]

            # Determine section properties from content
            is_login = any(kw in content.lower() for kw in ["login", "authenticate", "sign in", "password", "credential"])
            is_search = any(kw in content.lower() for kw in ["search", "filter", "navigate", "sort"])
            is_extract = any(kw in content.lower() for kw in ["extract", "read", "inspect", "check", "listing", "phone"])
            is_entry = any(kw in content.lower() for kw in ["enter", "fill", "submit", "form"])
            is_loop = any(kw in content.lower() for kw in ["for each", "repeat", "loop", "next page", "pagination"])

            section = Section(
                task_id=f"step_{step_num}_{_slugify(title)}",
                intent=content[:2000],
                max_steps=120 if is_login else (80 if is_loop else max_steps),
                max_retries=1 if is_login else 3,
                save_session=is_login or is_search,
                require_session=i > 0,
                start_url=base_url,
            )

            # URL detection
            urls = re.findall(r'https?://[^\s<>"]+', content)
            if urls:
                section.start_url = urls[0]

            sections.append(section)

        return sections

    # Strategy 2: Split by empty lines + heuristic grouping
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) <= 3:
        # Too few paragraphs — treat as single section
        return [Section(
            task_id="full_task",
            intent=text[:3000],
            max_steps=max_steps * 3,
            start_url=base_url,
        )]

    # Group paragraphs into logical sections
    current_section_text = []
    section_num = 0

    for para in paragraphs:
        current_section_text.append(para)

        # Section boundary heuristics
        is_boundary = (
            re.match(r'^\d+[\.\)]\s', para) and len(current_section_text) > 2
            or any(kw in para.lower() for kw in ["then", "next", "after that", "finally"])
            or len('\n'.join(current_section_text)) > 1500
        )

        if is_boundary:
            section_num += 1
            content = '\n\n'.join(current_section_text)
            sections.append(Section(
                task_id=f"section_{section_num}",
                intent=content[:2000],
                max_steps=max_steps,
                require_session=section_num > 1,
                save_session=section_num == 1,
                start_url=base_url,
            ))
            current_section_text = []

    # Remaining text
    if current_section_text:
        section_num += 1
        sections.append(Section(
            task_id=f"section_{section_num}",
            intent='\n\n'.join(current_section_text)[:2000],
            max_steps=max_steps,
            require_session=True,
            start_url=base_url,
        ))

    return sections if sections else [Section(
        task_id="full_task",
        intent=text[:3000],
        max_steps=max_steps * 3,
        start_url=base_url,
    )]


def _slugify(text: str) -> str:
    """Convert text to a safe task ID slug."""
    slug = re.sub(r'[^a-z0-9]+', '_', text.lower())
    return slug.strip('_')[:30]


def section_from_spec(
    spec_path: str,
    inputs: dict[str, str] | None = None,
    session_name: str = "workflow",
    output_path: str | None = None,
) -> dict:
    """Load a spec file and generate sectioned task JSON.

    Args:
        spec_path: Path to workflow spec (.md or .txt).
        inputs: Input variables.
        session_name: Session name.
        output_path: If set, save the task JSON to this path.

    Returns:
        Task suite dict.
    """
    with open(spec_path) as f:
        plan_text = f.read()

    # Extract base URL from plan
    url_match = re.search(r'(?:URL|url|Start):\s*(https?://[^\s]+)', plan_text)
    base_url = url_match.group(1) if url_match else ""

    result = section_workflow(
        plan_text=plan_text,
        inputs=inputs,
        session_name=session_name,
        base_url=base_url,
        max_steps_per_section=60,
    )

    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result
