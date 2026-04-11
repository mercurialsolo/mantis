"""Persistent plan files for web tasks.

Plans are YAML files that define multi-step browser workflows. They can be:
- Edited by hand (tweak steps, add validation)
- Tested independently (run a single plan against an env)
- Parameterized with user inputs ({{variable}} placeholders)
- Generated from natural language and saved for reuse

Plan format:
    name: login_crm
    description: Log in to StaffAI CRM
    url: https://staffai-test-crm.exe.xyz
    inputs:
      user_id:
        description: "CRM user ID"
        default: "sarah.connor"
      password:
        description: "CRM password"
        required: true

    steps:
      - action: navigate
        url: "{{url}}"

      - action: click
        target: "User ID input field"
        wait_for: input_focused

      - action: type
        text: "{{user_id}}"

      - action: click
        target: "Password input field"
        wait_for: input_focused

      - action: type
        text: "{{password}}"

      - action: click
        target: "Sign In button"
        wait_for: url_changed

      - action: verify
        check: url_contains
        value: "dashboard"

    on_input_needed:
      message: "Plan requires user input for: {missing_inputs}"
      block: true
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try YAML, fall back to JSON
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class PlanInput:
    """A user-configurable input variable for a plan."""
    name: str
    description: str = ""
    default: str | None = None
    required: bool = False


@dataclass
class PlanStep:
    """A single step in a plan."""
    action: str  # navigate, click, type, key, scroll, wait, verify, done
    params: dict[str, Any] = field(default_factory=dict)
    target: str = ""  # Natural language description of what to interact with
    wait_for: str = ""  # Condition to wait for after action
    on_fail: str = "continue"  # "continue", "retry", "abort", "ask_user"


@dataclass
class Plan:
    """A persistent, editable, parameterized task plan."""
    name: str
    description: str
    url: str
    steps: list[PlanStep]
    inputs: list[PlanInput] = field(default_factory=list)
    require_session: bool = False
    save_session: bool = False
    file_path: str | None = None

    def resolve_inputs(self, provided: dict[str, str] | None = None) -> tuple[str, list[str]]:
        """Resolve input placeholders and return (task_text, missing_inputs).

        Args:
            provided: Dict of input_name → value. If None, uses defaults.

        Returns:
            Tuple of (resolved task instruction, list of missing required inputs).
        """
        provided = provided or {}
        resolved: dict[str, str] = {}
        missing: list[str] = []

        for inp in self.inputs:
            if inp.name in provided:
                resolved[inp.name] = provided[inp.name]
            elif inp.default is not None:
                resolved[inp.name] = inp.default
            elif inp.required:
                missing.append(inp.name)

        # Also resolve URL
        resolved["url"] = self.url

        task_text = self.to_instruction()
        for key, value in resolved.items():
            task_text = task_text.replace(f"{{{{{key}}}}}", value)

        return task_text, missing

    def to_instruction(self) -> str:
        """Convert plan steps to a natural language instruction for the brain."""
        lines = [self.description]
        for i, step in enumerate(self.steps, 1):
            line = f"{i}. "
            match step.action:
                case "navigate":
                    line += f"Go to {step.params.get('url', self.url)}"
                case "click":
                    line += f"Click on {step.target}"
                case "type":
                    line += f"Type \"{step.params.get('text', '{{text}}')}\""
                    if step.target:
                        line += f" into {step.target}"
                case "key":
                    line += f"Press {step.params.get('keys', '')}"
                case "scroll":
                    line += f"Scroll {step.params.get('direction', 'down')}"
                case "wait":
                    line += f"Wait for {step.wait_for or 'page to load'}"
                case "verify":
                    check = step.params.get("check", "")
                    value = step.params.get("value", "")
                    line += f"Verify: {check} = \"{value}\""
                case "done":
                    line += "Task complete"
                case _:
                    line += f"{step.action}: {step.target or step.params}"
            lines.append(line)
        return "\n".join(lines)

    def to_task_config(self) -> dict:
        """Convert to the task config format expected by run_web_tasks.py."""
        return {
            "task_id": self.name,
            "intent": self.to_instruction(),
            "start_url": self.url,
            "require_session": self.require_session,
            "save_session": self.save_session,
            "verify": self._extract_verify(),
        }

    def _extract_verify(self) -> dict:
        """Extract verification config from the last verify step."""
        for step in reversed(self.steps):
            if step.action == "verify":
                return {
                    "type": step.params.get("check", ""),
                    "value": step.params.get("value", ""),
                }
        return {}


def load_plan(path: str, inputs: dict[str, str] | None = None) -> Plan:
    """Load a plan from a YAML or JSON file.

    Args:
        path: Path to the plan file.
        inputs: Optional input values to resolve placeholders.

    Returns:
        Plan instance with resolved inputs.
    """
    path_obj = Path(path)
    content = path_obj.read_text()

    if path_obj.suffix in (".yml", ".yaml") and HAS_YAML:
        data = yaml.safe_load(content)
    else:
        import json
        data = json.loads(content)

    # Parse inputs
    plan_inputs = []
    for name, spec in (data.get("inputs") or {}).items():
        if isinstance(spec, str):
            plan_inputs.append(PlanInput(name=name, description=spec))
        elif isinstance(spec, dict):
            plan_inputs.append(PlanInput(
                name=name,
                description=spec.get("description", ""),
                default=spec.get("default"),
                required=spec.get("required", False),
            ))

    # Parse steps
    steps = []
    for step_data in data.get("steps", []):
        params = {k: v for k, v in step_data.items()
                  if k not in ("action", "target", "wait_for", "on_fail")}
        steps.append(PlanStep(
            action=step_data.get("action", ""),
            params=params,
            target=step_data.get("target", ""),
            wait_for=step_data.get("wait_for", ""),
            on_fail=step_data.get("on_fail", "continue"),
        ))

    plan = Plan(
        name=data.get("name", path_obj.stem),
        description=data.get("description", ""),
        url=data.get("url", ""),
        steps=steps,
        inputs=plan_inputs,
        require_session=data.get("require_session", False),
        save_session=data.get("save_session", False),
        file_path=str(path_obj),
    )

    return plan


def save_plan(plan: Plan, path: str) -> str:
    """Save a plan to a YAML or JSON file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "name": plan.name,
        "description": plan.description,
        "url": plan.url,
    }

    if plan.require_session:
        data["require_session"] = True
    if plan.save_session:
        data["save_session"] = True

    if plan.inputs:
        data["inputs"] = {}
        for inp in plan.inputs:
            entry: dict[str, Any] = {"description": inp.description}
            if inp.default is not None:
                entry["default"] = inp.default
            if inp.required:
                entry["required"] = True
            data["inputs"][inp.name] = entry

    data["steps"] = []
    for step in plan.steps:
        step_data: dict[str, Any] = {"action": step.action}
        if step.target:
            step_data["target"] = step.target
        if step.wait_for:
            step_data["wait_for"] = step.wait_for
        if step.on_fail != "continue":
            step_data["on_fail"] = step.on_fail
        step_data.update(step.params)
        data["steps"].append(step_data)

    if path_obj.suffix in (".yml", ".yaml") and HAS_YAML:
        content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    else:
        import json
        content = json.dumps(data, indent=2)

    path_obj.write_text(content)
    logger.info(f"Plan saved: {path}")
    return str(path_obj)


def get_missing_inputs(plan: Plan, provided: dict[str, str] | None = None) -> list[PlanInput]:
    """Return list of required inputs that are not provided and have no default."""
    provided = provided or {}
    missing = []
    for inp in plan.inputs:
        if inp.required and inp.name not in provided and inp.default is None:
            missing.append(inp)
    return missing
