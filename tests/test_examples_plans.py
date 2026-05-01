"""Smoke-test that every committed example plan passes server-side validation.

Catches drift between the API schema (`api_schemas.validate_micro_steps`) and
the curated examples shipped under `examples/`. Without this, an `intent` /
`type` rename in the validator silently breaks the README's curl examples.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mantis_agent.api_schemas import validate_micro_steps

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
RECIPE_PLANS = list((REPO_ROOT / "recipes").glob("*/plan.json"))


def _example_plans() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("*.json"))


@pytest.mark.parametrize("plan_path", _example_plans(), ids=lambda p: p.name)
def test_example_plan_validates(plan_path: Path) -> None:
    with plan_path.open() as f:
        steps = json.load(f)
    validated = validate_micro_steps(steps)
    assert isinstance(validated, list)
    assert len(validated) == len(steps)


@pytest.mark.parametrize("plan_path", RECIPE_PLANS, ids=lambda p: f"{p.parent.name}/plan")
def test_recipe_plan_validates(plan_path: Path) -> None:
    with plan_path.open() as f:
        steps = json.load(f)
    validated = validate_micro_steps(steps)
    assert isinstance(validated, list)
    assert len(validated) == len(steps)


def test_examples_directory_is_not_empty() -> None:
    """Catch the regression where examples/ was gitignored away."""
    assert _example_plans(), "examples/ must contain at least one .json plan"
