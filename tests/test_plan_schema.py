"""Self-consistency tests for ``docs/reference/plan.schema.json``.

Two assertions:

1. The shipped schema is itself a valid JSON Schema (Draft 2020-12). Catches
   typos in the schema file before they reach an end-user's IDE.
2. Every committed example plan (``examples/*.json``) and recipe plan
   (``src/mantis_agent/recipes/*/plan.json``) validates against the schema.
   The schema is permissive (``additionalProperties: true`` everywhere),
   so this test is a structural smoke check, not a contract on every
   plan-specific field.

If a new step type or field lands in the shipped plans before the schema
catches up, this test fails with the exact offending file + JSON Pointer
to the field that doesn't validate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "docs" / "reference" / "plan.schema.json"
EXAMPLES_DIR = REPO_ROOT / "examples"
RECIPES_DIR = REPO_ROOT / "src" / "mantis_agent" / "recipes"


def _load_schema() -> dict:
    with SCHEMA_PATH.open() as fh:
        return json.load(fh)


def _example_plans() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("*.json"))


def _recipe_plans() -> list[Path]:
    return sorted(RECIPES_DIR.glob("*/plan.json"))


def test_schema_is_well_formed() -> None:
    """The shipped plan.schema.json itself must be a valid JSON Schema."""
    schema = _load_schema()
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("plan_path", _example_plans(), ids=lambda p: p.name)
def test_example_plan_matches_schema(plan_path: Path) -> None:
    schema = _load_schema()
    validator = Draft202012Validator(schema)
    with plan_path.open() as fh:
        plan = json.load(fh)
    errors = sorted(validator.iter_errors(plan), key=lambda e: e.absolute_path)
    if errors:
        msg = "\n".join(
            f"  {list(e.absolute_path)}: {e.message}" for e in errors
        )
        pytest.fail(f"{plan_path.name} does not match plan.schema.json:\n{msg}")


@pytest.mark.parametrize(
    "plan_path", _recipe_plans(), ids=lambda p: f"{p.parent.name}/plan"
)
def test_recipe_plan_matches_schema(plan_path: Path) -> None:
    schema = _load_schema()
    validator = Draft202012Validator(schema)
    with plan_path.open() as fh:
        plan = json.load(fh)
    errors = sorted(validator.iter_errors(plan), key=lambda e: e.absolute_path)
    if errors:
        msg = "\n".join(
            f"  {list(e.absolute_path)}: {e.message}" for e in errors
        )
        pytest.fail(
            f"{plan_path.parent.name}/plan.json does not match "
            f"plan.schema.json:\n{msg}"
        )
