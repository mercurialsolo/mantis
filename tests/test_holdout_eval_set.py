"""Validate the typed holdout eval set (experiments/holdout/eval_set.json).

Guards the manifest's shape + coverage so a malformed/lopsided holdout can't
silently degrade the champion/challenger gate.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

MANIFEST = Path(__file__).resolve().parent.parent / "experiments/holdout/eval_set.json"

# The real-world capability types the holdout must represent.
REQUIRED_TYPES = {
    "login", "scrape", "search", "form_fill",
    "crud_create", "crud_edit", "export", "navigate",
}


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="module")
def tasks(manifest) -> list[dict]:
    return manifest["tasks"]


def test_manifest_loads_and_has_tasks(tasks):
    assert len(tasks) >= 16, "holdout too small for a meaningful gate verdict"


def test_task_ids_unique(tasks):
    ids = [t["task_id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task_id in holdout"


def test_every_task_is_well_formed(tasks):
    for t in tasks:
        assert t["task_id"] and t["task_text"], t
        assert t["criteria"], f"{t['task_id']} has no grading criteria"
        m = t["metadata"]
        for k in ("type", "env", "oracle_task_id", "split"):
            assert m.get(k), f"{t['task_id']} missing metadata.{k}"
        assert m["split"] in ("visible", "sealed")
        # task_id is the canonical "<env-ish>.<oracle_task_id>" anchor.
        assert m["oracle_task_id"] in t["task_id"]


def test_all_capability_types_covered(tasks):
    present = {t["metadata"]["type"] for t in tasks}
    missing = REQUIRED_TYPES - present
    assert not missing, f"holdout missing capability types: {sorted(missing)}"


def test_each_required_type_has_at_least_two_instances(tasks):
    # Type coverage over site coverage — guard against overfitting one site.
    by_type = Counter(t["metadata"]["type"] for t in tasks)
    thin = {ty: by_type[ty] for ty in REQUIRED_TYPES if by_type[ty] < 2}
    assert not thin, f"capability types with <2 tasks (gameable): {thin}"


def test_has_a_real_sealed_holdout_split(tasks):
    sealed = [t for t in tasks if t["metadata"]["split"] == "sealed"]
    # The gate evaluates on sealed; need enough for the paired bootstrap.
    assert len(sealed) >= 8, "too few sealed tasks for a gate evaluation"
    # Sealed must span multiple capability types (not all one type).
    assert len({t["metadata"]["type"] for t in sealed}) >= 5


def test_spans_multiple_envs(tasks):
    envs = {t["metadata"]["env"] for t in tasks}
    assert len(envs) >= 6, f"holdout too env-concentrated: {sorted(envs)}"


def test_oracle_grading_criterion_present(tasks):
    # Every task is graded by its env oracle → task_success criterion.
    for t in tasks:
        types = {c["type"] for c in t["criteria"]}
        assert "task_success" in types, f"{t['task_id']} lacks oracle task_success"
