"""#920 — structural checks for the expanded holdout (v2) candidate set.

These pin the *shape* of the candidate plans authored offline (grounded from each
env's oracle + templates). They do NOT run anything — the live-verification pass
runs each through Mantis and freezes the survivors into ``mantis-holdout-v2``. The
point: enough well-formed, oracle-grounded tasks exist that the gate's paired
bootstrap can clear ``prob_improvement >= 0.95`` (3 tasks can't; ~15+ can).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "holdout"))

from sealed_plans import SANDBOXES, SEALED_TASKS, V2_HOLDOUT  # noqa: E402

_VALID_STEP_TYPES = {"navigate", "click", "submit", "fill_field"}


def test_v2_holdout_is_live_verified_and_wired():
    # The runnable v2 set: every entry exists, is marked verified, on a wired env.
    assert len(V2_HOLDOUT) >= 6, f"v2 holdout too small: {len(V2_HOLDOUT)}"
    assert len(set(V2_HOLDOUT)) == len(V2_HOLDOUT), "duplicate task in V2_HOLDOUT"
    for key in V2_HOLDOUT:
        assert key in SEALED_TASKS, f"{key} not in SEALED_TASKS"
        spec = SEALED_TASKS[key]
        assert spec.get("status", "verified") == "verified", f"{key} is not verified"
        assert SANDBOXES.get(spec["env"]), f"{key}'s env {spec['env']!r} not wired"


def test_holdout_has_enough_tasks_for_gate_significance():
    # 3 tasks top out at prob_improvement ~0.7; ~15+ can exceed 0.95.
    assert len(SEALED_TASKS) >= 15, f"only {len(SEALED_TASKS)} tasks — too few for gate significance"


def test_every_task_is_well_formed():
    for key, spec in SEALED_TASKS.items():
        assert spec.get("env"), f"{key}: missing env"
        assert spec.get("oracle_task_id"), f"{key}: missing oracle_task_id"
        assert spec.get("steps"), f"{key}: no steps"
        assert spec["env"] in SANDBOXES, f"{key}: env {spec['env']!r} not in SANDBOXES"


def test_every_step_uses_a_valid_primitive():
    for key, spec in SEALED_TASKS.items():
        for i, step in enumerate(spec["steps"]):
            assert step.get("type") in _VALID_STEP_TYPES, f"{key} step {i}: bad type {step.get('type')!r}"
            assert step.get("intent"), f"{key} step {i}: missing intent"


def test_every_plan_starts_with_a_navigate():
    for key, spec in SEALED_TASKS.items():
        assert spec["steps"][0]["type"] == "navigate", f"{key}: first step must be navigate"


def test_spans_multiple_capability_envs():
    # Representativeness: the holdout must span envs, not pile onto one site.
    envs = {spec["env"] for spec in SEALED_TASKS.values()}
    assert len(envs) >= 6, f"only {len(envs)} envs — not representative"


def test_candidate_vs_verified_split_is_marked():
    # v1-frozen tasks are 'verified'; new ones carry status='candidate' so the
    # freeze step (and operators) can tell what still needs a live pass.
    statuses = {spec.get("status", "verified") for spec in SEALED_TASKS.values()}
    assert "candidate" in statuses, "expected some status='candidate' (the v2 additions)"
    candidates = [k for k, v in SEALED_TASKS.items() if v.get("status") == "candidate"]
    assert len(candidates) >= 10, f"only {len(candidates)} candidates authored"


def test_oracle_task_ids_unique():
    ids = [spec["oracle_task_id"] for spec in SEALED_TASKS.values()]
    assert len(ids) == len(set(ids)), "duplicate oracle_task_id across tasks"


def test_fill_steps_carry_label_and_value():
    for key, spec in SEALED_TASKS.items():
        for i, step in enumerate(spec["steps"]):
            if step["type"] == "fill_field":
                p = step.get("params", {})
                assert p.get("label"), f"{key} step {i}: fill_field missing label"
                assert "value" in p, f"{key} step {i}: fill_field missing value"
