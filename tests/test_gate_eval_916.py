"""#916 — holdout-eval runner: suite generation + ArmResult mapping + gate wiring.

The live submit/grade arm is spend-gated; these pin the pure core (no network):
suite shape (built micro-plan, oracle fields, per-arm profile isolation, the #911
challenger adapter), the result→ArmResult mapping, the eval_harness tasks-JSON
emission, and that the mapped arms drive a sensible PromotionGate verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "holdout"))

from mantis_agent.learning.promotion_gate import ArmResult, PromotionGate  # noqa: E402

import run_gate_eval as rge  # noqa: E402
from sealed_plans import SEALED_TASKS  # noqa: E402

_INFO = {
    "url": "https://sandbox.example/app",
    "preview_token": "pv-tok",
    "admin_token": "admin-tok",
}
_KEY = "indeed.t01_search_save_remote"
_SPEC = SEALED_TASKS[_KEY]
_TID = _SPEC["oracle_task_id"]


# ── build_eval_suite ────────────────────────────────────────────────────


def test_suite_has_built_micro_plan_not_raw_steps():
    suite = rge.build_eval_suite(_SPEC, _INFO, arm=rge.CHAMPION, task_key=_KEY)
    # build_micro_suite output carries a non-empty _micro_plan (the 0/0/0 footgun
    # is passing raw {steps:[...]}). Steps are dicts, not the raw sealed-plan form.
    assert suite.get("_micro_plan"), "suite must carry a built _micro_plan"
    assert isinstance(suite["_micro_plan"], list) and suite["_micro_plan"]


def test_suite_carries_oracle_fields():
    suite = rge.build_eval_suite(_SPEC, _INFO, arm=rge.CHAMPION, task_key=_KEY)
    assert suite["_oracle_task_id"] == _TID
    assert suite["_oracle_url"] == _INFO["url"]
    assert suite["_oracle_admin_token"] == "admin-tok"
    assert suite["_oracle_preview_token"] == "pv-tok"


def test_champion_arm_has_no_adapter():
    suite = rge.build_eval_suite(_SPEC, _INFO, arm=rge.CHAMPION, task_key=_KEY)
    assert "_lora_adapter" not in suite


def test_challenger_arm_sets_adapter():
    suite = rge.build_eval_suite(
        _SPEC, _INFO, arm=rge.CHALLENGER, task_key=_KEY,
        lora_adapter="mantis-trainer-vol:/checkpoints/sft-x",
    )
    assert suite["_lora_adapter"] == "mantis-trainer-vol:/checkpoints/sft-x"


def test_challenger_model_full_swap_arm():
    # #918: full-model-swap challenger sets _challenger_model (not _lora_adapter).
    suite = rge.build_eval_suite(
        _SPEC, _INFO, arm=rge.CHALLENGER, task_key=_KEY,
        challenger_model="mantis-trainer-vol:/checkpoints/sft-x/merged.Q8_0.gguf",
    )
    assert suite["_challenger_model"].endswith("merged.Q8_0.gguf")
    assert "_lora_adapter" not in suite


def test_challenger_model_takes_precedence_over_adapter():
    suite = rge.build_eval_suite(
        _SPEC, _INFO, arm=rge.CHALLENGER, task_key=_KEY,
        lora_adapter="vol:/a.gguf", challenger_model="vol:/m.gguf",
    )
    assert suite["_challenger_model"] == "vol:/m.gguf"
    assert "_lora_adapter" not in suite


def test_arms_use_distinct_profile_ids():
    champ = rge.build_eval_suite(_SPEC, _INFO, arm=rge.CHAMPION, task_key=_KEY)
    chal = rge.build_eval_suite(_SPEC, _INFO, arm=rge.CHALLENGER, task_key=_KEY)
    # Distinct profile per (task, arm) → no 409 collision when run concurrently.
    assert champ["_profile_id"] != chal["_profile_id"]
    assert champ["_profile_id"].startswith("gate-champion-")
    assert chal["_profile_id"].startswith("gate-challenger-")


def test_run_nonce_makes_profiles_unique_across_runs():
    # #920: a per-invocation nonce → a prior run's stale Chrome lock can't 409 us.
    a = rge.build_eval_suite(_SPEC, _INFO, arm=rge.CHAMPION, task_key=_KEY, run_nonce="aaaa")
    b = rge.build_eval_suite(_SPEC, _INFO, arm=rge.CHAMPION, task_key=_KEY, run_nonce="bbbb")
    assert a["_profile_id"] == f"gate-champion-{_KEY}-aaaa"
    assert a["_profile_id"] != b["_profile_id"]  # different runs → different profiles
    # still distinct across arms within a run
    chal = rge.build_eval_suite(_SPEC, _INFO, arm=rge.CHALLENGER, task_key=_KEY, run_nonce="aaaa")
    assert chal["_profile_id"] != a["_profile_id"]


# ── arm_from_results ────────────────────────────────────────────────────


def test_arm_from_results_maps_pass_fail_to_scores():
    arm = rge.arm_from_results([
        {"task_id": "a", "oracle_passed": True, "cost": 0.30},
        {"task_id": "b", "oracle_passed": False, "cost": 0.12},
    ])
    assert arm.scores == {"a": 1.0, "b": 0.0}
    assert arm.costs == {"a": 0.30, "b": 0.12}


def test_arm_from_results_drops_entries_without_task_id():
    arm = rge.arm_from_results([{"oracle_passed": True}, {"task_id": "c", "oracle_passed": True}])
    assert arm.scores == {"c": 1.0}


def test_arm_from_results_cost_optional():
    arm = rge.arm_from_results([{"task_id": "a", "oracle_passed": True}])
    assert arm.scores == {"a": 1.0}
    assert arm.costs == {}


# ── to_eval_tasks_json ──────────────────────────────────────────────────


def test_tasks_json_populates_micro_plan():
    tasks = rge.to_eval_tasks_json([_KEY], {_SPEC["env"]: _INFO["url"]})
    assert len(tasks) == 1
    t = tasks[0]
    assert t["task_id"] == _TID
    assert t["micro_plan"], "generated tasks must carry a micro_plan (the #916 gap)"
    assert t["metadata"]["env"] == _SPEC["env"]


# ── gate wiring ─────────────────────────────────────────────────────────


def test_mapped_arms_drive_promotion_gate():
    # Challenger strictly better on every task → gate should promote.
    champ = rge.arm_from_results([
        {"task_id": "a", "oracle_passed": False},
        {"task_id": "b", "oracle_passed": False},
        {"task_id": "c", "oracle_passed": False},
    ])
    chal = rge.arm_from_results([
        {"task_id": "a", "oracle_passed": True},
        {"task_id": "b", "oracle_passed": True},
        {"task_id": "c", "oracle_passed": True},
    ])
    verdict = PromotionGate().evaluate(champ, chal)
    assert verdict.n_compared == 3
    assert verdict.win_rate == 1.0
    assert verdict.mean_delta == 1.0
    assert verdict.promote is True


def test_no_adapter_sanity_run_does_not_promote():
    # champion == challenger (both base) → zero delta → HOLD.
    same = rge.arm_from_results([
        {"task_id": "a", "oracle_passed": True},
        {"task_id": "b", "oracle_passed": False},
    ])
    verdict = PromotionGate().evaluate(same, ArmResult(label="b", scores=dict(same.scores)))
    assert verdict.mean_delta == 0.0
    assert verdict.promote is False
