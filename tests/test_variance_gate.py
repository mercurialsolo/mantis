"""Group-variance gate for GRPO sweeps (experiments/holdout/run_rollout_sweep.py).

mantis-trainer feedback: all-pass / all-fail groups have only step-cost "hair"
variance → ±1 noise advantages. A group is GRPO-usable only with mixed outcomes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/holdout"))

_SPEC = importlib.util.spec_from_file_location(
    "run_rollout_sweep", ROOT / "experiments/holdout/run_rollout_sweep.py"
)
rrs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rrs)  # type: ignore[union-attr]


def _r(passed):
    return {"oracle_passed": passed, "reward": 1.0 if passed else 0.0}


def test_mixed_group_is_grpo_usable():
    g = rrs._classify_group([_r(True), _r(False), _r(True)])
    assert g["grpo_usable"] is True
    assert g["distinct_outcomes"] == 2
    assert "mixed outcomes" in g["reason"]


def test_all_pass_no_shaping_is_degenerate():
    g = rrs._classify_group([_r(True), _r(True), _r(True)])
    assert g["grpo_usable"] is False
    assert "all pass" in g["reason"]


def test_all_pass_with_shaped_variance_is_usable():
    # #906 process/progress shaping: all-pass siblings vary by effort →
    # meaningful Augur episode_return spread → GRPO-usable.
    shaped = {"r0": 28.9, "r1": 28.2, "r2": 27.6}  # std ≈ 0.53 ≥ 0.05
    g = rrs._classify_group([_r(True), _r(True), _r(True)], shaped)
    assert g["grpo_usable"] is True
    assert g["shaped_episode_return_std"] >= rrs._MEANINGFUL_STD
    assert "shaped reward variance" in g["reason"]


def test_all_pass_with_hair_variance_stays_degenerate():
    # Step-cost "hair" (the original noise complaint) is below threshold.
    hair = {"r0": 18.8087, "r1": 18.8087, "r2": 18.8091}  # std ≈ 0.0002
    g = rrs._classify_group([_r(True), _r(True), _r(True)], hair)
    assert g["grpo_usable"] is False
    assert g["shaped_episode_return_std"] < rrs._MEANINGFUL_STD


def test_all_fail_group_is_degenerate():
    g = rrs._classify_group([_r(False), _r(False)])
    assert g["grpo_usable"] is False
    assert "all fail" in g["reason"]
    assert g["n_pass"] == 0


def test_single_sibling_not_usable():
    g = rrs._classify_group([_r(True)])
    assert g["grpo_usable"] is False
