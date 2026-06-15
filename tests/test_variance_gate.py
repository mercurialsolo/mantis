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
    assert g["reward_std"] > 0
    assert g["reason"] == ""


def test_all_pass_group_is_degenerate():
    g = rrs._classify_group([_r(True), _r(True), _r(True)])
    assert g["grpo_usable"] is False
    assert g["reward_std"] == 0.0
    assert "all pass" in g["reason"]


def test_all_fail_group_is_degenerate():
    g = rrs._classify_group([_r(False), _r(False)])
    assert g["grpo_usable"] is False
    assert "all fail" in g["reason"]
    assert g["n_pass"] == 0


def test_single_sibling_not_usable():
    g = rrs._classify_group([_r(True)])
    assert g["grpo_usable"] is False  # need ≥2 distinct outcomes
