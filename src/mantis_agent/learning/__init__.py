"""The Learning Allocator — budget-constrained substrate selection.

An adaptation of *"The Learning Allocator: Continual Agent Improvement as
Budget-Constrained Substrate Selection"* to the Mantis CUA testbed. See
``experiments/learning_allocator/PLAN.md`` for the full design.

The package has these parts, added across the phased plan:

    substrates/   the uniform action space (the ladder S0→S3)   — P0/P2/P3/P4
    reward.py     dual reward channels (oracle vs proxy)         — P0  (live)
    eval.py       the heterogeneous failure-cluster eval         — P0  (live)
    allocator.py  the myopic contextual-bandit allocator         — P2

The substrate interface, reward channels, and eval scaffold are present; the
concrete rungs and the allocator land in the P2–P4 PRs.
"""

from .eval import EvalManifest, EvalTask, load_manifest
from .reward import (
    DEFAULT_LAMBDA,
    RewardRecord,
    compute_reward,
    reward_from_run,
)
from .substrates.base import (
    Durability,
    LearningSubstrate,
    SubstrateContext,
    SubstrateResult,
)

__all__ = [
    # substrates
    "Durability",
    "LearningSubstrate",
    "SubstrateContext",
    "SubstrateResult",
    # reward
    "DEFAULT_LAMBDA",
    "RewardRecord",
    "compute_reward",
    "reward_from_run",
    # eval
    "EvalManifest",
    "EvalTask",
    "load_manifest",
]
