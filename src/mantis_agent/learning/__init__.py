"""The Learning Allocator — budget-constrained substrate selection.

An adaptation of *"The Learning Allocator: Continual Agent Improvement as
Budget-Constrained Substrate Selection"* to the Mantis CUA testbed. See
``experiments/learning_allocator/PLAN.md`` for the full design.

The package has three parts, added across the phased plan:

    substrates/   the uniform action space (the ladder S0→S3)   — P0/P2/P3/P4
    reward.py     dual reward channels (oracle vs proxy)         — P0
    allocator.py  the myopic contextual-bandit allocator         — P2

Today only the substrate interface (``substrates/base.py``) is present.
"""

from .substrates.base import (
    Durability,
    LearningSubstrate,
    SubstrateContext,
    SubstrateResult,
)

__all__ = [
    "Durability",
    "LearningSubstrate",
    "SubstrateContext",
    "SubstrateResult",
]
