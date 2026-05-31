"""Substrate adapters — the rungs of the ladder S0→S3.

Each adapter wraps an existing Mantis mechanism behind the uniform
:class:`~mantis_agent.learning.substrates.base.LearningSubstrate` protocol so
the allocator can compare them with one currency:

    retrieval.py      S0  hint_memory / grounding_cache    (LIVE)
    exemplar.py       S1  trace_exporter + replay          (replay TO BUILD)
    skill.py          S2  graph/learner + playbook         (PARTIAL)
    consolidation.py  S3  rollout_collector + distill      (NEVER RUN)

Only :mod:`base` exists today; the adapters land per the P2–P4 issues.
"""

from .base import (
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
