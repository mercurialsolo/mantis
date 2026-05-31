"""Substrate adapters — the rungs of the ladder S0→S3.

Each adapter wraps an existing Mantis mechanism behind the uniform
:class:`~mantis_agent.learning.substrates.base.LearningSubstrate` protocol so
the allocator can compare them with one currency:

    retrieval.py      S0  hint_memory overlay              (LIVE)
    exemplar.py       S1  trace_exporter + replay          (LIVE)
    skill.py          S2  graph/learner + playbook         (PARTIAL)
    consolidation.py  S3  rollout_collector + distill      (NEVER RUN)

S0 and S1 land in the P2 PR; S2/S3 land per the P3–P4 issues.
"""

from .base import (
    Durability,
    LearningSubstrate,
    SubstrateContext,
    SubstrateResult,
)
from .exemplar import ExemplarSubstrate
from .retrieval import RetrievalSubstrate

__all__ = [
    "Durability",
    "LearningSubstrate",
    "SubstrateContext",
    "SubstrateResult",
    "RetrievalSubstrate",
    "ExemplarSubstrate",
]
