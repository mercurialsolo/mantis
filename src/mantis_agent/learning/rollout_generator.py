"""Rollout generation for the continuous-improvement flywheel (#894).

A source of diverse, ground-truth-graded rollouts — the upstream piece the
flywheel was missing. Today the task set is a static manifest; these generators
turn it into a stream by exploiting the sim envs' two parameterizable surfaces:

  * ``POST /__env__/reset {seed}`` — one task template × N seeds = N distinct
    env instances (synthetic data volume).
  * ``GET /__env__/oracle?task_id`` — ground-truth reward per instance.

This module is the *stable* layer: the data types + selection policies. It does
NOT execute anything and imports neither Augur nor the executor — failure-cluster
counts are passed in as plain data (read from Augur ``list_failure_clusters`` by
the caller). Execution wiring (Daytona reset → run → oracle reward) is a separate
``RolloutRunner`` adapter (P1). See ``docs/proposals/rollout-generator.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class TaskTemplate:
    """A parameterizable task. Either ``plan_text`` (free-text →
    ``PlanDecomposer.decompose_text``) or ``plan_steps`` (authored micro-plan
    step dicts) must be set. ``oracle_task_id`` ties a rollout to its grader
    (``GET /__env__/oracle?task_id=<oracle_task_id>``)."""

    template_id: str
    cluster: str
    oracle_task_id: str
    plan_text: str | None = None
    plan_steps: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if not self.plan_text and not self.plan_steps:
            raise ValueError(
                f"TaskTemplate {self.template_id!r} needs plan_text or plan_steps"
            )


@dataclass(frozen=True)
class RolloutSpec:
    """One concrete graded run to execute: a template instantiated at a
    specific env seed. ``group_id`` groups GRPO siblings (same template+seed,
    different sampling) so the trainer can read a rollout group as one unit."""

    spec_id: str
    template: TaskTemplate
    env_seed: int
    group_id: str
    sibling_index: int = 0


@runtime_checkable
class RolloutGenerator(Protocol):
    """Emits the rollouts to execute this round."""

    def generate(self) -> Iterator[RolloutSpec]: ...


def _group_id(template_id: str, seed: int) -> str:
    return f"{template_id}__seed{seed}"


def _spec_id(template_id: str, seed: int, sibling: int) -> str:
    return f"{template_id}__seed{seed}__s{sibling}"


@dataclass
class SeedSweepGenerator:
    """Volume engine: ``templates × seeds × siblings_per_instance``.

    Deterministic and exhaustive — each ``(template, seed)`` is a distinct env
    instance; ``siblings_per_instance >= 2`` yields GRPO siblings sharing a
    ``group_id`` (vary sampling/temperature downstream, not here).
    """

    templates: list[TaskTemplate]
    seeds: list[int]
    siblings_per_instance: int = 1

    def __post_init__(self) -> None:
        if self.siblings_per_instance < 1:
            raise ValueError("siblings_per_instance must be >= 1")

    def generate(self) -> Iterator[RolloutSpec]:
        for template in self.templates:
            for seed in self.seeds:
                gid = _group_id(template.template_id, seed)
                for sib in range(self.siblings_per_instance):
                    yield RolloutSpec(
                        spec_id=_spec_id(template.template_id, seed, sib),
                        template=template,
                        env_seed=seed,
                        group_id=gid,
                        sibling_index=sib,
                    )


@dataclass
class FailureBiasedGenerator:
    """Targeting engine: spend a fixed instance budget across clusters in
    proportion to their failure share, generating where the agent is weakest.

    ``cluster_failure_counts`` is plain data read by the caller from Augur
    ``list_failure_clusters`` (this module never imports Augur). Clusters with
    no template are skipped. A deterministic RNG (``rng_seed``) makes the
    generated set reproducible for eval/replay.
    """

    templates_by_cluster: dict[str, list[TaskTemplate]]
    cluster_failure_counts: dict[str, int]
    seeds: list[int]
    total_instances: int
    siblings_per_instance: int = 1
    rng_seed: int = 0

    def _budget_by_cluster(self) -> dict[str, int]:
        # Only clusters we can actually generate for.
        live = {
            c: max(0, int(n))
            for c, n in self.cluster_failure_counts.items()
            if self.templates_by_cluster.get(c)
        }
        total = sum(live.values())
        if total <= 0 or self.total_instances <= 0:
            return {}
        # Largest-remainder apportionment so the budget sums exactly.
        raw = {c: self.total_instances * n / total for c, n in live.items()}
        floor = {c: int(v) for c, v in raw.items()}
        assigned = sum(floor.values())
        remainder = self.total_instances - assigned
        # Hand out the leftover to the largest fractional parts (stable order).
        order = sorted(
            live, key=lambda c: (raw[c] - floor[c], c), reverse=True,
        )
        for c in order[:remainder]:
            floor[c] += 1
        return {c: n for c, n in floor.items() if n > 0}

    def generate(self) -> Iterator[RolloutSpec]:
        import random

        rng = random.Random(self.rng_seed)
        if not self.seeds:
            return
        for cluster, budget in sorted(self._budget_by_cluster().items()):
            templates = self.templates_by_cluster.get(cluster) or []
            if not templates:
                continue
            for i in range(budget):
                template = templates[i % len(templates)]
                seed = rng.choice(self.seeds)
                gid = _group_id(template.template_id, seed)
                for sib in range(self.siblings_per_instance):
                    yield RolloutSpec(
                        spec_id=f"{_spec_id(template.template_id, seed, sib)}__b{i}",
                        template=template,
                        env_seed=seed,
                        group_id=gid,
                        sibling_index=sib,
                    )
