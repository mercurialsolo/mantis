"""Heterogeneous failure-cluster eval — engineered substrate heterogeneity.

The allocator-beats-any-fixed-substrate claim is only meaningful if the best
substrate genuinely *varies* across tasks. We don't hope for that — we engineer
it, by binning tasks into three clusters (PLAN §3):

    knowledge   — the answer is an indexable fact     → S0 (retrieval) should win
    capability  — needs a multi-step macro            → S2 (skill synth) should win
    policy      — recurring mis-handling under drift   → S1 early, S3 once frequent

Each task names a real oracle ``task_id`` and a seeded env. The sealed/visible
split (the overfitting check, PLAN §4) is by **seed**: the same task graded
against a different seeded DB is a held-out instance. ``runnable`` filters to
tasks whose oracle exists today — the manifest also carries roadmap entries
(``status="needs_oracle"``) so the structure is honest about what is and isn't
measurable yet.

The manifest data lives next to the experiment, not in the package:
``experiments/learning_allocator/eval/clusters.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CLUSTERS = ("knowledge", "capability", "policy")
SPLITS = ("visible", "sealed")
STATUSES = ("ready", "needs_oracle")

# experiments/learning_allocator/eval/clusters.json, relative to the repo root
# (this file is src/mantis_agent/learning/eval.py → up 4 to the repo root).
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "learning_allocator"
    / "eval"
    / "clusters.json"
)


@dataclass(frozen=True)
class EvalTask:
    """One row of the failure-cluster eval.

    ``task_id`` is the oracle id (``None`` until an oracle is built).
    ``expects_substrate`` is the *hypothesis* — which rung should win this
    cluster — not a constraint; the experiment is what confirms or falsifies it.
    """

    name: str
    cluster: str
    split: str
    seed: int
    env: str = "mantis_boattrader"
    task_id: str | None = None
    plan: str | None = None
    expects_substrate: str = ""
    status: str = "needs_oracle"
    notes: str = ""

    @property
    def runnable(self) -> bool:
        """True when this task can actually be graded today."""
        return self.status == "ready" and bool(self.task_id)


@dataclass
class EvalManifest:
    """The set of failure-cluster tasks + convenience views over it."""

    env: str
    tasks: list[EvalTask] = field(default_factory=list)
    description: str = ""

    def by_cluster(self, cluster: str) -> list[EvalTask]:
        return [t for t in self.tasks if t.cluster == cluster]

    def by_split(self, split: str) -> list[EvalTask]:
        return [t for t in self.tasks if t.split == split]

    def visible(self) -> list[EvalTask]:
        return self.by_split("visible")

    def sealed(self) -> list[EvalTask]:
        return self.by_split("sealed")

    def runnable(self) -> list[EvalTask]:
        """Tasks whose oracle exists — the subset measurable right now."""
        return [t for t in self.tasks if t.runnable]

    def clusters_covered(self) -> set[str]:
        """Clusters that have at least one runnable task."""
        return {t.cluster for t in self.runnable()}


def _validate(task: EvalTask) -> None:
    if task.cluster not in CLUSTERS:
        raise ValueError(
            f"task {task.name!r}: unknown cluster {task.cluster!r} "
            f"(expected one of {CLUSTERS})"
        )
    if task.split not in SPLITS:
        raise ValueError(
            f"task {task.name!r}: unknown split {task.split!r} "
            f"(expected one of {SPLITS})"
        )
    if task.status not in STATUSES:
        raise ValueError(
            f"task {task.name!r}: unknown status {task.status!r} "
            f"(expected one of {STATUSES})"
        )
    if task.status == "ready" and not task.task_id:
        raise ValueError(
            f"task {task.name!r}: status 'ready' but no task_id — a runnable "
            f"task must name an oracle"
        )


def load_manifest(path: str | Path | None = None) -> EvalManifest:
    """Load + validate the failure-cluster manifest from JSON.

    Fails fast with a descriptive error on an unknown cluster/split/status —
    a typo in the manifest should not silently drop a task from the eval.
    """
    p = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    raw: dict[str, Any] = json.loads(p.read_text())

    tasks: list[EvalTask] = []
    for entry in raw.get("tasks", []):
        task = EvalTask(
            name=entry["name"],
            cluster=entry["cluster"],
            split=entry["split"],
            seed=int(entry["seed"]),
            env=entry.get("env", raw.get("env", "mantis_boattrader")),
            task_id=entry.get("task_id"),
            plan=entry.get("plan"),
            expects_substrate=entry.get("expects_substrate", ""),
            status=entry.get("status", "needs_oracle"),
            notes=entry.get("notes", ""),
        )
        _validate(task)
        tasks.append(task)

    return EvalManifest(
        env=raw.get("env", "mantis_boattrader"),
        tasks=tasks,
        description=raw.get("description", ""),
    )


__all__ = [
    "CLUSTERS",
    "SPLITS",
    "STATUSES",
    "DEFAULT_MANIFEST_PATH",
    "EvalTask",
    "EvalManifest",
    "load_manifest",
]
