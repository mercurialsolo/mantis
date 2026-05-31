"""Phase-2 orchestrator — the bandit control loop that produces the data.

Ties the four Phase-0/2 pieces into one budget-constrained run:

    eval manifest  →  allocator.allocate  →  run the plan  →  dual reward
         ↑                                                        │
         └──────────────── allocator.observe ←───────────────────┘

For each runnable eval task it asks the :class:`MyopicAllocator` to pick a
substrate, applies it, executes the plan, scores the run through both
reward channels (oracle + proxy + cost via
:func:`~mantis_agent.learning.reward.reward_from_run`), and folds the
realised reward back into the bandit. A single hard budget ``B`` is
decremented by each run's realised dollar cost — the paper's budget
constraint — and scheduling stops when it runs dry.

Two I/O seams are injected so the whole loop is unit-testable with **no
env boots and no spend**:

* ``run_fn(task, plan, substrate_result) -> run_result`` — executes the
  (possibly substrate-modified) plan and returns the result dict the
  reward channels read (``dynamic_verification_summary.verdict`` +
  ``costs.total``). In production this submits to the Modal CUA server
  against the Daytona boattrader env; in tests it returns a canned dict.
* ``plan_loader(task) -> plan`` — returns the in-process plan object S0
  overlays grounding hints onto (and that ``run_fn`` submits). Optional:
  without it S0 simply finds nothing to overlay and S1 still runs.
* ``reward_fn(*, env_url, admin_token, task_id, run_result, lam)`` — scores
  a finished run into a :class:`~mantis_agent.learning.reward.RewardRecord`.
  Defaults to the live :func:`~mantis_agent.learning.reward.reward_from_run`
  (which calls the env's oracle over HTTP). Injecting it lets the experiment
  runner drive the loop fully offline — outcomes baked into ``run_result``
  rather than read from a live env — so Table 1 / Fig 1 can be produced and
  the renderers exercised with no env boots and no spend.

The experiment wiring that supplies the real Modal-submit ``run_fn`` lives
under ``experiments/learning_allocator/`` — that is the only spend
boundary; this module never calls Modal or Daytona itself.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .allocator import MyopicAllocator
from .eval import EvalTask
from .reward import DEFAULT_LAMBDA, RewardRecord, reward_from_run
from .substrates.base import SubstrateContext, SubstrateResult

logger = logging.getLogger(__name__)

# Execute a plan and return the run_result dict reward_from_run consumes.
RunFn = Callable[[EvalTask, Any, SubstrateResult], dict]
# Load the in-process plan object for a task (the thing S0 overlays onto).
PlanLoader = Callable[[EvalTask], Any]
# Map a task to the plan_signature that keys the hint / exemplar stores.
PlanSigFn = Callable[[EvalTask], str]
# Score a finished run into a RewardRecord. Defaults to reward_from_run (live
# oracle over HTTP); injectable so the loop can run fully offline.
RewardFn = Callable[..., RewardRecord]


@dataclass
class TaskOutcome:
    """One task's trip through the loop — the row behind every data point."""

    task_name: str
    task_id: str
    cluster: str
    substrate: str | None = None
    reward: float | None = None
    dollars: float | None = None
    reward_record: RewardRecord | None = None
    substrate_result: SubstrateResult | None = None
    skipped: bool = False
    note: str = ""


@dataclass
class Phase2Report:
    """The eval's collected output — Table 1 / Fig 1 + attribution tallies."""

    outcomes: list[TaskOutcome] = field(default_factory=list)
    value_table: dict[tuple[str, str], tuple[float, int]] = field(default_factory=dict)
    selection_distribution: dict[str, float] = field(default_factory=dict)
    budget_start: float = 0.0
    budget_spent: float = 0.0
    n_runs: int = 0
    n_skipped: int = 0
    false_passes: int = 0
    false_fails: int = 0

    @property
    def budget_remaining(self) -> float:
        return self.budget_start - self.budget_spent


class Phase2Orchestrator:
    """Drives the allocator over the eval under a hard dollar budget."""

    def __init__(
        self,
        *,
        allocator: MyopicAllocator,
        run_fn: RunFn,
        env_url: str,
        admin_token: str,
        budget: float,
        plan_loader: PlanLoader | None = None,
        plan_signature_fn: PlanSigFn | None = None,
        reward_fn: RewardFn | None = None,
        start_url: str = "",
        lam: float = DEFAULT_LAMBDA,
    ) -> None:
        self.allocator = allocator
        self.run_fn = run_fn
        self.env_url = env_url
        self.admin_token = admin_token
        self.budget_start = float(budget)
        self.budget_remaining = float(budget)
        self.lam = float(lam)
        self.start_url = start_url
        self._plan_loader = plan_loader
        self._plan_sig = plan_signature_fn or _default_plan_signature
        self._reward_fn = reward_fn
        self.outcomes: list[TaskOutcome] = []

    # ── per-task ────────────────────────────────────────────────────────

    def _context(self, task: EvalTask, plan: Any) -> SubstrateContext:
        extras: dict[str, Any] = {
            "plan_signature": self._plan_sig(task),
            "start_url": self.start_url,
        }
        if plan is not None:
            extras["plan"] = plan
        return SubstrateContext(
            task_id=task.task_id or task.name,
            cluster=task.cluster,
            budget_remaining=self.budget_remaining,
            extras=extras,
        )

    def run_task(self, task: EvalTask) -> TaskOutcome:
        """Allocate → run → reward → observe for one task.

        Skips (cheaply, no spend) when the task isn't runnable, the budget
        is exhausted, or the allocator finds nothing affordable.
        """
        if not task.runnable:
            return self._skip(task, "task not runnable (no oracle / not ready)")
        if self.budget_remaining <= 0.0:
            return self._skip(task, "budget exhausted")

        plan = self._plan_loader(task) if self._plan_loader else None
        context = self._context(task, plan)

        allocated = self.allocator.allocate(context)
        if allocated is None:
            return self._skip(task, "no affordable substrate")
        choice, substrate_result = allocated

        run_result = self.run_fn(task, plan, substrate_result)

        # Default resolves the module global at call time so existing tests
        # that monkeypatch ``orchestrator.reward_from_run`` keep working.
        reward_fn = self._reward_fn or reward_from_run
        record = reward_fn(
            env_url=self.env_url,
            admin_token=self.admin_token,
            task_id=task.task_id or task.name,
            run_result=run_result,
            lam=self.lam,
        )

        # Realised cost = the run's dollars plus whatever the substrate
        # spent applying itself (0 for S0/S1, nonzero once S2/S3 land).
        spent = float(record.dollars) + float(substrate_result.dollars_spent)
        self.budget_remaining -= spent

        self.allocator.observe(
            context, choice.name, record.reward, result=substrate_result,
        )

        outcome = TaskOutcome(
            task_name=task.name,
            task_id=task.task_id or task.name,
            cluster=task.cluster,
            substrate=choice.name,
            reward=record.reward,
            dollars=record.dollars,
            reward_record=record,
            substrate_result=substrate_result,
        )
        self.outcomes.append(outcome)
        logger.info(
            "[phase2] %s cluster=%s substrate=%s reward=%.3f $=%.3f budget_left=%.3f",
            task.name, task.cluster, choice.name, record.reward,
            record.dollars, self.budget_remaining,
        )
        return outcome

    def _skip(self, task: EvalTask, note: str) -> TaskOutcome:
        outcome = TaskOutcome(
            task_name=task.name,
            task_id=task.task_id or task.name,
            cluster=task.cluster,
            skipped=True,
            note=note,
        )
        self.outcomes.append(outcome)
        return outcome

    # ── driver ──────────────────────────────────────────────────────────

    def run(
        self, tasks: Iterable[EvalTask], *, rounds: int = 1,
    ) -> Phase2Report:
        """Run ``rounds`` passes over ``tasks`` and return the report.

        Repeating passes is how the bandit accumulates per-cluster
        evidence: round 1 is mostly warm-up exploration, later rounds let
        exploitation settle so Fig 2's selection drift becomes visible.
        """
        task_list = list(tasks)
        for _ in range(max(1, int(rounds))):
            for task in task_list:
                self.run_task(task)
        return self.report()

    def report(self) -> Phase2Report:
        graded = [o for o in self.outcomes if not o.skipped and o.reward_record]
        return Phase2Report(
            outcomes=list(self.outcomes),
            value_table=self.allocator.value_table(),
            selection_distribution=self.allocator.selection_distribution(),
            budget_start=self.budget_start,
            budget_spent=self.budget_start - self.budget_remaining,
            n_runs=len(graded),
            n_skipped=sum(1 for o in self.outcomes if o.skipped),
            false_passes=sum(1 for o in graded if o.reward_record.false_pass),
            false_fails=sum(1 for o in graded if o.reward_record.false_fail),
        )


def _default_plan_signature(task: EvalTask) -> str:
    """Fallback plan signature: the task's plan name, else its id / name."""
    return task.plan or task.task_id or task.name


__all__ = [
    "RunFn",
    "PlanLoader",
    "PlanSigFn",
    "RewardFn",
    "TaskOutcome",
    "Phase2Report",
    "Phase2Orchestrator",
]
