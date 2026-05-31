"""Phase-2 experiment runner — turn the bandit loop into Table 1 / Fig 1.

The orchestrator (:mod:`mantis_agent.learning.orchestrator`) drives *one*
policy through the eval. The paper's headline result is a *comparison*: the
Learning Allocator against every fixed-substrate baseline under one budget
(PLAN §4). This module is that comparison harness — it builds the baseline
and learner policies, runs each over the same eval tasks, and renders the
two artifacts #741 asks for:

* **Table 1** — per policy: sealed oracle score, score-per-dollar, and the
  visible−sealed overfitting gap.
* **Fig 1** — per policy: running oracle score vs cumulative dollars, so the
  allocator's "more competence per dollar" curve is plottable above the
  fixed rungs.

Policies (PLAN §4 baselines), each a :class:`MyopicAllocator` over a chosen
slice of the ladder:

* ``frozen``   — base agent, no learning (a single no-op rung).
* ``S0_only``  — retrieval/hint overlay only.
* ``S1_only``  — exemplar replay only.
* ``allocator``— the learner, choosing across S0+S1 (our policy).

``oracle_allocator`` (the upper-reference / headroom) is **derived** post
hoc — per task, the best oracle score any fixed policy achieved — not run as
its own policy.

Two run modes, one code path:

* **Offline** (this PR, no spend) — outcomes are baked into the
  ``run_result`` dict and scored by :func:`offline_reward_fn`. Used by the
  unit tests and the ``--out`` demo to exercise the whole pipeline and the
  renderers with no env boots and no Modal/Daytona calls.
* **Live** (the spend boundary, a follow-up) — a ``run_fn`` that submits each
  plan to the Modal CUA server against the Daytona boattrader env (no
  proxies) and a ``reward_fn`` left at its default
  (:func:`~mantis_agent.learning.reward.reward_from_run`, which calls the
  env's ``/__env__/oracle``). That adapter lands when we cross the spend
  line; this module deliberately ships only the no-spend half.

Run the offline demo (synthetic — validates wiring, NOT agent data)::

    uv run python -m experiments.learning_allocator.runner --out /tmp/la
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

from mantis_agent.learning.allocator import MyopicAllocator
from mantis_agent.learning.eval import EvalTask, load_manifest
from mantis_agent.learning.orchestrator import (
    Phase2Orchestrator,
    Phase2Report,
    TaskOutcome,
)
from mantis_agent.learning.reward import (
    DEFAULT_LAMBDA,
    RewardRecord,
    compute_reward,
    cost_channel,
    proxy_channel,
)
from mantis_agent.learning.substrates.base import (
    Durability,
    SubstrateContext,
    SubstrateResult,
)
from mantis_agent.learning.substrates.exemplar import ExemplarSubstrate
from mantis_agent.learning.substrates.retrieval import RetrievalSubstrate

# Policy → the substrate roles it may choose among. "frozen" is the base
# agent with no learning, modelled as a single no-op rung.
FROZEN = "frozen"
S0 = "S0_retrieval"
S1 = "S1_exemplar"

POLICY_SUBSTRATES: dict[str, tuple[str, ...]] = {
    "frozen": (FROZEN,),
    "S0_only": (S0,),
    "S1_only": (S1,),
    "allocator": (S0, S1),
}

# The fixed-substrate policies whose per-task best defines the oracle-allocator
# upper bound. The learner ("allocator") is excluded — it is what the upper
# bound is a reference *for*.
FIXED_POLICIES: tuple[str, ...] = ("frozen", "S0_only", "S1_only")


# ── frozen rung ─────────────────────────────────────────────────────────────


class NoOpSubstrate:
    """The ``frozen`` baseline: a rung that never acts, at zero cost.

    Lets the base (non-learning) agent ride the exact same loop as the
    learning policies — the allocator "chooses" it, ``apply`` overlays
    nothing, and the run reflects the frozen agent's own competence. Without
    this, ``frozen`` would need a separate code path; with it, every policy
    is just a different substrate slice.
    """

    durability = Durability.EPHEMERAL

    def __init__(self, *, name: str = FROZEN) -> None:
        self.name = name

    def cost_estimate(self, context: SubstrateContext) -> float:  # noqa: ARG002
        return 0.0

    def apply(self, context: SubstrateContext) -> SubstrateResult:  # noqa: ARG002
        return SubstrateResult(
            substrate=self.name,
            applied=False,
            dollars_spent=0.0,
            durability=self.durability,
            notes="frozen baseline — no learning applied",
        )

    def observe(
        self, context: SubstrateContext, result: SubstrateResult, reward: float,
    ) -> None:  # noqa: ARG002
        return None


# ── offline reward channel ──────────────────────────────────────────────────


def offline_reward_fn(
    *,
    env_url: str,  # noqa: ARG001 — kept for reward_fn signature parity
    admin_token: str,  # noqa: ARG001
    task_id: str,
    run_result: dict[str, Any],
    lam: float = DEFAULT_LAMBDA,
) -> RewardRecord:
    """Score a run from outcomes baked into ``run_result`` — no oracle call.

    The live channel (:func:`reward_from_run`) hits the env's oracle over
    HTTP. Offline, the ground-truth verdict is carried *in* the run_result
    (``oracle_score`` / ``oracle_passed``); the proxy verdict and dollar cost
    are read from the same dict by the shared channel readers. Same arithmetic
    (:func:`compute_reward`), no network — so the loop runs with zero spend.
    """
    return compute_reward(
        task_id=task_id,
        oracle_score=float(run_result.get("oracle_score", 0.0) or 0.0),
        oracle_passed=bool(run_result.get("oracle_passed", False)),
        proxy_verdict=proxy_channel(run_result),
        dollars=cost_channel(run_result),
        lam=lam,
    )


# ── policy assembly ─────────────────────────────────────────────────────────


def build_substrate(role: str, *, hint_store: Any = None, trace_dir: Any = None):
    """Instantiate one substrate by role name."""
    if role == FROZEN:
        return NoOpSubstrate()
    if role == S0:
        return RetrievalSubstrate(hint_store)
    if role == S1:
        # trace_dir may be None in offline runs — ExemplarSubstrate then finds
        # no corpus and simply applies nothing, which is the correct frozen-ish
        # behaviour for a cold start.
        return ExemplarSubstrate(trace_dir or Path("/nonexistent-trace-dir"))
    raise ValueError(f"unknown substrate role: {role!r}")


def build_policy_allocator(
    policy: str,
    *,
    hint_store: Any = None,
    trace_dir: Any = None,
    epsilon: float = 0.1,
    rng: Random | None = None,
    lam: float = DEFAULT_LAMBDA,
) -> MyopicAllocator:
    """Build the :class:`MyopicAllocator` for a named policy."""
    roles = POLICY_SUBSTRATES.get(policy)
    if roles is None:
        raise ValueError(f"unknown policy: {policy!r}")
    substrates = [
        build_substrate(r, hint_store=hint_store, trace_dir=trace_dir)
        for r in roles
    ]
    return MyopicAllocator(substrates, lam=lam, epsilon=epsilon, rng=rng)


@dataclass
class ExperimentResult:
    """Everything the renderers need: one report per policy + split metadata."""

    reports: dict[str, Phase2Report] = field(default_factory=dict)
    split_of: dict[str, str] = field(default_factory=dict)
    seed_of: dict[str, int] = field(default_factory=dict)
    cluster_of: dict[str, str] = field(default_factory=dict)
    budget: float = 0.0
    rounds: int = 1


def run_experiment(
    *,
    tasks: list[EvalTask],
    run_fn,
    reward_fn=None,
    env_url: str = "",
    admin_token: str = "",
    budget: float = 50.0,
    rounds: int = 1,
    epsilon: float = 0.1,
    seed: int = 0,
    policies: tuple[str, ...] = tuple(POLICY_SUBSTRATES),
    hint_store: Any = None,
    trace_dir: Any = None,
    start_url: str = "",
) -> ExperimentResult:
    """Run every policy over ``tasks`` under an identical budget and collect.

    Each policy gets its own allocator and its own fresh budget ``B`` (the
    paper runs all learning policies under the *same* budget, separately).
    ``run_fn`` / ``reward_fn`` are injected so this is spend-agnostic: offline
    tests pass canned ones, the live wiring passes the Modal-submit pair.
    """
    result = ExperimentResult(budget=budget, rounds=rounds)
    for t in tasks:
        result.split_of[t.name] = t.split
        result.seed_of[t.name] = t.seed
        result.cluster_of[t.name] = t.cluster

    for policy in policies:
        # Per-policy RNG keeps each policy deterministic yet decorrelated.
        # Seed with a string — Random hashes str/bytes stably (sha512), unlike
        # the builtin hash() which is per-process randomized.
        rng = Random(f"{seed}:{policy}")
        allocator = build_policy_allocator(
            policy, hint_store=hint_store, trace_dir=trace_dir,
            epsilon=epsilon, rng=rng,
        )
        orch = Phase2Orchestrator(
            allocator=allocator,
            run_fn=run_fn,
            env_url=env_url,
            admin_token=admin_token,
            budget=budget,
            reward_fn=reward_fn,
            start_url=start_url,
        )
        result.reports[policy] = orch.run(tasks, rounds=rounds)
    return result


# ── analysis: Table 1 + the oracle-allocator upper bound ────────────────────


def _graded(report: Phase2Report) -> list[TaskOutcome]:
    return [o for o in report.outcomes if not o.skipped and o.reward_record]


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


@dataclass
class Table1Row:
    """One policy's row in Table 1 (PLAN §5)."""

    policy: str
    sealed_score: float       # mean oracle score on the sealed split
    visible_score: float      # mean oracle score on the visible split
    visible_minus_sealed: float  # the overfitting gap
    score_per_dollar: float   # total oracle score / total dollars
    total_dollars: float
    n_runs: int


def build_table1(result: ExperimentResult) -> list[Table1Row]:
    """Per-policy Table 1 rows, plus the derived ``oracle_allocator`` row."""
    rows = [_policy_row(name, rep, result) for name, rep in result.reports.items()]
    oracle_row = _oracle_allocator_row(result)
    if oracle_row is not None:
        rows.append(oracle_row)
    return rows


def _policy_row(
    policy: str, report: Phase2Report, result: ExperimentResult,
) -> Table1Row:
    visible, sealed = [], []
    total_score = 0.0
    total_dollars = 0.0
    for o in _graded(report):
        score = o.reward_record.oracle_score
        total_score += score
        total_dollars += float(o.dollars or 0.0)
        if result.split_of.get(o.task_name) == "sealed":
            sealed.append(score)
        else:
            visible.append(score)
    sealed_mean = _mean(sealed)
    visible_mean = _mean(visible)
    return Table1Row(
        policy=policy,
        sealed_score=round(sealed_mean, 4),
        visible_score=round(visible_mean, 4),
        visible_minus_sealed=round(visible_mean - sealed_mean, 4),
        score_per_dollar=round(_safe_div(total_score, total_dollars), 4),
        total_dollars=round(total_dollars, 4),
        n_runs=len(_graded(report)),
    )


def _oracle_allocator_row(result: ExperimentResult) -> Table1Row | None:
    """Upper bound: per task, the best (oracle_score, dollars) any fixed
    policy got. The headroom the learner is chasing."""
    # task_name -> (best_score, dollars_at_best)
    best: dict[str, tuple[float, float]] = {}
    for policy in FIXED_POLICIES:
        report = result.reports.get(policy)
        if not report:
            continue
        for o in _graded(report):
            score = o.reward_record.oracle_score
            dollars = float(o.dollars or 0.0)
            cur = best.get(o.task_name)
            # Prefer higher score; on a tie prefer the cheaper run.
            if cur is None or score > cur[0] or (score == cur[0] and dollars < cur[1]):
                best[o.task_name] = (score, dollars)
    if not best:
        return None

    visible, sealed = [], []
    total_score = 0.0
    total_dollars = 0.0
    for name, (score, dollars) in best.items():
        total_score += score
        total_dollars += dollars
        if result.split_of.get(name) == "sealed":
            sealed.append(score)
        else:
            visible.append(score)
    sealed_mean = _mean(sealed)
    visible_mean = _mean(visible)
    return Table1Row(
        policy="oracle_allocator",
        sealed_score=round(sealed_mean, 4),
        visible_score=round(visible_mean, 4),
        visible_minus_sealed=round(visible_mean - sealed_mean, 4),
        score_per_dollar=round(_safe_div(total_score, total_dollars), 4),
        total_dollars=round(total_dollars, 4),
        n_runs=len(best),
    )


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ── analysis: Fig 1 series ──────────────────────────────────────────────────


@dataclass
class Fig1Point:
    index: int
    cum_dollars: float
    running_mean_oracle: float


def fig1_series(report: Phase2Report) -> list[Fig1Point]:
    """Running-mean oracle score vs cumulative dollars, in decision order.

    The x-axis is *cumulative dollars* (the paper's Fig 1, not steps); the
    y-axis is the agent's competence so far (running mean oracle score). A
    learner that buys competence efficiently rises left of the fixed rungs.
    """
    points: list[Fig1Point] = []
    cum_dollars = 0.0
    score_sum = 0.0
    n = 0
    for o in _graded(report):
        n += 1
        cum_dollars += float(o.dollars or 0.0)
        score_sum += o.reward_record.oracle_score
        points.append(Fig1Point(
            index=n,
            cum_dollars=round(cum_dollars, 4),
            running_mean_oracle=round(score_sum / n, 4),
        ))
    return points


# ── writers ─────────────────────────────────────────────────────────────────

_SYNTHETIC_BANNER = (
    "# SOURCE=offline-demo — SYNTHETIC outcomes for pipeline validation. "
    "NOT agent data. Real results require the live Modal/Daytona run."
)


def write_results(
    result: ExperimentResult, out_dir: str | Path, *, banner: str = _SYNTHETIC_BANNER,
) -> dict[str, Path]:
    """Write results.tsv + table1.tsv + fig1.tsv under ``out_dir``.

    ``banner`` is stamped as the first commented line of every file so an
    offline-demo run can never be mistaken for real agent data.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": out / "results.tsv",
        "table1": out / "table1.tsv",
        "fig1": out / "fig1.tsv",
    }
    _write_outcomes_tsv(paths["results"], result, banner)
    _write_table1_tsv(paths["table1"], build_table1(result), banner)
    _write_fig1_tsv(paths["fig1"], result, banner)
    return paths


def _write_outcomes_tsv(path: Path, result: ExperimentResult, banner: str) -> None:
    cols = [
        "policy", "task_name", "task_id", "cluster", "split", "seed",
        "substrate", "oracle_score", "oracle_passed", "proxy_verdict",
        "dollars", "reward", "false_pass", "false_fail", "skipped", "note",
    ]
    with path.open("w", newline="") as fh:
        fh.write(banner + "\n")
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for policy, report in result.reports.items():
            for o in report.outcomes:
                rr = o.reward_record
                w.writerow([
                    policy, o.task_name, o.task_id, o.cluster,
                    result.split_of.get(o.task_name, ""),
                    result.seed_of.get(o.task_name, ""),
                    o.substrate or "",
                    rr.oracle_score if rr else "",
                    rr.oracle_passed if rr else "",
                    rr.proxy_verdict if rr else "",
                    o.dollars if o.dollars is not None else "",
                    o.reward if o.reward is not None else "",
                    rr.false_pass if rr else "",
                    rr.false_fail if rr else "",
                    o.skipped, o.note,
                ])


def _write_table1_tsv(path: Path, rows: list[Table1Row], banner: str) -> None:
    cols = [
        "policy", "sealed_score", "visible_score", "visible_minus_sealed",
        "score_per_dollar", "total_dollars", "n_runs",
    ]
    with path.open("w", newline="") as fh:
        fh.write(banner + "\n")
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r.policy, r.sealed_score, r.visible_score,
                r.visible_minus_sealed, r.score_per_dollar,
                r.total_dollars, r.n_runs,
            ])


def _write_fig1_tsv(path: Path, result: ExperimentResult, banner: str) -> None:
    with path.open("w", newline="") as fh:
        fh.write(banner + "\n")
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["policy", "index", "cum_dollars", "running_mean_oracle"])
        for policy, report in result.reports.items():
            for p in fig1_series(report):
                w.writerow([policy, p.index, p.cum_dollars, p.running_mean_oracle])


def format_table1(rows: list[Table1Row]) -> str:
    """Render Table 1 as a fixed-width text table for stdout."""
    header = (
        f"{'policy':<18} {'sealed':>8} {'visible':>8} {'vis−seal':>9} "
        f"{'score/$':>9} {'$total':>8} {'n':>4}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.policy:<18} {r.sealed_score:>8.4f} {r.visible_score:>8.4f} "
            f"{r.visible_minus_sealed:>9.4f} {r.score_per_dollar:>9.4f} "
            f"{r.total_dollars:>8.4f} {r.n_runs:>4d}"
        )
    return "\n".join(lines)


# ── offline demo (synthetic — wiring validation only) ───────────────────────

# Synthetic per-(cluster, substrate-role) outcomes encoding the PLAN §3
# heterogeneity: knowledge→S0 wins, capability/policy→S1 wins, frozen loses
# everywhere. ONLY for validating the harness end-to-end with no spend; these
# numbers are invented, not measured.
_DEMO_PROFILE: dict[tuple[str, str], dict[str, Any]] = {
    ("knowledge", FROZEN): {"oracle_score": 0.0, "oracle_passed": False, "verdict": "fail", "cost": 0.40},
    ("knowledge", S0):     {"oracle_score": 0.9, "oracle_passed": True,  "verdict": "pass", "cost": 0.42},
    ("knowledge", S1):     {"oracle_score": 0.4, "oracle_passed": False, "verdict": "partial", "cost": 0.45},
    ("capability", FROZEN): {"oracle_score": 0.1, "oracle_passed": False, "verdict": "fail", "cost": 0.50},
    ("capability", S0):     {"oracle_score": 0.4, "oracle_passed": False, "verdict": "partial", "cost": 0.52},
    ("capability", S1):     {"oracle_score": 0.8, "oracle_passed": True,  "verdict": "pass", "cost": 0.55},
    ("policy", FROZEN): {"oracle_score": 0.0, "oracle_passed": False, "verdict": "fail", "cost": 0.45},
    ("policy", S0):     {"oracle_score": 0.3, "oracle_passed": False, "verdict": "fail", "cost": 0.46},
    ("policy", S1):     {"oracle_score": 0.85, "oracle_passed": True, "verdict": "pass", "cost": 0.50},
}


def make_demo_run_fn(cluster_of: dict[str, str]):
    """A deterministic offline ``run_fn`` for the synthetic demo.

    Looks the outcome up by the task's cluster and the substrate the
    allocator applied, returning the ``run_result`` shape the offline reward
    channel reads. Pure — no network, no spend.
    """

    def run_fn(task: EvalTask, plan: Any, result: SubstrateResult) -> dict:  # noqa: ARG001
        cluster = cluster_of.get(task.name, task.cluster)
        profile = _DEMO_PROFILE.get((cluster, result.substrate))
        if profile is None:
            profile = {"oracle_score": 0.0, "oracle_passed": False, "verdict": "fail", "cost": 0.4}
        return {
            "oracle_score": profile["oracle_score"],
            "oracle_passed": profile["oracle_passed"],
            "dynamic_verification_summary": {"verdict": profile["verdict"]},
            "costs": {"total": profile["cost"]},
        }

    return run_fn


def run_offline_demo(
    *, rounds: int = 3, budget: float = 50.0, epsilon: float = 0.1, seed: int = 0,
) -> ExperimentResult:
    """Run the whole comparison offline on synthetic outcomes (no spend)."""
    tasks = load_manifest().runnable()
    cluster_of = {t.name: t.cluster for t in tasks}
    return run_experiment(
        tasks=tasks,
        run_fn=make_demo_run_fn(cluster_of),
        reward_fn=offline_reward_fn,
        budget=budget,
        rounds=rounds,
        epsilon=epsilon,
        seed=seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase-2 Learning Allocator runner (offline demo).",
    )
    parser.add_argument("--out", default="", help="dir to write results TSVs (optional)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    print(
        "Learning Allocator — Phase-2 OFFLINE DEMO (synthetic outcomes).\n"
        "This validates the runner wiring + renderers with NO spend. Real\n"
        "Table 1 / Fig 1 require the live Modal/Daytona run (spend boundary).\n"
    )
    result = run_offline_demo(
        rounds=args.rounds, budget=args.budget,
        epsilon=args.epsilon, seed=args.seed,
    )
    print(format_table1(build_table1(result)))
    if args.out:
        paths = write_results(result, args.out)
        print("\nwrote (SYNTHETIC):")
        for label, p in paths.items():
            print(f"  {label:<8} {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
