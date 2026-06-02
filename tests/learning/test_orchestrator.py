"""Tests for the Phase-2 orchestrator (the bandit control loop).

The loop's two I/O seams — ``run_fn`` (the Modal/Daytona submit) and the
oracle call inside ``reward_from_run`` — are injected / monkeypatched, so
these run with no env boots and no spend. Most tests stub
``reward_from_run`` to drive reward + cost deterministically; one drives
the *real* reward path with only ``grade_run`` mocked, to catch wiring
bugs. One more exercises the real S0 overlay end-to-end through the loop.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mantis_agent.gym.grading import GradingResult
from mantis_agent.gym.hint_memory import (
    HintRecord,
    InMemoryHintStore,
    hint_key_for,
)
from mantis_agent.learning import orchestrator as O
from mantis_agent.learning import reward as R
from mantis_agent.learning.allocator import MyopicAllocator
from mantis_agent.learning.eval import EvalTask
from mantis_agent.learning.substrates.base import (
    Durability,
    SubstrateContext,
    SubstrateResult,
)
from mantis_agent.learning.substrates.retrieval import RetrievalSubstrate

PLAN_SIG = "sig123456789"
START_URL = "https://www.boattrader.com/boat/x/"


# ── fakes ───────────────────────────────────────────────────────────────


class FakeSub:
    def __init__(
        self, name: str, *, cost: float = 0.0,
        durability: Durability = Durability.EPHEMERAL,
    ) -> None:
        self.name = name
        self._cost = cost
        self.durability = durability
        self.observed: list[float] = []

    def cost_estimate(self, context: SubstrateContext) -> float:  # noqa: ARG002
        return self._cost

    def apply(self, context: SubstrateContext) -> SubstrateResult:  # noqa: ARG002
        return SubstrateResult(
            substrate=self.name, applied=True,
            dollars_spent=self._cost, durability=self.durability,
            delta_artifacts={"tag": self.name},
        )

    def observe(self, context, result, reward: float) -> None:  # noqa: ARG002
        self.observed.append(reward)


def _task(
    name: str = "bt01_visible", *, task_id: str = "BT01",
    cluster: str = "capability", status: str = "ready",
) -> EvalTask:
    return EvalTask(
        name=name, cluster=cluster, split="visible", seed=42,
        task_id=task_id, status=status, plan="boattrader_scrape",
    )


def _stub_reward(monkeypatch, **defaults):
    """Stub orchestrator.reward_from_run to read the run_result dict.

    run_result may carry ``reward`` / ``costs.total`` / ``false_pass`` /
    ``false_fail`` to drive the loop deterministically.
    """

    def fake(*, env_url, admin_token, task_id, run_result, lam):  # noqa: ARG001
        return R.RewardRecord(
            task_id=task_id,
            oracle_score=run_result.get("oracle_score", 0.0),
            oracle_passed=run_result.get("oracle_passed", False),
            proxy_verdict=run_result.get("verdict", "unknown"),
            proxy_score=0.0,
            dollars=run_result.get("costs", {}).get("total", defaults.get("dollars", 0.0)),
            reward=run_result.get("reward", defaults.get("reward", 0.0)),
            false_pass=run_result.get("false_pass", False),
            false_fail=run_result.get("false_fail", False),
        )

    monkeypatch.setattr(O, "reward_from_run", fake)


def _orch(allocator, run_fn, *, budget=100.0, **kw) -> O.Phase2Orchestrator:
    return O.Phase2Orchestrator(
        allocator=allocator, run_fn=run_fn,
        env_url="https://env.example", admin_token="tok",
        budget=budget, start_url=START_URL, **kw,
    )


# ── basic loop ──────────────────────────────────────────────────────────


def test_runs_task_and_records_outcome(monkeypatch) -> None:
    _stub_reward(monkeypatch)
    sub = FakeSub("S0")
    alloc = MyopicAllocator([sub], epsilon=0.0)
    calls = []

    def run_fn(task, plan, result):
        calls.append((task.name, result.substrate))
        return {"reward": 0.7, "costs": {"total": 0.2}}

    orch = _orch(alloc, run_fn)
    outcome = orch.run_task(_task())

    assert outcome.skipped is False
    assert outcome.substrate == "S0"
    assert outcome.reward == 0.7
    assert outcome.dollars == 0.2
    assert calls == [("bt01_visible", "S0")]
    # reward folded back into the bandit
    assert sub.observed == [0.7]
    assert alloc.value_of("capability", "S0") == pytest.approx(0.7)


def test_skips_non_runnable_task(monkeypatch) -> None:
    _stub_reward(monkeypatch)
    alloc = MyopicAllocator([FakeSub("S0")], epsilon=0.0)
    ran = []
    orch = _orch(alloc, lambda *a: ran.append(a) or {})

    outcome = orch.run_task(_task(status="needs_oracle", task_id=""))

    assert outcome.skipped is True
    assert "not runnable" in outcome.note
    assert ran == []  # run_fn never called


def test_no_affordable_substrate_skips(monkeypatch) -> None:
    _stub_reward(monkeypatch)
    pricey = FakeSub("S3", cost=3.0, durability=Durability.WEIGHTS)
    alloc = MyopicAllocator([pricey], epsilon=0.0)
    orch = _orch(alloc, lambda *a: {}, budget=0.5)  # >0 so the budget guard passes

    outcome = orch.run_task(_task())

    assert outcome.skipped is True
    assert "no affordable substrate" in outcome.note


# ── budget ──────────────────────────────────────────────────────────────


def test_budget_decrements_by_realised_cost(monkeypatch) -> None:
    _stub_reward(monkeypatch)
    alloc = MyopicAllocator([FakeSub("S0")], epsilon=0.0)
    orch = _orch(alloc, lambda *a: {"reward": 1.0, "costs": {"total": 0.3}}, budget=1.0)

    orch.run_task(_task())

    assert orch.budget_remaining == pytest.approx(0.7)


def test_budget_exhaustion_skips_remaining(monkeypatch) -> None:
    _stub_reward(monkeypatch)
    alloc = MyopicAllocator([FakeSub("S0")], epsilon=0.0)
    # First run costs the whole budget; the second task must be skipped.
    orch = _orch(alloc, lambda *a: {"reward": 1.0, "costs": {"total": 1.0}}, budget=1.0)

    first = orch.run_task(_task("t1"))
    second = orch.run_task(_task("t2"))

    assert first.skipped is False
    assert second.skipped is True
    assert "budget exhausted" in second.note


# ── substrate delta threading ───────────────────────────────────────────


def test_run_fn_receives_substrate_delta(monkeypatch) -> None:
    _stub_reward(monkeypatch)
    alloc = MyopicAllocator([FakeSub("S1", durability=Durability.SESSION)], epsilon=0.0)
    seen = {}

    def run_fn(task, plan, result):
        seen["delta"] = result.delta_artifacts
        return {"reward": 0.5, "costs": {"total": 0.1}}

    _orch(alloc, run_fn).run_task(_task())

    assert seen["delta"] == {"tag": "S1"}


def test_s0_overlay_fires_through_the_loop(monkeypatch) -> None:
    # End-to-end: a stored anchor lands on the plan step the loop submits.
    _stub_reward(monkeypatch)
    step = SimpleNamespace(type="click", intent="click Show More", hints=None)
    plan = SimpleNamespace(steps=[step])

    store = InMemoryHintStore()
    store.add(
        hint_key_for(PLAN_SIG, step, START_URL),
        HintRecord(anchor_text="Show More", confidence=1.0),
    )
    alloc = MyopicAllocator([RetrievalSubstrate(store)], epsilon=0.0)

    submitted = {}

    def run_fn(task, plan_arg, result):
        submitted["plan"] = plan_arg
        submitted["applied"] = result.applied
        return {"reward": 1.0, "costs": {"total": 0.2}}

    orch = _orch(
        alloc, run_fn,
        plan_loader=lambda task: plan,
        plan_signature_fn=lambda task: PLAN_SIG,
    )
    orch.run_task(_task())

    # The real overlay mutated the plan step, and run_fn got that same plan.
    assert step.hints["preferred_target_description"] == "Show More"
    assert submitted["plan"] is plan
    assert submitted["applied"] is True


# ── driver + report ─────────────────────────────────────────────────────


def test_rounds_repeats_passes(monkeypatch) -> None:
    _stub_reward(monkeypatch)
    sub = FakeSub("S0")
    alloc = MyopicAllocator([sub], epsilon=0.0)
    orch = _orch(alloc, lambda *a: {"reward": 0.5, "costs": {"total": 0.1}})

    report = orch.run([_task()], rounds=3)

    assert report.n_runs == 3
    assert alloc.count_of("capability", "S0") == 3


def test_report_aggregates_value_table_and_distribution(monkeypatch) -> None:
    _stub_reward(monkeypatch)
    s0 = FakeSub("S0")
    s1 = FakeSub("S1", durability=Durability.SESSION)
    alloc = MyopicAllocator([s0, s1], epsilon=0.0)
    orch = _orch(alloc, lambda *a: {"reward": 0.6, "costs": {"total": 0.1}})

    report = orch.run([_task()], rounds=2)

    # Warm-up tried both rungs once across the two rounds.
    assert ("capability", "S0") in report.value_table
    assert ("capability", "S1") in report.value_table
    assert sum(report.selection_distribution.values()) == pytest.approx(1.0)
    assert report.n_runs == 2
    assert report.budget_spent == pytest.approx(0.2)


def test_report_counts_false_pass_and_fail(monkeypatch) -> None:
    _stub_reward(monkeypatch)
    alloc = MyopicAllocator([FakeSub("S0")], epsilon=0.0)
    orch = _orch(alloc, lambda *a: {"reward": 0.0, "costs": {"total": 0.1},
                                    "false_pass": True})

    report = orch.run([_task()])

    assert report.false_passes == 1
    assert report.false_fails == 0


# ── on_outcome streaming callback ────────────────────────────────────────


def test_on_outcome_fires_once_per_run(monkeypatch) -> None:
    # The loop emits each outcome the moment it lands, so a caller can stream
    # rows to disk instead of waiting for the whole matrix to finish.
    _stub_reward(monkeypatch)
    alloc = MyopicAllocator([FakeSub("S0")], epsilon=0.0)
    seen: list[tuple[str, str | None, bool]] = []

    orch = _orch(
        alloc, lambda *a: {"reward": 0.5, "costs": {"total": 0.1}},
        on_outcome=lambda task, o: seen.append((task.name, o.substrate, o.skipped)),
    )
    orch.run([_task("t1")], rounds=2)

    # One emission per pass, carrying the live task + the outcome just appended.
    assert seen == [("t1", "S0", False), ("t1", "S0", False)]


def test_on_outcome_fires_on_skip(monkeypatch) -> None:
    # Skips are outcomes too — they must stream so the file mirrors the report.
    _stub_reward(monkeypatch)
    alloc = MyopicAllocator([FakeSub("S0")], epsilon=0.0)
    seen: list[tuple[str, bool, str]] = []

    orch = _orch(
        alloc, lambda *a: {"reward": 0.5, "costs": {"total": 0.1}},
        on_outcome=lambda task, o: seen.append((task.name, o.skipped, o.note)),
    )
    orch.run_task(_task("nope", status="needs_oracle", task_id=""))

    assert len(seen) == 1
    name, skipped, note = seen[0]
    assert name == "nope" and skipped is True and "not runnable" in note


def test_on_outcome_absent_is_a_no_op(monkeypatch) -> None:
    # The callback is optional; the loop must run unchanged without one.
    _stub_reward(monkeypatch)
    alloc = MyopicAllocator([FakeSub("S0")], epsilon=0.0)
    orch = _orch(alloc, lambda *a: {"reward": 0.5, "costs": {"total": 0.1}})

    report = orch.run([_task()])

    assert report.n_runs == 1


def test_run_skips_non_runnable_in_batch(monkeypatch) -> None:
    _stub_reward(monkeypatch)
    alloc = MyopicAllocator([FakeSub("S0")], epsilon=0.0)
    orch = _orch(alloc, lambda *a: {"reward": 0.5, "costs": {"total": 0.1}})

    report = orch.run([_task("ok"), _task("road", status="needs_oracle", task_id="")])

    assert report.n_runs == 1
    assert report.n_skipped == 1


# ── real reward path (only grade_run mocked) ────────────────────────────


def test_full_reward_path_with_mocked_oracle(monkeypatch) -> None:
    # No reward_from_run stub — exercise the real arithmetic + cost channel,
    # mocking only the live oracle call.
    def fake_grade(env_url, admin_token, task_id, **kw):  # noqa: ARG001
        return GradingResult(task_id=task_id, passed=True, score=0.9, reasons=["ok"])

    monkeypatch.setattr(R, "grade_run", fake_grade)

    alloc = MyopicAllocator([FakeSub("S0")], epsilon=0.0)
    orch = _orch(
        alloc,
        lambda *a: {"dynamic_verification_summary": {"verdict": "pass"},
                    "costs": {"total": 1.0}},
        budget=5.0,
    )

    outcome = orch.run_task(_task())

    # reward = oracle_score − λ·dollars = 0.9 − 0.1·1.0 = 0.8
    assert outcome.reward == pytest.approx(0.8)
    assert outcome.dollars == pytest.approx(1.0)
    assert orch.budget_remaining == pytest.approx(4.0)
    assert alloc.value_of("capability", "S0") == pytest.approx(0.8)
