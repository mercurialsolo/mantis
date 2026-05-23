"""LocalFanoutRunner + plan-rewriter unit tests (#616).

The plan rewriter is pure (no I/O) so we test it directly. The local
runner spawns ProcessPoolExecutor workers; we test the partitioning,
sub-plan construction, and merge logic via an in-process executor
stub that bypasses pickling — production picks up
``ProcessPoolExecutor`` automatically because the unit under test is
the runner's orchestration, not the IPC layer.

Acceptance for #616 is:

  - Rewriter drops the loop body's click/extract_url/navigate_back
    anchor and inserts a parameterised navigate.
  - Rewriter refuses sequential groups (returns plan unchanged).
  - Partition is round-robin across N workers.
  - build_worker_subplan fills the navigate URL per worker.
  - LocalFanoutRunner spawns workers and merges StepResults.
"""

from __future__ import annotations

from typing import Any

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.fanout_runner import (
    LocalFanoutRunner,
    build_worker_subplan,
    fanout_enabled,
    partition_urls,
    rewrite_for_fanout,
)
from mantis_agent.plan_decomposer import (
    LoopGroup,
    MicroIntent,
    MicroPlan,
    PlanDecomposer,
)


def _canonical_extraction_plan() -> MicroPlan:
    """The shape PlanDecomposer.DECOMPOSE_PROMPT documents at line 589.

    setup: navigate
    extraction body: click → extract_url → scroll → extract_data → navigate_back
    loop step (target=1, count=20)
    """
    plan = MicroPlan(source_plan="", domain="example.com")
    plan.steps = [
        MicroIntent(
            intent="Navigate to https://example.com/listings",
            type="navigate", section="setup",
            params={"url": "https://example.com/listings"},
        ),
        MicroIntent(
            intent="Click the next listing", type="click",
            section="extraction",
        ),
        MicroIntent(intent="Read URL", type="extract_url", section="extraction"),
        MicroIntent(intent="Scroll the page", type="scroll", section="extraction"),
        MicroIntent(
            intent="Extract fields", type="extract_data", section="extraction",
        ),
        MicroIntent(intent="Go back", type="navigate_back", section="extraction"),
        MicroIntent(
            intent="Loop body", type="loop", section="extraction",
            loop_target=1, loop_count=20,
        ),
    ]
    PlanDecomposer._classify_loop_groups(plan)
    return plan


# ── fanout_enabled gate ────────────────────────────────────────────────


def test_fanout_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MANTIS_FANOUT", raising=False)
    assert fanout_enabled() is False


def test_fanout_enabled_when_local(monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_FANOUT", "local")
    assert fanout_enabled() is True


def test_fanout_disabled_for_unknown_value(monkeypatch) -> None:
    """Defensive: anything that isn't ``local`` leaves sequential path on."""
    monkeypatch.setenv("MANTIS_FANOUT", "modal")
    assert fanout_enabled() is False


# ── rewrite_for_fanout ─────────────────────────────────────────────────


def test_rewriter_drops_click_extract_url_navigate_back() -> None:
    plan = _canonical_extraction_plan()
    group = plan.loop_groups[0]
    rewritten = rewrite_for_fanout(plan, group)
    types = [s.type for s in rewritten.steps]
    # Setup navigate is preserved; loop body becomes navigate → scroll
    # → extract_data; loop step is gone.
    assert types == ["navigate", "navigate", "scroll", "extract_data"]


def test_rewriter_inserts_url_sentinel_on_body_navigate() -> None:
    plan = _canonical_extraction_plan()
    rewritten = rewrite_for_fanout(plan, plan.loop_groups[0])
    body_navigate = rewritten.steps[1]
    assert body_navigate.type == "navigate"
    assert body_navigate.params["url"] == ""
    # Setup navigate is untouched.
    assert rewritten.steps[0].params["url"] == "https://example.com/listings"


def test_rewriter_refuses_sequential_group() -> None:
    plan = _canonical_extraction_plan()
    group = LoopGroup(loop_step_idx=6, body_range=(1, 6), shape="sequential")
    rewritten = rewrite_for_fanout(plan, group)
    # Sequential group → original plan returned unchanged.
    assert rewritten is plan


def test_rewriter_preserves_post_loop_steps() -> None:
    plan = _canonical_extraction_plan()
    # Append a paginate after the loop so we can pin that the rewriter
    # carries trailing steps through.
    plan.steps.append(
        MicroIntent(intent="Paginate", type="paginate", section="pagination")
    )
    PlanDecomposer._classify_loop_groups(plan)
    rewritten = rewrite_for_fanout(plan, plan.loop_groups[0])
    assert rewritten.steps[-1].type == "paginate"


def test_rewriter_drops_loop_groups_on_output() -> None:
    plan = _canonical_extraction_plan()
    rewritten = rewrite_for_fanout(plan, plan.loop_groups[0])
    # The loop is gone — no group should survive on the rewrite.
    assert rewritten.loop_groups == []


def test_rewriter_does_not_mutate_input() -> None:
    plan = _canonical_extraction_plan()
    original_types = [s.type for s in plan.steps]
    rewrite_for_fanout(plan, plan.loop_groups[0])
    assert [s.type for s in plan.steps] == original_types


# ── partition_urls ─────────────────────────────────────────────────────


def test_partition_round_robin() -> None:
    chunks = partition_urls(["a", "b", "c", "d", "e"], workers=2)
    assert chunks == [["a", "c", "e"], ["b", "d"]]


def test_partition_when_workers_exceed_urls() -> None:
    chunks = partition_urls(["a", "b"], workers=4)
    assert chunks == [["a"], ["b"], [], []]


def test_partition_clamps_workers_to_one() -> None:
    chunks = partition_urls(["a", "b"], workers=0)
    assert chunks == [["a", "b"]]


# ── build_worker_subplan ───────────────────────────────────────────────


def test_build_worker_subplan_fills_url_per_worker() -> None:
    plan = _canonical_extraction_plan()
    rewritten = rewrite_for_fanout(plan, plan.loop_groups[0])
    urls = ["https://example.com/boat/1/", "https://example.com/boat/2/"]
    subs = build_worker_subplan(rewritten, urls)
    assert len(subs) == 2
    body_urls = [
        next(
            s.params["url"] for s in sub.steps
            if s.type == "navigate"
            and s.params.get("url", "").startswith("https://example.com/boat/")
        )
        for sub in subs
    ]
    assert body_urls == urls


def test_build_worker_subplan_does_not_share_step_refs() -> None:
    """Two sub-plans should have independent params dicts — mutating
    one mustn't leak into the other (a common process-pool footgun)."""
    plan = _canonical_extraction_plan()
    rewritten = rewrite_for_fanout(plan, plan.loop_groups[0])
    subs = build_worker_subplan(rewritten, ["url1", "url2"])
    nav_1 = next(s for s in subs[0].steps if s.params.get("url") == "url1")
    nav_2 = next(s for s in subs[1].steps if s.params.get("url") == "url2")
    nav_1.params["url"] = "MUTATED"
    assert nav_2.params["url"] == "url2"


# ── LocalFanoutRunner orchestration ────────────────────────────────────


class _StubRunner:
    """Records the plan it received and returns a deterministic
    StepResult per step. Picklable so ProcessPoolExecutor can dispatch
    it (but the test in this file uses an in-process pool stub
    instead to keep the test fast and debuggable)."""

    def run(self, plan: MicroPlan) -> list[StepResult]:
        url = ""
        for s in plan.steps:
            if s.type == "navigate" and s.params.get("url", "").startswith("http"):
                url = s.params["url"]
        return [
            StepResult(
                step_index=0, intent=f"extract {url}", success=True,
                data=f"OK:{url}",
            )
        ]


class _InProcessPool:
    """Drop-in replacement for ProcessPoolExecutor that runs futures
    inline. Lets the test exercise the merge + accounting logic
    without paying process-spawn cost or worrying about pickleability
    of test fixtures.

    Patched in via monkeypatch on the module's ``ProcessPoolExecutor``
    symbol; tests that DO need process boundaries can fall back to
    the real executor with a top-level-defined factory.
    """

    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers

    def __enter__(self) -> "_InProcessPool":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def submit(self, fn, *args, **kwargs):
        from concurrent.futures import Future
        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001
            fut.set_exception(exc)
        return fut


def test_local_fanout_dispatches_n_workers(monkeypatch) -> None:
    monkeypatch.setattr(
        "mantis_agent.gym.fanout_runner.ProcessPoolExecutor", _InProcessPool,
    )
    plan = _canonical_extraction_plan()
    runner = LocalFanoutRunner(runner_factory=_StubRunner, workers=4)
    urls = [f"https://example.com/boat/{i}/" for i in range(8)]
    res = runner.run(plan, urls, group=plan.loop_groups[0])

    # 8 URLs, 8 sub-plans, 8 merged results (one per stub run).
    assert res.urls_dispatched == 8
    assert res.workers == 4
    assert len(res.results) == 8
    assert res.failures == 0
    # Each StepResult.data carries the per-worker URL → confirms each
    # sub-plan got a distinct navigate URL.
    seen = sorted(r.data.split(":", 1)[1] for r in res.results)
    assert seen == sorted(urls)


def test_local_fanout_worker_failure_does_not_take_down_run(monkeypatch) -> None:
    monkeypatch.setattr(
        "mantis_agent.gym.fanout_runner.ProcessPoolExecutor", _InProcessPool,
    )

    class _PartialFailFactory:
        calls = 0

        def __new__(cls):
            cls.calls += 1
            if cls.calls == 2:
                # Build a runner whose run() raises.
                class _BoomRunner:
                    def run(self, plan):
                        raise RuntimeError("simulated crash")
                return _BoomRunner()
            return _StubRunner()

    plan = _canonical_extraction_plan()
    runner = LocalFanoutRunner(runner_factory=_PartialFailFactory, workers=2)
    urls = ["https://example.com/boat/1/", "https://example.com/boat/2/"]
    res = runner.run(plan, urls, group=plan.loop_groups[0])

    assert res.failures == 1
    # One worker succeeded → one result merged through.
    assert len(res.results) == 1
