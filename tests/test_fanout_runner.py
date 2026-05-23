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
    DEFAULT_PAGINATION_URL_TEMPLATE,
    LocalFanoutRunner,
    build_worker_subplan,
    fanout_enabled,
    partition_urls,
    partition_urls_for_pagination,
    prepare_modal_partitions,
    read_partition_result,
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


# ── partition_urls_for_pagination ──────────────────────────────────────


def test_pagination_url_synth_default_template() -> None:
    urls = partition_urls_for_pagination(
        "https://example.com/listings/", max_pages=3,
    )
    assert urls == [
        "https://example.com/listings/",
        "https://example.com/listings/page-2/",
        "https://example.com/listings/page-3/",
    ]


def test_pagination_url_synth_custom_template() -> None:
    urls = partition_urls_for_pagination(
        "https://example.com/search",
        max_pages=2,
        template="{base}?page={n}",
    )
    assert urls == [
        "https://example.com/search",
        "https://example.com/search?page=2",
    ]


def test_pagination_url_synth_clamps_zero() -> None:
    """``max_pages == 0`` is a misconfigured loop_count — clamp to 1
    so the partition list isn't accidentally empty."""
    urls = partition_urls_for_pagination("https://example.com/", max_pages=0)
    assert urls == ["https://example.com/"]


# ── prepare_modal_partitions ───────────────────────────────────────────


def _pagination_plan_suite() -> dict:
    """A suite_dict mimicking what build_micro_suite produces for a
    boattrader-shaped plan with a pagination loop wrapping the
    extraction body."""
    plan = MicroPlan(source_plan="", domain="example.com")
    plan.steps = [
        MicroIntent(
            intent="Navigate", type="navigate", section="setup",
            params={"url": "https://example.com/listings/"},
        ),
        MicroIntent(intent="Click card", type="click", section="extraction"),
        MicroIntent(intent="Read URL", type="extract_url", section="extraction"),
        MicroIntent(intent="Scroll", type="scroll", section="extraction"),
        MicroIntent(intent="Extract", type="extract_data", section="extraction"),
        MicroIntent(intent="Go back", type="navigate_back", section="extraction"),
        MicroIntent(
            intent="Inner loop", type="loop", section="extraction",
            loop_target=1, loop_count=25,
        ),
        MicroIntent(intent="Paginate", type="paginate", section="pagination"),
        MicroIntent(
            intent="Outer loop", type="loop", section="pagination",
            loop_target=7, loop_count=4,  # 4 pages
        ),
    ]
    PlanDecomposer._classify_loop_groups(plan)
    suite = {
        "session_name": "example",
        "_micro_plan": [
            {
                "intent": s.intent, "type": s.type, "section": s.section,
                "params": dict(s.params or {}),
                "loop_target": s.loop_target, "loop_count": s.loop_count,
                "claude_only": s.claude_only, "gate": s.gate,
                "required": s.required, "grounding": s.grounding,
                "verify": s.verify, "reverse": s.reverse,
                "budget": s.budget, "hints": dict(s.hints or {}),
            }
            for s in plan.steps
        ],
        "_loop_groups": [
            {
                "loop_step_idx": g.loop_step_idx,
                "body_range": list(g.body_range),
                "shape": g.shape,
            }
            for g in plan.loop_groups
        ],
    }
    return suite


def test_prepare_modal_partitions_synthesizes_per_page_suites() -> None:
    suite = _pagination_plan_suite()
    partitions = prepare_modal_partitions(suite, workers=4)
    # 4 pages → 4 partition sub-suites.
    assert len(partitions) == 4
    # Each carries a distinct setup-navigate URL.
    nav_urls = [
        next(s["params"]["url"] for s in p["_micro_plan"] if s["type"] == "navigate")
        for p in partitions
    ]
    assert nav_urls == [
        "https://example.com/listings/",
        "https://example.com/listings/page-2/",
        "https://example.com/listings/page-3/",
        "https://example.com/listings/page-4/",
    ]


def test_prepare_modal_partitions_drops_pagination_loop() -> None:
    """Each worker is single-page — the outer pagination loop step
    and the inner paginate step must not appear in the sub-plan,
    otherwise the worker would try to paginate inside its slice."""
    suite = _pagination_plan_suite()
    partitions = prepare_modal_partitions(suite, workers=2)
    for p in partitions:
        types = [s["type"] for s in p["_micro_plan"]]
        assert "paginate" not in types
        # The outer loop (the LAST loop step in the original plan)
        # should be gone; the inner extraction loop stays so the worker
        # iterates across the listings on its single page.
        assert types.count("loop") == 1


def test_prepare_modal_partitions_preserves_inner_extraction_body() -> None:
    """Regression for the first deploy: my rewriter was dropping the
    entire pagination-loop body_range (steps 2..loop_idx), which also
    deletes the inner extraction body. Each worker then ran only the
    setup-navigate + verify step → 0 leads per partition. The fix:
    drop ONLY the outer loop step + any paginate inside the body.
    Keep click/extract_url/scroll/extract_data/navigate_back/loop —
    that's the per-page extraction work the worker actually does."""
    suite = _pagination_plan_suite()
    partitions = prepare_modal_partitions(suite, workers=2)
    expected = ["navigate", "click", "extract_url", "scroll", "extract_data",
                "navigate_back", "loop"]
    for p in partitions:
        types = [s["type"] for s in p["_micro_plan"]]
        assert types == expected, types


def test_prepare_modal_partitions_returns_empty_without_loop_groups() -> None:
    """Plans with no parallelizable group → empty list → caller falls
    through to single-worker dispatch."""
    suite = {
        "session_name": "x",
        "_micro_plan": [
            {"type": "navigate", "intent": "n", "section": "setup",
             "params": {"url": "https://x.com/"}},
        ],
        "_loop_groups": [],
    }
    assert prepare_modal_partitions(suite, workers=4) == []


def test_prepare_modal_partitions_url_collect_falls_through() -> None:
    """parallelizable_url_collect isn't yet wired into the Modal path —
    it's a Phase 5 follow-up. Test that the orchestrator returns empty
    (caller falls through) and logs a WARNING."""
    plan = MicroPlan(source_plan="", domain="x.com")
    plan.steps = [
        MicroIntent(
            intent="Navigate", type="navigate", section="setup",
            params={"url": "https://x.com/listings"},
        ),
        MicroIntent(intent="Click", type="click", section="extraction"),
        MicroIntent(intent="URL", type="extract_url", section="extraction"),
        MicroIntent(intent="Extract", type="extract_data", section="extraction"),
        MicroIntent(intent="Back", type="navigate_back", section="extraction"),
        MicroIntent(
            intent="Loop", type="loop", section="extraction",
            loop_target=1, loop_count=20,
        ),
    ]
    PlanDecomposer._classify_loop_groups(plan)
    assert plan.loop_groups[0].shape == "parallelizable_url_collect"
    suite = {
        "session_name": "x",
        "_micro_plan": [
            {
                "intent": s.intent, "type": s.type, "section": s.section,
                "params": dict(s.params or {}),
                "loop_target": s.loop_target, "loop_count": s.loop_count,
                "claude_only": s.claude_only, "gate": s.gate,
                "required": s.required, "grounding": s.grounding,
                "verify": s.verify, "reverse": s.reverse,
                "budget": s.budget, "hints": dict(s.hints or {}),
            }
            for s in plan.steps
        ],
        "_loop_groups": [
            {
                "loop_step_idx": g.loop_step_idx,
                "body_range": list(g.body_range),
                "shape": g.shape,
            }
            for g in plan.loop_groups
        ],
    }
    assert prepare_modal_partitions(suite, workers=4) == []


def test_pagination_template_used_from_paginate_step_hint() -> None:
    """Operator-provided ``url_template`` on the paginate step overrides
    the default. Lets Zillow / FB-style URL schemes work without code change."""
    suite = _pagination_plan_suite()
    # Inject a custom template on the paginate step.
    for s in suite["_micro_plan"]:
        if s["type"] == "paginate":
            s["params"]["url_template"] = "{base}?page={n}"
            break
    partitions = prepare_modal_partitions(suite, workers=4)
    nav_urls = [
        next(s["params"]["url"] for s in p["_micro_plan"] if s["type"] == "navigate")
        for p in partitions
    ]
    # Pages 2-4 use the override; page 1 is the bare base URL.
    assert nav_urls[1] == "https://example.com/listings?page=2"
    assert nav_urls[3] == "https://example.com/listings?page=4"


def test_default_pagination_template_constant() -> None:
    assert DEFAULT_PAGINATION_URL_TEMPLATE == "{base}/page-{n}/"


# ── #623: read_partition_result ────────────────────────────────────────


def test_read_partition_result_canonical_shape() -> None:
    """build_micro_result returns ``viable`` + ``leads_with_phone`` + ``leads``.
    The reader must surface those — NOT the legacy ``leads_count`` /
    ``score`` keys the gemma4 worker used to return."""
    fake = {
        "viable": 27,
        "leads_with_phone": 1,
        "leads": [{"listing_url": "https://example.com/boat/1/"}],
        # Extra keys (cost, time, etc) must not interfere.
        "cost": 10.03,
        "score": 0.9,  # MUST be ignored — legacy field.
    }
    out = read_partition_result(fake)
    assert out["viable"] == 27
    assert out["with_phone"] == 1
    assert out["leads"] == [{"listing_url": "https://example.com/boat/1/"}]


def test_read_partition_result_legacy_keys_zeroed() -> None:
    """A worker that returns only the legacy gemma4 shape (``leads_count``
    / ``score``, no ``viable``) yields zeros — the orchestrator must NOT
    silently confuse counts. This pins the bug #623 fixed: the legacy
    reader saw ``leads_count`` and trusted it; the new reader requires
    the canonical ``viable`` key."""
    legacy = {"leads_count": 27, "score": 0.9}
    out = read_partition_result(legacy)
    assert out["viable"] == 0
    assert out["with_phone"] == 0
    assert out["leads"] == []


def test_read_partition_result_none_input_safe() -> None:
    """``handle.get()`` returning None (worker crash) shouldn't crash
    the orchestrator — the reader returns the zero shape and the
    orchestrator's exception handler logs the failure separately."""
    out = read_partition_result(None)
    assert out == {"viable": 0, "with_phone": 0, "leads": []}


def test_read_partition_result_tolerates_missing_leads_list() -> None:
    """A worker that returns counts but elides the leads list (some
    paths drop it to save bandwidth) must not crash the dedup pass
    (#621); ``leads`` defaults to an empty list."""
    out = read_partition_result({"viable": 5, "leads_with_phone": 1})
    assert out["viable"] == 5
    assert out["with_phone"] == 1
    assert out["leads"] == []


def test_read_partition_result_coerces_string_counts() -> None:
    """Defensive — some serialisation paths upcast ints to strings.
    The reader must coerce to int so ``merged_total += summary['viable']``
    doesn't TypeError on the orchestrator side."""
    out = read_partition_result({"viable": "27", "leads_with_phone": "1"})
    assert out["viable"] == 27
    assert out["with_phone"] == 1
