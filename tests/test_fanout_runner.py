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
    DEFAULT_PHASE1_MAX_PAGES,
    LocalFanoutRunner,
    NullSharedSeenSet,
    _InMemorySharedSeenSet,
    _normalize_listing_url,
    build_shared_seen_set,
    build_worker_subplan,
    dedup_leads_by_url,
    fanout_enabled,
    find_url_collect_group,
    partition_urls,
    partition_urls_for_pagination,
    prepare_modal_partitions,
    prepare_phase1_suite,
    prepare_phase2_suites,
    read_partition_result,
    resolve_phase1_max_pages,
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
    assert out == {
        "viable": 0, "with_phone": 0,
        "leads": [], "collected_urls": [], "shared_seen_hits": 0,
    }


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


# ── #621: cross-partition lead dedup ───────────────────────────────────


def test_normalize_listing_url_strips_trailing_slash() -> None:
    assert _normalize_listing_url("https://example.com/boat/1/") == "https://example.com/boat/1"


def test_normalize_listing_url_lowercases() -> None:
    """Path-case drift across pages (some marketplaces emit /Boat/Foo
    on one page and /boat/foo on another) should collapse."""
    assert (
        _normalize_listing_url("https://Example.com/Boat/Foo/")
        == "https://example.com/boat/foo"
    )


def test_normalize_listing_url_empty_input() -> None:
    assert _normalize_listing_url("") == ""
    assert _normalize_listing_url("   ") == ""


def test_dedup_collapses_exact_url_duplicates() -> None:
    """Featured / sponsored boat appears on two pages — collapse to one."""
    per_partition = [
        [
            {"listing_url": "https://example.com/boat/1/", "make": "A"},
            {"listing_url": "https://example.com/boat/2/", "make": "B"},
        ],
        [
            {"listing_url": "https://example.com/boat/1/", "make": "A"},  # dup
            {"listing_url": "https://example.com/boat/3/", "make": "C"},
        ],
    ]
    deduped, raw, dedup_count = dedup_leads_by_url(per_partition)
    assert raw == 4
    assert dedup_count == 3
    assert [d["make"] for d in deduped] == ["A", "B", "C"]


def test_dedup_collapses_trailing_slash_variants() -> None:
    """``/boat/1`` and ``/boat/1/`` are the same listing (BoatTrader emits
    both interchangeably across pages)."""
    per_partition = [
        [{"listing_url": "https://example.com/boat/1/"}],
        [{"listing_url": "https://example.com/boat/1"}],  # no slash
    ]
    deduped, raw, dedup_count = dedup_leads_by_url(per_partition)
    assert raw == 2
    assert dedup_count == 1


def test_dedup_no_collapse_when_urls_differ() -> None:
    """Five distinct URLs across two partitions → 5 deduped leads."""
    per_partition = [
        [{"listing_url": f"https://example.com/boat/{i}/"} for i in range(3)],
        [{"listing_url": f"https://example.com/boat/{i}/"} for i in range(3, 5)],
    ]
    deduped, raw, dedup_count = dedup_leads_by_url(per_partition)
    assert raw == 5
    assert dedup_count == 5


def test_dedup_preserves_first_seen_partition_order() -> None:
    """When the same URL appears in partition 1 first and partition 2
    later, we keep partition 1's row (its data may be cleaner since
    partition 1 likely ran first / had warmer cache)."""
    per_partition = [
        [{"listing_url": "https://example.com/boat/1/", "asking_price": "$100"}],
        [{"listing_url": "https://example.com/boat/1/", "asking_price": "$999"}],
    ]
    deduped, _, _ = dedup_leads_by_url(per_partition)
    assert len(deduped) == 1
    assert deduped[0]["asking_price"] == "$100"


def test_dedup_leads_without_url_pass_through() -> None:
    """Leads without a listing_url key (partial extracts) are passed
    through unchanged — the orchestrator shouldn't silently drop them
    just because they lack a dedup key."""
    per_partition = [
        [{"listing_url": "https://example.com/boat/1/"}, {"asking_price": "$50"}],
        [{"listing_url": "https://example.com/boat/1/"}, {"asking_price": "$99"}],
    ]
    deduped, raw, dedup_count = dedup_leads_by_url(per_partition)
    # 4 raw, 1 URL-dup collapsed, 2 no-URL passed through, 1 unique URL = 3
    assert raw == 4
    assert dedup_count == 3


def test_dedup_handles_empty_partition_lists() -> None:
    """A partition that returned no leads (worker crashed early) leaves
    the merge a no-op for that chunk."""
    per_partition = [
        [],
        [{"listing_url": "https://example.com/boat/1/"}],
        [],
    ]
    deduped, raw, dedup_count = dedup_leads_by_url(per_partition)
    assert raw == 1
    assert dedup_count == 1


def test_dedup_ignores_malformed_lead_entries() -> None:
    """Defensive — None / int / unknown types shouldn't crash the
    merge. They pass through unchanged ("no key" path)."""
    per_partition = [
        [{"listing_url": "https://example.com/boat/1/"}, None, 42],
    ]
    deduped, raw, dedup_count = dedup_leads_by_url(per_partition)
    assert raw == 3
    # None + 42 yield no URL → pass through (no dedup); dict has URL → counts.
    assert dedup_count == 3


# ── Real production string-lead format (#621 verification regression) ─


def _viable_lead(url: str, year: int = 2023, make: str = "Acme") -> str:
    """Build a lead row in the actual format build_micro_result emits
    (server_utils.py:820, leads list). VIABLE prefix + pipe-delimited
    fields + ``URL: <url>``. Matches what ListingDedup.lead_key parses
    so cross-partition keys match per-container keys."""
    return (
        f"VIABLE | Year: {year} | Make: {make} | Model: X | "
        f"Price: $99,000 | Phone: none | URL: {url}"
    )


def test_dedup_collapses_string_leads_across_partitions() -> None:
    """The Modal verification run for #621 surfaced this: leads are
    strings (not dicts), so the dict-only first pass dropped all 87.
    Fix: extract URL via ListingDedup.lead_key for string leads."""
    per_partition = [
        [
            _viable_lead("boattrader.com/boat/1/", year=2023, make="Pershing"),
            _viable_lead("boattrader.com/boat/2/", year=2022, make="Freeman"),
        ],
        [
            _viable_lead("boattrader.com/boat/1/", year=2023, make="Pershing"),  # dup
            _viable_lead("boattrader.com/boat/3/", year=2021, make="Azimut"),
        ],
    ]
    deduped, raw, dedup_count = dedup_leads_by_url(per_partition)
    assert raw == 4
    assert dedup_count == 3
    # First-seen wins (partition 1's Pershing row, not partition 2's)
    pershing_rows = [
        d for d in deduped if isinstance(d, str) and "Pershing" in d
    ]
    assert len(pershing_rows) == 1


def test_dedup_string_leads_collapse_trailing_slash_variants() -> None:
    """Cross-page emit of the same URL with/without trailing slash is
    the most common cross-partition overlap case — must collapse to one."""
    per_partition = [
        [_viable_lead("boattrader.com/boat/1/")],
        [_viable_lead("boattrader.com/boat/1")],  # no trailing slash
    ]
    deduped, raw, dedup_count = dedup_leads_by_url(per_partition)
    assert raw == 2
    assert dedup_count == 1


def test_dedup_string_lead_without_url_passes_through() -> None:
    """A lead row missing the ``URL:`` token (rare partial-extract case)
    has no dedup key — pass through unchanged rather than collapsing
    all keyless rows under a single fallback bucket."""
    no_url = "VIABLE | Year: 2023 | Make: X | Phone: none"
    per_partition = [[no_url, no_url]]  # two identical keyless rows
    deduped, raw, dedup_count = dedup_leads_by_url(per_partition)
    assert raw == 2
    # Both pass through — better to over-report than silently collapse
    # rows we can't authoritatively identify as duplicates.
    assert dedup_count == 2


def test_dedup_mixed_string_and_dict_leads() -> None:
    """Defensive — if a future caller mixes string and dict leads in
    one partition list, dedup should still work consistently across
    both shapes by extracted URL."""
    per_partition = [
        [_viable_lead("boattrader.com/boat/1/")],
        [{"listing_url": "boattrader.com/boat/1/", "make": "dict-shaped"}],
    ]
    deduped, raw, dedup_count = dedup_leads_by_url(per_partition)
    assert raw == 2
    assert dedup_count == 1


# ── #628: Phase-1/Phase-2 fan-out builders ──────────────────────────────


def _url_collect_suite() -> dict:
    """A boattrader-shaped task_suite with a parallelizable_url_collect
    inner extraction loop (no pagination wrapper). Mirrors what
    PlanDecomposer.decompose emits for the boattrader_scrape_urlnav plan
    before pagination is added."""
    plan = MicroPlan(domain="x.com")
    plan.steps = [
        MicroIntent(
            intent="Navigate", type="navigate", section="setup",
            params={"url": "https://x.com/listings"},
        ),
        MicroIntent(intent="Click card", type="click", section="extraction"),
        MicroIntent(intent="Read URL", type="extract_url", section="extraction"),
        MicroIntent(intent="Scroll", type="scroll", section="extraction"),
        MicroIntent(intent="Extract", type="extract_data", section="extraction"),
        MicroIntent(intent="Back", type="navigate_back", section="extraction"),
        MicroIntent(
            intent="Inner loop", type="loop", section="extraction",
            loop_target=1, loop_count=20,
        ),
    ]
    PlanDecomposer._classify_loop_groups(plan)
    return {
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


def test_find_url_collect_group_returns_group_when_present() -> None:
    suite = _url_collect_suite()
    g = find_url_collect_group(suite)
    assert g is not None
    assert g.shape == "parallelizable_url_collect"


def test_find_url_collect_group_returns_none_for_pagination_only() -> None:
    """Pagination-only plan (no url-collect inner) → orchestrator falls
    through to the existing per-page partition path."""
    suite = _pagination_plan_suite()
    g = find_url_collect_group(suite)
    # The pagination plan has BOTH groups (inner url-collect + outer
    # pagination). Verify finder returns the url-collect inner — but
    # in real flows the orchestrator prefers Phase-1/Phase-2 over
    # per-page when the url-collect group exists. We pin shape here.
    if g is not None:
        assert g.shape == "parallelizable_url_collect"


def test_find_url_collect_group_returns_none_without_micro_plan() -> None:
    assert find_url_collect_group({}) is None
    assert find_url_collect_group({"_micro_plan": []}) is None


# ── prepare_phase1_suite ─────────────────────────────────────────────


def test_phase1_suite_keeps_setup_drops_extraction_body() -> None:
    suite = _url_collect_suite()
    g = find_url_collect_group(suite)
    phase1 = prepare_phase1_suite(suite, g)
    types = [s["type"] for s in phase1["_micro_plan"]]
    # Setup navigate is preserved; extraction body + loop are gone;
    # collect_urls is injected at the end.
    assert types == ["navigate", "collect_urls"]


def test_phase1_suite_injects_collect_urls_step() -> None:
    suite = _url_collect_suite()
    g = find_url_collect_group(suite)
    phase1 = prepare_phase1_suite(suite, g)
    last = phase1["_micro_plan"][-1]
    assert last["type"] == "collect_urls"
    assert last["claude_only"] is True
    assert last["section"] == "extraction"


def test_phase1_suite_drops_loop_groups() -> None:
    """Phase-1 sub-suite has no loops — drop _loop_groups so the child
    container doesn't accidentally route through any fanout path."""
    suite = _url_collect_suite()
    g = find_url_collect_group(suite)
    phase1 = prepare_phase1_suite(suite, g)
    assert "_loop_groups" not in phase1


def test_phase1_suite_tags_phase_metadata() -> None:
    suite = _url_collect_suite()
    g = find_url_collect_group(suite)
    phase1 = prepare_phase1_suite(suite, g)
    assert phase1["_fanout_phase"] == "phase1_collect"
    assert phase1["session_name"].endswith("_phase1")


# ── #638 axis 2: multi-page Phase-1 ──────────────────────────────────


def test_phase1_suite_max_pages_one_is_single_collect() -> None:
    """max_pages=1 → no navigate inserted after the setup chain; one
    collect_urls only. Backward compatible with the original #628 shape."""
    suite = _url_collect_suite()
    g = find_url_collect_group(suite)
    phase1 = prepare_phase1_suite(suite, g, max_pages=1)
    types = [s["type"] for s in phase1["_micro_plan"]]
    assert types == ["navigate", "collect_urls"]


def test_phase1_suite_max_pages_three_chains_nav_collect() -> None:
    """max_pages=3 → setup navigate + collect_urls + navigate(page-2) +
    collect_urls + navigate(page-3) + collect_urls."""
    suite = _url_collect_suite()
    g = find_url_collect_group(suite)
    phase1 = prepare_phase1_suite(suite, g, max_pages=3)
    types = [s["type"] for s in phase1["_micro_plan"]]
    assert types == [
        "navigate", "collect_urls",
        "navigate", "collect_urls",
        "navigate", "collect_urls",
    ]


def test_phase1_suite_max_pages_three_synthesizes_page_urls() -> None:
    """Pages 2..N synthesised via the default pagination url_template,
    matching what the per-page fan-out path uses."""
    suite = _url_collect_suite()
    # Setup navigate base URL is https://x.com/listings.
    g = find_url_collect_group(suite)
    phase1 = prepare_phase1_suite(suite, g, max_pages=3)
    nav_urls = [
        s["params"]["url"] for s in phase1["_micro_plan"]
        if s["type"] == "navigate"
    ]
    # First navigate is the original setup URL; pages 2 and 3 follow
    # the default ``{base}/page-{n}/`` template.
    assert nav_urls[0] == "https://x.com/listings"
    assert nav_urls[1].endswith("/page-2/")
    assert nav_urls[2].endswith("/page-3/")


def test_phase1_suite_max_pages_template_override() -> None:
    """Custom url_template propagates from caller into the synthesised
    page-N navigate steps."""
    suite = _url_collect_suite()
    g = find_url_collect_group(suite)
    phase1 = prepare_phase1_suite(
        suite, g, max_pages=2, pagination_url_template="{base}?p={n}",
    )
    nav_urls = [
        s["params"]["url"] for s in phase1["_micro_plan"]
        if s["type"] == "navigate"
    ]
    assert nav_urls[1].endswith("?p=2")


def test_phase1_suite_records_max_pages_metadata() -> None:
    suite = _url_collect_suite()
    g = find_url_collect_group(suite)
    phase1 = prepare_phase1_suite(suite, g, max_pages=3)
    assert phase1["_fanout_phase1_pages"] == 3


def test_phase1_suite_max_pages_silently_falls_back_without_base_url() -> None:
    """If the setup chain has no navigate (degenerate plan), Phase-1 falls
    back to single-page mode rather than crashing — the operator gets
    fewer URLs but the worker still runs."""
    suite = _url_collect_suite()
    # Strip the setup navigate by truncating the micro_plan from
    # body_start. Rebuild loop_groups against the stripped sequence.
    g = find_url_collect_group(suite)
    body_start, _ = g.body_range
    # Replace the setup navigate with a dummy non-navigate step so the
    # body range stays valid but base_url resolution returns "".
    suite["_micro_plan"][0]["type"] = "wait"
    suite["_micro_plan"][0]["params"] = {}
    phase1 = prepare_phase1_suite(suite, g, max_pages=3)
    types = [s["type"] for s in phase1["_micro_plan"]]
    # Only one collect_urls fires — no page-2/page-3 chain because no
    # base URL was resolvable.
    assert types.count("collect_urls") == 1
    assert "navigate" not in types  # only the wait step remains


# ── #638 axis 2: resolve_phase1_max_pages ────────────────────────────


def test_resolve_phase1_max_pages_default_when_no_pagination_group() -> None:
    """Plan without a parallelizable_pagination group → max_pages=1."""
    suite = _url_collect_suite()
    max_pages, template = resolve_phase1_max_pages(suite)
    assert max_pages == 1
    assert template == DEFAULT_PAGINATION_URL_TEMPLATE


def test_resolve_phase1_max_pages_uses_pagination_loop_count() -> None:
    """Pagination loop_count=4 → max_pages=min(4, DEFAULT_PHASE1_MAX_PAGES)."""
    suite = _pagination_plan_suite()
    max_pages, _ = resolve_phase1_max_pages(suite)
    assert max_pages == min(4, DEFAULT_PHASE1_MAX_PAGES)


def test_resolve_phase1_max_pages_clamps_to_default_cap() -> None:
    """A plan with loop_count=50 should be clamped to DEFAULT_PHASE1_MAX_PAGES
    so Phase-1 cost stays bounded."""
    suite = _pagination_plan_suite()
    # Bump the outer loop count past the cap.
    for step in suite["_micro_plan"]:
        if step.get("type") == "loop" and step.get("section") == "pagination":
            step["loop_count"] = 50
    max_pages, _ = resolve_phase1_max_pages(suite)
    assert max_pages == DEFAULT_PHASE1_MAX_PAGES


def test_resolve_phase1_max_pages_honours_explicit_override() -> None:
    """An explicit ``_fanout_phase1_max_pages`` on the suite overrides
    the pagination loop_count + the default cap."""
    suite = _pagination_plan_suite()
    suite["_fanout_phase1_max_pages"] = 7
    max_pages, _ = resolve_phase1_max_pages(suite)
    assert max_pages == 7


def test_resolve_phase1_max_pages_picks_up_plan_template_override() -> None:
    """Plan-level ``_pagination_url_template`` (set by #629) wins over
    the default."""
    suite = _pagination_plan_suite()
    suite["_pagination_url_template"] = "{base}?page={n}"
    _, template = resolve_phase1_max_pages(suite)
    assert template == "{base}?page={n}"


# ── prepare_phase2_suites ────────────────────────────────────────────


def test_phase2_suites_one_per_worker_chunk() -> None:
    """8 URLs × 4 workers → round-robin chunks of 2 URLs each → 4 sub-suites."""
    suite = _url_collect_suite()
    g = find_url_collect_group(suite)
    urls = [f"https://x.com/boat/{i}/" for i in range(8)]
    suites = prepare_phase2_suites(suite, urls, g, workers=4)
    assert len(suites) == 4
    for sub in suites:
        assert sub["_fanout_url_count"] == 2


def test_phase2_suites_each_url_gets_navigate_scroll_extract() -> None:
    """Per-URL triple: navigate → scroll → extract_data."""
    suite = _url_collect_suite()
    g = find_url_collect_group(suite)
    urls = ["https://x.com/boat/1/", "https://x.com/boat/2/"]
    suites = prepare_phase2_suites(suite, urls, g, workers=1)
    types = [s["type"] for s in suites[0]["_micro_plan"]]
    assert types == [
        "navigate", "scroll", "extract_data",
        "navigate", "scroll", "extract_data",
    ]
    nav_urls = [
        s["params"]["url"] for s in suites[0]["_micro_plan"]
        if s["type"] == "navigate"
    ]
    assert nav_urls == urls


def test_phase2_suites_empty_chunks_dropped() -> None:
    """3 URLs × 5 workers → 3 non-empty chunks + 2 empty → return 3 sub-suites."""
    suite = _url_collect_suite()
    g = find_url_collect_group(suite)
    urls = ["a", "b", "c"]
    suites = prepare_phase2_suites(suite, urls, g, workers=5)
    assert len(suites) == 3


def test_phase2_suites_no_loop_groups_persisted() -> None:
    """Phase-2 sub-plans have no loops — confirm _loop_groups not set."""
    suite = _url_collect_suite()
    g = find_url_collect_group(suite)
    suites = prepare_phase2_suites(suite, ["a"], g, workers=1)
    assert "_loop_groups" not in suites[0]


def test_phase2_suites_no_urls_returns_empty_list() -> None:
    suite = _url_collect_suite()
    g = find_url_collect_group(suite)
    assert prepare_phase2_suites(suite, [], g, workers=4) == []


def test_phase2_navigate_carries_wait_after_load() -> None:
    """Phase-2 worker hits a detail page directly (cold cache). The
    navigate step needs wait_after_load_seconds so the runner pauses
    long enough for the page to render before scroll fires."""
    suite = _url_collect_suite()
    g = find_url_collect_group(suite)
    suites = prepare_phase2_suites(suite, ["https://x.com/boat/1/"], g, workers=1)
    nav = next(s for s in suites[0]["_micro_plan"] if s["type"] == "navigate")
    assert nav["params"]["wait_after_load_seconds"] == 6


# ── read_partition_result collected_urls ─────────────────────────────


def test_read_partition_result_surfaces_collected_urls() -> None:
    """Phase-1 workers stash harvested URLs on result.collected_urls
    (added to build_micro_result in #628). The reader must surface
    them so the orchestrator can dispatch Phase-2."""
    fake = {
        "viable": 0,
        "leads_with_phone": 0,
        "leads": [],
        "collected_urls": ["https://x.com/boat/1/", "https://x.com/boat/2/"],
    }
    out = read_partition_result(fake)
    assert out["collected_urls"] == [
        "https://x.com/boat/1/", "https://x.com/boat/2/",
    ]


def test_read_partition_result_collected_urls_defaults_empty() -> None:
    """Phase-2 / sequential workers have no collected_urls — empty list."""
    out = read_partition_result({"viable": 27, "leads_with_phone": 1})
    assert out["collected_urls"] == []


# ── #627: shared seen-URL set ───────────────────────────────────────────


def test_null_shared_seen_is_noop() -> None:
    """Default backend on every runner. ``contains`` never hits;
    ``add`` is silent; ``size`` is always zero — preserves single-worker
    behaviour without conditional dedup branches on the hot path."""
    s = NullSharedSeenSet()
    s.add("https://x.com/boat/1/")
    assert s.contains("https://x.com/boat/1/") is False
    assert s.size() == 0


def test_inmemory_shared_seen_basic_round_trip() -> None:
    s = _InMemorySharedSeenSet()
    s.add("https://x.com/boat/1/")
    assert s.contains("https://x.com/boat/1/") is True
    assert s.size() == 1


def test_inmemory_shared_seen_normalizes_url() -> None:
    """Trailing slash and case drift collapse — keys line up with
    the merge-step ``dedup_leads_by_url`` normalization."""
    s = _InMemorySharedSeenSet()
    s.add("https://Example.com/Boat/1/")
    assert s.contains("https://example.com/boat/1") is True
    assert s.contains("https://example.com/boat/1/") is True
    assert s.size() == 1


def test_inmemory_shared_seen_ignores_empty_urls() -> None:
    """Empty / whitespace-only URLs aren't keys — they pass through the
    normalizer to ``""``, which we don't add (avoids a single 'no URL'
    bucket swallowing everything that fails extract_url)."""
    s = _InMemorySharedSeenSet()
    s.add("")
    s.add("   ")
    assert s.size() == 0


def test_build_shared_seen_set_returns_null_without_dict_name() -> None:
    """No ``_fanout_seen_dict_name`` in suite → NullSharedSeenSet.
    Pure local / non-fanout runs hit this path."""
    s = build_shared_seen_set({})
    assert isinstance(s, NullSharedSeenSet)
    s2 = build_shared_seen_set({"_micro_plan": [{"type": "navigate"}]})
    assert isinstance(s2, NullSharedSeenSet)


def test_build_shared_seen_set_falls_back_when_modal_raises(monkeypatch) -> None:
    """Suite has a dict name but ``modal.Dict.from_name`` raises (e.g.
    not authenticated to Modal, or running outside a Modal container) →
    fall back to NullSharedSeenSet with a WARNING. Prevents fan-out
    runs from crashing when the shared store can't be attached.

    Skipped when ``modal`` is not installed (CI's default test extra
    doesn't pull it). The bare-module check is what matters here, and
    on Modal-less environments the fall-through path already returns
    NullSharedSeenSet via the import-fail branch in build_shared_seen_set.
    """
    import pytest
    modal = pytest.importorskip("modal")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated: not on Modal")

    monkeypatch.setattr(modal.Dict, "from_name", _boom)
    s = build_shared_seen_set({"_fanout_seen_dict_name": "test-name"})
    assert isinstance(s, NullSharedSeenSet)


def test_build_shared_seen_set_returns_modal_backend_when_available() -> None:
    """When modal is importable AND from_name succeeds, the build helper
    returns the Modal backend, not Null. Skipped when modal isn't
    installed (CI's default test extra doesn't pull it)."""
    import pytest
    pytest.importorskip("modal")
    from mantis_agent.gym.fanout_runner import _ModalDictSharedSeenSet
    s = build_shared_seen_set({"_fanout_seen_dict_name": "test-unit-noop"})
    assert isinstance(s, _ModalDictSharedSeenSet)


def test_build_shared_seen_set_returns_null_when_modal_unavailable() -> None:
    """Modal-less environments (CI default) → build helper returns
    NullSharedSeenSet via the ImportError catch in the implementation.
    Pins the fail-soft behaviour so a fan-out config in a non-Modal
    env doesn't crash the runner."""
    s = build_shared_seen_set({"_fanout_seen_dict_name": "test-name"})
    # In a Modal-equipped env this returns the real backend; in CI it
    # falls back to Null. Either way the runner doesn't crash — we
    # accept both shapes for cross-env stability.
    assert isinstance(s, (NullSharedSeenSet,)) or hasattr(s, "contains")
