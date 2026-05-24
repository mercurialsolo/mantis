"""#644 — parallelize Phase-1 across M page-slice workers.

Pins the contracts for:
  - ``partition_pages``: round-robin distribution of pages 1..N across
    M workers; empty chunks dropped.
  - ``prepare_phase1_partitions``: returns M sub-suites, each with the
    proper navigate-collect chain for its assigned pages.
  - ``dedup_urls_across_workers``: order-preserving cross-worker URL
    merge.
  - Backward compat: ``prepare_phase1_suite(max_pages=N)`` still emits
    the same steps as before (no ``pages`` arg).
"""

from __future__ import annotations

from mantis_agent.gym.fanout_runner import (
    dedup_urls_across_workers,
    find_url_collect_group,
    partition_pages,
    prepare_phase1_partitions,
    prepare_phase1_suite,
)
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan, PlanDecomposer


def _suite() -> dict:
    """Boattrader-shaped suite with a parallelizable_url_collect group."""
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


# ── partition_pages ───────────────────────────────────────────────────


def test_partition_pages_round_robin_distribution():
    """6 pages, 3 workers → round-robin assignment."""
    chunks = partition_pages(6, 3)
    assert chunks == [[1, 4], [2, 5], [3, 6]]


def test_partition_pages_two_workers_six_pages():
    """6 pages, 2 workers → odd pages to w1, even to w2."""
    chunks = partition_pages(6, 2)
    assert chunks == [[1, 3, 5], [2, 4, 6]]


def test_partition_pages_workers_exceed_pages_caps_to_pages():
    """Asking for 10 workers when there are only 3 pages → 3 chunks,
    one page each. We don't spawn idle Phase-1 workers."""
    chunks = partition_pages(3, 10)
    assert chunks == [[1], [2], [3]]


def test_partition_pages_one_worker_returns_full_range():
    chunks = partition_pages(5, 1)
    assert chunks == [[1, 2, 3, 4, 5]]


def test_partition_pages_clamps_zero_workers_to_one():
    chunks = partition_pages(3, 0)
    assert chunks == [[1, 2, 3]]


def test_partition_pages_clamps_zero_pages_to_one():
    chunks = partition_pages(0, 3)
    # min(1, 3) → 1 worker, walking page 1
    assert chunks == [[1]]


# ── prepare_phase1_suite backward compat ──────────────────────────────


def test_prepare_phase1_suite_default_max_pages_one_unchanged():
    suite = _suite()
    g = find_url_collect_group(suite)
    phase1 = prepare_phase1_suite(suite, g)
    types = [s["type"] for s in phase1["_micro_plan"]]
    assert types == ["navigate", "collect_urls"]


def test_prepare_phase1_suite_max_pages_three_unchanged():
    suite = _suite()
    g = find_url_collect_group(suite)
    phase1 = prepare_phase1_suite(suite, g, max_pages=3)
    types = [s["type"] for s in phase1["_micro_plan"]]
    assert types == [
        "navigate", "collect_urls",      # page 1 (setup + collect)
        "navigate", "collect_urls",      # page 2
        "navigate", "collect_urls",      # page 3
    ]
    # All page-2/3 navigates have the proper synthesised URL.
    nav_steps = [
        s for s in phase1["_micro_plan"][2:] if s["type"] == "navigate"
    ]
    assert nav_steps[0]["params"]["url"].endswith("/page-2/")
    assert nav_steps[1]["params"]["url"].endswith("/page-3/")


# ── prepare_phase1_suite with explicit pages ──────────────────────────


def test_prepare_phase1_suite_pages_subset_skips_setup_collect():
    """Worker assigned pages=[2, 4] should NOT collect on the setup
    URL (page 1) — only the assigned pages."""
    suite = _suite()
    g = find_url_collect_group(suite)
    phase1 = prepare_phase1_suite(suite, g, pages=[2, 4])
    types = [s["type"] for s in phase1["_micro_plan"]]
    # setup navigate + (navigate page2 + collect) + (navigate page4 + collect)
    assert types == [
        "navigate",                      # setup
        "navigate", "collect_urls",      # page 2
        "navigate", "collect_urls",      # page 4
    ]
    # Verify URLs synthesized correctly.
    nav_urls = [
        s["params"]["url"] for s in phase1["_micro_plan"]
        if s["type"] == "navigate"
    ]
    assert nav_urls[0] == "https://x.com/listings"  # setup
    assert nav_urls[1].endswith("/page-2/")
    assert nav_urls[2].endswith("/page-4/")


def test_prepare_phase1_suite_pages_includes_one_keeps_setup_collect():
    """Worker assigned pages=[1, 3] should collect on the setup URL
    + navigate to page 3."""
    suite = _suite()
    g = find_url_collect_group(suite)
    phase1 = prepare_phase1_suite(suite, g, pages=[1, 3])
    types = [s["type"] for s in phase1["_micro_plan"]]
    assert types == [
        "navigate", "collect_urls",      # page 1 (setup + collect)
        "navigate", "collect_urls",      # page 3
    ]


def test_prepare_phase1_suite_pages_sorts_internally():
    """Worker assigned pages=[3, 1] should walk in page order — pages
    are sorted internally so the chain is always ascending."""
    suite = _suite()
    g = find_url_collect_group(suite)
    phase1 = prepare_phase1_suite(suite, g, pages=[3, 1])
    nav_urls = [
        s["params"]["url"] for s in phase1["_micro_plan"]
        if s["type"] == "navigate"
    ]
    # setup, then page 3 (because pages sorted to [1, 3] and 1 is the setup).
    assert nav_urls == ["https://x.com/listings", "https://x.com/listings/page-3/"]


def test_prepare_phase1_suite_tags_page_set_metadata():
    suite = _suite()
    g = find_url_collect_group(suite)
    phase1 = prepare_phase1_suite(suite, g, pages=[2, 5])
    assert phase1["_fanout_phase1_page_set"] == [2, 5]
    # Length of the page set, not max(pages) — the operator wants to
    # know "how many pages this worker walked", not "what max-page was
    # asked for".
    assert phase1["_fanout_phase1_pages"] == 2


# ── prepare_phase1_partitions ────────────────────────────────────────


def test_prepare_phase1_partitions_emits_one_suite_per_worker():
    suite = _suite()
    g = find_url_collect_group(suite)
    subs = prepare_phase1_partitions(
        suite, g, n_workers=3, max_pages=6,
    )
    assert len(subs) == 3
    # Each worker gets a session_name suffix _w1 / _w2 / _w3.
    names = [s["session_name"] for s in subs]
    assert all(
        names[i].endswith(f"_phase1_w{i + 1}") for i in range(len(names))
    )


def test_prepare_phase1_partitions_assigns_distinct_pages():
    suite = _suite()
    g = find_url_collect_group(suite)
    subs = prepare_phase1_partitions(
        suite, g, n_workers=3, max_pages=6,
    )
    page_sets = [s["_fanout_phase1_page_set"] for s in subs]
    assert page_sets == [[1, 4], [2, 5], [3, 6]]


def test_prepare_phase1_partitions_drops_empty_chunks():
    """3 workers, 2 pages → only 2 sub-suites (we don't spawn idle
    Phase-1 workers)."""
    suite = _suite()
    g = find_url_collect_group(suite)
    subs = prepare_phase1_partitions(
        suite, g, n_workers=3, max_pages=2,
    )
    assert len(subs) == 2


def test_prepare_phase1_partitions_single_worker_matches_serial():
    """n_workers=1 → returns one suite whose plan-step shape matches
    the serial ``prepare_phase1_suite(max_pages=N)`` output."""
    suite = _suite()
    g = find_url_collect_group(suite)
    serial = prepare_phase1_suite(
        suite, g, max_pages=3,
    )
    parallel = prepare_phase1_partitions(
        suite, g, n_workers=1, max_pages=3,
    )
    assert len(parallel) == 1
    assert (
        [s["type"] for s in parallel[0]["_micro_plan"]]
        == [s["type"] for s in serial["_micro_plan"]]
    )


def test_prepare_phase1_partitions_passes_worker_index_tag():
    suite = _suite()
    g = find_url_collect_group(suite)
    subs = prepare_phase1_partitions(
        suite, g, n_workers=2, max_pages=4,
    )
    assert subs[0]["_fanout_phase1_worker_index"] == 0
    assert subs[1]["_fanout_phase1_worker_index"] == 1
    assert all(s["_fanout_phase1_worker_count"] == 2 for s in subs)


# ── dedup_urls_across_workers ─────────────────────────────────────────


def test_dedup_urls_across_workers_preserves_first_seen_order():
    per_worker = [
        ["a", "b", "c"],
        ["b", "d"],
        ["a", "e"],
    ]
    assert dedup_urls_across_workers(per_worker) == ["a", "b", "c", "d", "e"]


def test_dedup_urls_across_workers_skips_empty_strings():
    per_worker = [["", "a", ""], ["b", ""]]
    assert dedup_urls_across_workers(per_worker) == ["a", "b"]


def test_dedup_urls_across_workers_handles_no_workers():
    assert dedup_urls_across_workers([]) == []


def test_dedup_urls_across_workers_handles_empty_worker_lists():
    assert dedup_urls_across_workers([[], [], []]) == []
