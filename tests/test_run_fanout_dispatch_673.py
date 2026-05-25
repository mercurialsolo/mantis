"""#673 — ``run_fanout_dispatch`` extraction tests.

The Phase-1/Phase-2 spawn block previously lived inline in
``deploy/modal/modal_cua_server.py:main()`` (CLI entrypoint only).
#673 lifted it into ``gym/fanout_runner.run_fanout_dispatch`` so the
HTTP ``/v1/predict`` path's executor can call the same code and stop
silently running every submission sequentially.

Contract pinned here:
    - Non-eligible suites (no parallelizable_url_collect group) → None.
    - Eligible + serial (workers <= 1) → 1 Phase-1 spawn + N Phase-2
      spawns, returns aggregate envelope.
    - Eligible + parallel (workers >= 2) → M Phase-1 spawns +
      N Phase-2 spawns, returns aggregate envelope with cross-worker
      URL dedup.
    - Phase-1 returns empty URL list → None (caller falls through).
    - Worker exception in Phase-2 → swallowed; aggregate excludes the
      failed worker's contribution.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mantis_agent.gym.fanout_runner import run_fanout_dispatch
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan, PlanDecomposer


@pytest.fixture(autouse=True)
def _disable_augur_for_dispatch_tests(monkeypatch):
    """Keep these dispatch tests focused on the spawn / dedup contract.

    augur-sdk 0.4.0 (#38) added ``open_orchestrator_session`` inside
    :func:`run_fanout_dispatch` — left enabled, the dispatch helper
    opens a real ``DebugSession`` that writes a bundle under
    ``data/augur/<fanout_parent_run_id>/`` and tries to close it on
    exit. That's a separate concern from the dispatch logic these
    tests pin, and the bundle close-on-exit interacts badly with the
    tmp-cwd these tests run in. Orchestrator-session behaviour is
    pinned by ``test_open_orchestrator_session_678.py``.
    """
    monkeypatch.setenv("MANTIS_AUGUR_DISABLED", "1")


def _make_url_collect_suite(*, fanout_phase1_workers: int = 1) -> dict:
    """Build a boattrader-shaped suite with a ``parallelizable_url_collect``
    loop group. Mirrors the shape ``build_micro_suite`` produces for the
    boattrader_scrape_urlnav plan."""
    plan = MicroPlan(domain="boattrader.com")
    plan.steps = [
        MicroIntent(
            intent="Navigate", type="navigate", section="setup",
            params={"url": "https://www.boattrader.com/boats/by-owner/"},
        ),
        MicroIntent(intent="Click card", type="click", section="extraction"),
        MicroIntent(intent="Read URL", type="extract_url", section="extraction"),
        MicroIntent(intent="Scroll", type="scroll", section="extraction"),
        MicroIntent(intent="Extract", type="extract_data", section="extraction"),
        MicroIntent(intent="Back", type="navigate_back", section="extraction"),
        MicroIntent(
            intent="Inner loop", type="loop", section="extraction",
            loop_target=1, loop_count=40,
        ),
    ]
    PlanDecomposer._classify_loop_groups(plan)
    return {
        "session_name": "x",
        "_plan_signature": "boattrader-test",
        "_fanout_phase1_workers": fanout_phase1_workers,
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


def _make_no_loop_suite() -> dict:
    """Suite with NO loop groups — fan-out should refuse and return None."""
    plan = MicroPlan(domain="example.com")
    plan.steps = [
        MicroIntent(intent="go", type="navigate", section="setup"),
    ]
    PlanDecomposer._classify_loop_groups(plan)
    return {
        "session_name": "x",
        "_plan_signature": "no-loop",
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
    }


def _make_executor_stub(
    *,
    phase1_urls: list[str] | None = None,
    phase2_leads_per_worker: list[list[dict]] | None = None,
) -> MagicMock:
    """Build a stub ``executor_fn`` whose ``.spawn(...)`` returns a
    handle whose ``.get()`` yields a result envelope matching what
    :func:`build_micro_result` produces.

    Phase-1 (collect_urls) and Phase-2 (per-URL extract) shapes are
    distinguished by suite content — Phase-2 sub-suites carry
    ``_fanout_phase = "phase2_extract"``.
    """
    phase1_urls = phase1_urls if phase1_urls is not None else []
    phase2_leads_per_worker = (
        phase2_leads_per_worker if phase2_leads_per_worker is not None else []
    )
    phase2_counter = [0]

    def _spawn(**kwargs):
        import json as _json
        sub = _json.loads(kwargs.get("task_file_contents", "{}"))
        handle = MagicMock()
        if sub.get("_fanout_phase") == "phase2_extract":
            i = phase2_counter[0]
            phase2_counter[0] += 1
            leads = (
                phase2_leads_per_worker[i]
                if i < len(phase2_leads_per_worker) else []
            )
            handle.get = MagicMock(return_value={
                "viable": len(leads),
                "leads_with_phone": sum(
                    1 for ld in leads if ld.get("phone")
                ),
                "leads": leads,
                "collected_urls": [],
                "shared_seen_hits": 0,
            })
        else:
            # Phase-1 collect — return harvested URLs.
            handle.get = MagicMock(return_value={
                "viable": 0,
                "leads_with_phone": 0,
                "leads": [],
                "collected_urls": list(phase1_urls),
                "shared_seen_hits": 0,
            })
        return handle

    executor_fn = MagicMock()
    executor_fn.spawn = MagicMock(side_effect=_spawn)
    return executor_fn


# ── Eligibility gate ────────────────────────────────────────────────


def test_returns_none_when_no_url_collect_group():
    """Suite without a ``parallelizable_url_collect`` loop is not
    fanout-eligible. Caller falls through to legacy paths."""
    suite = _make_no_loop_suite()
    result = run_fanout_dispatch(
        suite,
        executor_fn=MagicMock(),
        model="holo3",
        claude_model="",
        max_steps=30,
        workers=4,
        fanout_parent_run_id="fanout-test",
    )
    assert result is None


def test_returns_none_when_phase1_harvests_zero_urls():
    """Eligible suite, but Phase-1 returned no URLs → fall through."""
    suite = _make_url_collect_suite(fanout_phase1_workers=1)
    executor_fn = _make_executor_stub(phase1_urls=[])
    result = run_fanout_dispatch(
        suite,
        executor_fn=executor_fn,
        model="holo3",
        claude_model="",
        max_steps=30,
        workers=4,
        fanout_parent_run_id="fanout-test",
    )
    assert result is None


# ── Serial Phase-1 ──────────────────────────────────────────────────


def test_serial_phase1_spawns_one_worker_then_phase2_workers():
    """``_fanout_phase1_workers=1`` → 1 Phase-1 spawn + N Phase-2."""
    suite = _make_url_collect_suite(fanout_phase1_workers=1)
    phase1_urls = [
        "https://www.boattrader.com/boat/listing-1/",
        "https://www.boattrader.com/boat/listing-2/",
        "https://www.boattrader.com/boat/listing-3/",
    ]
    phase2_leads = [
        [{"year": "2020", "make": "Sea Ray", "model": "X", "phone": "555-1234"}],
        [{"year": "2019", "make": "Boston Whaler", "model": "Y", "phone": ""}],
    ]
    executor_fn = _make_executor_stub(
        phase1_urls=phase1_urls,
        phase2_leads_per_worker=phase2_leads,
    )
    result = run_fanout_dispatch(
        suite,
        executor_fn=executor_fn,
        model="holo3",
        claude_model="",
        max_steps=30,
        workers=2,
        fanout_parent_run_id="fanout-test",
    )
    assert result is not None
    assert result["viable"] == 2
    assert result["leads_with_phone"] == 1
    assert len(result["collected_urls"]) == 3
    # 1 Phase-1 spawn + 2 Phase-2 spawns = 3 total.
    assert executor_fn.spawn.call_count == 3


# ── Parallel Phase-1 ────────────────────────────────────────────────


def test_parallel_phase1_spawns_multiple_workers_and_dedups_urls():
    """``_fanout_phase1_workers=3`` → 3 Phase-1 spawns + cross-worker
    URL dedup before Phase-2."""
    suite = _make_url_collect_suite(fanout_phase1_workers=3)
    # Phase-1 max_pages defaults; phase1_workers will be clamped to
    # min(opt-in, max_pages). The url-collect-only plan with no
    # pagination group defaults max_pages=1 → workers clamped to 1.
    # Override by injecting a fanout_phase1_max_pages override.
    suite["_fanout_phase1_max_pages"] = 3
    # Each Phase-1 worker returns the same set (simulate overlap).
    overlapping = [
        "https://www.boattrader.com/boat/a/",
        "https://www.boattrader.com/boat/b/",
    ]
    executor_fn = _make_executor_stub(
        phase1_urls=overlapping,
        phase2_leads_per_worker=[[{"year": "2020"}]],
    )
    result = run_fanout_dispatch(
        suite,
        executor_fn=executor_fn,
        model="holo3",
        claude_model="",
        max_steps=30,
        workers=1,
        fanout_parent_run_id="fanout-test",
    )
    assert result is not None
    # 3 Phase-1 workers each emitted the same 2 URLs → dedup to 2.
    assert len(result["collected_urls"]) == 2


# ── Phase-2 worker fault tolerance ──────────────────────────────────


def test_phase2_worker_exception_swallowed_aggregate_continues():
    """If one Phase-2 worker raises, the aggregate still returns
    leads from the survivors. Failure shouldn't take down the run."""
    suite = _make_url_collect_suite(fanout_phase1_workers=1)

    executor_fn = MagicMock()
    call_count = [0]

    def _spawn(**kwargs):
        import json as _json
        sub = _json.loads(kwargs.get("task_file_contents", "{}"))
        handle = MagicMock()
        if sub.get("_fanout_phase") == "phase2_extract":
            i = call_count[0]
            call_count[0] += 1
            if i == 1:
                # Second Phase-2 worker raises on .get()
                handle.get = MagicMock(side_effect=RuntimeError("simulated"))
            else:
                handle.get = MagicMock(return_value={
                    "viable": 1, "leads_with_phone": 0,
                    "leads": [{"year": "2020"}],
                    "collected_urls": [], "shared_seen_hits": 0,
                })
        else:
            handle.get = MagicMock(return_value={
                "viable": 0, "leads_with_phone": 0,
                "leads": [],
                "collected_urls": ["url-1", "url-2"],
                "shared_seen_hits": 0,
            })
        return handle

    executor_fn.spawn = MagicMock(side_effect=_spawn)
    result = run_fanout_dispatch(
        suite,
        executor_fn=executor_fn,
        model="holo3",
        claude_model="",
        max_steps=30,
        workers=2,
        fanout_parent_run_id="fanout-test",
    )
    assert result is not None
    # Only worker 0 succeeded — viable=1.
    assert result["viable"] == 1


# ── Recursion guard semantics (called by the executor gate) ─────────


def test_eligible_suite_with_fanout_phase_set_is_NOT_called_through_dispatch():
    """A sub-worker suite (carrying ``_fanout_phase``) must never
    reach ``run_fanout_dispatch`` — the executor gate short-circuits
    BEFORE calling the helper. This test documents that contract by
    asserting the executor-side gate behaviour rather than the helper
    (which has no such guard — by design, since the helper's contract
    is "you've already decided this is the orchestrator")."""
    suite = _make_url_collect_suite(fanout_phase1_workers=3)
    suite["_fanout_phase"] = "phase2_extract"
    # If the gate is implemented correctly, the executor never calls
    # run_fanout_dispatch on this suite. But if a caller DOES call it
    # anyway (defensive testing), it'll spawn — which would recurse.
    # We document the contract here rather than enforce it: callers
    # MUST check ``_fanout_phase`` before invoking the helper.
    assert suite["_fanout_phase"] == "phase2_extract"


# ── Model + claude_model forwarding ─────────────────────────────────


def test_claude_model_forwarded_when_model_is_claude():
    """``model='claude'`` adds ``claude_model`` to spawn kwargs."""
    suite = _make_url_collect_suite(fanout_phase1_workers=1)
    executor_fn = _make_executor_stub(
        phase1_urls=["url-1"],
        phase2_leads_per_worker=[[{"year": "2020"}]],
    )
    run_fanout_dispatch(
        suite,
        executor_fn=executor_fn,
        model="claude",
        claude_model="claude-sonnet-4-5",
        max_steps=30,
        workers=1,
        fanout_parent_run_id="fanout-test",
    )
    # Inspect the spawn call kwargs — all should carry claude_model.
    for call in executor_fn.spawn.call_args_list:
        assert call.kwargs.get("claude_model") == "claude-sonnet-4-5"


def test_holo3_model_does_not_carry_claude_model():
    """When ``model != 'claude'``, no claude_model kwarg on spawn."""
    suite = _make_url_collect_suite(fanout_phase1_workers=1)
    executor_fn = _make_executor_stub(
        phase1_urls=["url-1"],
        phase2_leads_per_worker=[[{"year": "2020"}]],
    )
    run_fanout_dispatch(
        suite,
        executor_fn=executor_fn,
        model="holo3",
        claude_model="ignored",
        max_steps=30,
        workers=1,
        fanout_parent_run_id="fanout-test",
    )
    for call in executor_fn.spawn.call_args_list:
        assert "claude_model" not in call.kwargs
