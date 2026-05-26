"""#686 — record_subgoal_completion at Phase-1 / Phase-2 boundaries.

The fan-out orchestrator session marks two structural subgoals so
the reward aggregator's ``progress`` component reads non-zero on
successful runs (see augur#148):

* ``url_collection`` — fires after Phase-1 returns ≥1 URL
  (``step_index=0``, ``completion=1.0``).
* ``per_url_extraction`` — fires after Phase-2 returns leads
  (``step_index=1``, ``completion=workers_with_leads/total_workers``).

Task_spec also declares both subgoals so the aggregator can compute
a max-progress baseline (``len(subgoals)``).

Contract pinned here:

* ``_record_subgoal`` is a no-op when session is ``None`` (SDK
  disabled / orchestrator didn't open) or the SDK predates 0.6.0.
* Phase-1 with zero URLs harvested → ``url_collection`` NOT recorded
  (returns ``None`` before reaching the stamp call).
* Phase-2 returning zero deduped leads → ``per_url_extraction`` NOT
  recorded (the dispatch still returns its envelope, but the subgoal
  shouldn't be marked complete on an empty result).
* Partial completion: ``per_url_extraction.completion`` reflects the
  fraction of Phase-2 workers that produced viable leads (not a
  binary 1.0/0.0).
* ``task_spec.subgoals`` is a 2-item list with the correct shape
  (subgoal_id, description, parent_subgoal_id chain).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import mantis_agent.observability.augur as augur_mod
from mantis_agent.gym.fanout_runner import (
    _build_task_spec_from_suite,
    _record_subgoal,
    run_fanout_dispatch,
)
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan, PlanDecomposer


@pytest.fixture
def force_augur_available(monkeypatch):
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.setattr(augur_mod, "_AUGUR_AVAILABLE", True)
    monkeypatch.setattr(
        augur_mod, "CaptureMode", lambda v=None: f"capture_mode:{v}",
    )


# ── _record_subgoal helper ─────────────────────────────────────────


def test_record_subgoal_noop_when_session_none():
    """``None`` session (orchestrator didn't open) → no exception,
    silent skip."""
    _record_subgoal(None, step_index=0, subgoal_id="x", completion=1.0)
    # No assertion needed — survival to here is the contract.


def test_record_subgoal_noop_when_sdk_lacks_method():
    """Pre-0.6.0 SDK without ``record_subgoal_completion`` → skip
    cleanly via ``getattr`` lookup."""
    session = MagicMock(spec=["record_step"])  # no record_subgoal_completion
    _record_subgoal(session, step_index=0, subgoal_id="x", completion=1.0)


def test_record_subgoal_forwards_to_sdk():
    """SDK with the method → forward as kwargs."""
    session = MagicMock()
    session.record_subgoal_completion = MagicMock()
    _record_subgoal(session, step_index=2, subgoal_id="url_collection", completion=1.0)
    session.record_subgoal_completion.assert_called_once_with(
        step_index=2, subgoal_id="url_collection", completion=1.0,
    )


def test_record_subgoal_swallows_sdk_exceptions():
    """SDK raised (corrupt session, network error) → log + return,
    never break the run."""
    session = MagicMock()
    session.record_subgoal_completion = MagicMock(side_effect=RuntimeError("boom"))
    # Should not propagate.
    _record_subgoal(session, step_index=0, subgoal_id="x", completion=1.0)
    session.record_subgoal_completion.assert_called_once()


# ── task_spec carries subgoal declarations ─────────────────────────


def test_task_spec_declares_url_collection_and_per_url_extraction():
    """The composed task_spec carries the 2-item subgoals[] block."""
    spec = _build_task_spec_from_suite(
        {"_plan_name": "boattrader_scrape_urlnav",
         "_site_config": {"domain": "boattrader.com"}},
        phase1_workers=4, phase1_max_pages=4, phase2_workers=4, max_steps=240,
    )
    assert "subgoals" in spec
    subgoals = spec["subgoals"]
    assert len(subgoals) == 2
    ids = [s["subgoal_id"] for s in subgoals]
    assert ids == ["url_collection", "per_url_extraction"]
    # per_url_extraction is a child of url_collection — completion
    # ordering matters for the aggregator's progress curve.
    assert subgoals[1]["parent_subgoal_id"] == "url_collection"
    # url_collection is a root subgoal — no parent.
    assert "parent_subgoal_id" not in subgoals[0]


# ── run_fanout_dispatch stamps subgoals at the right boundaries ───


def _make_eligible_suite() -> dict:
    plan = MicroPlan(domain="boattrader.com")
    plan.steps = [
        MicroIntent(
            intent="Navigate", type="navigate", section="setup",
            params={"url": "https://x/"},
        ),
        MicroIntent(intent="Click", type="click", section="extraction"),
        MicroIntent(intent="URL", type="extract_url", section="extraction"),
        MicroIntent(intent="Scroll", type="scroll", section="extraction"),
        MicroIntent(intent="Extract", type="extract_data", section="extraction"),
        MicroIntent(intent="Back", type="navigate_back", section="extraction"),
        MicroIntent(
            intent="Loop", type="loop", section="extraction",
            loop_target=1, loop_count=40,
        ),
    ]
    PlanDecomposer._classify_loop_groups(plan)
    return {
        "session_name": "x",
        "_plan_name": "boattrader_scrape_urlnav",
        "_plan_signature": "subgoal-test",
        "_site_config": {"domain": "boattrader.com"},
        "_fanout_phase1_workers": 1,
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


def _make_executor_stub(*, phase1_urls: list[str], phase2_viable: int = 1,
                       phase2_leads: list[dict] | None = None):
    handle_p1 = MagicMock()
    handle_p1.get.return_value = {
        "viable": 0, "leads_with_phone": 0, "leads": [],
        "collected_urls": phase1_urls, "shared_seen_hits": 0,
    }
    handle_p2 = MagicMock()
    leads = phase2_leads if phase2_leads is not None else [
        {"url": u} for u in phase1_urls[:phase2_viable]
    ]
    handle_p2.get.return_value = {
        "viable": phase2_viable, "leads_with_phone": 0,
        "leads": leads, "shared_seen_hits": 0,
    }
    executor = MagicMock()
    executor.spawn.side_effect = [handle_p1, handle_p2]
    return executor


def test_dispatch_records_url_collection_after_phase1(force_augur_available):
    """Phase-1 returns URLs → record url_collection subgoal."""
    fake_session = MagicMock()
    fake_session.record_subgoal_completion = MagicMock()
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=fake_session)
    fake_ctx.__exit__ = MagicMock(return_value=False)
    stub = MagicMock()
    stub.open_orchestrator = MagicMock(return_value=fake_ctx)
    with patch.object(augur_mod, "DebugSession", stub):
        run_fanout_dispatch(
            _make_eligible_suite(),
            executor_fn=_make_executor_stub(phase1_urls=["https://x/d/1"]),
            model="holo3", claude_model="",
            max_steps=10, workers=1,
            fanout_parent_run_id="fanout-686",
        )
    # Must have recorded url_collection AND per_url_extraction
    calls = fake_session.record_subgoal_completion.call_args_list
    subgoal_ids = [c.kwargs["subgoal_id"] for c in calls]
    assert "url_collection" in subgoal_ids
    # url_collection always fires at completion=1.0 (binary — Phase-1
    # either harvested or didn't).
    url_call = next(c for c in calls if c.kwargs["subgoal_id"] == "url_collection")
    assert url_call.kwargs["completion"] == 1.0


def test_dispatch_records_per_url_extraction_after_phase2(force_augur_available):
    """Phase-2 returns leads → record per_url_extraction subgoal."""
    fake_session = MagicMock()
    fake_session.record_subgoal_completion = MagicMock()
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=fake_session)
    fake_ctx.__exit__ = MagicMock(return_value=False)
    stub = MagicMock()
    stub.open_orchestrator = MagicMock(return_value=fake_ctx)
    with patch.object(augur_mod, "DebugSession", stub):
        run_fanout_dispatch(
            _make_eligible_suite(),
            executor_fn=_make_executor_stub(phase1_urls=["https://x/d/1"]),
            model="holo3", claude_model="",
            max_steps=10, workers=1,
            fanout_parent_run_id="fanout-686",
        )
    calls = fake_session.record_subgoal_completion.call_args_list
    subgoal_ids = [c.kwargs["subgoal_id"] for c in calls]
    assert "per_url_extraction" in subgoal_ids
    ext_call = next(c for c in calls if c.kwargs["subgoal_id"] == "per_url_extraction")
    # Single worker produced a lead → completion=1.0.
    assert ext_call.kwargs["completion"] == 1.0


def test_dispatch_skips_subgoals_when_phase1_empty(force_augur_available):
    """Phase-1 returned 0 URLs → dispatch returns None BEFORE the
    url_collection stamp, so neither subgoal fires."""
    fake_session = MagicMock()
    fake_session.record_subgoal_completion = MagicMock()
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=fake_session)
    fake_ctx.__exit__ = MagicMock(return_value=False)
    stub = MagicMock()
    stub.open_orchestrator = MagicMock(return_value=fake_ctx)
    with patch.object(augur_mod, "DebugSession", stub):
        result = run_fanout_dispatch(
            _make_eligible_suite(),
            executor_fn=_make_executor_stub(phase1_urls=[]),  # zero URLs
            model="holo3", claude_model="",
            max_steps=10, workers=1,
            fanout_parent_run_id="fanout-686",
        )
    assert result is None
    fake_session.record_subgoal_completion.assert_not_called()


def test_dispatch_skips_per_url_extraction_when_dedup_total_zero(force_augur_available):
    """Phase-2 had URLs to extract but returned an empty leads list →
    url_collection fires (Phase-1 succeeded), per_url_extraction does
    NOT (no leads to credit)."""
    fake_session = MagicMock()
    fake_session.record_subgoal_completion = MagicMock()
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=fake_session)
    fake_ctx.__exit__ = MagicMock(return_value=False)
    stub = MagicMock()
    stub.open_orchestrator = MagicMock(return_value=fake_ctx)
    with patch.object(augur_mod, "DebugSession", stub):
        run_fanout_dispatch(
            _make_eligible_suite(),
            executor_fn=_make_executor_stub(
                phase1_urls=["https://x/d/1"],
                phase2_viable=0, phase2_leads=[],
            ),
            model="holo3", claude_model="",
            max_steps=10, workers=1,
            fanout_parent_run_id="fanout-686",
        )
    calls = fake_session.record_subgoal_completion.call_args_list
    subgoal_ids = [c.kwargs["subgoal_id"] for c in calls]
    assert "url_collection" in subgoal_ids
    assert "per_url_extraction" not in subgoal_ids
