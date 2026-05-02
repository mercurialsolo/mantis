"""Tests for #115 step 6 — CheckpointManager extracted from MicroPlanRunner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mantis_agent.gym.checkpoint import RunCheckpoint, StepResult
from mantis_agent.gym.checkpoint_manager import CheckpointManager


# ── Plan signature (pure / static) ──────────────────────────────────────


class _FakePlanStep:
    def __init__(self, **kwargs: Any) -> None:
        # Defaults match every field _compute_plan_signature serializes.
        self.intent = kwargs.get("intent", "click button")
        self.type = kwargs.get("type", "click")
        self.verify = kwargs.get("verify", "")
        self.budget = kwargs.get("budget", 5)
        self.reverse = kwargs.get("reverse", False)
        self.grounding = kwargs.get("grounding", False)
        self.claude_only = kwargs.get("claude_only", False)
        self.loop_target = kwargs.get("loop_target", 0)
        self.loop_count = kwargs.get("loop_count", 0)
        self.section = kwargs.get("section", "")
        self.required = kwargs.get("required", False)
        self.gate = kwargs.get("gate", False)


class _FakePlan:
    def __init__(self, *steps: _FakePlanStep) -> None:
        self.steps = list(steps)


def test_plan_signature_is_64_char_hex() -> None:
    sig = CheckpointManager.compute_plan_signature(_FakePlan(_FakePlanStep()))
    assert len(sig) == 64
    assert all(c in "0123456789abcdef" for c in sig)


def test_plan_signature_stable_across_calls() -> None:
    plan = _FakePlan(_FakePlanStep(), _FakePlanStep(type="scroll"))
    a = CheckpointManager.compute_plan_signature(plan)
    b = CheckpointManager.compute_plan_signature(plan)
    assert a == b


def test_plan_signature_changes_when_step_field_changes() -> None:
    a = CheckpointManager.compute_plan_signature(
        _FakePlan(_FakePlanStep(verify="check button"))
    )
    b = CheckpointManager.compute_plan_signature(
        _FakePlan(_FakePlanStep(verify="check submit"))
    )
    assert a != b


def test_plan_signature_ignores_attributes_outside_payload() -> None:
    """Adding an unrelated attribute to a step should not affect the SHA."""
    s1 = _FakePlanStep()
    s2 = _FakePlanStep()
    s2.unrelated_attr = "extra"
    a = CheckpointManager.compute_plan_signature(_FakePlan(s1))
    b = CheckpointManager.compute_plan_signature(_FakePlan(s2))
    assert a == b


# ── Persist / restore via a fake runner ─────────────────────────────────


class _FakeBrowserState:
    def reentry_url_for_step(self, plan, idx: int) -> str:
        return "https://x.test/reentry"


class _FakeDynamicVerifier:
    def __init__(self) -> None:
        self.loaded: dict | None = None

    def report(self, status: str | None = None) -> dict:
        return {"status": status, "pages": []}

    def load_report(self, report: dict) -> None:
        self.loaded = report


def _make_runner(checkpoint_path: str, on_checkpoint: Any = None) -> Any:
    """Fake runner exposing the surface CheckpointManager reads/writes."""
    runner = type("FakeRunner", (), {})()
    runner.run_key = "rk"
    runner.plan_signature = "sig"
    runner.session_name = "sess"
    runner.checkpoint_path = checkpoint_path
    runner.on_checkpoint = on_checkpoint
    runner.dynamic_verifier = _FakeDynamicVerifier()
    runner._final_status = ""
    runner._current_page = 1
    runner._last_known_url = ""
    runner._seen_urls = set()
    runner._extracted_titles = []
    runner._page_listings = []
    runner._page_listing_index = 0
    runner._viewport_stage = 0
    runner._results_base_url = ""
    runner._required_filter_tokens = ()
    runner._scroll_state = {}
    runner._last_extracted = {}
    runner.costs = {
        "gpu_steps": 0, "gpu_seconds": 0.0,
        "claude_extract": 0, "claude_grounding": 0,
        "proxy_mb": 0.0,
    }
    runner.browser_state = _FakeBrowserState()
    runner._unique_leads_from_results = staticmethod(lambda results: [])
    runner.dynamic_verification_report = lambda status=None: {
        "status": status, "pages": []
    }
    return runner


def test_persist_writes_full_state_to_checkpoint(tmp_path: Path) -> None:
    target = tmp_path / "cp.json"
    runner = _make_runner(str(target))
    runner._current_page = 3
    runner._last_known_url = "https://x.test/page"
    runner._seen_urls = {"https://x.test/a", "https://x.test/b"}
    runner._scroll_state = {"page_downs": 2, "context": "results"}
    runner.costs["gpu_steps"] = 17

    cp = RunCheckpoint()
    mgr = CheckpointManager(runner)
    mgr.persist(
        checkpoint=cp,
        plan=_FakePlan(_FakePlanStep()),
        results=[],
        loop_counters={1: 2, 3: 4},
        listings_on_page=5,
        next_step_index=2,
        status="running",
    )
    # Round-trip through disk to verify the saved JSON is well-formed.
    payload = json.loads(target.read_text())
    assert payload["run_key"] == "rk"
    assert payload["current_page"] == 3
    assert payload["current_url"] == "https://x.test/page"
    assert sorted(payload["seen_urls"]) == ["https://x.test/a", "https://x.test/b"]
    assert payload["scroll_state"] == {"page_downs": 2, "context": "results"}
    assert payload["costs"]["gpu_steps"] == 17
    assert payload["loop_counters"] == {"1": 2, "3": 4}
    assert payload["listings_on_page"] == 5
    assert payload["step_index"] == 2
    assert payload["status"] == "running"
    assert payload["reentry_url"] == "https://x.test/reentry"


def test_persist_invokes_on_checkpoint_hook(tmp_path: Path) -> None:
    target = tmp_path / "cp.json"
    calls: list[None] = []
    runner = _make_runner(str(target), on_checkpoint=lambda: calls.append(None))
    cp = RunCheckpoint()
    CheckpointManager(runner).persist(
        checkpoint=cp, plan=_FakePlan(_FakePlanStep()),
        results=[], loop_counters={}, listings_on_page=0, next_step_index=0,
    )
    assert len(calls) == 1


def test_persist_swallows_on_checkpoint_exception(tmp_path: Path) -> None:
    target = tmp_path / "cp.json"

    def boom() -> None:
        raise RuntimeError("hook failed")

    runner = _make_runner(str(target), on_checkpoint=boom)
    cp = RunCheckpoint()
    # Should NOT raise.
    CheckpointManager(runner).persist(
        checkpoint=cp, plan=_FakePlan(_FakePlanStep()),
        results=[], loop_counters={}, listings_on_page=0, next_step_index=0,
    )
    # Save still happened despite hook failure.
    assert target.exists()


def test_restore_replays_state_attributes() -> None:
    runner = _make_runner("/tmp/unused.json")
    cp = RunCheckpoint(
        seen_urls=["https://x.test/a"],
        extracted_titles=["t1", "t2"],
        page_listings=[[100, 200, "card1"]],
        page_listing_index=3,
        viewport_stage=2,
        results_base_url="https://x.test/results/",
        required_filter_tokens=["price", "owner"],
        current_page=4,
        page=4,
        current_url="https://x.test/p4",
        scroll_state={"page_downs": 5},
        last_extracted={"url": "https://x.test/last"},
        costs={"gpu_steps": 9, "gpu_seconds": 27.0},
        listings_on_page=12,
        step_results=[
            {"step_index": 1, "intent": "click", "success": True}
        ],
        loop_counters={"1": 2, "3": 4},
    )
    results, loop_counters, listings = CheckpointManager(runner).restore(cp)
    assert runner._seen_urls == {"https://x.test/a"}
    assert runner._extracted_titles == ["t1", "t2"]
    assert runner._page_listings == [(100, 200, "card1")]
    assert runner._page_listing_index == 3
    assert runner._viewport_stage == 2
    assert runner._current_page == 4
    assert runner._last_known_url == "https://x.test/p4"
    assert runner._scroll_state == {"page_downs": 5}
    assert runner.costs["gpu_steps"] == 9
    assert listings == 12
    assert len(results) == 1
    assert isinstance(results[0], StepResult)
    assert loop_counters == {1: 2, 3: 4}


def test_save_active_progress_is_noop_when_no_context(tmp_path: Path) -> None:
    target = tmp_path / "cp.json"
    runner = _make_runner(str(target))
    runner._active_checkpoint_context = None
    CheckpointManager(runner).save_active_progress()
    assert not target.exists()


def test_save_active_progress_persists_using_pending_context(tmp_path: Path) -> None:
    target = tmp_path / "cp.json"
    runner = _make_runner(str(target))
    cp = RunCheckpoint()
    runner._active_checkpoint_context = {
        "checkpoint": cp,
        "plan": _FakePlan(_FakePlanStep()),
        "results": [],
        "loop_counters": {},
        "listings_on_page": 0,
        "step_index": 7,
    }
    CheckpointManager(runner).save_active_progress(halt_reason="cancellation")
    payload = json.loads(target.read_text())
    assert payload["status"] == "running"
    assert payload["halt_reason"] == "cancellation"
    assert payload["step_index"] == 7


# ── Backward-compat: MicroPlanRunner shims still resolve ───────────────


def test_runner_shims_delegate_to_checkpoint_manager() -> None:
    from mantis_agent.gym.micro_runner import MicroPlanRunner

    plan = _FakePlan(_FakePlanStep(intent="click X"))
    sig = MicroPlanRunner._compute_plan_signature(plan)
    assert sig == CheckpointManager.compute_plan_signature(plan)
