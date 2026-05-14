"""Self-healing audit log helpers — epic #377 Phase C.

Tests the producer surface: each ``record_*`` helper appends a
canonical-shape dict to ``runner._healing_events``. The consumer
surface (``snapshot()`` + ``build_micro_result`` integration) is
covered in ``test_server_utils.py``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from mantis_agent.gym import healing_events


# ── append-side: each helper writes a canonical dict ────────────────────


def test_record_rewrite_appends_canonical_dict() -> None:
    runner = SimpleNamespace(_healing_events=[])
    healing_events.record_rewrite(
        runner, step_index=3,
        from_intent="Scroll to reveal title", to_intent="Scroll down by one viewport",
        source="intent_rewriter", failure_class="brain_loop_exhausted",
    )
    assert len(runner._healing_events) == 1
    ev = runner._healing_events[0]
    assert ev["kind"] == "rewrite"
    assert ev["step_index"] == 3
    assert ev["source"] == "intent_rewriter"
    assert ev["from_intent"] == "Scroll to reveal title"
    assert ev["to_intent"] == "Scroll down by one viewport"
    assert ev["failure_class"] == "brain_loop_exhausted"
    assert "at" in ev


def test_record_demotion_appends_canonical_dict() -> None:
    runner = SimpleNamespace(_healing_events=[])
    healing_events.record_demotion(
        runner, step_index=0, step_type="click",
        reason="env URL unchanged", source="demote_click",
    )
    ev = runner._healing_events[0]
    assert ev["kind"] == "demotion"
    assert ev["source"] == "demote_click"
    assert ev["step_type"] == "click"


def test_record_handler_escalation_appends_canonical_dict() -> None:
    runner = SimpleNamespace(_healing_events=[])
    healing_events.record_handler_escalation(
        runner, step_index=5,
        from_handler="default", to_handler="holo3",
        trigger="2x_no_state_change",
    )
    ev = runner._healing_events[0]
    assert ev["kind"] == "handler_escalation"
    assert ev["from_handler"] == "default"
    assert ev["to_handler"] == "holo3"


def test_record_insert_step_appends_canonical_dict() -> None:
    runner = SimpleNamespace(_healing_events=[])
    healing_events.record_insert_step(
        runner, after_step_index=6,
        inserted_intent="Navigate to https://example.com/discover",
        inserted_type="navigate",
        reason="navigate_back hit brain_loop_exhausted",
    )
    ev = runner._healing_events[0]
    assert ev["kind"] == "insert_step"
    assert ev["source"] == "critic"
    assert ev["inserted_type"] == "navigate"


# ── multiple events accumulate in order ─────────────────────────────────


def test_events_accumulate_in_emission_order() -> None:
    runner = SimpleNamespace(_healing_events=[])
    healing_events.record_demotion(
        runner, step_index=1, step_type="click", reason="r1",
    )
    healing_events.record_handler_escalation(
        runner, step_index=1, from_handler="default",
        to_handler="holo3", trigger="2x_no_state_change",
    )
    healing_events.record_rewrite(
        runner, step_index=1, from_intent="X", to_intent="Y",
        source="intent_rewriter",
    )
    kinds = [e["kind"] for e in runner._healing_events]
    assert kinds == ["demotion", "handler_escalation", "rewrite"]


# ── defensive: runner without the attr, malformed inputs ────────────────


def test_record_creates_attr_when_missing() -> None:
    """First record on a runner that lacks ``_healing_events``
    should create the list, not raise."""
    runner = SimpleNamespace()  # no _healing_events
    healing_events.record_rewrite(
        runner, step_index=0, from_intent="a", to_intent="b",
        source="intent_rewriter",
    )
    assert isinstance(runner._healing_events, list)
    assert len(runner._healing_events) == 1


def test_string_fields_clamp_at_200_chars() -> None:
    """Intent strings are clamped at 200 chars so result.json stays
    reasonable on noisy LLM outputs."""
    runner = SimpleNamespace(_healing_events=[])
    long_intent = "x" * 500
    healing_events.record_rewrite(
        runner, step_index=0, from_intent=long_intent, to_intent=long_intent,
        source="intent_rewriter",
    )
    ev = runner._healing_events[0]
    assert len(ev["from_intent"]) == 200
    assert len(ev["to_intent"]) == 200


# ── snapshot() ───────────────────────────────────────────────────────────


def test_snapshot_returns_list_copy() -> None:
    runner = SimpleNamespace(_healing_events=[])
    healing_events.record_demotion(
        runner, step_index=0, step_type="click", reason="r",
    )
    snap = healing_events.snapshot(runner)
    assert snap == runner._healing_events
    # Mutating the snapshot must not affect the runner's log (and v.v.).
    snap[0]["kind"] = "MUTATED"
    assert runner._healing_events[0]["kind"] == "demotion"


def test_snapshot_handles_missing_attr() -> None:
    assert healing_events.snapshot(SimpleNamespace()) == []


def test_snapshot_handles_magicmock_runner() -> None:
    """Tests / hosts that stub the runner with MagicMock get a
    Mock for any attribute access. snapshot must return [] then,
    not splice Mock repr into result.json."""
    assert healing_events.snapshot(MagicMock()) == []
