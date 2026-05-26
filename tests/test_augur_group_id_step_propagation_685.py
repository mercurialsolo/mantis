"""#685 — verify ``step.group_id`` propagates from session to step traces.

augur-sdk 0.6.0's contract: when ``DebugSession(group_id=X)`` is set
at session open, the SDK auto-stamps ``step.group_id = X`` on every
recorded step trace. PR #681 (closes #680) wired the mantis side:

* Orchestrator stamps ``_fanout_group_id`` on the suite envelope.
* Modal worker entrypoint pulls it back onto
  ``MicroRunner._fanout_group_id``.
* ``RunExecutor`` forwards as ``AugurAdapter(group_id=...)``.
* ``AugurAdapter.__init__`` forwards to ``DebugSession(group_id=...)``.
* SDK's ``record_step`` reads ``self._group_id`` and stamps
  ``step["group_id"]`` (augur-sdk 0.6.0+ line 451-452).

This test pins the end-to-end: open AugurAdapter with group_id,
record a step, close, read the bundle, assert step.group_id ==
expected.

Without this pinned, an SDK change (e.g. moving the propagation to
``set_group_id()`` only, removing the constructor-time propagation,
renaming the schema field) would silently break GRPO sibling
correlation in ``/runs/sample`` without any mantis-side regression
test catching it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from mantis_agent.observability.augur import AugurAdapter


def _stub_step_result(*, step_index: int = 0):
    return SimpleNamespace(
        step_index=step_index, intent=f"step{step_index}",
        success=True,
        verdict=SimpleNamespace(kind="ok", reason="", confidence=1.0),
        data="", failure_class="", last_action=None, duration=0.0,
        page_title="", executor_backend="som", reasoning="",
    )


def test_group_id_lands_on_step_trace():
    """End-to-end: open adapter with group_id → record_step →
    close → bundle's step trace carries the group_id."""
    with tempfile.TemporaryDirectory() as tmp:
        a = AugurAdapter(
            run_id="685_e2e", tenant_id="t", session_name="s",
            out_dir=tmp, group_id="fanout-parent-685",
        )
        if not a.active:
            pytest.skip("AugurAdapter inactive — SDK not installed")
        a.record_step(step_result=_stub_step_result(step_index=0))
        a.close()

        trace = json.loads((Path(tmp) / "trace.json").read_text())
        steps = trace.get("steps", [])
        assert steps, "No steps in trace.json"
        # SDK auto-propagates session.group_id → every step.
        assert steps[0]["group_id"] == "fanout-parent-685"


def test_group_id_omitted_when_session_opened_without_it():
    """No ``group_id=`` on the adapter → step trace doesn't carry
    one. Confirms the propagation is conditional (production non-
    fanout runs don't pollute the bundle with a synthetic id)."""
    with tempfile.TemporaryDirectory() as tmp:
        a = AugurAdapter(
            run_id="685_no_group", tenant_id="t", session_name="s",
            out_dir=tmp,  # no group_id
        )
        if not a.active:
            pytest.skip("AugurAdapter inactive")
        a.record_step(step_result=_stub_step_result(step_index=0))
        a.close()

        trace = json.loads((Path(tmp) / "trace.json").read_text())
        steps = trace.get("steps", [])
        assert steps, "No steps in trace.json"
        assert steps[0].get("group_id") is None


def test_group_id_propagates_across_multiple_steps():
    """Every step carries the same group_id — not just the first
    (the SDK's auto-stamp fires per record_step call)."""
    with tempfile.TemporaryDirectory() as tmp:
        a = AugurAdapter(
            run_id="685_multi", tenant_id="t", session_name="s",
            out_dir=tmp, group_id="fanout-parent-685",
        )
        if not a.active:
            pytest.skip("AugurAdapter inactive")
        for i in range(3):
            a.record_step(step_result=_stub_step_result(step_index=i))
        a.close()

        trace = json.loads((Path(tmp) / "trace.json").read_text())
        steps = trace.get("steps", [])
        assert len(steps) >= 3
        group_ids = [s.get("group_id") for s in steps[:3]]
        assert group_ids == ["fanout-parent-685"] * 3
