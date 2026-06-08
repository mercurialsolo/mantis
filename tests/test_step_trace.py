"""Tests for `StepTraceEnvelope` + `StepTraceCollector` + TraceExporter
integration (#783, PR 6)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mantis_agent.gym.step_trace import (
    DispatchRecord,
    GroundingTarget,
    StepTraceCollector,
    StepTraceEnvelope,
    VerifierVerdict,
)


# ── GroundingTarget / DispatchRecord / VerifierVerdict ────────────


def test_grounding_target_serializes_bbox_as_list():
    g = GroundingTarget(description="story title", bbox=(10, 20, 100, 40), confidence=0.92)
    d = g.as_dict()
    assert d["bbox"] == [10, 20, 100, 40]
    assert d["confidence"] == 0.92


def test_grounding_target_handles_no_bbox():
    g = GroundingTarget(description="vague")
    assert g.as_dict()["bbox"] is None


def test_dispatch_record_xdotool_shape():
    d = DispatchRecord(
        kind="xdotool",
        argv=["mousemove", "100", "200", "click", "1"],
        coordinates=(100, 200),
    )
    blob = d.as_dict()
    assert blob["kind"] == "xdotool"
    assert blob["argv"] == ["mousemove", "100", "200", "click", "1"]
    assert blob["coordinates"] == [100, 200]


def test_dispatch_record_browser_use_shape():
    d = DispatchRecord(kind="click", params={"x": 100, "y": 200, "button": "left"})
    blob = d.as_dict()
    assert blob["kind"] == "click"
    assert blob["params"]["x"] == 100


def test_verifier_verdict_default_skipped():
    v = VerifierVerdict()
    assert v.status == "skipped"


def test_verifier_verdict_demoted_carries_reason():
    v = VerifierVerdict(status="demoted", reason="critic disagreed; required=False")
    assert v.as_dict()["status"] == "demoted"
    assert "critic" in v.as_dict()["reason"]


# ── StepTraceEnvelope ────────────────────────────────────────────


def test_envelope_default_is_empty():
    env = StepTraceEnvelope()
    assert env.is_empty() is True


def test_envelope_with_url_is_not_empty():
    env = StepTraceEnvelope(url_before="https://x.com/")
    assert env.is_empty() is False


def test_envelope_with_only_grounding_not_empty():
    env = StepTraceEnvelope(grounding=GroundingTarget(description="x"))
    assert env.is_empty() is False


def test_envelope_serializes_all_optional_blocks():
    env = StepTraceEnvelope(
        url_before="https://hn.com/",
        url_after="https://example.com/article",
        screenshot_pre_path="/run/abc/0001_pre.png",
        screenshot_post_path="/run/abc/0001_post.png",
        grounding=GroundingTarget(description="title link", bbox=(10, 20, 100, 40)),
        dispatch=DispatchRecord(kind="click", coordinates=(50, 30)),
        verifier=VerifierVerdict(status="pass"),
        retry_count=1,
        retry_reason="no_state_change",
        emitted_count=1,
        emitted_sample={"url": "https://example.com/article"},
    )
    d = env.as_dict()
    assert d["url_before"] == "https://hn.com/"
    assert d["grounding"]["description"] == "title link"
    assert d["dispatch"]["kind"] == "click"
    assert d["verifier"]["status"] == "pass"
    assert d["emitted_count"] == 1


# ── StepTraceCollector ───────────────────────────────────────────


def test_collector_record_and_get():
    c = StepTraceCollector(run_id="r1")
    env = StepTraceEnvelope(url_before="https://x.com/")
    c.record(3, env)
    assert c.get(3) is env
    assert c.get(99) is None


def test_collector_update_merges_fields():
    c = StepTraceCollector(run_id="r1")
    c.update(0, url_before="https://hn.com/")
    c.update(0, url_after="https://example.com/")
    c.update(0, retry_count=2, retry_reason="dispatch_5xx")
    env = c.get(0)
    assert env is not None
    assert env.url_before == "https://hn.com/"
    assert env.url_after == "https://example.com/"
    assert env.retry_count == 2
    assert env.retry_reason == "dispatch_5xx"


def test_collector_update_with_unknown_field_is_ignored():
    c = StepTraceCollector(run_id="r1")
    c.update(0, url_before="https://x.com/", nonexistent_field="ignored")
    env = c.get(0)
    assert env is not None
    assert env.url_before == "https://x.com/"
    # Unknown field should not be set as an attribute.
    assert not hasattr(env, "nonexistent_field")


def test_collector_as_dict_skips_empty_envelopes():
    c = StepTraceCollector(run_id="r1")
    c.record(0, StepTraceEnvelope())  # empty
    c.record(1, StepTraceEnvelope(url_before="https://x.com/"))
    out = c.as_dict_by_index()
    assert "0" not in out
    assert "1" in out
    assert out["1"]["url_before"] == "https://x.com/"


def test_collector_as_dict_keys_are_strings():
    c = StepTraceCollector(run_id="r1")
    c.record(5, StepTraceEnvelope(url_before="x"))
    out = c.as_dict_by_index()
    assert "5" in out
    assert 5 not in out  # type: ignore[operator]


# ── TraceExporter integration ────────────────────────────────────


def _make_step_result(step_index: int = 0, intent: str = "x"):
    """Build a minimal `StepResult` shape that `_step_to_dict` accepts."""
    from mantis_agent.gym.checkpoint import StepResult

    return StepResult(
        step_index=step_index,
        intent=intent,
        success=True,
        data="",
    )


def test_exporter_merges_envelope_into_step_block():
    from mantis_agent.gym.trace_exporter import (
        SCHEMA_VERSION,
        TraceExporter,
        _serialize_steps,
    )

    assert SCHEMA_VERSION >= 3
    step = _make_step_result(step_index=0, intent="navigate")
    collector = StepTraceCollector(run_id="r1")
    collector.update(
        0,
        url_before="https://hn.com/",
        url_after="https://hn.com/news",
        dispatch=DispatchRecord(kind="click", coordinates=(50, 30)),
        verifier=VerifierVerdict(status="pass"),
        emitted_count=1,
    )
    out = _serialize_steps([step], collector)
    assert len(out) == 1
    assert "envelope" in out[0]
    assert out[0]["envelope"]["url_before"] == "https://hn.com/"
    assert out[0]["envelope"]["dispatch"]["kind"] == "click"


def test_exporter_omits_envelope_block_when_empty():
    from mantis_agent.gym.trace_exporter import _serialize_steps

    step = _make_step_result(step_index=0, intent="navigate")
    collector = StepTraceCollector(run_id="r1")
    collector.record(0, StepTraceEnvelope())
    out = _serialize_steps([step], collector)
    assert "envelope" not in out[0]


def test_exporter_omits_envelope_block_when_no_collector():
    from mantis_agent.gym.trace_exporter import _serialize_steps

    step = _make_step_result(step_index=0, intent="navigate")
    out = _serialize_steps([step], None)
    assert "envelope" not in out[0]


def test_exporter_round_trip_end_to_end(tmp_path: Path, monkeypatch):
    """Drive `maybe_export` with a step_traces argument; verify the
    JSON on disk carries the envelope block."""
    from mantis_agent.gym.trace_exporter import TraceExporter

    monkeypatch.setenv("MANTIS_TRACE_EXPORT_DIR", str(tmp_path))
    exporter = TraceExporter.from_env()
    assert exporter.enabled is True

    runner = MagicMock()
    runner.tenant_id = "t1"
    runner.run_key = "run-001"
    runner.session_name = "test"
    runner.plan_signature = "abcd"
    runner.shadow_variant = ""
    runner._final_status = "complete"
    runner._run_start = 1000.0
    runner._cost_totals = lambda: (0.0, 0.0, 0.0, 0.0)

    step = _make_step_result(step_index=0, intent="navigate")
    collector = StepTraceCollector(run_id="run-001")
    collector.update(0, url_before="https://hn.com/", emitted_count=3)

    out_path = exporter.maybe_export(
        runner, [step], status="complete", step_traces=collector
    )
    assert out_path is not None
    payload = json.loads(Path(out_path).read_text())
    assert payload["schema_version"] == 3
    assert payload["steps"][0]["envelope"]["url_before"] == "https://hn.com/"
    assert payload["steps"][0]["envelope"]["emitted_count"] == 3


def test_exporter_no_step_traces_kwarg_still_works_legacy(tmp_path: Path, monkeypatch):
    """Backwards compat: legacy callers don't pass step_traces."""
    from mantis_agent.gym.trace_exporter import TraceExporter

    monkeypatch.setenv("MANTIS_TRACE_EXPORT_DIR", str(tmp_path))
    exporter = TraceExporter.from_env()

    runner = MagicMock()
    runner.tenant_id = "t1"
    runner.run_key = "run-002"
    runner.session_name = "legacy"
    runner.plan_signature = ""
    runner.shadow_variant = ""
    runner._final_status = "complete"
    runner._run_start = 1000.0
    runner._cost_totals = lambda: (0.0, 0.0, 0.0, 0.0)

    step = _make_step_result(step_index=0, intent="navigate")
    out_path = exporter.maybe_export(runner, [step], status="complete")
    assert out_path is not None
    payload = json.loads(Path(out_path).read_text())
    # Legacy path: no envelope in the step blocks.
    assert "envelope" not in payload["steps"][0]
