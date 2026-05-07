"""Tests for #155 step 2 — TraceLabeller heuristics + CLI surface.

Pins the labeller's heuristic ladder and verifies the CLI subcommands
produce the right exit codes / output shapes. Runs end-to-end against
trace JSON exactly as :class:`~.trace_exporter.TraceExporter` would
emit, so the two pieces stay schema-locked.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from mantis_agent.cli import EXIT_ERROR, EXIT_OK, main
from mantis_agent.gym.trace_labeller import LabelledTrace, TraceLabeller


# ── Helpers ────────────────────────────────────────────────────────────


def _step(
    *,
    step_index: int = 0,
    success: bool = True,
    data: str = "",
    intent: str = "step",
    type: str = "click",
    last_action: dict[str, Any] | None = None,
    predicted_outcome: str = "",
    observed_outcome: str = "",
) -> dict[str, Any]:
    return {
        "step_index": step_index,
        "intent": intent,
        "type": type,
        "success": success,
        "data": data,
        "last_action": last_action,
        "predicted_outcome": predicted_outcome,
        "observed_outcome": observed_outcome,
    }


def _trace_payload(steps: list[dict[str, Any]], **overrides) -> dict[str, Any]:
    base = {
        "schema_version": 1,
        "run_id": "rid_abc",
        "tenant_id": "t-one",
        "session_name": "sess",
        "plan_signature": "sig",
        "status": "completed",
        "started_at": 100.0,
        "ended_at": 200.0,
        "total_time_s": 100.0,
        "costs": {"gpu": 1.0, "claude": 0.5, "proxy": 0.1, "total": 1.6},
        "step_count": len(steps),
        "steps": steps,
    }
    base.update(overrides)
    return base


# ── Heuristic ladder ───────────────────────────────────────────────────


def test_escalation_marker_in_data_labels_negative():
    labeller = TraceLabeller()
    step = _step(success=True, data="REJECTED_INCOMPLETE|missing required field")
    out = labeller.label_step(step)
    assert out.label == "negative"
    assert out.label_reason == "escalation"


def test_failed_step_without_escalation_labels_negative():
    labeller = TraceLabeller()
    step = _step(success=False, data="generic failure")
    out = labeller.label_step(step)
    assert out.label == "negative"
    assert out.label_reason == "failed_step"


def test_gate_pass_labels_positive():
    labeller = TraceLabeller()
    step = _step(success=True, data="gate:PASS:URL contains expected token", type="extract_data")
    out = labeller.label_step(step)
    assert out.label == "positive"
    assert out.label_reason == "gate_verify_pass"


def test_success_with_observed_delta_labels_positive():
    labeller = TraceLabeller()
    step = _step(
        success=True, data="",
        observed_outcome="page navigated to /detail/42",
    )
    out = labeller.label_step(step)
    assert out.label == "positive"
    assert out.label_reason == "success_with_observed_delta"


def test_success_no_observed_delta_labels_neutral():
    labeller = TraceLabeller()
    step = _step(success=True, data="", observed_outcome="")
    out = labeller.label_step(step)
    assert out.label == "neutral"
    assert out.label_reason == "success_no_delta"


def test_escalation_outranks_failure():
    """A failed step that ALSO has an escalation marker should land in
    'escalation' — that's the higher-priority label per the heuristic
    ladder. Pinned because reviewers triage escalations first."""
    labeller = TraceLabeller()
    step = _step(success=False, data="page_blocked: cloudflare challenge")
    out = labeller.label_step(step)
    assert out.label_reason == "escalation"


# ── Trace + summary rollup ─────────────────────────────────────────────


def test_label_trace_aggregates_summary_counts():
    labeller = TraceLabeller()
    payload = _trace_payload([
        _step(step_index=0, success=True, data="gate:PASS:done"),
        _step(step_index=1, success=False, data="generic"),
        _step(step_index=2, success=True, observed_outcome="ok"),
        _step(step_index=3, success=True, observed_outcome=""),
    ])
    labelled = labeller.label_trace(payload)
    assert isinstance(labelled, LabelledTrace)
    assert labelled.summary() == {"positive": 2, "negative": 1, "neutral": 1}


def test_label_trace_round_trips_through_dict():
    labeller = TraceLabeller()
    payload = _trace_payload([_step()])
    labelled = labeller.label_trace(payload)
    out = labelled.to_dict()
    assert out["run_id"] == "rid_abc"
    assert out["label_summary"]["neutral"] == 1
    assert out["steps"][0]["label"] == "neutral"


def test_label_trace_file_reads_path(tmp_path):
    payload = _trace_payload([_step(success=True, data="gate:PASS")])
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(payload))
    labelled = TraceLabeller().label_trace_file(path)
    assert labelled.source_path == str(path)
    assert labelled.summary()["positive"] == 1


# ── Directory mode ─────────────────────────────────────────────────────


def test_label_directory_mirrors_subtree(tmp_path):
    # Two tenant subdirs, one trace each — labelling preserves layout.
    (tmp_path / "input" / "acme").mkdir(parents=True)
    (tmp_path / "input" / "globex").mkdir(parents=True)
    (tmp_path / "input" / "acme" / "run1.json").write_text(
        json.dumps(_trace_payload([_step(success=True, data="gate:PASS")]))
    )
    (tmp_path / "input" / "globex" / "run2.json").write_text(
        json.dumps(_trace_payload([_step(success=False)]))
    )
    summary = TraceLabeller().label_directory(
        tmp_path / "input", tmp_path / "output",
    )
    assert (tmp_path / "output" / "acme" / "run1.json").exists()
    assert (tmp_path / "output" / "globex" / "run2.json").exists()
    assert "acme/run1.json" in summary
    assert summary["acme/run1.json"]["positive"] == 1


def test_label_directory_skips_unreadable_files(tmp_path):
    """Bad JSON in one file mustn't kill the whole batch."""
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "good.json").write_text(
        json.dumps(_trace_payload([_step()]))
    )
    (tmp_path / "input" / "broken.json").write_text("{not json")
    summary = TraceLabeller().label_directory(
        tmp_path / "input", tmp_path / "output",
    )
    # Only the good file shows up.
    assert "good.json" in summary
    assert "broken.json" not in summary


# ── CLI: trace label ───────────────────────────────────────────────────


def test_cli_trace_label_single_file(tmp_path, capsys):
    src = tmp_path / "trace.json"
    src.write_text(json.dumps(_trace_payload([_step(success=True, data="gate:PASS")])))
    out_dir = tmp_path / "out"
    rc = main(["trace", "label", str(src), "--output", str(out_dir)])
    assert rc == EXIT_OK
    target = out_dir / "trace.json"
    assert target.exists()
    payload = json.loads(target.read_text())
    assert payload["label_summary"]["positive"] == 1


def test_cli_trace_label_directory(tmp_path, capsys):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "a.json").write_text(json.dumps(_trace_payload([_step()])))
    out_dir = tmp_path / "out"
    rc = main(["trace", "label", str(inp), "--output", str(out_dir)])
    assert rc == EXIT_OK
    assert (out_dir / "a.json").exists()


def test_cli_trace_label_missing_input_exits_error(tmp_path, capsys):
    rc = main([
        "trace", "label", str(tmp_path / "nope"),
        "--output", str(tmp_path / "out"),
    ])
    assert rc == EXIT_ERROR
    assert "input not found" in capsys.readouterr().err


def test_cli_trace_label_json_mode(tmp_path, capsys):
    src = tmp_path / "trace.json"
    src.write_text(json.dumps(_trace_payload([_step(success=True, data="gate:PASS")])))
    rc = main([
        "trace", "label", str(src),
        "--output", str(tmp_path / "out"), "--json",
    ])
    assert rc == EXIT_OK
    parsed = json.loads(capsys.readouterr().out)
    # Single-file mode keys by absolute input path.
    assert any("positive" in v for v in parsed.values())


# ── CLI: trace review ──────────────────────────────────────────────────


def test_cli_trace_review_prints_table(tmp_path, capsys):
    src = tmp_path / "trace.json"
    src.write_text(json.dumps(_trace_payload([
        _step(step_index=0, success=True, data="gate:PASS"),
        _step(step_index=1, success=False, data="cloudflare"),
    ])))
    rc = main(["trace", "review", str(src)])
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "[00]" in out and "[01]" in out
    assert "positive" in out and "negative" in out
    assert "gate_verify_pass" in out


def test_cli_trace_review_json_mode(tmp_path, capsys):
    src = tmp_path / "trace.json"
    src.write_text(json.dumps(_trace_payload([_step(success=True, data="gate:PASS")])))
    rc = main(["trace", "review", str(src), "--json"])
    assert rc == EXIT_OK
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["label_summary"]["positive"] == 1


def test_cli_trace_review_missing_path(capsys):
    rc = main(["trace", "review", "/does/not/exist.json"])
    assert rc == EXIT_ERROR
    assert "trace not found" in capsys.readouterr().err


def test_cli_trace_review_invalid_json(tmp_path, capsys):
    bad = tmp_path / "broken.json"
    bad.write_text("{not json")
    rc = main(["trace", "review", str(bad)])
    assert rc == EXIT_ERROR
    assert "invalid JSON" in capsys.readouterr().err


# ── Argument parsing contracts ─────────────────────────────────────────


def test_cli_trace_with_no_subcommand_errors():
    with pytest.raises(SystemExit):
        main(["trace"])


def test_cli_trace_label_requires_output():
    with pytest.raises(SystemExit):
        main(["trace", "label", "/tmp/x.json"])
