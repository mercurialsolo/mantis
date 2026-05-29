"""Tests for the Claude cost meter — per-source attribution. (#675 follow-up)"""

from __future__ import annotations

import json
import os

import pytest

from mantis_agent.observability.claude_cost_meter import (
    ClaudeCostMeter,
    estimate_cost,
    finalize_to_disk,
    rates_for,
    record_from_response,
    set_current_meter,
)


@pytest.fixture()
def temp_runs_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> str:
    monkeypatch.setenv("MANTIS_RUN_ARTIFACTS_DIR", str(tmp_path))
    return str(tmp_path)


# ── cost model ───────────────────────────────────────────────────────


def test_rates_for_known_models() -> None:
    # Rates are per-TOKEN (Anthropic per-Mtok rates / 1e6).
    assert rates_for("claude-opus-4-7").input == 15.0 / 1_000_000
    assert rates_for("claude-sonnet-4-6").input == 3.0 / 1_000_000
    assert rates_for("claude-haiku-4-5-20251001").input == 0.8 / 1_000_000


def test_rates_for_versioned_haiku_falls_back_to_base() -> None:
    """Versioned model names match by prefix."""
    rates = rates_for("claude-haiku-4-5-20260201")
    assert rates.input == 0.8 / 1_000_000  # haiku rate


def test_rates_for_unknown_model_uses_opus_conservative() -> None:
    """Unknown model → conservative (Opus) so cost never under-reports."""
    assert rates_for("claude-future-x").input == 15.0 / 1_000_000


def test_estimate_cost_combines_input_output_cache() -> None:
    # Haiku-4-5: input $0.80/Mtok, output $4/Mtok,
    # cache_read $0.08/Mtok, cache_creation $1/Mtok
    cost = estimate_cost(
        model="claude-haiku-4-5-20251001",
        input_tokens=1_000_000, output_tokens=100_000,
        cache_read_tokens=5_000_000, cache_creation_tokens=0,
    )
    # 1Mtok * $0.80 + 100Ktok * $4 + 5Mtok * $0.08
    expected = 0.80 + 0.40 + 0.40
    assert abs(cost - expected) < 1e-6


# ── accumulator ──────────────────────────────────────────────────────


def test_meter_records_call() -> None:
    meter = ClaudeCostMeter()
    meter.record(
        source="brain_claude", model="claude-sonnet-4-6",
        input_tokens=1000, output_tokens=200,
    )
    snap = meter.snapshot()
    assert snap["totals"]["calls"] == 1
    assert snap["totals"]["input_tokens"] == 1000
    assert snap["totals"]["output_tokens"] == 200
    assert snap["totals"]["cost_usd"] > 0


def test_meter_accumulates_across_calls() -> None:
    meter = ClaudeCostMeter()
    for _ in range(5):
        meter.record(
            source="extract_multi", model="claude-sonnet-4-6",
            input_tokens=500, output_tokens=50,
        )
    snap = meter.snapshot()
    assert snap["totals"]["calls"] == 5
    assert snap["totals"]["input_tokens"] == 2500
    assert snap["totals"]["output_tokens"] == 250


def test_meter_separates_by_source_and_model() -> None:
    """Two sources + two models = four buckets."""
    meter = ClaudeCostMeter()
    meter.record(source="brain_claude", model="claude-opus-4-7",
                 input_tokens=100, output_tokens=10)
    meter.record(source="brain_claude", model="claude-sonnet-4-6",
                 input_tokens=100, output_tokens=10)
    meter.record(source="extract_multi", model="claude-sonnet-4-6",
                 input_tokens=100, output_tokens=10)
    snap = meter.snapshot()
    # 3 distinct (source, model) tuples
    assert len(snap["by_path"]) == 3


def test_meter_ignores_empty_source_or_model() -> None:
    meter = ClaudeCostMeter()
    meter.record(source="", model="claude-opus-4-7", input_tokens=100)
    meter.record(source="brain_claude", model="", input_tokens=100)
    assert meter.snapshot()["totals"]["calls"] == 0


# ── record_from_response ─────────────────────────────────────────────


def test_record_from_response_no_op_without_meter() -> None:
    """Bound meter is None → no-op (doesn't raise)."""
    set_current_meter(None)
    record_from_response(
        source="brain_claude", model="claude-sonnet-4-6",
        response_json={"usage": {"input_tokens": 100}},
    )


def test_record_from_response_pulls_usage_fields() -> None:
    meter = ClaudeCostMeter()
    set_current_meter(meter)
    try:
        record_from_response(
            source="extract_multi", model="claude-sonnet-4-6",
            response_json={
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "cache_read_input_tokens": 500,
                    "cache_creation_input_tokens": 0,
                },
            },
        )
        snap = meter.snapshot()
        bucket = next(iter(snap["by_path"].values()))
        assert bucket["input_tokens"] == 1000
        assert bucket["output_tokens"] == 200
        assert bucket["cache_read_tokens"] == 500
    finally:
        set_current_meter(None)


def test_record_from_response_swallows_malformed_response() -> None:
    meter = ClaudeCostMeter()
    set_current_meter(meter)
    try:
        record_from_response(
            source="brain_claude", model="claude-opus-4-7",
            response_json=None,
        )
        record_from_response(
            source="brain_claude", model="claude-opus-4-7",
            response_json={"usage": "not a dict"},
        )
        assert meter.snapshot()["totals"]["calls"] == 0
    finally:
        set_current_meter(None)


# ── finalize_to_disk ─────────────────────────────────────────────────


def test_finalize_writes_json_at_canonical_path(temp_runs_dir: str) -> None:
    meter = ClaudeCostMeter()
    meter.record(
        source="brain_claude", model="claude-sonnet-4-6",
        input_tokens=1000, output_tokens=100,
    )
    path = finalize_to_disk(
        meter=meter, run_id="run_xyz", tenant_id="acme",
    )
    assert path
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data["run_id"] == "run_xyz"
    assert data["tenant_id"] == "acme"
    assert data["totals"]["calls"] == 1
    assert "brain_claude::claude-sonnet-4-6" in data["by_path"]


def test_finalize_no_op_when_no_calls(temp_runs_dir: str) -> None:
    """Empty meter → no file written (avoid empty-file noise)."""
    meter = ClaudeCostMeter()
    path = finalize_to_disk(
        meter=meter, run_id="empty_run", tenant_id="acme",
    )
    assert path == ""


def test_finalize_no_op_without_run_id(temp_runs_dir: str) -> None:
    meter = ClaudeCostMeter()
    meter.record(source="x", model="claude-opus-4-7", input_tokens=10)
    assert finalize_to_disk(meter=meter, run_id="", tenant_id="acme") == ""


def test_finalize_uses_current_meter_by_default(temp_runs_dir: str) -> None:
    meter = ClaudeCostMeter()
    meter.record(source="brain", model="claude-sonnet-4-6", input_tokens=10)
    set_current_meter(meter)
    try:
        path = finalize_to_disk(run_id="run_default", tenant_id="acme")
        assert path
    finally:
        set_current_meter(None)


def test_path_includes_sanitized_tenant_and_run_id(temp_runs_dir: str) -> None:
    """Slashes / spaces in tenant_id or run_id are sanitized — no directory escape."""
    meter = ClaudeCostMeter()
    meter.record(source="x", model="claude-opus-4-7", input_tokens=10)
    path = finalize_to_disk(
        meter=meter,
        run_id="../escape/attempt",
        tenant_id="../../bad/tenant",
    )
    # File must land under temp_runs_dir, never outside it
    assert path.startswith(temp_runs_dir)
