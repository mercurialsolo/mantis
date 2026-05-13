"""``TimeMeter`` — per-run wall-time bookkeeping (epic #362 Phase A).

Pins the context-manager contract, the per-step accounting, the
``overhead`` residual semantics, and the Prometheus emission path
(skipped when ``tenant_id`` is empty)."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from mantis_agent.gym.time_meter import BUCKETS, TimeMeter


# ── BUCKETS vocabulary ──────────────────────────────────────────────────


def test_buckets_vocabulary_is_stable() -> None:
    """A widening of the bucket vocabulary should be deliberate — pinned
    here so a stealth rename / drop forces a docs + dashboard update."""
    assert BUCKETS == (
        "perceive", "think", "act", "settle", "claude_ground",
        "claude_extract", "claude_verify", "load", "overhead",
    )


# ── measure() primitive ─────────────────────────────────────────────────


def test_measure_credits_elapsed_time_to_the_bucket() -> None:
    meter = TimeMeter()
    with meter.measure("act"):
        time.sleep(0.01)
    assert meter.totals["act"] >= 0.01
    # Other buckets stay at zero.
    for b in BUCKETS:
        if b == "act":
            continue
        assert meter.totals[b] == 0.0


def test_measure_accumulates_across_calls() -> None:
    meter = TimeMeter()
    with meter.measure("think"):
        time.sleep(0.005)
    with meter.measure("think"):
        time.sleep(0.005)
    assert meter.totals["think"] >= 0.01


def test_measure_with_step_idx_records_to_per_step() -> None:
    meter = TimeMeter()
    with meter.measure("act", step_idx=2):
        time.sleep(0.01)
    # per_step grew to accommodate index 2.
    assert len(meter.per_step) == 3
    # Step 0 + step 1 never measured — all-zero records.
    assert meter.per_step[0]["act"] == 0.0
    assert meter.per_step[1]["act"] == 0.0
    # Step 2 carries the measurement.
    assert meter.per_step[2]["act"] >= 0.01


def test_measure_without_step_idx_does_not_touch_per_step() -> None:
    meter = TimeMeter()
    with meter.measure("think"):
        time.sleep(0.001)
    assert meter.per_step == []
    assert meter.totals["think"] >= 0.001


def test_measure_raises_on_unknown_bucket() -> None:
    meter = TimeMeter()
    with pytest.raises(KeyError):
        with meter.measure("ungoverned_bucket"):
            pass


def test_measure_credits_even_when_block_raises() -> None:
    """Exceptions don't skip accounting — operators want to see where
    time went on failed runs too."""
    meter = TimeMeter()
    with pytest.raises(RuntimeError):
        with meter.measure("act"):
            time.sleep(0.005)
            raise RuntimeError("synthetic")
    assert meter.totals["act"] >= 0.005


# ── record() direct credit ──────────────────────────────────────────────


def test_record_credits_without_context_manager() -> None:
    meter = TimeMeter()
    meter.record("settle", 2.5, step_idx=0)
    assert meter.totals["settle"] == 2.5
    assert meter.per_step[0]["settle"] == 2.5


def test_record_ignores_negative_seconds() -> None:
    """Defensive: a misbehaving caller shouldn't poison the totals."""
    meter = TimeMeter()
    meter.record("settle", -1.0)
    assert meter.totals["settle"] == 0.0


def test_record_raises_on_unknown_bucket() -> None:
    meter = TimeMeter()
    with pytest.raises(KeyError):
        meter.record("nope", 1.0)


# ── overhead residual ───────────────────────────────────────────────────


def test_breakdown_fills_overhead_with_residual() -> None:
    """Unmeasured wall time lands in ``overhead`` automatically — so the
    sum of breakdown values tracks total wall-clock without the caller
    having to wrap orchestration code."""
    meter = TimeMeter()
    with meter.measure("act"):
        time.sleep(0.01)
    time.sleep(0.02)  # unaccounted-for orchestration time
    bd = meter.breakdown()
    assert bd["act"] >= 0.01
    # Sum should approximate elapsed (within timing jitter).
    total = sum(bd.values())
    assert total == pytest.approx(meter.elapsed_seconds(), rel=0.5)
    assert bd["overhead"] >= 0.015  # at least the unwrapped sleep


def test_breakdown_overhead_floors_at_zero() -> None:
    """Overlapping measurements (one bucket runs inside another) could
    push the residual negative; the floor keeps the dict JSON-friendly
    and avoids surprising consumers."""
    meter = TimeMeter()
    # Two measurements that together exceed elapsed wall time.
    meter.record("act", 100.0)
    meter.record("think", 50.0)
    bd = meter.breakdown()
    assert bd["overhead"] == 0.0


def test_step_breakdown_returns_zero_dict_for_unmeasured_step() -> None:
    meter = TimeMeter()
    bd = meter.step_breakdown(7)
    assert set(bd) == set(BUCKETS)
    assert all(v == 0.0 for v in bd.values())


def test_step_breakdown_returns_recorded_values() -> None:
    meter = TimeMeter()
    meter.record("act", 1.5, step_idx=0)
    meter.record("think", 0.75, step_idx=0)
    bd = meter.step_breakdown(0)
    assert bd["act"] == 1.5
    assert bd["think"] == 0.75
    assert bd["perceive"] == 0.0  # untouched bucket


# ── Prometheus emission ─────────────────────────────────────────────────


def test_prometheus_emit_skipped_when_tenant_unset() -> None:
    """Local / script runs (no tenant) must not pollute the registry
    with a default-label series."""
    meter = TimeMeter(tenant_id="")
    with patch("mantis_agent.metrics.STEP_LATENCY_SECONDS") as mock_hist:
        with meter.measure("act"):
            time.sleep(0.001)
    mock_hist.labels.assert_not_called()


def test_prometheus_emit_observes_on_tenant_run() -> None:
    meter = TimeMeter(tenant_id="acme")
    with patch("mantis_agent.metrics.STEP_LATENCY_SECONDS") as mock_hist:
        with meter.measure("act"):
            time.sleep(0.001)
    mock_hist.labels.assert_called_once_with(tenant_id="acme", phase="act")
    mock_hist.labels.return_value.observe.assert_called_once()
    observed = mock_hist.labels.return_value.observe.call_args.args[0]
    assert observed >= 0.001


def test_prometheus_emit_is_best_effort() -> None:
    """A metric backend error must not crash the run."""
    meter = TimeMeter(tenant_id="acme")
    with patch("mantis_agent.metrics.STEP_LATENCY_SECONDS") as mock_hist:
        mock_hist.labels.side_effect = RuntimeError("registry down")
        with meter.measure("act"):
            time.sleep(0.001)
    # Despite the metric failure, totals still updated.
    assert meter.totals["act"] >= 0.001
