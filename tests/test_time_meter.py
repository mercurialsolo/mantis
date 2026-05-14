"""``TimeMeter`` — per-run wall-time bookkeeping (epic #362 Phase A).

Pins the context-manager contract, the per-step accounting, the
``overhead`` residual semantics, and the Prometheus emission path
(skipped when ``tenant_id`` is empty)."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from mantis_agent.gym.time_meter import (
    BUCKETS,
    TimeMeter,
    current_dispatch,
    publish_dispatch,
    record_to_current,
)


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


# ── Dispatch context (contextvars) ──────────────────────────────────────


def test_current_dispatch_is_none_by_default() -> None:
    assert current_dispatch() is None


def test_publish_dispatch_routes_record_to_current_meter() -> None:
    """Deep helpers call ``record_to_current(bucket, seconds)``; the
    contextvar-published meter receives the credit on the right step."""
    meter = TimeMeter()
    with publish_dispatch(meter, step_idx=2):
        record_to_current("settle", 1.5)
        record_to_current("load", 0.5)
    assert meter.totals["settle"] == 1.5
    assert meter.totals["load"] == 0.5
    assert meter.per_step[2]["settle"] == 1.5
    assert meter.per_step[2]["load"] == 0.5


def test_publish_dispatch_resets_on_exit() -> None:
    """Context must clear after the ``with`` block — no leakage into
    later helpers running outside any step."""
    meter = TimeMeter()
    with publish_dispatch(meter, step_idx=0):
        assert current_dispatch() == (meter, 0)
    assert current_dispatch() is None


def test_publish_dispatch_resets_on_exception() -> None:
    """A crashing step must NOT leave the contextvar pointing at a
    stale meter — the next runner would silently inherit it."""
    meter = TimeMeter()
    with pytest.raises(RuntimeError):
        with publish_dispatch(meter, step_idx=0):
            raise RuntimeError("synthetic")
    assert current_dispatch() is None


def test_record_to_current_is_noop_outside_dispatch() -> None:
    """Helpers invoked outside any dispatch (script mode, tests) must
    not raise — they silently skip recording."""
    # No exception, no state change.
    record_to_current("settle", 2.0)


def test_publish_dispatch_with_none_meter_clears_context() -> None:
    """Tests sometimes want to verify a helper bails out cleanly when
    no meter is published — ``publish_dispatch(None, ...)`` makes that
    inline-testable."""
    real_meter = TimeMeter()
    with publish_dispatch(real_meter, 0):
        with publish_dispatch(None, 0):
            record_to_current("settle", 1.0)
        # Outer context restored after inner exits.
        assert current_dispatch() == (real_meter, 0)
    # The recording inside the inner block (None meter) should have
    # been a no-op.
    assert real_meter.totals["settle"] == 0.0


# ── adaptive_settle integration ─────────────────────────────────────────


def test_adaptive_settle_credits_settle_bucket_via_dispatch() -> None:
    """The real ``settle_after_action`` should record its elapsed wait
    into the ``settle`` bucket on whichever meter the executor has
    published — zero call-site changes in step handlers."""
    from PIL import Image

    from mantis_agent.gym import adaptive_settle

    class _Env:
        def screenshot(self) -> Image.Image:
            return Image.new("RGB", (32, 32), "white")

    meter = TimeMeter()
    with publish_dispatch(meter, step_idx=4):
        elapsed = adaptive_settle.settle_after_action(_Env(), max_seconds=0.5)

    assert elapsed > 0.0
    assert meter.totals["settle"] == pytest.approx(elapsed, rel=0.01)
    assert meter.per_step[4]["settle"] == pytest.approx(elapsed, rel=0.01)


def test_adaptive_settle_is_silent_outside_dispatch() -> None:
    """Without a published dispatch, ``settle_after_action`` still
    works — it just doesn't credit any meter."""
    from PIL import Image

    from mantis_agent.gym import adaptive_settle

    class _Env:
        def screenshot(self) -> Image.Image:
            return Image.new("RGB", (32, 32), "white")

    # No publish_dispatch wrapping → record_to_current is a no-op.
    elapsed = adaptive_settle.settle_after_action(_Env(), max_seconds=0.3)
    assert elapsed > 0.0  # The wait still happened, just unrecorded.
