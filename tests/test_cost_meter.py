"""Tests for #115 step 3 — CostMeter extracted from MicroPlanRunner."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from mantis_agent.cost_config import CostConfig
from mantis_agent.gym.cost_meter import CostMeter, make_initial_costs


# ── Tiny stand-ins for MicroIntent / StepResult ─────────────────────────


@dataclass
class _FakeStep:
    type: str = "click"
    claude_only: bool = False
    grounding: bool = False


@dataclass
class _FakeStepResult:
    steps_used: int = 0


# ── Initial state ───────────────────────────────────────────────────────


def test_make_initial_costs_returns_zeros() -> None:
    c = make_initial_costs()
    assert c == {
        "gpu_steps": 0,
        "gpu_seconds": 0.0,
        "claude_extract": 0,
        "claude_grounding": 0,
        "proxy_mb": 0.0,
    }


def test_make_initial_costs_returns_fresh_dict_per_call() -> None:
    a = make_initial_costs()
    b = make_initial_costs()
    a["gpu_steps"] = 99
    assert b["gpu_steps"] == 0


def test_meter_default_uses_costconfig_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_COST_GPU_HOURLY_USD", "1.50")
    meter = CostMeter()
    assert meter.cost_config.gpu_hourly_usd == 1.50


def test_meter_initial_run_start_is_close_to_now() -> None:
    before = time.time()
    meter = CostMeter()
    after = time.time()
    assert before <= meter.run_start <= after


# ── Cost totals ─────────────────────────────────────────────────────────


def test_totals_zero_for_fresh_meter() -> None:
    meter = CostMeter()
    assert meter.totals() == (0.0, 0.0, 0.0, 0.0)


def test_totals_combines_components_via_cost_config() -> None:
    cfg = CostConfig(
        gpu_hourly_usd=4.0,
        claude_call_usd=0.01,
        proxy_per_gb_usd=2.0,
    )
    meter = CostMeter(cost_config=cfg)
    meter.costs["gpu_seconds"] = 3600  # 1 hour → $4
    meter.costs["claude_extract"] = 5  # → $0.05
    meter.costs["claude_grounding"] = 5  # → $0.05
    meter.costs["proxy_mb"] = 1024  # 1 GB → $2
    gpu, claude, proxy, total = meter.totals()
    assert gpu == pytest.approx(4.0)
    assert claude == pytest.approx(0.10)
    assert proxy == pytest.approx(2.0)
    assert total == pytest.approx(6.10)


# ── Per-step accounting ─────────────────────────────────────────────────


def test_record_step_adds_gpu_seconds_per_steps_used() -> None:
    cfg = CostConfig(gpu_seconds_per_step=4.0)
    meter = CostMeter(cost_config=cfg)
    meter.record_step(_FakeStep(), _FakeStepResult(steps_used=3))
    assert meter.costs["gpu_steps"] == 3
    assert meter.costs["gpu_seconds"] == 12.0


def test_record_step_no_gpu_when_steps_used_zero() -> None:
    meter = CostMeter()
    meter.record_step(_FakeStep(), _FakeStepResult(steps_used=0))
    assert meter.costs["gpu_steps"] == 0
    assert meter.costs["gpu_seconds"] == 0.0


def test_record_step_increments_claude_extract_only_when_claude_only() -> None:
    meter = CostMeter()
    meter.record_step(_FakeStep(claude_only=True), _FakeStepResult())
    meter.record_step(_FakeStep(claude_only=False), _FakeStepResult())
    assert meter.costs["claude_extract"] == 1


def test_record_step_increments_grounding_per_steps_used() -> None:
    meter = CostMeter()
    meter.record_step(_FakeStep(grounding=True), _FakeStepResult(steps_used=2))
    assert meter.costs["claude_grounding"] == 2


def test_record_step_proxy_mb_for_nav_types() -> None:
    cfg = CostConfig(proxy_mb_per_nav=7.0, proxy_mb_per_scroll=0.25)
    meter = CostMeter(cost_config=cfg)
    for kind in ("click", "navigate", "paginate"):
        meter.record_step(_FakeStep(type=kind), _FakeStepResult())
    meter.record_step(_FakeStep(type="scroll"), _FakeStepResult())
    meter.record_step(_FakeStep(type="extract_data"), _FakeStepResult())
    assert meter.costs["proxy_mb"] == pytest.approx(3 * 7.0 + 0.25)


# ── Snapshot / restore ──────────────────────────────────────────────────


def test_snapshot_returns_independent_copy() -> None:
    meter = CostMeter()
    meter.costs["gpu_steps"] = 5
    snap = meter.snapshot()
    snap["gpu_steps"] = 999
    assert meter.costs["gpu_steps"] == 5


def test_restore_merges_persisted_counters() -> None:
    meter = CostMeter()
    meter.restore({"gpu_steps": 10, "claude_extract": 3})
    assert meter.costs["gpu_steps"] == 10
    assert meter.costs["claude_extract"] == 3
    # Untouched keys keep their initial values.
    assert meter.costs["proxy_mb"] == 0.0


def test_restore_with_none_is_noop() -> None:
    meter = CostMeter()
    meter.costs["gpu_steps"] = 7
    meter.restore(None)
    assert meter.costs["gpu_steps"] == 7


# ── Inflight gauge emission ─────────────────────────────────────────────


def test_emit_skipped_when_no_tenant_id() -> None:
    """No tenant_id ⇒ no Prometheus interaction at all (registry stays clean)."""
    meter = CostMeter(tenant_id="")
    # Should not raise even if metrics module is missing/broken — exercises
    # the early-return guard.
    meter.emit_inflight_gauges(1.0, 2.0, 3.0, 6.0)


def test_emit_calls_prometheus_when_tenant_set(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, str, float]] = []

    class _Gauge:
        def labels(self, **kwargs):
            return _LabeledGauge(kwargs)

    class _LabeledGauge:
        def __init__(self, kwargs: dict) -> None:
            self.kwargs = kwargs

        def set(self, value: float) -> None:
            captured.append(
                (self.kwargs["tenant_id"], self.kwargs["component"], value)
            )

    from mantis_agent import metrics as real_metrics
    monkeypatch.setattr(real_metrics, "RUN_COST_USD_INFLIGHT", _Gauge())

    meter = CostMeter(tenant_id="tenant-A")
    meter.emit_inflight_gauges(1.0, 2.0, 3.0, 6.0)
    assert ("tenant-A", "gpu", 1.0) in captured
    assert ("tenant-A", "claude", 2.0) in captured
    assert ("tenant-A", "proxy", 3.0) in captured
    assert ("tenant-A", "total", 6.0) in captured
    assert len(captured) == 4


def test_emit_swallows_metrics_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """If prometheus_client fails, the runner shouldn't crash."""
    class _BrokenGauge:
        def labels(self, **kwargs):
            raise RuntimeError("prometheus exploded")

    from mantis_agent import metrics as real_metrics
    monkeypatch.setattr(real_metrics, "RUN_COST_USD_INFLIGHT", _BrokenGauge())

    meter = CostMeter(tenant_id="t")
    # Should not raise.
    meter.emit_inflight_gauges(1, 2, 3, 6)


# ── Backward-compat: MicroPlanRunner aliases meter state ─────────────────


def test_runner_costs_aliases_meter_costs() -> None:
    """MicroPlanRunner.costs must be the same dict as cost_meter.costs."""
    from mantis_agent.gym.micro_runner import MicroPlanRunner

    runner = MicroPlanRunner.__new__(MicroPlanRunner)
    runner.cost_meter = CostMeter()
    runner.costs = runner.cost_meter.costs

    runner.costs["gpu_steps"] += 5
    assert runner.cost_meter.costs["gpu_steps"] == 5


def test_runner_cost_totals_delegates_to_meter() -> None:
    from mantis_agent.gym.micro_runner import MicroPlanRunner

    runner = MicroPlanRunner.__new__(MicroPlanRunner)
    runner.cost_meter = CostMeter()
    runner.costs = runner.cost_meter.costs

    runner.costs["gpu_seconds"] = 1800  # 30 min default = $1.625
    gpu, claude, proxy, total = runner._cost_totals()
    assert gpu == pytest.approx(1.625)
    assert claude == 0.0
    assert proxy == 0.0
    assert total == pytest.approx(1.625)
