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
        # #514 token counters (Anthropic's per-call usage block)
        "claude_input_tokens": 0,
        "claude_output_tokens": 0,
        "claude_cached_input_tokens": 0,
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


# ── #351: sync_gpu_seconds_from_time_meter ───────────────────────────────


class _FakeTimeMeter:
    def __init__(self, think_seconds: float = 0.0, **extra) -> None:
        self.totals: dict[str, float] = {"think": think_seconds, **extra}


def test_sync_gpu_seconds_overwrites_with_think_bucket() -> None:
    meter = CostMeter()
    meter.costs["gpu_seconds"] = 999.0  # pre-existing synthetic value
    meter.sync_gpu_seconds_from_time_meter(_FakeTimeMeter(think_seconds=42.5))
    assert meter.costs["gpu_seconds"] == 42.5


def test_sync_gpu_seconds_idempotent_on_repeated_calls() -> None:
    meter = CostMeter()
    tm = _FakeTimeMeter(think_seconds=10.0)
    meter.sync_gpu_seconds_from_time_meter(tm)
    meter.sync_gpu_seconds_from_time_meter(tm)
    assert meter.costs["gpu_seconds"] == 10.0


def test_sync_gpu_seconds_missing_think_bucket_leaves_value_alone() -> None:
    meter = CostMeter()
    meter.costs["gpu_seconds"] = 5.0

    class _NoThink:
        totals: dict[str, float] = {}

    meter.sync_gpu_seconds_from_time_meter(_NoThink())
    assert meter.costs["gpu_seconds"] == 5.0


def test_sync_gpu_seconds_negative_value_rejected() -> None:
    meter = CostMeter()
    meter.costs["gpu_seconds"] = 7.0
    meter.sync_gpu_seconds_from_time_meter(_FakeTimeMeter(think_seconds=-1.0))
    assert meter.costs["gpu_seconds"] == 7.0


def test_record_step_with_time_meter_syncs_gpu_seconds() -> None:
    """When a TimeMeter is wired in, gpu_seconds comes from ``think``
    rather than the legacy ``steps × per-step`` multiplier."""
    meter = CostMeter(cost_config=CostConfig(gpu_seconds_per_step=99.0))
    tm = _FakeTimeMeter(think_seconds=8.0)
    meter.record_step(_FakeStep(), _FakeStepResult(steps_used=2), time_meter=tm)
    assert meter.costs["gpu_steps"] == 2
    # Real wall-time wins over the 99×2=198 synthetic value.
    assert meter.costs["gpu_seconds"] == 8.0


def test_record_step_without_time_meter_uses_legacy_multiplier() -> None:
    """Tests / legacy callers without a TimeMeter keep the pre-#351 behaviour."""
    meter = CostMeter(cost_config=CostConfig(gpu_seconds_per_step=4.0))
    meter.record_step(_FakeStep(), _FakeStepResult(steps_used=3))
    assert meter.costs["gpu_seconds"] == 12.0


def test_record_step_with_time_meter_syncs_even_when_steps_used_zero() -> None:
    """Per-step deltas must absorb think-bucket growth from steps with
    ``steps_used == 0`` (e.g. ``extract_data``, verify) and from
    between-step handlers (critic, recovery). Was previously skipped
    when ``steps_used == 0``, leaking inter-step think into the
    finalize sync where it landed on the run total but on no per-step
    row."""
    meter = CostMeter()
    tm = _FakeTimeMeter(think_seconds=12.5)
    meter.record_step(_FakeStep(), _FakeStepResult(steps_used=0), time_meter=tm)
    assert meter.costs["gpu_steps"] == 0  # not bumped — still gated
    assert meter.costs["gpu_seconds"] == 12.5  # but synced


# ── #350: totals_from helper ─────────────────────────────────────────────


def test_totals_from_returns_breakdown_dict() -> None:
    cfg = CostConfig(
        gpu_hourly_usd=3.6,  # → $0.001/sec
        claude_call_usd=0.01,
        proxy_per_gb_usd=10.0,
    )
    meter = CostMeter(cost_config=cfg)
    delta = {
        "gpu_steps": 5,
        "gpu_seconds": 100.0,  # → $0.10
        "claude_extract": 3,
        "claude_grounding": 2,  # 5 calls × $0.01 = $0.05
        "proxy_mb": 102.4,  # 0.1 GB × $10 = $1.00
    }
    out = meter.totals_from(delta)
    assert out["gpu"] == pytest.approx(0.10)
    assert out["claude"] == pytest.approx(0.05)
    assert out["proxy"] == pytest.approx(1.00)
    assert out["total"] == pytest.approx(1.15)
    # Counters echo through for downstream reporters.
    assert out["gpu_steps"] == 5
    assert out["gpu_seconds"] == 100.0
    assert out["claude_extract"] == 3
    assert out["claude_grounding"] == 2
    assert out["proxy_mb"] == 102.4


def test_totals_from_tolerates_partial_dict() -> None:
    meter = CostMeter()
    out = meter.totals_from({"gpu_seconds": 3600})
    assert out["gpu"] == pytest.approx(3.25)
    assert out["claude"] == 0.0
    assert out["proxy"] == 0.0
    assert out["claude_grounding"] == 0
    assert out["proxy_mb"] == 0.0


def test_totals_from_matches_totals_for_full_dict() -> None:
    """For a costs dict identical to meter.costs, totals_from() must
    agree with totals() — the two helpers are the same arithmetic
    seen from different angles."""
    meter = CostMeter()
    meter.costs.update({
        "gpu_steps": 4,
        "gpu_seconds": 12.0,
        "claude_extract": 2,
        "claude_grounding": 1,
        "proxy_mb": 50.0,
    })
    gpu, claude, proxy, total = meter.totals()
    out = meter.totals_from(meter.costs)
    assert out["gpu"] == pytest.approx(gpu)
    assert out["claude"] == pytest.approx(claude)
    assert out["proxy"] == pytest.approx(proxy)
    assert out["total"] == pytest.approx(total)


# ── #514 Claude token cost ──────────────────────────────────────────────


def test_record_claude_tokens_accumulates() -> None:
    """``record_claude_tokens`` adds to existing counters (per-call
    increment, not set). Three calls of 10k input + 1k output should
    leave 30k input + 3k output on the meter."""
    meter = CostMeter()
    for _ in range(3):
        meter.record_claude_tokens(input_tokens=10_000, output_tokens=1_000)
    assert meter.costs["claude_input_tokens"] == 30_000
    assert meter.costs["claude_output_tokens"] == 3_000
    assert meter.costs["claude_cached_input_tokens"] == 0


def test_totals_prefers_token_billing_when_populated() -> None:
    """When ``claude_input_tokens`` / ``output_tokens`` are set, the
    Claude component is computed from per-token rates, NOT from the
    legacy call counters. The default Sonnet 4 rates make a 30k/2k
    call land at ~$0.12."""
    meter = CostMeter()
    # Also set the legacy call counter to a high value to prove tokens
    # take precedence — without #514 this'd produce $0.003 × 5 = $0.015.
    meter.costs["claude_extract"] = 5
    meter.record_claude_tokens(input_tokens=30_000, output_tokens=2_000)
    _, claude_cost, _, _ = meter.totals()
    # 30k × $3/M + 2k × $15/M = $0.09 + $0.030 = $0.12
    assert claude_cost == pytest.approx(0.12, abs=1e-6)


def test_totals_falls_back_to_call_count_when_no_tokens() -> None:
    """Legacy callers that don't go through the Anthropic-client
    token-credit hook still get a Claude cost via the per-call counter.
    This keeps the surface backwards compatible while #514 is rolling
    out across all call sites."""
    meter = CostMeter()
    meter.costs["claude_extract"] = 4
    meter.costs["claude_grounding"] = 2
    _, claude_cost, _, _ = meter.totals()
    # 6 calls × $0.003/call default
    assert claude_cost == pytest.approx(0.018, abs=1e-6)


def test_cached_input_tokens_billed_at_lower_rate() -> None:
    """Anthropic's cached input is ~10% of the standard input rate
    (Sonnet 4: $0.30/M vs $3/M). 10k input with 5k cached should bill
    the cached portion at the discount rate."""
    meter = CostMeter()
    meter.record_claude_tokens(
        input_tokens=10_000, output_tokens=0, cached_input_tokens=5_000,
    )
    _, claude_cost, _, _ = meter.totals()
    # 5k plain × $3/M + 5k cached × $0.30/M = $0.015 + $0.0015 = $0.0165
    assert claude_cost == pytest.approx(0.0165, abs=1e-6)


def test_env_override_changes_per_token_rate(monkeypatch) -> None:
    """Per-mtok rates honour env-var overrides so operators can pin
    to whatever model their deployment actually targets."""
    from mantis_agent.cost_config import CostConfig
    monkeypatch.setenv("MANTIS_COST_CLAUDE_INPUT_PER_MTOK", "5.0")
    monkeypatch.setenv("MANTIS_COST_CLAUDE_OUTPUT_PER_MTOK", "25.0")
    cfg = CostConfig.from_env()
    cost = cfg.claude_token_cost(input_tokens=1_000_000, output_tokens=100_000)
    assert cost == pytest.approx(5.0 + 2.5, abs=1e-6)


def test_publish_cost_meter_scoping() -> None:
    """``publish_cost_meter`` makes the meter visible to
    ``record_claude_tokens_to_current`` for the wrapped block, and
    cleanly clears on exit."""
    from mantis_agent.gym import cost_meter as cm
    assert cm.current_cost_meter() is None
    meter = cm.CostMeter()
    with cm.publish_cost_meter(meter):
        assert cm.current_cost_meter() is meter
        cm.record_claude_tokens_to_current(input_tokens=2_000, output_tokens=500)
    # Context popped — no leakage into the next scope.
    assert cm.current_cost_meter() is None
    assert meter.costs["claude_input_tokens"] == 2_000
    assert meter.costs["claude_output_tokens"] == 500
    # No published meter → silent no-op.
    cm.record_claude_tokens_to_current(input_tokens=99_999, output_tokens=99_999)
    assert meter.costs["claude_input_tokens"] == 2_000


def test_credit_helper_parses_anthropic_response() -> None:
    """``credit_claude_tokens_from_response`` reads the standard
    Anthropic ``usage`` block (``input_tokens``, ``output_tokens``,
    ``cache_read_input_tokens``) off a response dict and credits the
    currently-published meter."""
    from mantis_agent._anthropic.client import credit_claude_tokens_from_response
    from mantis_agent.gym import cost_meter as cm
    meter = cm.CostMeter()
    response_json = {
        "id": "msg_abc",
        "model": "claude-sonnet-4-5",
        "usage": {
            "input_tokens": 25_000,
            "output_tokens": 1_500,
            "cache_read_input_tokens": 3_000,
        },
    }
    with cm.publish_cost_meter(meter):
        credit_claude_tokens_from_response(response_json)
    assert meter.costs["claude_input_tokens"] == 25_000
    assert meter.costs["claude_output_tokens"] == 1_500
    assert meter.costs["claude_cached_input_tokens"] == 3_000


def test_credit_helper_is_noop_without_published_meter() -> None:
    """Tests + scripts that run the Anthropic client outside a runner
    context shouldn't crash on missing meter — the helper silently
    drops the credit."""
    from mantis_agent._anthropic.client import credit_claude_tokens_from_response
    # No meter published → no-op (no exception)
    credit_claude_tokens_from_response({"usage": {"input_tokens": 100, "output_tokens": 50}})
    credit_claude_tokens_from_response(None)
    credit_claude_tokens_from_response({"no_usage_block": True})
