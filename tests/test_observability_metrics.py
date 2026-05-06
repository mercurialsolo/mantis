"""Per-action observability metrics — emission tests for #156.

Pin the metric handles + the emission sites that produce them. Uses
the prometheus_client REGISTRY to read counter values back; tests are
deterministic because each one reads a delta against its own baseline.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.run_executor import _emit_action_metric
from mantis_agent.metrics import (
    ACTION_TOTAL,
    BRAIN_ESCALATION_TOTAL,
    LOOP_TERMINATION_TOTAL,
    is_available,
)
from mantis_agent.plan_decomposer import MicroIntent


pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="prometheus_client not installed — metrics are no-ops",
)


def _counter_value(counter, **labels) -> float:
    """Read a labelled counter's current value, handling missing labels."""
    try:
        sample = counter.labels(**labels)
        return sample._value.get()
    except Exception:
        return 0.0


# ── ACTION_TOTAL ───────────────────────────────────────────────────────


def test_action_metric_emits_success_for_passing_step():
    runner = MagicMock()
    runner.tenant_id = "tenant-abc"
    step = MicroIntent(intent="go", type="navigate")
    result = StepResult(step_index=0, intent="go", success=True)

    before = _counter_value(
        ACTION_TOTAL, tenant_id="tenant-abc", step_kind="navigate", outcome="success",
    )
    _emit_action_metric(runner, step, result)
    after = _counter_value(
        ACTION_TOTAL, tenant_id="tenant-abc", step_kind="navigate", outcome="success",
    )
    assert after - before == pytest.approx(1.0)


def test_action_metric_emits_failed_outcome():
    runner = MagicMock()
    runner.tenant_id = "tenant-fail"
    step = MicroIntent(intent="x", type="click")
    result = StepResult(step_index=0, intent="x", success=False)

    before = _counter_value(
        ACTION_TOTAL, tenant_id="tenant-fail", step_kind="click", outcome="failed",
    )
    _emit_action_metric(runner, step, result)
    after = _counter_value(
        ACTION_TOTAL, tenant_id="tenant-fail", step_kind="click", outcome="failed",
    )
    assert after - before == pytest.approx(1.0)


def test_action_metric_emits_duplicate_when_data_contains_duplicate():
    runner = MagicMock()
    runner.tenant_id = ""
    step = MicroIntent(intent="ex", type="extract_url")
    # Success is True but data carries DUPLICATE — runtime treats this as a
    # dedup skip, not a green-path completion.
    result = StepResult(
        step_index=0, intent="ex", success=True, data="DUPLICATE:url-already-seen",
    )

    before = _counter_value(
        ACTION_TOTAL, tenant_id="", step_kind="extract_url", outcome="duplicate",
    )
    _emit_action_metric(runner, step, result)
    after = _counter_value(
        ACTION_TOTAL, tenant_id="", step_kind="extract_url", outcome="duplicate",
    )
    assert after - before == pytest.approx(1.0)


def test_action_metric_emits_filters_not_applied_outcome():
    runner = MagicMock()
    runner.tenant_id = "t"
    step = MicroIntent(intent="paginate", type="paginate")
    result = StepResult(
        step_index=0, intent="paginate", success=False, data="filters_not_applied",
    )

    before = _counter_value(
        ACTION_TOTAL,
        tenant_id="t", step_kind="paginate", outcome="filters_not_applied",
    )
    _emit_action_metric(runner, step, result)
    after = _counter_value(
        ACTION_TOTAL,
        tenant_id="t", step_kind="paginate", outcome="filters_not_applied",
    )
    assert after - before == pytest.approx(1.0)


def test_action_metric_swallows_telemetry_errors():
    """If the metric emit ever raises, the runner must not crash. We force
    a label-set explosion by passing a step with type=None — the helper
    falls back to ``unknown`` rather than raising."""
    runner = MagicMock()
    runner.tenant_id = "t"
    step = MicroIntent(intent="x", type=None)  # type: ignore[arg-type]
    result = StepResult(step_index=0, intent="x", success=True)

    # Should not raise.
    _emit_action_metric(runner, step, result)
    assert _counter_value(
        ACTION_TOTAL, tenant_id="t", step_kind="unknown", outcome="success",
    ) >= 1.0


# ── LOOP_TERMINATION_TOTAL ─────────────────────────────────────────────


def test_loop_termination_metric_increments_for_each_status():
    """Every call to ``LOOP_TERMINATION_TOTAL.labels(...).inc()`` shows up
    in the registry. The actual emission happens in
    :meth:`RunExecutor._finalize`; we validate the metric handle here so
    a future refactor that drops the call gets caught by the dedicated
    executor tests below."""
    before_completed = _counter_value(
        LOOP_TERMINATION_TOTAL, tenant_id="t", reason="completed",
    )
    LOOP_TERMINATION_TOTAL.labels(tenant_id="t", reason="completed").inc()
    after_completed = _counter_value(
        LOOP_TERMINATION_TOTAL, tenant_id="t", reason="completed",
    )
    assert after_completed - before_completed == pytest.approx(1.0)

    before_halted = _counter_value(
        LOOP_TERMINATION_TOTAL, tenant_id="t", reason="halted",
    )
    LOOP_TERMINATION_TOTAL.labels(tenant_id="t", reason="halted").inc()
    after_halted = _counter_value(
        LOOP_TERMINATION_TOTAL, tenant_id="t", reason="halted",
    )
    assert after_halted - before_halted == pytest.approx(1.0)


# ── BRAIN_ESCALATION_TOTAL ─────────────────────────────────────────────


# ── GroundingCache metrics (#117) ──────────────────────────────────────


def test_grounding_cache_emits_hit_and_miss_counters():
    """Cache get/put cycles bump the labelled counters."""
    from PIL import Image

    from mantis_agent.grounding import GroundingResult
    from mantis_agent.grounding_cache import GroundingCache
    from mantis_agent.metrics import (
        GROUNDING_CACHE_HITS_TOTAL,
        GROUNDING_CACHE_MISSES_TOTAL,
    )

    cache = GroundingCache(tenant_id="t-grounding")
    img = Image.new("RGB", (256, 256), (50, 50, 50))
    result = GroundingResult(x=10, y=20, confidence=0.9, description="d")

    miss_before = _counter_value(
        GROUNDING_CACHE_MISSES_TOTAL, tenant_id="t-grounding",
    )
    hit_before = _counter_value(
        GROUNDING_CACHE_HITS_TOTAL, tenant_id="t-grounding",
    )

    key = cache.make_key(img, "click first card", initial_x=80, initial_y=80)
    assert cache.get(key) is None  # miss
    cache.put(key, result)
    assert cache.get(key) is not None  # hit

    miss_after = _counter_value(
        GROUNDING_CACHE_MISSES_TOTAL, tenant_id="t-grounding",
    )
    hit_after = _counter_value(
        GROUNDING_CACHE_HITS_TOTAL, tenant_id="t-grounding",
    )
    assert miss_after - miss_before == pytest.approx(1.0)
    assert hit_after - hit_before == pytest.approx(1.0)


def test_grounding_cache_emits_eviction_counter_at_capacity():
    from mantis_agent.grounding import GroundingResult
    from mantis_agent.grounding_cache import GroundingCache
    from mantis_agent.metrics import GROUNDING_CACHE_EVICTIONS_TOTAL

    cache = GroundingCache(max_entries=2, tenant_id="t-evict")
    r = GroundingResult(x=1, y=1, confidence=0.9)

    before = _counter_value(GROUNDING_CACHE_EVICTIONS_TOTAL, tenant_id="t-evict")
    cache.put("a", r)
    cache.put("b", r)
    cache.put("c", r)  # evicts "a"
    after = _counter_value(GROUNDING_CACHE_EVICTIONS_TOTAL, tenant_id="t-evict")
    assert after - before == pytest.approx(1.0)


def test_brain_escalation_metric_increments_on_ladder_think():
    """Smoke test the brain ladder emits one escalation counter per
    ``think()`` call, labelled with which brain handled the call."""
    from mantis_agent.brain_ladder import BrainLadder

    primary = MagicMock()
    primary.think.return_value = MagicMock(
        action=MagicMock(action_type=MagicMock(value="click")),
        thinking="t",
    )
    fallback = MagicMock()
    fallback.think.return_value = MagicMock(
        action=MagicMock(action_type=MagicMock(value="click")),
        thinking="t",
    )

    ladder = BrainLadder(primary, fallback)

    before_primary = _counter_value(
        BRAIN_ESCALATION_TOTAL,
        tenant_id="", from_brain="primary", to_brain="primary",
    )
    ladder.think(frames=[], task="t")
    after_primary = _counter_value(
        BRAIN_ESCALATION_TOTAL,
        tenant_id="", from_brain="primary", to_brain="primary",
    )
    assert after_primary - before_primary == pytest.approx(1.0)

    before_fallback = _counter_value(
        BRAIN_ESCALATION_TOTAL,
        tenant_id="", from_brain="primary", to_brain="fallback",
    )
    ladder.force_fallback()
    ladder.think(frames=[], task="t")
    after_fallback = _counter_value(
        BRAIN_ESCALATION_TOTAL,
        tenant_id="", from_brain="primary", to_brain="fallback",
    )
    assert after_fallback - before_fallback == pytest.approx(1.0)
