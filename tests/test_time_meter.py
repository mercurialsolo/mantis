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
    here so a stealth rename / drop forces a docs + dashboard update.

    #421 added ``claude_verify_haiku`` + ``claude_verify_opus_escalation``
    so the cost report can show the Haiku/Opus split per run.
    """
    assert BUCKETS == (
        "perceive", "think", "act", "settle", "claude_ground",
        "claude_extract", "claude_verify",
        "claude_verify_haiku", "claude_verify_opus_escalation",
        "load", "overhead",
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


# ── XdotoolGymEnv / PlaywrightGymEnv credit ``act`` + ``perceive`` ──────


def test_xdotool_env_step_credits_act_via_dispatch() -> None:
    """``XdotoolGymEnv.step`` wraps ``_execute_action`` in ``act``;
    deeper helpers credit themselves. Verified end-to-end by patching
    ``_execute_action`` + ``_capture`` to instant returns and watching
    the totals."""
    from unittest.mock import patch

    from mantis_agent.actions import Action, ActionType
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    env = object.__new__(XdotoolGymEnv)
    env._human_speed = False
    env._settle_time = 0.0
    env._screenshot = lambda: None  # type: ignore[attr-defined]
    env._capture = lambda: object()  # type: ignore[attr-defined]
    env._execute_action = lambda _a: time.sleep(0.01)  # type: ignore[attr-defined]

    meter = TimeMeter()
    with publish_dispatch(meter, step_idx=0):
        # Patch adaptive_settle to a no-op so this test only measures
        # _execute_action time on the ``act`` bucket.
        with patch(
            "mantis_agent.gym.xdotool_env.adaptive_settle.is_enabled",
            return_value=False,
        ):
            env.step(Action(action_type=ActionType.CLICK, params={"x": 1, "y": 1}))
    assert meter.totals["act"] >= 0.01
    assert meter.per_step[0]["act"] >= 0.01


def test_xdotool_env_screenshot_credits_perceive_via_dispatch() -> None:
    """``XdotoolGymEnv.screenshot()`` credits ``perceive`` —
    ``_screenshot()`` (the private path used by adaptive_settle's frame
    polling) deliberately does NOT, to avoid double-counting settle."""
    from PIL import Image

    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    env = object.__new__(XdotoolGymEnv)
    img = Image.new("RGB", (8, 8), "white")
    env._screenshot = lambda: img  # type: ignore[attr-defined]

    meter = TimeMeter()
    with publish_dispatch(meter, step_idx=0):
        env.screenshot()
    assert meter.totals["perceive"] > 0.0
    assert meter.per_step[0]["perceive"] > 0.0


def test_xdotool_env_private_screenshot_does_not_credit_perceive() -> None:
    """Regression guard: adaptive_settle uses ``_screenshot`` directly
    inside its settle wait; that path must NOT credit ``perceive`` (it
    already credits ``settle`` once at the loop's end). Otherwise the
    two buckets compound for every settle call."""
    from PIL import Image

    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    env = object.__new__(XdotoolGymEnv)
    img = Image.new("RGB", (8, 8), "white")

    # Override _screenshot to skip the OS calls but keep the method on
    # the instance for the test.
    def _take():
        return img

    env._screenshot = _take  # type: ignore[attr-defined]

    meter = TimeMeter()
    with publish_dispatch(meter, step_idx=0):
        # Call ``_screenshot`` directly — must not credit any bucket.
        env._screenshot()
    assert meter.totals["perceive"] == 0.0


# ── ClaudeExtractor / ClaudeGrounding credit Claude buckets ────────────


def test_claude_extractor_call_credits_claude_extract_by_default() -> None:
    """``ClaudeExtractor._call`` records elapsed time to
    ``claude_extract`` via the dispatch context. Mocks out requests so
    no actual HTTP happens — only the timer matters."""
    from unittest.mock import MagicMock, patch

    from PIL import Image

    from mantis_agent.extraction import ClaudeExtractor

    extractor = ClaudeExtractor(api_key="x")
    img = Image.new("RGB", (8, 8), "white")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"content": [{"type": "text", "text": "ok"}]}

    def _post(*_a, **_kw):
        time.sleep(0.01)
        return fake_resp

    meter = TimeMeter()
    with publish_dispatch(meter, step_idx=0):
        with patch("requests.post", _post):
            extractor._call(img, "prompt")
    assert meter.totals["claude_extract"] >= 0.01
    assert meter.per_step[0]["claude_extract"] >= 0.01


def test_claude_extractor_verify_calls_credit_claude_verify_haiku() -> None:
    """``ClaudeExtractor.verify_gate`` routes through the Haiku verify
    client with ``time_bucket="claude_verify_haiku"`` so cost reports
    can split Haiku-default verify time from Opus escalation time
    (#421). Neither bucket should leak into ``claude_extract``."""
    from unittest.mock import MagicMock, patch

    from PIL import Image

    from mantis_agent.extraction import ClaudeExtractor

    extractor = ClaudeExtractor(api_key="x")
    img = Image.new("RGB", (8, 8), "white")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "content": [
            {"type": "tool_use", "name": "report_gate_verification",
             "input": {"passed": True, "reason": "ok"}},
        ],
    }

    def _post(*_a, **_kw):
        time.sleep(0.01)
        return fake_resp

    meter = TimeMeter()
    with publish_dispatch(meter, step_idx=0):
        with patch("requests.post", _post):
            extractor.verify_gate(img, "filters visible")
    # Haiku PASS — no escalation fires, so only the haiku bucket
    # should accrue time.
    assert meter.totals["claude_verify_haiku"] >= 0.01
    assert meter.totals["claude_verify_opus_escalation"] == 0.0
    assert meter.totals["claude_extract"] == 0.0


def test_claude_grounding_credits_claude_ground() -> None:
    """``ClaudeGrounding.ground`` wraps both cache hits and API calls in
    ``claude_ground``."""
    from unittest.mock import MagicMock, patch

    from PIL import Image

    from mantis_agent.grounding import ClaudeGrounding

    grounding = ClaudeGrounding(api_key="x")
    img = Image.new("RGB", (32, 32), "white")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"content": [{"type": "text", "text": "10 12"}]}

    def _post(*_a, **_kw):
        time.sleep(0.01)
        return fake_resp

    meter = TimeMeter()
    with publish_dispatch(meter, step_idx=0):
        with patch("requests.post", _post):
            grounding.ground(img, "click sign in", initial_x=5, initial_y=5)
    assert meter.totals["claude_ground"] >= 0.01


# ── think bucket ────────────────────────────────────────────────────────


def test_record_to_current_credits_think_bucket() -> None:
    """``GymRunner`` calls ``record_to_current("think", elapsed)``
    around every ``brain.think`` call. This unit test exercises the
    contract directly — the integration is verified via the wire-in
    in ``gym/runner.py``."""
    meter = TimeMeter()
    with publish_dispatch(meter, step_idx=2):
        record_to_current("think", 1.25)
    assert meter.totals["think"] == 1.25
    assert meter.per_step[2]["think"] == 1.25
