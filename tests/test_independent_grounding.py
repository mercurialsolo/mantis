"""Tests for #181 — independent grounding routing + force-compute bypass + metrics.

Pin the policy decision (when to bypass the GroundingCache), the
``force_compute`` plumbing through the grounder, and the new Prometheus
counters/histogram.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from mantis_agent.grounding import (
    ClaudeGrounding,
    GroundingResult,
    LLMGrounding,
    PassthroughGrounding,
    RegionGrounding,
)
from mantis_agent.grounding_cache import GroundingCache
from mantis_agent.gym.step_handlers.click import (
    _emit_grounding_metrics,
    _should_force_independent_grounding,
)
from mantis_agent.metrics import (
    GROUNDING_CALL_TOTAL,
    GROUNDING_CORRECTION_DISTANCE,
    is_available,
)
from mantis_agent.site_config import SiteConfig


def _step(
    *,
    type_: str = "click",
    section: str = "extraction",
    hints: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        type=type_,
        section=section,
        hints=hints or {},
    )


# ── _should_force_independent_grounding ────────────────────────────────


def test_force_off_by_default():
    """Without site config or step hint, no force."""
    sc = SiteConfig()
    assert _should_force_independent_grounding(_step(), sc) is False


def test_force_via_step_hint():
    """A plan author can pin a single step to force grounding."""
    sc = SiteConfig()
    step = _step(hints={"independent_grounding": True})
    assert _should_force_independent_grounding(step, sc) is True


def test_force_via_site_config_layout():
    """SiteConfig flag opts whole layouts in via the ``layout`` hint."""
    sc = SiteConfig(require_independent_grounding=("listings",))
    step = _step(hints={"layout": "listings"})
    assert _should_force_independent_grounding(step, sc) is True


def test_force_via_site_config_section():
    """Or via the ``section`` field."""
    sc = SiteConfig(require_independent_grounding=("extraction",))
    step = _step(section="extraction")
    assert _should_force_independent_grounding(step, sc) is True


def test_force_via_site_config_step_type():
    """Or via the literal step.type."""
    sc = SiteConfig(require_independent_grounding=("click",))
    step = _step(type_="click")
    assert _should_force_independent_grounding(step, sc) is True


def test_force_off_when_site_config_doesnt_match():
    sc = SiteConfig(require_independent_grounding=("paginate",))
    step = _step(hints={"layout": "listings"}, type_="click")
    assert _should_force_independent_grounding(step, sc) is False


def test_force_off_when_no_site_config():
    """Defensive: a runner built without site_config still works."""
    assert _should_force_independent_grounding(_step(), None) is False


def test_step_hint_truthy_value_counts():
    """``independent_grounding`` is treated as truthy — any non-falsy
    value enables forcing. Lets plans use ``"yes"`` or ``1`` interchangeably."""
    sc = SiteConfig()
    assert _should_force_independent_grounding(
        _step(hints={"independent_grounding": "yes"}), sc,
    ) is True
    assert _should_force_independent_grounding(
        _step(hints={"independent_grounding": 0}), sc,
    ) is False


# ── force_compute plumbing through grounders ───────────────────────────


def _img() -> Image.Image:
    return Image.new("RGB", (256, 256))


def test_passthrough_accepts_force_compute_kwarg():
    """Sanity: every concrete grounder accepts the new kwarg without
    breaking the legacy positional API."""
    g = PassthroughGrounding()
    out = g.ground(_img(), "click X", 100, 200, force_compute=True)
    assert isinstance(out, GroundingResult)


def test_region_accepts_force_compute_kwarg():
    g = RegionGrounding()
    out = g.ground(_img(), "click X", 100, 200, force_compute=True)
    assert isinstance(out, GroundingResult)


def test_claude_grounding_force_compute_bypasses_cache(monkeypatch):
    """``force_compute=True`` must skip the GroundingCache hit-or-miss
    path entirely and call ``_ground_remote`` directly."""
    cache = GroundingCache()
    grounder = ClaudeGrounding(api_key="dummy", cache=cache)

    # Pre-populate the cache so a normal call would short-circuit.
    img = _img()
    key = cache.make_key(img, "title-text", 100, 200)
    cached = GroundingResult(x=999, y=999, confidence=0.99, description="cached")
    cache.put(key, cached)

    # Spy the remote call.
    called: list[bool] = []

    def _stub_remote(self, screenshot, description, ix=None, iy=None):
        called.append(True)
        return GroundingResult(x=42, y=43, confidence=0.8)

    monkeypatch.setattr(ClaudeGrounding, "_ground_remote", _stub_remote, raising=True)

    # Without force_compute → cache hit.
    out = grounder.ground(img, "title-text", 100, 200, force_compute=False)
    assert out.x == 999  # came from cache
    assert called == []

    # With force_compute → cache bypassed.
    out = grounder.ground(img, "title-text", 100, 200, force_compute=True)
    assert out.x == 42  # fresh remote
    assert called == [True]


def test_llm_grounding_force_compute_bypasses_cache(monkeypatch):
    cache = GroundingCache()
    grounder = LLMGrounding(cache=cache)

    img = _img()
    key = cache.make_key(img, "title", 100, 200)
    cache.put(key, GroundingResult(x=11, y=22, confidence=0.9))

    called: list[bool] = []

    def _stub_remote(self, screenshot, description, ix=None, iy=None):
        called.append(True)
        return GroundingResult(x=33, y=44, confidence=0.8)

    monkeypatch.setattr(LLMGrounding, "_ground_remote", _stub_remote, raising=True)

    out = grounder.ground(img, "title", 100, 200, force_compute=False)
    assert out.x == 11
    out = grounder.ground(img, "title", 100, 200, force_compute=True)
    assert out.x == 33
    assert called == [True]


# ── Metric emission ────────────────────────────────────────────────────


pytestmark_metrics = pytest.mark.skipif(
    not is_available(), reason="prometheus_client not installed",
)


@pytestmark_metrics
def test_grounding_call_metric_increments_with_force_label():
    runner = MagicMock()
    runner.tenant_id = "t-181"
    sample = GROUNDING_CALL_TOTAL.labels(
        tenant_id="t-181", outcome="accepted", force_compute="true",
    )
    before = sample._value.get()
    _emit_grounding_metrics(runner, dx=10, dy=20, outcome="accepted", force=True)
    assert sample._value.get() - before == pytest.approx(1.0)


@pytestmark_metrics
def test_correction_distance_observation_recorded():
    """Histogram doesn't expose a single-counter view, but we can read
    the sum for a labelled stream and confirm it increased."""
    runner = MagicMock()
    runner.tenant_id = "t-181-dist"
    hist = GROUNDING_CORRECTION_DISTANCE.labels(
        tenant_id="t-181-dist", force_compute="false",
    )
    before = hist._sum.get()
    # 3-4-5 triangle: distance = 5
    _emit_grounding_metrics(runner, dx=3, dy=4, outcome="accepted", force=False)
    after = hist._sum.get()
    assert after - before == pytest.approx(5.0, rel=0.01)


@pytestmark_metrics
def test_emit_does_not_raise_on_telemetry_error(monkeypatch):
    """If the Prometheus call ever raises, the runner must keep going."""
    runner = MagicMock()
    runner.tenant_id = "t"
    # Force a TypeError inside the labelled call.
    bad = MagicMock()
    bad.labels.side_effect = RuntimeError("boom")
    monkeypatch.setattr(
        "mantis_agent.metrics.GROUNDING_CALL_TOTAL", bad,
    )
    # Should not raise.
    _emit_grounding_metrics(runner, dx=1, dy=1, outcome="accepted", force=False)


# ── SiteConfig round-trip ──────────────────────────────────────────────


def test_site_config_round_trip_preserves_tuple_typing():
    """``require_independent_grounding`` round-trips as JSON list and
    must come back as a tuple so equality comparisons stay stable."""
    sc = SiteConfig(require_independent_grounding=("listings", "card_grid"))
    payload = sc.to_dict()
    assert isinstance(payload["require_independent_grounding"], list)
    restored = SiteConfig.from_dict(payload)
    assert restored.require_independent_grounding == ("listings", "card_grid")
    assert isinstance(restored.require_independent_grounding, tuple)
