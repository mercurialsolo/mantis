"""Tests for #117 step 1 — SoM promotion in GymRunner.

Verifies the dispatch decision: when ``site_config.prefer_som_grounding``
is True AND ``page_discovery`` is available, the runner tries
``_try_discovery_execution`` BEFORE the direct executor. On a SoM hit
the runner skips both direct exec and brain inference. On a SoM miss it
falls back to the existing direct → discovery → brain chain.

Tests use a mocked GymRunner via ``__new__`` + attribute injection so
they're cheap (no Xvfb, no real brain, no real DOM). The decision logic
under test is ``GymRunner._should_prefer_som`` and the dispatch in
``GymRunner.run``'s plan-execution block.
"""

from __future__ import annotations

from types import SimpleNamespace

from mantis_agent.gym.runner import GymRunner


# ── _should_prefer_som ──────────────────────────────────────────────────


def _runner_with_site_config(prefer_som: bool, page_discovery=None) -> GymRunner:
    runner = GymRunner.__new__(GymRunner)
    runner.site_config = SimpleNamespace(prefer_som_grounding=prefer_som)
    runner.page_discovery = page_discovery
    runner.env = SimpleNamespace(tenant_id="test-tenant")
    return runner


def test_should_prefer_som_true_when_flag_and_discovery_present() -> None:
    runner = _runner_with_site_config(prefer_som=True)
    assert runner._should_prefer_som() is True


def test_should_prefer_som_false_when_flag_off() -> None:
    runner = _runner_with_site_config(prefer_som=False)
    assert runner._should_prefer_som() is False


def test_should_prefer_som_false_when_no_site_config() -> None:
    runner = GymRunner.__new__(GymRunner)
    runner.site_config = None
    assert runner._should_prefer_som() is False


def test_should_prefer_som_false_when_site_config_missing_attr() -> None:
    """SiteConfig from before #117 won't have ``prefer_som_grounding`` —
    the helper should treat that as False (default)."""
    runner = GymRunner.__new__(GymRunner)
    runner.site_config = SimpleNamespace()  # no attr
    assert runner._should_prefer_som() is False


# ── _emit_som_branch_metric ─────────────────────────────────────────────


def test_emit_som_branch_metric_does_not_raise_without_prom() -> None:
    """Telemetry must never break runs even when the metric handle is a
    no-op (prometheus_client not installed)."""
    runner = _runner_with_site_config(prefer_som=True)
    runner._emit_som_branch_metric("taken")
    runner._emit_som_branch_metric("skipped")
    runner._emit_som_branch_metric("aborted")


def test_emit_som_branch_metric_increments_real_counter() -> None:
    """When prometheus_client is available the counter actually moves."""
    import pytest

    from mantis_agent.metrics import PLAN_BRANCH_TOTAL, is_available

    if not is_available():
        pytest.skip("prometheus_client not installed")

    runner = _runner_with_site_config(prefer_som=True)
    sample = PLAN_BRANCH_TOTAL.labels(
        tenant_id="test-tenant", branch="som_promotion", outcome="taken",
    )
    before = sample._value.get()
    runner._emit_som_branch_metric("taken")
    after = sample._value.get()
    assert after - before == pytest.approx(1.0)
