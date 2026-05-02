"""Tests for #122 — CostConfig env-overridable rates."""

from __future__ import annotations

import pytest

from mantis_agent.cost_config import CostConfig


def test_defaults_match_legacy_hardcoded_values() -> None:
    """The previous MicroPlanRunner hardcoded these constants; preserve them."""
    cfg = CostConfig()
    assert cfg.gpu_hourly_usd == 3.25
    assert cfg.claude_call_usd == 0.003
    assert cfg.proxy_per_gb_usd == 5.0
    assert cfg.gpu_seconds_per_step == 3.0
    assert cfg.proxy_mb_per_nav == 5.0
    assert cfg.proxy_mb_per_scroll == 0.5


def test_from_env_with_no_overrides_returns_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "MANTIS_COST_GPU_HOURLY_USD",
        "MANTIS_COST_CLAUDE_CALL_USD",
        "MANTIS_COST_PROXY_PER_GB_USD",
        "MANTIS_COST_GPU_SECONDS_PER_STEP",
        "MANTIS_COST_PROXY_MB_PER_NAV",
        "MANTIS_COST_PROXY_MB_PER_SCROLL",
    ):
        monkeypatch.delenv(key, raising=False)
    assert CostConfig.from_env() == CostConfig()


def test_from_env_applies_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_COST_GPU_HOURLY_USD", "1.50")
    monkeypatch.setenv("MANTIS_COST_CLAUDE_CALL_USD", "0.005")
    monkeypatch.setenv("MANTIS_COST_PROXY_PER_GB_USD", "10.0")
    cfg = CostConfig.from_env()
    assert cfg.gpu_hourly_usd == 1.50
    assert cfg.claude_call_usd == 0.005
    assert cfg.proxy_per_gb_usd == 10.0


def test_from_env_invalid_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_COST_GPU_HOURLY_USD", "not-a-number")
    with pytest.raises(ValueError, match="MANTIS_COST_GPU_HOURLY_USD"):
        CostConfig.from_env()


def test_gpu_cost_scales_linearly() -> None:
    cfg = CostConfig(gpu_hourly_usd=4.0)
    # 30 minutes at $4/h = $2
    assert cfg.gpu_cost(1800) == pytest.approx(2.0)


def test_claude_cost_scales_linearly() -> None:
    cfg = CostConfig(claude_call_usd=0.005)
    assert cfg.claude_cost(10) == pytest.approx(0.05)


def test_proxy_cost_per_gb() -> None:
    cfg = CostConfig(proxy_per_gb_usd=8.0)
    # 512 MB → 0.5 GB → $4
    assert cfg.proxy_cost(512) == pytest.approx(4.0)


def test_dataclass_is_frozen() -> None:
    cfg = CostConfig()
    with pytest.raises(Exception):  # noqa: PT011 — dataclasses raises FrozenInstanceError
        cfg.gpu_hourly_usd = 99.0  # type: ignore[misc]
