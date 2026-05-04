"""Unit tests for the end-of-run reporter (Phase 1 of EPIC #161).

Pure-formatting tests: exercise the three public RunReporter helpers and
pin the output strings + costs-dict shape to what the pre-refactor
``MicroPlanRunner.run()`` emitted. Adding a new field to ``final_costs_dict``
is a downstream-visible change (build_micro_result reads it) — these
tests pin the existing field set so a careless rename surfaces here.

No browser, no Xvfb, no GymRunner, no extractor: pure dataclass + format.
"""

from __future__ import annotations

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.run_reporter import RunReporter


def _viable(idx: int, summary: str) -> StepResult:
    return StepResult(step_index=idx, intent="extract_data", success=True, data=summary)


def _fail(idx: int) -> StepResult:
    return StepResult(step_index=idx, intent="extract_data", success=False, data="")


def test_step_progress_line_zero_leads():
    """No leads in results → ``0 leads (0 phone)``, divisor falls back to 1."""
    line = RunReporter.step_progress_line(
        step_index=2,
        success=True,
        results=[],
        gpu_cost=0.10,
        claude_cost=0.20,
        proxy_cost=0.05,
        total_cost=0.35,
        elapsed_seconds=120.0,
    )
    assert "[ 2] OK" in line
    assert "0 leads (0 phone)" in line
    assert "$0.35 total" in line
    assert "GPU $0.10 Claude $0.20 Proxy $0.05" in line
    assert "2m" in line


def test_step_progress_line_failed_step():
    line = RunReporter.step_progress_line(
        step_index=7,
        success=False,
        results=[_fail(7)],
        gpu_cost=0.0,
        claude_cost=0.05,
        proxy_cost=0.0,
        total_cost=0.05,
        elapsed_seconds=30.0,
    )
    assert "[ 7] FAIL" in line


def test_step_progress_line_with_viable_lead():
    """Lead summary in result.data → counted by ListingDedup, divisor reflected."""
    results = [_viable(3, "VIABLE | year:2020 | make:Acme | url:http://x")]
    line = RunReporter.step_progress_line(
        step_index=3,
        success=True,
        results=results,
        gpu_cost=0.10,
        claude_cost=0.50,
        proxy_cost=0.05,
        total_cost=0.65,
        elapsed_seconds=600.0,
    )
    # ListingDedup reports 1 unique lead → cost-per-lead = total / 1 = $0.65
    assert "1 leads" in line
    assert "($0.65/lead" in line


def test_final_summary_lines_format():
    block = RunReporter.final_summary_lines(
        results=[],
        gpu_cost=0.12,
        claude_cost=0.34,
        proxy_cost=0.05,
        total_cost=0.51,
        elapsed_seconds=185.0,
        gpu_steps=4,
        claude_extract_calls=2,
        claude_grounding_calls=1,
        proxy_mb=12.3,
    )
    assert block[0] == "=" * 60
    assert block[1] == "MICRO-PLAN COMPLETE"
    assert block[2] == "  Time:     3m"
    assert block[3] == "  Steps:    0"
    assert block[4] == "  Leads:    0"
    assert block[5] == "  Phone:    0"
    assert block[6] == "  Cost:     $0.51 total ($0.51/lead, $0.51/phone lead)"
    assert block[7] == "    GPU:    $0.12 (4 steps)"
    assert block[8] == "    Claude: $0.34 (2 extract + 1 grounding)"
    assert block[9] == "    Proxy:  $0.05 (12 MB)"
    assert block[10] == "=" * 60


def test_final_costs_dict_shape_and_rounding():
    """Pin field set + 3-decimal rounding — build_micro_result reads this dict."""
    d = RunReporter.final_costs_dict(
        results=[],
        gpu_cost=0.123456,
        claude_cost=0.234567,
        proxy_cost=0.030001,
        total_cost=0.388024,
        final_status="completed",
        checkpoint_path="/data/checkpoints/r.json",
    )
    assert set(d.keys()) == {
        "total", "gpu", "claude", "proxy",
        "leads", "leads_with_phone",
        "per_lead", "per_phone_lead",
        "status", "checkpoint_path",
    }
    assert d["total"] == 0.388
    assert d["gpu"] == 0.123
    assert d["claude"] == 0.235
    assert d["proxy"] == 0.030
    assert d["leads"] == 0
    assert d["leads_with_phone"] == 0
    assert d["per_lead"] == 0.388        # divisor max(0,1) = 1
    assert d["per_phone_lead"] == 0.388
    assert d["status"] == "completed"
    assert d["checkpoint_path"] == "/data/checkpoints/r.json"


def test_final_costs_with_viable_leads_divides_correctly():
    """When unique leads > 0, cost-per-lead reflects the actual divisor."""
    results = [
        _viable(1, "VIABLE | year:2020 | make:A | url:http://x"),
        _viable(2, "VIABLE | year:2021 | make:B | url:http://y"),
    ]
    d = RunReporter.final_costs_dict(
        results=results,
        gpu_cost=0.0,
        claude_cost=0.40,
        proxy_cost=0.0,
        total_cost=0.40,
        final_status="completed",
        checkpoint_path="",
    )
    assert d["leads"] == 2
    # Both leads lack a phone → leads_with_phone=0; per_phone_lead falls back to total / 1
    assert d["leads_with_phone"] == 0
    assert d["per_lead"] == 0.20
    assert d["per_phone_lead"] == 0.40
