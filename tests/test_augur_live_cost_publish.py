"""#659 follow-up — ``set_costs`` publishes the run-level cost
rollup on every step emission, not just at finalize.

Before this fix the Augur Runs-list COST column showed a stale
value (typically $0.00 or the last per-step delta) for the entire
duration of a live run — ``_emit_augur_aggregate_metrics`` was the
only caller of ``set_costs`` and it only ran from ``_finalize``.
For a 60-minute run the user saw misleadingly understated cost
right up until terminal status flipped.

Contract pinned here:
    - ``_publish_run_costs_to_augur`` reads ``cost_meter.totals()``
      and forwards them to ``AugurAdapter.set_costs`` (the SDK's
      canonical run-cost surface, augur-sdk 0.1.8+).
    - ``_emit_augur_step`` calls the helper after every step.
    - ``_emit_augur_aggregate_metrics`` still calls the helper at
      finalize (one source of truth — same call shape).
    - Best-effort: a missing augur adapter / cost meter / SDK method
      never breaks the run.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.gym.run_executor import RunExecutor


def _build_executor_with_augur(total: float = 4.61) -> tuple:
    """Construct a minimal RunExecutor with a spy AugurAdapter + a
    cost-meter that returns the given total. Returns (executor, spy)."""
    executor = RunExecutor.__new__(RunExecutor)
    runner = MagicMock()
    augur = MagicMock()
    augur.active = True
    runner._augur = augur
    # totals() returns (gpu, claude, proxy, total).
    runner.cost_meter.totals = MagicMock(return_value=(0.0, 2.20, 2.41, total))
    runner.cost_meter.costs = {
        "claude_input_tokens": 1000,
        "claude_output_tokens": 500,
        "claude_cached_input_tokens": 200,
    }
    runner.cost_meter.elapsed_seconds = MagicMock(return_value=120.0)
    executor.parent = runner
    return executor, augur


def test_publish_run_costs_forwards_totals_to_set_costs():
    executor, augur = _build_executor_with_augur(total=4.61)
    executor._publish_run_costs_to_augur()
    augur.set_costs.assert_called_once()
    kwargs = augur.set_costs.call_args.kwargs
    assert kwargs["total_usd"] == 4.61
    assert kwargs["model_usd"] == 2.20
    assert kwargs["proxy_usd"] == 2.41
    assert kwargs["gpu_usd"] == 0.0
    assert kwargs["tokens_in"] == 1000
    assert kwargs["tokens_out"] == 500
    assert kwargs["cache_hit_tokens"] == 200


def test_publish_run_costs_idempotent_repeated_calls_keep_publishing():
    """Multiple calls re-publish the latest totals — the per-step
    hook will call this hundreds of times during a long inner loop."""
    executor, augur = _build_executor_with_augur(total=1.0)
    for _ in range(5):
        executor._publish_run_costs_to_augur()
    assert augur.set_costs.call_count == 5


def test_publish_run_costs_no_op_when_augur_inactive():
    executor = RunExecutor.__new__(RunExecutor)
    runner = MagicMock()
    augur = MagicMock()
    augur.active = False
    runner._augur = augur
    executor.parent = runner
    executor._publish_run_costs_to_augur()
    augur.set_costs.assert_not_called()


def test_publish_run_costs_no_op_when_no_augur():
    executor = RunExecutor.__new__(RunExecutor)
    runner = MagicMock()
    runner._augur = None
    executor.parent = runner
    executor._publish_run_costs_to_augur()  # no crash


def test_publish_run_costs_no_op_when_no_cost_meter():
    executor = RunExecutor.__new__(RunExecutor)
    runner = MagicMock()
    augur = MagicMock()
    augur.active = True
    runner._augur = augur
    runner.cost_meter = None
    executor.parent = runner
    executor._publish_run_costs_to_augur()
    augur.set_costs.assert_not_called()


def test_publish_run_costs_swallows_totals_exception():
    """A failing cost-meter must never break a run — the helper
    catches and demotes to debug log."""
    executor = RunExecutor.__new__(RunExecutor)
    runner = MagicMock()
    augur = MagicMock()
    augur.active = True
    runner._augur = augur
    runner.cost_meter.totals = MagicMock(side_effect=RuntimeError("simulated"))
    executor.parent = runner
    # Should not raise.
    executor._publish_run_costs_to_augur()
    augur.set_costs.assert_not_called()


def test_publish_run_costs_handles_none_token_counts():
    """``set_costs`` accepts ``None`` for token fields — preserve
    the ``or None`` chain that converts 0 to None to keep the
    Runs-list inspector tidy (a 0-token reading is rendered as
    blank instead of '0 tokens')."""
    executor, augur = _build_executor_with_augur()
    executor.parent.cost_meter.costs = {
        "claude_input_tokens": 0,
        "claude_output_tokens": 0,
        "claude_cached_input_tokens": 0,
    }
    executor._publish_run_costs_to_augur()
    kwargs = augur.set_costs.call_args.kwargs
    assert kwargs["tokens_in"] is None
    assert kwargs["tokens_out"] is None
    assert kwargs["cache_hit_tokens"] is None
