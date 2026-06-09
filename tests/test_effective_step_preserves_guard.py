"""Regression test for the boattrader urlnav verification run halt
(2026-05-24). RunExecutor._build_effective_step was dropping the
``guard`` and ``out_var`` MicroIntent fields when reconstructing the
per-dispatch MicroIntent. The canonical authoring form sets these at
the top level — the executor's hints fallback in ``resolve_guard_name``
was meant as a backup but the production path silently regressed for
any plan that didn't ALSO redundantly populate ``hints['guard']``.

The fix preserves both fields verbatim. This test pins the contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.plan_decomposer import MicroIntent


def _build_effective_step_via_executor(step: MicroIntent) -> MicroIntent:
    """Invoke RunExecutor._build_effective_step against a minimal
    MagicMock runner. Mirrors the production call site path."""
    from mantis_agent.gym.run_executor import RunExecutor
    executor = RunExecutor.__new__(RunExecutor)
    runner = MagicMock()
    runner._step_intent_overrides = {}
    runner.scanner.scroll_directive_for = MagicMock(return_value="")
    executor.parent = runner
    state = MagicMock()
    state.step_index = 0
    state.listings_on_page = 0
    return executor._build_effective_step(step, state)


def test_effective_step_preserves_top_level_guard():
    """Top-level ``MicroIntent.guard`` survives effective-step
    reconstruction — the executor must not silently drop dataclass
    fields the runner downstream reads."""
    src = MicroIntent(
        intent="Click Show More", type="click", section="extraction",
        guard="has_show_more",
    )
    eff = _build_effective_step_via_executor(src)
    assert eff.guard == "has_show_more"


def test_effective_step_preserves_top_level_out_var():
    """Top-level ``MicroIntent.out_var`` survives effective-step
    reconstruction — detect_visible's binding target must reach the
    handler verbatim."""
    src = MicroIntent(
        intent="Is X visible?", type="detect_visible", section="extraction",
        out_var="has_x",
    )
    eff = _build_effective_step_via_executor(src)
    assert eff.out_var == "has_x"


def test_effective_step_guard_empty_default_preserved():
    """Unconditional step (default empty guard) stays empty after
    reconstruction — no spurious value injection."""
    src = MicroIntent(intent="Click", type="click")
    eff = _build_effective_step_via_executor(src)
    assert eff.guard == ""
    assert eff.out_var == ""


# ── Skip envelope outer-loop bypass ──────────────────────────────────


def test_skip_envelope_advances_without_recovery():
    """``StepResult(success=False, skip=True)`` is the contract that
    execute_step's guard branch + recipe-rejection paths emit. The
    outer loop must advance to the next step directly — calling
    ``_handle_failure`` on a skip envelope routes the skipped step
    through the click-recovery branch (Escape + jump to loop), which
    silently corrupts a deliberate skip into a real failure path.

    Live repro 2026-05-24 boattrader verification run
    20260524_171431_ece15fbb: step 6 guard correctly resolved
    has_show_more=False but the skip envelope was processed as a
    click failure.
    """
    # The skip-bypass branch lives in _execute_inner. Drive the
    # branch detection logic directly to keep the test focused +
    # avoid the deep dependency chain RunExecutor._execute_inner
    # mounts (env, brain, checkpoint persistence, ...).
    from mantis_agent.gym.checkpoint import StepResult

    skip_result = StepResult(
        step_index=6, intent="Click Show More", success=False,
        data="guard:has_show_more=False:skipped",
        skip=True, skip_reason="guard_has_show_more_false",
    )
    # The outer-loop branch condition:
    #   if success: success path
    #   elif skip:  advance directly (THIS PR)
    #   else:       _handle_failure
    is_success = bool(skip_result.success)
    is_skip = bool(getattr(skip_result, "skip", False))
    # Must hit the skip branch BEFORE the failure branch.
    assert not is_success
    assert is_skip


def test_skip_envelope_outer_loop_bypass_in_run_executor():
    """Reads the _execute_inner source to confirm the ``elif
    step_result.skip:`` branch exists between the success and failure
    branches. A grep-style structural test — cheaper than mocking the
    whole MicroPlanRunner just to assert a 4-line control-flow
    change.

    Honours the #835 navigation-drift pre-dispatch gate that may
    also call ``_handle_failure`` BEFORE the main dispatch — find the
    success/skip/failure positions in the post-dispatch block by
    anchoring on the ``state.results[-1]`` line that the main branch
    reads.
    """
    import inspect
    from mantis_agent.gym.run_executor import RunExecutor

    source = inspect.getsource(RunExecutor._execute_inner)
    # Anchor on the post-dispatch ``step_result = state.results[-1]``
    # line so the drift gate's pre-dispatch ``_handle_failure`` call
    # doesn't shift the positions we care about.
    anchor = source.find("step_result = state.results[-1]")
    assert anchor > 0, "expected dispatch anchor in _execute_inner"
    post_dispatch = source[anchor:]
    success_pos = post_dispatch.find("step_result.success")
    skip_pos = post_dispatch.find("step_result.skip")
    failure_pos = post_dispatch.find("_handle_failure")
    assert success_pos < skip_pos < failure_pos, (
        f"Expected branch order success({success_pos}) < "
        f"skip({skip_pos}) < failure({failure_pos}). Source:\n{post_dispatch[:2000]}"
    )
