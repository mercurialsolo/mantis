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
