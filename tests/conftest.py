"""Shared pytest fixtures.

The autouse fixture below short-circuits the #294 adaptive-settle helpers
to instant returns for the broad set of unit tests that drive step
handlers and the runner. Those tests historically rely on
``monkeypatch.setattr("step_handlers.X.time.sleep", lambda *_: None)`` to
avoid real waits. The #294 extension routes settles through
:mod:`mantis_agent.gym.adaptive_settle`, which calls ``time.sleep`` in
its own module namespace — the per-test patches don't intercept it, and
tests that pin a ``MagicMock`` env hang for the full ``max_seconds`` cap.

Patching the three adaptive-settle entry points to instant returns is
both pragmatic and faithful to the test intent ("don't actually wait
in this unit test"). Production behaviour is unchanged.

The adaptive-settle helper's own tests opt out by filename so the gate
logic is still exercised end-to-end.
"""

from __future__ import annotations

import pytest


_SETTLE_TEST_FILES: frozenset[str] = frozenset({
    "test_adaptive_settle.py",
    "test_adaptive_content_settle.py",
    "test_adaptive_settle_end_to_end_budget.py",
    # Epic #362: TimeMeter tests need the real adaptive_settle to verify
    # the dispatch-context credit path. None of the existing TimeMeter
    # cases trigger a real wait (they call adaptive_settle directly
    # with tiny budgets), so the suite stays fast.
    "test_time_meter.py",
})


@pytest.fixture(autouse=True)
def _instant_adaptive_settle_in_tests(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make adaptive-settle helpers instant-return in unit tests."""
    fname = request.node.fspath.basename
    if fname in _SETTLE_TEST_FILES:
        return

    def _instant(*args: object, **kwargs: object) -> float:
        return 0.0

    monkeypatch.setattr(
        "mantis_agent.gym.adaptive_settle.settle_after_action",
        _instant,
        raising=False,
    )
    monkeypatch.setattr(
        "mantis_agent.gym.adaptive_settle.wait_until_stable",
        _instant,
        raising=False,
    )
    monkeypatch.setattr(
        "mantis_agent.gym.adaptive_settle.wait_for_networkidle",
        _instant,
        raising=False,
    )
