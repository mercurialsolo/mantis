"""Tests for the per-state-key fan-out dispatcher (experiments/holdout)."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "holdout"))

from state_key_dispatcher import Call, StateKeyDispatcher  # noqa: E402


class _OverlapTracker:
    """Records peak simultaneous executions and per-key overlap."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.live = 0
        self.peak = 0
        self.per_key_live: dict[str, int] = {}
        self.per_key_overlap: dict[str, int] = {}

    def enter(self, key: str) -> None:
        with self._lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
            cur = self.per_key_live.get(key, 0) + 1
            self.per_key_live[key] = cur
            if cur > 1:
                self.per_key_overlap[key] = self.per_key_overlap.get(key, 0) + 1

    def exit(self, key: str) -> None:
        with self._lock:
            self.live -= 1
            self.per_key_live[key] -= 1


def _make_work(tracker: _OverlapTracker, order: list, dwell: float = 0.05):
    def work(state_key: str):
        tracker.enter(state_key)
        order.append(state_key)
        time.sleep(dwell)
        tracker.exit(state_key)
        return state_key

    return work


def test_session_true_requires_state_key():
    with StateKeyDispatcher() as d:
        with pytest.raises(ValueError, match="requires an explicit state_key"):
            d.submit(lambda k: k, session=True)


def test_independent_calls_get_distinct_fresh_keys():
    tracker = _OverlapTracker()
    order: list[str] = []
    with StateKeyDispatcher(max_parallel=4) as d:
        results = d.run_all([Call(_make_work(tracker, order)) for _ in range(4)])
    # Each independent call ran under its own unique resolved key.
    assert len(set(results)) == 4
    # Distinct keys → no per-key overlap anywhere.
    assert tracker.per_key_overlap == {}


def test_independent_calls_run_in_parallel():
    tracker = _OverlapTracker()
    order: list[str] = []
    with StateKeyDispatcher(max_parallel=4) as d:
        d.run_all([Call(_make_work(tracker, order, dwell=0.1)) for _ in range(4)])
    # All four distinct-key works overlapped.
    assert tracker.peak >= 2


def test_independent_key_prefix_is_used():
    with StateKeyDispatcher() as d:
        key = d.submit(lambda k: k, state_key="rollout").result()
    assert key.startswith("rollout-")


def test_session_same_key_serializes_fifo():
    tracker = _OverlapTracker()
    order: list[str] = []

    def numbered(idx: int):
        def work(state_key: str):
            tracker.enter(state_key)
            order.append(idx)
            time.sleep(0.05)
            tracker.exit(state_key)
            return idx

        return work

    with StateKeyDispatcher(max_parallel=4) as d:
        futures = [
            d.submit(numbered(i), state_key="acme-session", session=True)
            for i in range(5)
        ]
        results = [f.result() for f in futures]

    # Never two at once on the same key, and strict submission (FIFO) order.
    assert tracker.per_key_overlap == {}
    assert order == [0, 1, 2, 3, 4]
    assert results == [0, 1, 2, 3, 4]


def test_distinct_session_keys_run_in_parallel():
    tracker = _OverlapTracker()
    order: list[str] = []
    with StateKeyDispatcher(max_parallel=4) as d:
        futures = [
            d.submit(
                _make_work(tracker, order, dwell=0.1),
                state_key=f"sess-{i}",
                session=True,
            )
            for i in range(3)
        ]
        [f.result() for f in futures]
    assert tracker.peak >= 2
    assert tracker.per_key_overlap == {}


def test_max_parallel_cap_respected():
    tracker = _OverlapTracker()
    order: list[str] = []
    with StateKeyDispatcher(max_parallel=2) as d:
        d.run_all([Call(_make_work(tracker, order, dwell=0.08)) for _ in range(6)])
    assert tracker.peak <= 2


def test_exception_propagates_through_future():
    def boom(state_key: str):
        raise RuntimeError("kaboom")

    with StateKeyDispatcher() as d:
        fut = d.submit(boom)
        with pytest.raises(RuntimeError, match="kaboom"):
            fut.result()


def test_session_failure_does_not_block_successor():
    order: list[int] = []

    def fail(state_key: str):
        raise RuntimeError("first fails")

    def ok(state_key: str):
        order.append(2)
        return "ok"

    with StateKeyDispatcher() as d:
        f1 = d.submit(fail, state_key="k", session=True)
        f2 = d.submit(ok, state_key="k", session=True)
        with pytest.raises(RuntimeError):
            f1.result()
        assert f2.result() == "ok"
    assert order == [2]


def test_mixed_independent_and_session():
    tracker = _OverlapTracker()
    order: list[str] = []
    with StateKeyDispatcher(max_parallel=4) as d:
        calls = [
            Call(_make_work(tracker, order)),  # independent
            Call(_make_work(tracker, order), state_key="s", session=True),
            Call(_make_work(tracker, order), state_key="s", session=True),
            Call(_make_work(tracker, order)),  # independent
        ]
        d.run_all(calls)
    # The two session-"s" calls never overlapped; independents free to overlap.
    assert "s" not in tracker.per_key_overlap


def test_tail_is_cleared_after_drain():
    with StateKeyDispatcher() as d:
        d.submit(lambda k: k, state_key="k", session=True).result()
        time.sleep(0.02)  # let the done-callback clear the tail
        assert d._tails.get("k") is None
