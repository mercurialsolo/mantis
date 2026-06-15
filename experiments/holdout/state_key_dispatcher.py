"""Client-side fan-out dispatcher that respects Mantis's per-state-key rule.

Mantis cannot service two concurrent calls that share a ``state_key`` — the key
resolves (``server_utils.resolve_ids``) to a Chrome ``profile_id`` (user-data-dir)
*and* a ``workflow_id`` (checkpoint key), so two in-flight runs on one key race on
the same browser profile + checkpoint. Distinct state_keys are safe to run in
parallel (Modal scales out containers via ``@modal.concurrent``).

This dispatcher encodes exactly that constraint, with the collision policy chosen
**per call** (the user's "Both, caller chooses per-call"):

* **independent** (``session=False``, the default) — the call has no affinity to
  any prior run, so the dispatcher auto-allocates a *fresh* unique state_key and
  runs it immediately, in parallel up to ``max_parallel``. This is the
  max-throughput path (e.g. GRPO siblings, each its own profile/checkpoint).
* **session** (``session=True``) — the call must reuse a caller-supplied
  ``state_key`` (a logged-in profile, a resumable checkpoint), so it is queued
  **FIFO** behind any other session call on that same key and run one-at-a-time.
  Different session keys still run in parallel.

The ``work`` callable receives the *resolved* state_key as its only argument, so
it can stamp ``profile_id``/``workflow_id`` on the submitted suite. Everything is
pure-Python + threads (no network), so the queueing/parallelism logic is
unit-testable — see ``tests/test_state_key_dispatcher.py``.

    with StateKeyDispatcher(max_parallel=4) as d:
        # 4 independent rollouts in parallel, each its own fresh key
        results = d.run_all([Call(make_work(spec)) for spec in specs])

        # two calls that must share one logged-in profile → serialized FIFO
        f1 = d.submit(step_a, state_key="acme-session", session=True)
        f2 = d.submit(step_b, state_key="acme-session", session=True)  # waits for f1
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

# A unit of work. ``work`` is called with the resolved state_key (a str) and
# returns whatever the caller wants back from ``Future.result()``.
Work = Callable[[str], Any]


@dataclass
class Call:
    """One dispatch request. ``state_key`` is a *prefix* for independent calls
    (the dispatcher appends a unique suffix) and the *exact* serialization key
    for session calls (required when ``session=True``)."""

    work: Work
    state_key: str | None = None
    session: bool = False


class StateKeyDispatcher:
    """Run callables under the per-state-key serialization rule.

    ``max_parallel`` caps the number of *simultaneously executing* works across
    all keys (a client-side throttle so we don't outrun Modal's scale-out).
    Session calls waiting on a predecessor do **not** occupy a worker slot — the
    real work is only submitted to the pool once its predecessor on the same key
    completes — so a backlog of same-key calls can't deadlock the pool.
    """

    def __init__(self, max_parallel: int = 4) -> None:
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        self._max_parallel = max_parallel
        self._pool = ThreadPoolExecutor(max_workers=max_parallel)
        self._guard = threading.Lock()
        # Per session key: the most recently queued result future (the FIFO tail).
        self._tails: dict[str, Future] = {}
        self._counter = itertools.count()

    # ── public API ────────────────────────────────────────────────────
    def submit(
        self, work: Work, *, state_key: str | None = None, session: bool = False
    ) -> Future:
        """Queue ``work`` and return a Future for its result.

        ``session=False`` (default): auto-allocate a fresh state_key and run in
        parallel. ``session=True``: serialize FIFO on ``state_key`` (required).
        """
        if session:
            if not state_key:
                raise ValueError(
                    "session=True requires an explicit state_key to serialize on"
                )
            return self._submit_session(work, state_key)
        return self._pool.submit(work, self._fresh_key(state_key))

    def run_all(self, calls: list[Call]) -> list[Any]:
        """Submit every Call, then block for all results in submission order.

        Exceptions raised by a work propagate from the matching ``.result()``.
        """
        futures = [
            self.submit(c.work, state_key=c.state_key, session=c.session) for c in calls
        ]
        return [f.result() for f in futures]

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)

    def __enter__(self) -> StateKeyDispatcher:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.shutdown()

    # ── internals ─────────────────────────────────────────────────────
    def _fresh_key(self, base: str | None) -> str:
        n = next(self._counter)
        return f"{(base or 'auto')}-{n:04d}"

    def _submit_session(self, work: Work, key: str) -> Future:
        # The outward-facing future the caller awaits; resolved when the *inner*
        # pool task (launched after our predecessor finishes) completes.
        result: Future = Future()

        def _launch(_prev: Future | None = None) -> None:
            try:
                inner = self._pool.submit(work, key)
            except Exception as exc:  # noqa: BLE001 — pool already shut down
                result.set_exception(exc)
                self._clear_tail(key, result)
                return

            def _propagate(f: Future) -> None:
                if f.cancelled():
                    result.cancel()
                elif f.exception() is not None:
                    result.set_exception(f.exception())
                else:
                    result.set_result(f.result())
                self._clear_tail(key, result)

            inner.add_done_callback(_propagate)

        # Linearize submission order under the guard so the FIFO chain is exact.
        with self._guard:
            prev = self._tails.get(key)
            self._tails[key] = result

        if prev is None or prev.done():
            _launch()
        else:
            prev.add_done_callback(_launch)
        return result

    def _clear_tail(self, key: str, result: Future) -> None:
        # Drop the key only if we're still the tail (a later call may have
        # replaced us — then the chain continues through it).
        with self._guard:
            if self._tails.get(key) is result:
                del self._tails[key]
