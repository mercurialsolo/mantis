"""Plan-Observe-Act-Verify-Emit lifecycle substrate (#486).

The CUA design's durable-execution model splits each step into
five phases the runner orchestrates explicitly:

* **Plan**     — pick the next step (from a pre-decomposed plan or
                 by asking the planner).
* **Observe**  — capture the pre-action screenshot + URL + viewport
                 (the substrate ``Observation`` reference).
* **Act**      — dispatch the typed action through the env. The
                 only phase that mutates the outside world.
* **Verify**   — read the post-action screenshot, ask the verifier
                 whether the intent was achieved, stamp a typed
                 :class:`Verdict`.
* **Emit**     — append the canonical :class:`TrajectoryEvent` to
                 the event store.

Today's :class:`~mantis_agent.gym.run_executor.RunExecutor` already
walks these phases implicitly. This module defines the typed
substrate (the phase enum + an :class:`Activity` protocol) so a
future refactor can split each phase into a serializable
input/output pair and isolate side-effectful operations as
"activities" in the Temporal sense.

**Substrate-only in this PR.** No runner refactor yet — the goal
is to give downstream work (a real workflow boundary, replay-safe
activity classes, Temporal migration) a typed surface to hook
into. Production behaviour is unchanged.

Design notes for the upcoming refactor (#486 follow-up):

* Plan / Verify can be **pure** (deterministic given inputs +
  model versions). They become workflow code in a Temporal sense.
* Observe / Act / Emit are **side-effectful** (touch browser,
  network, disk). They become Activities in a Temporal sense —
  retried at the activity boundary with idempotency keys.
* Each Activity's inputs must be replay-safe and serializable —
  the :class:`Activity` protocol enforces this with a typed
  result + a contract that no Python-only state crosses the
  boundary.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable


class LifecyclePhase(str, Enum):
    """The five phases of a single CUA step (#486).

    String values match the design doc's phase names so logs /
    metrics / event annotations can use the enum's ``.value``
    directly without a translation layer.

    Phase ordering is fixed: ``PLAN → OBSERVE → ACT → VERIFY →
    EMIT``. The :class:`~..gym.run_executor.RunExecutor` walks them
    in order for every step.
    """

    PLAN = "plan"
    OBSERVE = "observe"
    ACT = "act"
    VERIFY = "verify"
    EMIT = "emit"


# Phases that mutate the outside world. Used by the upcoming
# workflow/activity split to decide which calls become activities
# (retried at the activity boundary with idempotency keys) vs
# workflow code (replayed deterministically by the orchestrator).
SIDE_EFFECTFUL_PHASES: frozenset[LifecyclePhase] = frozenset({
    LifecyclePhase.OBSERVE,  # env.screenshot mutates capture state
    LifecyclePhase.ACT,      # the only phase that touches the browser
    LifecyclePhase.EMIT,     # disk / network write of the canonical event
})


# Phases that are pure / deterministic given their typed inputs.
# Become workflow code in a Temporal sense — replayed by the
# orchestrator without re-execution of side effects.
PURE_PHASES: frozenset[LifecyclePhase] = frozenset({
    LifecyclePhase.PLAN,
    LifecyclePhase.VERIFY,
})


@runtime_checkable
class Activity(Protocol):
    """The narrow contract a side-effectful phase implementation
    must satisfy.

    An Activity is a unit of work the orchestrator hands off to a
    worker pool — its inputs and outputs cross the workflow /
    activity boundary, so both must be **replay-safe** and
    **serializable** (the orchestrator's replay path will rebuild
    the workflow state by replaying activity outputs from history).

    The two requirements:

    * ``phase`` returns the :class:`LifecyclePhase` this activity
      implements. Used by the orchestrator's dispatcher to route
      ``ACT`` work to the actor pool, ``OBSERVE`` work to the
      browser pool, etc.
    * ``execute`` takes a JSON-serializable input dict and returns
      a JSON-serializable output dict. The orchestrator records
      both in the activity history for replay.

    For v1 the protocol is intentionally minimal. The follow-up
    refactor adds idempotency keys + retry-budget metadata + a
    typed-input/output binding for each phase. Keeping the v1
    surface narrow means handlers can satisfy it by adapting their
    existing callable signatures without a wholesale rewrite.

    ``runtime_checkable`` so the orchestrator can ``isinstance``
    a registered activity for assertion clarity in tests.
    """

    @property
    def phase(self) -> LifecyclePhase:
        """Which lifecycle phase this activity implements."""
        ...

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run the activity. Both ``payload`` and the return value
        must be JSON-serializable so the activity's input/output
        round-trips through the orchestrator's history.
        """
        ...
