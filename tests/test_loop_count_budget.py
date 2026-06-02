"""Loop-count budget tests (#622).

Two layers under test:

  - DECOMPOSE_PROMPT carries the concrete loop_count guidance (Layer A).
    Pure prompt-content assertion — verifies the prompt tells Claude
    to emit 30 / 10 and to never emit 0.
  - RunState's _handle_loop_step falls through to section-aware defaults
    when step.loop_count is 0 (Layer B). Direct unit test of the
    resolver logic with the inner run-executor body invoked.

The end-to-end behaviour (decomposer emits non-zero loop_count;
runtime caps a buggy 0 at section default) is verified on the next
Modal redeploy.
"""

from __future__ import annotations

from mantis_agent.gym.checkpoint import RunCheckpoint
from mantis_agent.gym.run_executor import RunState
from mantis_agent.plan_decomposer import DECOMPOSE_PROMPT, MicroIntent, MicroPlan


# ── Layer A: prompt guidance ───────────────────────────────────────────


def test_prompt_documents_concrete_extraction_loop_count() -> None:
    """The extraction-loop example in LOOP STRUCTURE shows ``count=30``,
    not the old ``count=N`` placeholder."""
    assert "count=30" in DECOMPOSE_PROMPT
    assert "count=N)" not in DECOMPOSE_PROMPT  # legacy placeholder gone


def test_prompt_documents_concrete_pagination_loop_count() -> None:
    """Pagination-loop example shows ``count=10``, not ``count=pages``."""
    assert "count=10" in DECOMPOSE_PROMPT
    assert "count=pages" not in DECOMPOSE_PROMPT


def test_prompt_forbids_loop_count_zero() -> None:
    """The prompt explicitly tells Claude never to emit 0 — that was
    the pre-#622 default that fell through to the 200-iter ceiling
    and burned $2.50/loop on dedup-skipped retries."""
    assert "Never 0" in DECOMPOSE_PROMPT or "never 0" in DECOMPOSE_PROMPT


def test_prompt_loop_step_type_doc_updated() -> None:
    """The ``- loop:`` type-definition line carries the same guidance,
    not just the LOOP STRUCTURE heading above it. Claude reads the
    type-definitions section when picking step parameters."""
    # The new doc reads "ALWAYS emit a concrete loop_count integer:"
    # Pin that exact phrase so the doc can't silently regress.
    assert "ALWAYS emit a concrete loop_count integer" in DECOMPOSE_PROMPT


# ── Layer B: runtime section-aware fallback ────────────────────────────


def _state() -> RunState:
    """Fresh RunState with the default section-aware caps."""
    return RunState(
        checkpoint=RunCheckpoint(run_key="t", plan_signature="x"),
    )


def test_default_section_caps_present() -> None:
    """RunState ships with the section-aware ceiling dict populated."""
    state = _state()
    assert state.max_loop_iterations_by_section == {
        "extraction": 30, "pagination": 10, "default": 50,
    }


def test_extraction_loop_with_zero_count_caps_at_30() -> None:
    """When the decomposer (legacy / buggy) emits loop_count=0 on an
    extraction-section loop, the runtime falls back to 30 — NOT 200."""
    from mantis_agent.gym.run_executor import RunExecutor

    plan = MicroPlan(domain="x")
    plan.steps = [
        MicroIntent(intent="click", type="click", section="extraction"),
        MicroIntent(
            intent="loop", type="loop", section="extraction",
            loop_target=0, loop_count=0,
        ),
    ]
    state = _state()
    state.step_index = 1  # at the loop step

    class _StubExecutor(RunExecutor):
        def __init__(self): pass  # bypass parent back-ref
        def _persist(self, plan, state): pass

    exec_ = _StubExecutor()
    # Simulate 31 loop fires to confirm the cap kicks in at 30.
    for i in range(31):
        state.step_index = 1
        exec_._handle_loop_step(plan, plan.steps[1], state)
    # After 30 iterations the loop step advances past itself (step_index
    # becomes 2, off the end of the plan). On iteration 31 the counter
    # has tripped — verify by re-firing once more and asserting we're
    # past the loop.
    assert state.loop_counters[1] >= 30
    # The 31st invocation should have already advanced.
    assert state.step_index == 2


def test_pagination_loop_with_zero_count_caps_at_10() -> None:
    """Pagination-section loop with loop_count=0 caps at 10 (not 30)."""
    from mantis_agent.gym.run_executor import RunExecutor

    plan = MicroPlan(domain="x")
    plan.steps = [
        MicroIntent(intent="paginate", type="paginate", section="pagination"),
        MicroIntent(
            intent="loop", type="loop", section="pagination",
            loop_target=0, loop_count=0,
        ),
    ]
    state = _state()

    class _StubExecutor(RunExecutor):
        def __init__(self): pass
        def _persist(self, plan, state): pass

    exec_ = _StubExecutor()
    for i in range(11):
        state.step_index = 1
        exec_._handle_loop_step(plan, plan.steps[1], state)
    assert state.loop_counters[1] >= 10
    assert state.step_index == 2


def test_explicit_loop_count_overrides_section_default() -> None:
    """When a plan specifies loop_count=50, that wins over the
    extraction-section 30 default."""
    from mantis_agent.gym.run_executor import RunExecutor

    plan = MicroPlan(domain="x")
    plan.steps = [
        MicroIntent(intent="click", type="click", section="extraction"),
        MicroIntent(
            intent="loop", type="loop", section="extraction",
            loop_target=0, loop_count=50,
        ),
    ]
    state = _state()

    class _StubExecutor(RunExecutor):
        def __init__(self): pass
        def _persist(self, plan, state): pass

    exec_ = _StubExecutor()
    # Fire 40 times — should still be looping (under the 50 explicit cap).
    for i in range(40):
        state.step_index = 1
        exec_._handle_loop_step(plan, plan.steps[1], state)
    # Still on the loop's jump-back target (step 0).
    assert state.step_index == 0
    assert state.loop_counters[1] == 40


def test_unsectioned_loop_falls_back_to_default_50() -> None:
    """A loop step with empty ``section`` falls through to the
    ``default`` bucket (50), not the legacy 200."""
    from mantis_agent.gym.run_executor import RunExecutor

    plan = MicroPlan(domain="x")
    plan.steps = [
        MicroIntent(intent="click", type="click"),
        MicroIntent(intent="loop", type="loop", loop_target=0, loop_count=0),
    ]
    state = _state()

    class _StubExecutor(RunExecutor):
        def __init__(self): pass
        def _persist(self, plan, state): pass

    exec_ = _StubExecutor()
    for i in range(51):
        state.step_index = 1
        exec_._handle_loop_step(plan, plan.steps[1], state)
    assert state.loop_counters[1] >= 50
    assert state.step_index == 2


# ── Loop stop-on-var (LA discriminator) ────────────────────────────────


class _StubParent:
    """Minimal stand-in for MicroPlanRunner — carries the state-var bag the
    loop's ``stop_var`` check reads (same bag ``execute_step`` uses for
    ``guard``)."""

    def __init__(self, state_vars: dict) -> None:
        self._state_vars = state_vars


def _exec_with_state_vars(state_vars: dict):
    """A RunExecutor whose ``self.parent._state_vars`` is ``state_vars``."""
    from mantis_agent.gym.run_executor import RunExecutor

    class _StubExecutor(RunExecutor):
        def __init__(self, parent): self.parent = parent
        def _persist(self, plan, state): pass

    return _StubExecutor(_StubParent(state_vars))


def _loop_plan(stop_var: str) -> MicroPlan:
    plan = MicroPlan(domain="x")
    plan.steps = [
        MicroIntent(intent="click", type="click", section="extraction"),
        MicroIntent(
            intent="loop", type="loop", section="extraction",
            loop_target=0, loop_count=30, stop_var=stop_var,
        ),
    ]
    return plan


def test_loop_stop_var_truthy_exits_before_count_reached() -> None:
    """A loop with ``stop_var`` bound truthy exits on the first fire even
    though only 1 of 30 iterations has run — the lever that turns an
    exhaustive search into a one-shot jump once the target is found."""
    plan = _loop_plan("is_caterpillar")
    state = _state()
    state.step_index = 1

    exec_ = _exec_with_state_vars({"is_caterpillar": True})
    exec_._handle_loop_step(plan, plan.steps[1], state)

    # Advanced PAST the loop (step 2 = off the end), not back to target 0.
    assert state.step_index == 2
    assert state.loop_counters[1] == 1


def test_loop_stop_var_falsy_continues_looping() -> None:
    """``stop_var`` bound False does not short-circuit — the loop jumps
    back to its target like an ordinary count-bounded loop."""
    plan = _loop_plan("is_caterpillar")
    state = _state()
    state.step_index = 1

    exec_ = _exec_with_state_vars({"is_caterpillar": False})
    exec_._handle_loop_step(plan, plan.steps[1], state)

    assert state.step_index == 0  # jumped back to loop_target
    assert state.loop_counters[1] == 1


def test_loop_stop_var_absent_continues_looping() -> None:
    """A ``stop_var`` naming a variable that no prior step ever bound is
    treated as falsy (``.get(name, False)``) — the loop keeps iterating
    rather than exiting on an unbound name."""
    plan = _loop_plan("never_bound")
    state = _state()
    state.step_index = 1

    exec_ = _exec_with_state_vars({})  # var never bound
    exec_._handle_loop_step(plan, plan.steps[1], state)

    assert state.step_index == 0
    assert state.loop_counters[1] == 1


def test_loop_without_stop_var_ignores_state_vars() -> None:
    """Back-compat: a loop with no ``stop_var`` never consults the state
    bag, so a truthy variable with the same shape can't accidentally stop
    an unrelated loop."""
    plan = MicroPlan(domain="x")
    plan.steps = [
        MicroIntent(intent="click", type="click", section="extraction"),
        MicroIntent(
            intent="loop", type="loop", section="extraction",
            loop_target=0, loop_count=30,  # no stop_var
        ),
    ]
    state = _state()
    state.step_index = 1

    exec_ = _exec_with_state_vars({"is_caterpillar": True})
    exec_._handle_loop_step(plan, plan.steps[1], state)

    assert state.step_index == 0  # still looping
    assert state.loop_counters[1] == 1


def test_stop_var_round_trips_through_serialize_and_rebuild() -> None:
    """``stop_var`` survives the MicroIntent → dict → MicroIntent wire the
    Modal submit path uses (``micro_plan_steps_to_dicts`` then the
    ``MicroIntent(**s)`` splat in modal_cua_server). A dropped key here
    would silently disable the discriminator on the remote run."""
    from mantis_agent.plan_decomposer import PlanDecomposer
    from mantis_agent.server_utils import micro_plan_steps_to_dicts

    step = MicroIntent(
        intent="loop", type="loop", loop_target=0, loop_count=30,
        stop_var="is_caterpillar",
    )
    [as_dict] = micro_plan_steps_to_dicts([step])
    assert as_dict["stop_var"] == "is_caterpillar"

    # Both rebuild paths: the **splat (Modal) and _build_intent (local JSON).
    assert MicroIntent(**as_dict).stop_var == "is_caterpillar"
    assert PlanDecomposer._build_intent(as_dict).stop_var == "is_caterpillar"


def test_legacy_max_loop_iterations_still_honoured_when_no_section_caps() -> None:
    """Caller passes a RunState with the section-cap dict cleared — the
    code falls through to the legacy single-value field. Preserves
    behaviour for any external caller / test fixture that hasn't
    migrated."""
    from mantis_agent.gym.run_executor import RunExecutor

    plan = MicroPlan(domain="x")
    plan.steps = [
        MicroIntent(intent="click", type="click", section="extraction"),
        MicroIntent(
            intent="loop", type="loop", section="extraction",
            loop_target=0, loop_count=0,
        ),
    ]
    state = _state()
    state.max_loop_iterations_by_section = {}  # legacy caller
    state.max_loop_iterations = 5  # set tight so we can observe the cap

    class _StubExecutor(RunExecutor):
        def __init__(self): pass
        def _persist(self, plan, state): pass

    exec_ = _StubExecutor()
    for i in range(6):
        state.step_index = 1
        exec_._handle_loop_step(plan, plan.steps[1], state)
    assert state.loop_counters[1] >= 5
    assert state.step_index == 2
