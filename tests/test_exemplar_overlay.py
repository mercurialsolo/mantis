"""S1 exemplar injection seam — ``apply_exemplar_overlay``.

The S1 substrate (``learning/substrates/exemplar.py``) indexes positive
steps from prior runs but, until this seam, nothing *consumed* them at
runtime — unlike S0, whose ``apply_hint_overlay`` stamps a grounding
anchor onto the plan. ``apply_exemplar_overlay`` is the S1 parallel: it
stamps a PROCEDURAL ``exemplar_replay`` hint (the action that worked →
the outcome it produced) onto the matching step.

CUA purity (``feedback_cua_no_dom_access`` / ``feedback_tab_walk_label_matcher``):
the injected signal is what WORKED, never a stale coordinate. The brain
re-grounds the target from the current screenshot — so the hint must not
leak the exemplar's recorded x/y.
"""

from __future__ import annotations

from types import SimpleNamespace

from mantis_agent.gym.exemplar_memory import apply_exemplar_overlay


def _step(intent: str, type_: str = "click", hints: dict | None = None):
    return SimpleNamespace(intent=intent, type=type_, params={}, hints=hints or {})


def _exemplar(
    intent: str,
    *,
    type_: str = "click",
    outcome: str = "phone revealed",
    action: dict | None = None,
    source_run: str = "run1",
) -> dict:
    # Same shape ExemplarSubstrate.apply() emits in delta_artifacts.
    return {
        "intent": intent,
        "type": type_,
        "last_action": action if action is not None else {
            "action_type": "click", "params": {"x": 412, "y": 663},
        },
        "observed_outcome": outcome,
        "label_reason": "success + non-empty outcome",
        "source_run": source_run,
    }


# ── nothing to do ───────────────────────────────────────────────────────


def test_empty_exemplars_stamps_nothing():
    plan = SimpleNamespace(steps=[_step("reveal phone on owner listing")])
    assert apply_exemplar_overlay(plan, []) == 0
    assert plan.steps[0].hints == {}


def test_no_steps_is_safe():
    plan = SimpleNamespace(steps=[])
    assert apply_exemplar_overlay(plan, [_exemplar("reveal phone")]) == 0


# ── matching ────────────────────────────────────────────────────────────


def test_matching_exemplar_stamps_replay_hint():
    step = _step("reveal phone on owner listing")
    plan = SimpleNamespace(steps=[step])

    n = apply_exemplar_overlay(
        plan, [_exemplar("reveal phone on owner listing")],
    )

    assert n == 1
    assert "exemplar_replay" in step.hints
    assert step.hints["exemplar_source_run"] == "run1"


def test_replay_hint_is_procedural_action_and_outcome():
    step = _step("reveal phone on owner listing")
    plan = SimpleNamespace(steps=[step])
    apply_exemplar_overlay(plan, [_exemplar("reveal phone on owner listing")])

    replay = step.hints["exemplar_replay"].lower()
    assert "click" in replay              # the action_type that worked
    assert "phone revealed" in replay     # the outcome it produced


def test_replay_hint_never_leaks_coordinates():
    """CUA purity: the worked-step's recorded x/y must NOT appear in the
    injected hint. S1 conveys the procedure; the brain re-grounds by sight."""
    step = _step("reveal phone on owner listing")
    plan = SimpleNamespace(steps=[step])
    apply_exemplar_overlay(
        plan,
        [_exemplar(
            "reveal phone on owner listing",
            action={"action_type": "click", "params": {"x": 412, "y": 663}},
        )],
    )
    replay = step.hints["exemplar_replay"]
    assert "412" not in replay
    assert "663" not in replay


def test_type_mismatch_does_not_match():
    step = _step("reveal phone on owner listing", type_="extract_data")
    plan = SimpleNamespace(steps=[step])

    n = apply_exemplar_overlay(
        plan, [_exemplar("reveal phone on owner listing", type_="click")],
    )

    assert n == 0
    assert "exemplar_replay" not in step.hints


def test_no_intent_overlap_does_not_match():
    step = _step("apply length filter")
    plan = SimpleNamespace(steps=[step])

    n = apply_exemplar_overlay(
        plan, [_exemplar("reveal phone on owner listing")],
    )

    assert n == 0
    assert "exemplar_replay" not in step.hints


def test_best_intent_overlap_wins():
    step = _step("reveal phone on owner listing")
    plan = SimpleNamespace(steps=[step])
    weak = _exemplar("reveal something", outcome="weak", source_run="weak")
    strong = _exemplar(
        "reveal phone on owner listing", outcome="phone revealed", source_run="strong",
    )

    apply_exemplar_overlay(plan, [weak, strong])

    assert step.hints["exemplar_source_run"] == "strong"
    assert "phone revealed" in step.hints["exemplar_replay"].lower()


def test_each_step_gets_its_own_match():
    s1 = _step("apply used filter")
    s2 = _step("reveal phone on owner listing")
    plan = SimpleNamespace(steps=[s1, s2])

    n = apply_exemplar_overlay(plan, [
        _exemplar("apply used filter", outcome="filter applied", source_run="a"),
        _exemplar("reveal phone on owner listing", outcome="phone revealed", source_run="b"),
    ])

    assert n == 2
    assert s1.hints["exemplar_source_run"] == "a"
    assert s2.hints["exemplar_source_run"] == "b"


# ── author hints win ────────────────────────────────────────────────────


def test_existing_replay_hint_not_overwritten():
    step = _step(
        "reveal phone on owner listing",
        hints={"exemplar_replay": "operator override"},
    )
    plan = SimpleNamespace(steps=[step])

    n = apply_exemplar_overlay(
        plan, [_exemplar("reveal phone on owner listing")],
    )

    assert n == 0
    assert step.hints["exemplar_replay"] == "operator override"


def test_preserves_unrelated_hints():
    step = _step("reveal phone on owner listing", hints={"region": "card"})
    plan = SimpleNamespace(steps=[step])
    apply_exemplar_overlay(plan, [_exemplar("reveal phone on owner listing")])
    assert step.hints["region"] == "card"
    assert "exemplar_replay" in step.hints


# ── outcome-less exemplar still injects the action ──────────────────────


def test_outcome_less_exemplar_still_stamps_action():
    step = _step("reveal phone on owner listing")
    plan = SimpleNamespace(steps=[step])
    apply_exemplar_overlay(
        plan, [_exemplar("reveal phone on owner listing", outcome="")],
    )
    assert "exemplar_replay" in step.hints
    assert "click" in step.hints["exemplar_replay"].lower()


# ── brain surfacing (holo3 prompt) ──────────────────────────────────────


def test_holo3_prompt_surfaces_exemplar_replay():
    """The stamped ``exemplar_replay`` must reach the brain prompt as a
    'Worked example' section, with the don't-reuse-coordinates guardrail."""
    from mantis_agent.gym.step_handlers.holo3 import _build_scoped_task

    step = _step(
        "reveal phone on owner listing",
        hints={
            "exemplar_replay": "a click produced 'phone revealed'",
            "exemplar_source_run": "run42",
        },
    )
    prompt = _build_scoped_task(step, SimpleNamespace(), step_index=0)

    assert "Worked example" in prompt
    assert "a click produced 'phone revealed'" in prompt
    assert "run42" in prompt
    # CUA guardrail — the brain must re-ground by sight, not replay coords.
    assert "do NOT reuse old coordinates" in prompt


def test_holo3_prompt_omits_section_when_no_exemplar():
    from mantis_agent.gym.step_handlers.holo3 import _build_scoped_task

    step = _step("reveal phone on owner listing")
    prompt = _build_scoped_task(step, SimpleNamespace(), step_index=0)
    assert "Worked example" not in prompt
