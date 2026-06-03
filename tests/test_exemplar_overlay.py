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


def _inject_exemplar(
    intent: str,
    *,
    inject_before: str,
    type_: str = "click",
    outcome: str = "contact reason selected",
    source_run: str = "inj1",
    required: bool | None = None,
    params: dict | None = None,
    grounding: bool | None = None,
    hints: dict | None = None,
) -> dict:
    # A nudge exemplar plus the ``inject_before`` successor that turns it into
    # an INJECT exemplar — the worked sub-goal has no matching plan step.
    ex = _exemplar(intent, type_=type_, outcome=outcome, source_run=source_run)
    ex["inject_before"] = inject_before
    if required is not None:
        ex["required"] = required
    if params is not None:
        ex["params"] = params
    if grounding is not None:
        ex["grounding"] = grounding
    if hints is not None:
        ex["hints"] = hints
    return ex


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


# ── injection: supply a missing step the base plan omits ────────────────


def test_inject_exemplar_inserts_new_step_before_anchor():
    s1 = _step("apply used filter")
    s2 = _step("reveal phone on owner listing")
    plan = SimpleNamespace(steps=[s1, s2])

    n = apply_exemplar_overlay(plan, [
        _inject_exemplar(
            "select a contact reason",
            inject_before="reveal phone on owner listing",
        ),
    ])

    assert n == 1
    assert [s.intent for s in plan.steps] == [
        "apply used filter",
        "select a contact reason",
        "reveal phone on owner listing",
    ]


def test_injected_step_carries_procedural_replay_hint():
    anchor = _step("reveal phone on owner listing")
    plan = SimpleNamespace(steps=[anchor])

    apply_exemplar_overlay(plan, [
        _inject_exemplar(
            "select a contact reason",
            inject_before="reveal phone on owner listing",
            outcome="contact reason selected",
            source_run="inj42",
        ),
    ])

    injected = plan.steps[0]
    assert injected.intent == "select a contact reason"
    assert "click" in injected.hints["exemplar_replay"].lower()
    assert "contact reason selected" in injected.hints["exemplar_replay"]
    assert injected.hints["exemplar_source_run"] == "inj42"


def test_injected_step_never_leaks_coordinates():
    anchor = _step("reveal phone on owner listing")
    plan = SimpleNamespace(steps=[anchor])
    apply_exemplar_overlay(plan, [
        _inject_exemplar(
            "select a contact reason",
            inject_before="reveal phone on owner listing",
        ),
    ])
    replay = plan.steps[0].hints["exemplar_replay"]
    assert "412" not in replay
    assert "663" not in replay


def test_injected_step_is_required_by_default():
    anchor = _step("reveal phone on owner listing")
    plan = SimpleNamespace(steps=[anchor])
    apply_exemplar_overlay(plan, [
        _inject_exemplar(
            "select a contact reason",
            inject_before="reveal phone on owner listing",
        ),
    ])
    assert plan.steps[0].required is True


def test_injected_step_respects_required_override():
    anchor = _step("reveal phone on owner listing")
    plan = SimpleNamespace(steps=[anchor])
    apply_exemplar_overlay(plan, [
        _inject_exemplar(
            "select a contact reason",
            inject_before="reveal phone on owner listing",
            required=False,
        ),
    ])
    assert plan.steps[0].required is False


def test_inject_skipped_when_no_step_matches_successor():
    plan = SimpleNamespace(steps=[_step("apply used filter")])

    n = apply_exemplar_overlay(plan, [
        _inject_exemplar(
            "select a contact reason",
            inject_before="reveal phone on owner listing",
        ),
    ])

    assert n == 0
    assert len(plan.steps) == 1
    assert plan.steps[0].intent == "apply used filter"


def test_mixed_nudge_and_inject_both_apply():
    s1 = _step("apply used filter")
    s2 = _step("reveal phone on owner listing")
    plan = SimpleNamespace(steps=[s1, s2])

    n = apply_exemplar_overlay(plan, [
        _exemplar("apply used filter", outcome="filter applied", source_run="nudge"),
        _inject_exemplar(
            "select a contact reason",
            inject_before="reveal phone on owner listing",
            source_run="inject",
        ),
    ])

    assert n == 2
    assert s1.hints["exemplar_source_run"] == "nudge"  # existing step nudged
    assert [s.intent for s in plan.steps] == [
        "apply used filter",
        "select a contact reason",
        "reveal phone on owner listing",
    ]
    assert plan.steps[1].hints["exemplar_source_run"] == "inject"


def test_injected_step_threads_params_and_grounding():
    """An injection exemplar can specify the executable knobs a fresh step needs
    (params + grounding) — so the injected click is well-formed for the brain."""
    anchor = _step("reveal phone on owner listing")
    plan = SimpleNamespace(steps=[anchor])
    apply_exemplar_overlay(plan, [
        _inject_exemplar(
            "start the contact request",
            inject_before="reveal phone on owner listing",
            params={"label": "Start contact request"},
            grounding=True,
        ),
    ])
    inj = plan.steps[0]
    assert inj.params == {"label": "Start contact request"}
    assert inj.grounding is True


def test_injected_step_defaults_grounding_true():
    """A fresh injected action almost always needs vision grounding, so grounding
    defaults to True when the exemplar doesn't say otherwise."""
    anchor = _step("reveal phone on owner listing")
    plan = SimpleNamespace(steps=[anchor])
    apply_exemplar_overlay(plan, [
        _inject_exemplar(
            "start the contact request",
            inject_before="reveal phone on owner listing",
        ),
    ])
    assert plan.steps[0].grounding is True


def test_injected_step_merges_author_hints():
    """Author-provided hints (e.g. expect_url_contains) survive, and the procedural
    exemplar_replay is stamped alongside them rather than replacing them."""
    anchor = _step("reveal phone on owner listing")
    plan = SimpleNamespace(steps=[anchor])
    apply_exemplar_overlay(plan, [
        _inject_exemplar(
            "start the contact request",
            inject_before="reveal phone on owner listing",
            hints={"expect_url_contains": ["/boat/"]},
        ),
    ])
    inj = plan.steps[0]
    assert inj.hints["expect_url_contains"] == ["/boat/"]
    assert "exemplar_replay" in inj.hints


def test_injected_step_is_brain_ready_via_holo3():
    from mantis_agent.gym.step_handlers.holo3 import _build_scoped_task

    anchor = _step("reveal phone on owner listing")
    plan = SimpleNamespace(steps=[anchor])
    apply_exemplar_overlay(plan, [
        _inject_exemplar(
            "select a contact reason",
            inject_before="reveal phone on owner listing",
            outcome="contact reason selected",
            source_run="inj7",
        ),
    ])

    injected = plan.steps[0]
    prompt = _build_scoped_task(injected, SimpleNamespace(), step_index=0)
    assert "Worked example" in prompt
    assert "contact reason selected" in prompt
    assert "inj7" in prompt


# ── loop-target safety: injection must not silently break a loop body ────


def _loop(intent: str, *, loop_target: int):
    return SimpleNamespace(
        intent=intent, type="loop", params={}, hints={}, loop_target=loop_target,
    )


def test_injection_inside_loop_body_leaves_target_unchanged():
    """The real BT03 shape: the loop rewinds to OPEN-LISTING (before the reveal),
    so the injected contact-start lands INSIDE the loop body and re-runs every
    iteration. loop_target points before the insert → it must NOT move."""
    s_open = _step("open the next owner listing")
    s_reveal = _step("reveal the hidden phone number via Show Phone Number")
    s_loop = _loop("loop back to the next listing", loop_target=0)
    plan = SimpleNamespace(steps=[s_open, s_reveal, s_loop])

    apply_exemplar_overlay(plan, [
        _inject_exemplar(
            "start the contact request to unlock the phone",
            inject_before="reveal the hidden phone number via Show Phone Number",
        ),
    ])

    assert [s.intent for s in plan.steps] == [
        "open the next owner listing",
        "start the contact request to unlock the phone",
        "reveal the hidden phone number via Show Phone Number",
        "loop back to the next listing",
    ]
    # loop_target=0 (open listing) is before the insert at idx 1 → unchanged,
    # and the injected step now sits within the loop body (idx 1..2).
    assert plan.steps[3].loop_target == 0


def test_injection_at_loop_target_shifts_it_to_follow_the_step():
    """Edge case mirroring agentic_recovery.splice_inserted_steps: when the loop
    rewinds to the very step we inject before, loop_target must shift +1 so it
    keeps pointing at that same logical step (not at the freshly inserted one)."""
    s_open = _step("open the next owner listing")
    s_reveal = _step("reveal the hidden phone number via Show Phone Number")
    s_loop = _loop("loop back to re-reveal", loop_target=1)
    plan = SimpleNamespace(steps=[s_open, s_reveal, s_loop])

    apply_exemplar_overlay(plan, [
        _inject_exemplar(
            "start the contact request to unlock the phone",
            inject_before="reveal the hidden phone number via Show Phone Number",
        ),
    ])

    # The reveal step moved from idx 1 → 2; loop_target follows it.
    assert plan.steps[2].intent.startswith("reveal the hidden phone")
    assert plan.steps[3].loop_target == 2


def test_injection_does_not_touch_non_loop_steps_without_loop_target():
    """SimpleNamespace stubs (and real non-loop MicroIntents) without a meaningful
    loop_target must be left alone — the renumber guard keys off ``>= anchor``."""
    anchor = _step("reveal the hidden phone number via Show Phone Number")
    plan = SimpleNamespace(steps=[anchor])
    apply_exemplar_overlay(plan, [
        _inject_exemplar(
            "start the contact request",
            inject_before="reveal the hidden phone number via Show Phone Number",
        ),
    ])
    # Neither the injected step nor the anchor grew a stray loop_target.
    assert not hasattr(plan.steps[0], "loop_target")
    assert not hasattr(plan.steps[1], "loop_target")
