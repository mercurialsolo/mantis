"""The BT03 gated-reveal frozen-vs-S1 discriminator, end-to-end on the SHIPPED
artifacts.

Two files have to agree for the policy-cluster discriminator to mean anything:

* ``plans/bt03_gated_reveal.json`` — the base plan a frozen agent runs as-is. It
  navigates the by-owner SRP, opens a listing, and clicks "Show Phone Number".
  It deliberately OMITS the "start contact request" prerequisite the gated env
  (``BT03_REVEAL_GATE=1``) demands, so frozen's reveal is a server-side no-op.
* ``experiments/learning_allocator/eval/bt03_gated_inject_exemplar.json`` — the
  S1 worked-step exemplar. Its ``inject_before`` must token-overlap the plan's
  reveal step so :func:`apply_exemplar_overlay` inserts the missing prerequisite
  immediately before it. That overlap is the whole linchpin; an edit to either
  file's wording can silently sever it and collapse S1 back onto frozen.

This loads both files through the EXACT path ``live_runner`` uses (``substitute_
env_url`` → ``MicroPlan.from_dict``) and proves: (a) the frozen plan has no
prerequisite, (b) S1 injects exactly one grounded prerequisite right before the
reveal, and (c) the loop still rewinds to the open-listing step so the injected
per-boat prerequisite re-runs each iteration. No network, no spend.
"""

from __future__ import annotations

import json
from pathlib import Path

from mantis_agent.gym.exemplar_memory import apply_exemplar_overlay
from mantis_agent.plan_decomposer import MicroPlan
from mantis_agent.sim_envs.templating import substitute_env_url

_REPO = Path(__file__).resolve().parents[2]
_PLAN = _REPO / "plans" / "bt03_gated_reveal.json"
_EXEMPLAR = (
    _REPO / "experiments" / "learning_allocator" / "eval"
    / "bt03_gated_inject_exemplar.json"
)
_ENV_URL = "https://bt03-gated.example"

# Distinctive to the reveal step's intent. (The detect_visible step also names
# the "Show Phone Number" button, so a bare "show phone number" is ambiguous.)
_REVEAL_TOKEN = "reveal the hidden phone number via"
_PREREQ_TOKEN = "contact request"            # appears only in the injected step


def _load_plan() -> MicroPlan:
    payload = json.loads(_PLAN.read_text())
    payload = substitute_env_url(payload, _ENV_URL)
    return MicroPlan.from_dict(payload)


def _load_exemplars() -> list[dict]:
    return json.loads(_EXEMPLAR.read_text())


def _reveal_index(plan: MicroPlan) -> int:
    hits = [
        i for i, s in enumerate(plan.steps)
        if _REVEAL_TOKEN in (s.intent or "").lower()
    ]
    assert len(hits) == 1, f"expected exactly one reveal step, found {hits}"
    return hits[0]


# ── the shipped artifacts are well-formed ───────────────────────────────


def test_plan_loads_and_substitutes_env_url() -> None:
    plan = _load_plan()
    assert plan.steps, "plan decoded to zero steps"
    blob = json.dumps([s.intent for s in plan.steps] + [
        s.params.get("url", "") for s in plan.steps
    ])
    assert "{{ENV_URL}}" not in blob, "env-url token survived substitution"
    assert _ENV_URL in plan.steps[0].params["url"], "navigate url not substituted"


def test_exemplar_is_an_inject_shape() -> None:
    exemplars = _load_exemplars()
    assert isinstance(exemplars, list) and len(exemplars) == 1
    ex = exemplars[0]
    assert ex.get("inject_before"), "exemplar is missing inject_before (would nudge, not inject)"
    assert _PREREQ_TOKEN in ex["intent"].lower()


# ── frozen baseline: the prerequisite is absent ─────────────────────────


def test_frozen_plan_omits_the_prerequisite() -> None:
    """A frozen agent runs the plan as-authored; the only path to the gate is the
    S1 injection, so the base plan must not already contain a contact-start step."""
    plan = _load_plan()
    assert not any(
        _PREREQ_TOKEN in (s.intent or "").lower() for s in plan.steps
    ), "base plan already contains the prerequisite — frozen would pass and void the discriminator"
    # And the reveal it DOES contain is a real, groundable step.
    rev = plan.steps[_reveal_index(plan)]
    assert rev.type == "click" and rev.grounding is True


# ── S1: the injection lands exactly where the discriminator needs it ─────


def test_s1_injects_one_prerequisite_immediately_before_the_reveal() -> None:
    plan = _load_plan()
    reveal_before = _reveal_index(plan)
    n_steps_before = len(plan.steps)

    injected = apply_exemplar_overlay(plan, _load_exemplars())

    assert injected == 1, "exemplar did not inject — check inject_before token overlap"
    assert len(plan.steps) == n_steps_before + 1
    reveal_after = _reveal_index(plan)
    # The new step sits in the slot the reveal used to occupy.
    assert reveal_after == reveal_before + 1
    prereq = plan.steps[reveal_after - 1]
    assert _PREREQ_TOKEN in (prereq.intent or "").lower()


def test_injected_prerequisite_is_brain_executable_and_coordinate_free() -> None:
    plan = _load_plan()
    apply_exemplar_overlay(plan, _load_exemplars())
    prereq = plan.steps[_reveal_index(plan) - 1]

    assert prereq.type == "click"
    assert prereq.grounding is True            # a fresh action needs vision grounding
    assert prereq.required is False            # a hiccup retries via the loop, never halts
    replay = prereq.hints["exemplar_replay"].lower()
    assert "click" in replay and "unlocked" in replay
    # CUA purity: the procedure is conveyed, never a stale coordinate.
    assert "x=" not in replay and "y=" not in replay
    assert prereq.hints["exemplar_source_run"] == "bt03-gated-contact-start-seed42"


def test_loop_target_survives_injection_so_prerequisite_reruns_each_iteration() -> None:
    """The loop rewinds to the open-listing step (before the insert), so the gate
    cookie — which is per-boat — gets re-established for every listing. The insert
    must not perturb that target."""
    plan = _load_plan()
    loop_before = [s for s in plan.steps if s.type == "loop"]
    assert len(loop_before) == 1
    target_step_intent = plan.steps[loop_before[0].loop_target].intent

    apply_exemplar_overlay(plan, _load_exemplars())

    loop_after = [s for s in plan.steps if s.type == "loop"][0]
    # loop_target may renumber, but it must still point at the SAME logical step…
    assert plan.steps[loop_after.loop_target].intent == target_step_intent
    # …and that step is upstream of the injected prerequisite (so it's in the body).
    assert loop_after.loop_target < _reveal_index(plan) - 1
