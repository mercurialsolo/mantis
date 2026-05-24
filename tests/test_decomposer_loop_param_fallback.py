"""Regression test for the boattrader urlnav 1-lead-instead-of-40 gap.

The canonical plan author put ``loop_target`` and ``loop_count`` inside
``params`` instead of at the top level. ``PlanDecomposer._build_intent``
read only the top-level keys, so loops landed with
``loop_target=-1, loop_count=0``. ``_fix_loop_targets`` rescues
``loop_target`` for extraction loops by retargeting to the first
extraction click step; nothing rescued ``loop_count``. The runner's
``_handle_loop_step`` self-spun on the loop step's own index when
``loop_target < 0`` (pre-_fix_loop_targets) and exhausted the
section-cap fallback without iterating the body.

The fix lets ``_build_intent`` fall back to ``params['loop_target']``
and ``params['loop_count']`` when the top-level keys are absent.
Backward compatible — Claude-emitted plans still set both at the top
level and win.
"""

from __future__ import annotations

from mantis_agent.plan_decomposer import PlanDecomposer


def test_build_intent_reads_loop_target_from_params_when_top_level_absent():
    src = {
        "type": "loop",
        "section": "extraction",
        "params": {"loop_target": 2, "loop_count": 40},
    }
    mi = PlanDecomposer._build_intent(src)
    assert mi.loop_target == 2
    assert mi.loop_count == 40


def test_build_intent_top_level_loop_target_wins_over_params():
    """When BOTH top-level and params have values, top-level wins.
    Preserves the canonical decomposer output shape."""
    src = {
        "type": "loop",
        "section": "extraction",
        "loop_target": 5,
        "loop_count": 15,
        "params": {"loop_target": 2, "loop_count": 40},
    }
    mi = PlanDecomposer._build_intent(src)
    assert mi.loop_target == 5
    assert mi.loop_count == 15


def test_build_intent_loop_defaults_when_neither_present():
    """No top-level, no params keys → default sentinels (-1, 0).
    Preserves the ``_fix_loop_targets`` rescue path's expectations."""
    src = {"type": "loop", "section": "extraction"}
    mi = PlanDecomposer._build_intent(src)
    assert mi.loop_target == -1
    assert mi.loop_count == 0


def test_build_intent_loop_count_falls_back_to_params_alone():
    """Mixed shape: top-level loop_target present, loop_count only in
    params — each falls back independently."""
    src = {
        "type": "loop",
        "section": "extraction",
        "loop_target": 3,
        "params": {"loop_count": 25},
    }
    mi = PlanDecomposer._build_intent(src)
    assert mi.loop_target == 3
    assert mi.loop_count == 25


def test_canonical_boattrader_loop_steps_resolve_correctly():
    """Mimics the canonical ``plans/boattrader_scrape_urlnav`` shape.
    Both inner and outer loops put loop_target / loop_count in params."""
    inner_loop = {
        "index": 9,
        "type": "loop",
        "intent": "Loop back to process the next listing",
        "params": {"loop_target": 2, "loop_count": 40},
        "section": "extraction",
        "budget": 0,
        "reverse": "No reverse needed for loop",
    }
    outer_loop = {
        "index": 11,
        "type": "loop",
        "intent": "Loop back to extract listings on the next page",
        "params": {"loop_target": 2, "loop_count": 20},
        "section": "pagination",
        "budget": 0,
        "reverse": "No reverse needed for loop",
    }
    mi_inner = PlanDecomposer._build_intent(inner_loop)
    mi_outer = PlanDecomposer._build_intent(outer_loop)
    assert (mi_inner.loop_target, mi_inner.loop_count) == (2, 40)
    assert (mi_outer.loop_target, mi_outer.loop_count) == (2, 20)
