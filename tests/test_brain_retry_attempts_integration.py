"""Brain adapter integration with retry_attempts + per_step_action_history.

Roadmap #435 items 2 + 7. All three adapters (Claude, Fara, Holo3) now
accept structured prior-failure records and a sub-goal-scoped action
slice as kwargs to ``think()`` / ``_build_messages()``. The rendered
prompt carries a shared ``Recent attempts on this sub-goal:`` block
and uses the scoped slice (when set) instead of the global tail.

These tests pin the shape that lands in the request payload — without
making real API calls. The brain adapter ``_build_messages`` is the
unit under test; ``think()`` just forwards.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest
from PIL import Image

from mantis_agent.actions import Action, ActionType


def _img() -> Image.Image:
    return Image.new("RGB", (200, 100), color=(255, 255, 255))


@pytest.fixture(scope="module")
def _has_holo3() -> bool:
    return importlib.util.find_spec("mantis_agent.brain_holo3") is not None


@pytest.fixture(scope="module")
def _has_claude() -> bool:
    return importlib.util.find_spec("mantis_agent.brain_claude") is not None


@pytest.fixture(scope="module")
def _has_fara() -> bool:
    return importlib.util.find_spec("mantis_agent.brain_fara") is not None


# ── brain_holo3._build_messages ────────────────────────────────────


def test_holo3_build_messages_renders_retry_attempts_block(_has_holo3) -> None:
    """A non-empty ``retry_attempts`` kwarg lands in the user content
    as an outcome-tagged ``Recent attempts on this sub-goal:`` block.
    The shared formatter (``gym.retry_attempts``) renders, the brain
    adapter just splices."""
    if not _has_holo3:
        pytest.skip("brain_holo3 not importable")
    from mantis_agent.brain_holo3 import Holo3Brain

    brain = Holo3Brain.__new__(Holo3Brain)  # bypass __init__
    msgs = brain._build_messages(
        frames=[_img()], task="Click Contacted",
        action_history=None, screen_size=(1280, 720),
        retry_attempts=[
            {"x": 44, "y": 390, "matched_label": "Qualified (1)",
             "kind": "wrong_target", "reason": "URL missing status=Contacted"},
            {"x": 47, "y": 376, "matched_label": "Contacted (8)",
             "kind": "no_state_change", "reason": ""},
        ],
    )
    user_msg = msgs[1]
    text_blocks = [b["text"] for b in user_msg["content"] if b.get("type") == "text"]
    joined = "\n".join(text_blocks)
    assert "Recent attempts on this sub-goal" in joined
    assert "(44, 390)" in joined
    assert "Qualified (1)" in joined
    assert "wrong target" in joined
    assert "(47, 376)" in joined
    assert "no observable state change" in joined


def test_holo3_build_messages_omits_block_when_no_retry_attempts(_has_holo3) -> None:
    """No prior failures → no ``Recent attempts`` block. Keeps the
    prompt tight on a fresh sub-goal where the runner hasn't recorded
    any failures yet."""
    if not _has_holo3:
        pytest.skip("brain_holo3 not importable")
    from mantis_agent.brain_holo3 import Holo3Brain

    brain = Holo3Brain.__new__(Holo3Brain)
    msgs = brain._build_messages(
        frames=[_img()], task="Click X",
        action_history=None, screen_size=(1280, 720),
    )
    user_msg = msgs[1]
    text_blocks = [b["text"] for b in user_msg["content"] if b.get("type") == "text"]
    joined = "\n".join(text_blocks)
    assert "Recent attempts on this sub-goal" not in joined


def test_holo3_build_messages_per_step_history_overrides_global(_has_holo3) -> None:
    """When ``per_step_action_history`` is set, the prompt uses THAT
    slice for the ``Recent actions:`` block instead of the last-5 of
    ``action_history``. Caller (the per-step handler) supplies the
    pre-trimmed slice — runner takes it verbatim."""
    if not _has_holo3:
        pytest.skip("brain_holo3 not importable")
    from mantis_agent.brain_holo3 import Holo3Brain

    brain = Holo3Brain.__new__(Holo3Brain)
    # 8 global actions — would tail to last-5 by default.
    global_history = [
        Action(ActionType.CLICK, {"x": 100 + i, "y": 200, "button": "left"})
        for i in range(8)
    ]
    # Per-step slice carries just one action — distinct (x, y) so we
    # can tell them apart.
    per_step = [Action(ActionType.SCROLL, {"direction": "down"})]
    msgs = brain._build_messages(
        frames=[_img()], task="Click X",
        action_history=global_history,
        screen_size=(1280, 720),
        per_step_action_history=per_step,
    )
    text_blocks = [
        b["text"] for b in msgs[1]["content"]
        if b.get("type") == "text"
    ]
    joined = "\n".join(text_blocks)
    # Per-step action lands.
    assert "scroll" in joined.lower()
    # Global tail does NOT — no click coords from x=103..107.
    for x in range(103, 108):
        assert f"x={x}" not in joined and f"x: {x}" not in joined


def test_holo3_build_messages_falls_back_to_global_when_per_step_absent(_has_holo3) -> None:
    """When the caller doesn't supply a per-step slice, behaviour is
    exactly the pre-#435 default (last-5 of ``action_history``).
    Preserves compatibility with non-MicroPlanRunner callers."""
    if not _has_holo3:
        pytest.skip("brain_holo3 not importable")
    from mantis_agent.brain_holo3 import Holo3Brain

    brain = Holo3Brain.__new__(Holo3Brain)
    history = [
        Action(ActionType.CLICK, {"x": 100 + i, "y": 200, "button": "left"})
        for i in range(8)
    ]
    msgs = brain._build_messages(
        frames=[_img()], task="X",
        action_history=history, screen_size=(1280, 720),
    )
    text_blocks = [
        b["text"] for b in msgs[1]["content"]
        if b.get("type") == "text"
    ]
    joined = "\n".join(text_blocks)
    assert "Recent actions:" in joined
    # Last-5: x=103..107 should appear (5 entries).
    appearance = sum(1 for x in range(103, 108) if str(x) in joined)
    assert appearance == 5


# ── brain_claude._build_messages ───────────────────────────────────


def test_claude_build_messages_renders_retry_attempts_block(_has_claude) -> None:
    if not _has_claude:
        pytest.skip("brain_claude not importable")
    from mantis_agent.brain_claude import ClaudeBrain

    brain = ClaudeBrain.__new__(ClaudeBrain)
    brain._FRAMES_KEEP_AS_IMAGE = 2  # bypass __init__ defaults
    msgs = brain._build_messages(
        frames=[_img()], task="Click Contacted",
        action_history=None, screen_size=(1280, 720),
        retry_attempts=[
            {"x": 44, "y": 390, "matched_label": "Qualified (1)",
             "kind": "wrong_target", "reason": "URL did not navigate"},
        ],
    )
    text_blocks = [
        b["text"] for b in msgs[0]["content"]
        if b.get("type") == "text"
    ]
    joined = "\n".join(text_blocks)
    assert "Recent attempts on this sub-goal" in joined
    assert "Qualified (1)" in joined
    assert "wrong target" in joined


def test_claude_build_messages_omits_block_when_no_retry_attempts(_has_claude) -> None:
    if not _has_claude:
        pytest.skip("brain_claude not importable")
    from mantis_agent.brain_claude import ClaudeBrain

    brain = ClaudeBrain.__new__(ClaudeBrain)
    brain._FRAMES_KEEP_AS_IMAGE = 2
    msgs = brain._build_messages(
        frames=[_img()], task="x",
        action_history=None, screen_size=(1280, 720),
    )
    text_blocks = [
        b["text"] for b in msgs[0]["content"]
        if b.get("type") == "text"
    ]
    joined = "\n".join(text_blocks)
    assert "Recent attempts" not in joined


def test_claude_per_step_history_overrides_global(_has_claude) -> None:
    if not _has_claude:
        pytest.skip("brain_claude not importable")
    from mantis_agent.brain_claude import ClaudeBrain

    brain = ClaudeBrain.__new__(ClaudeBrain)
    brain._FRAMES_KEEP_AS_IMAGE = 2
    global_history = [
        Action(ActionType.CLICK, {"x": 500 + i, "y": 300, "button": "left"})
        for i in range(12)
    ]
    per_step = [Action(ActionType.KEY_PRESS, {"keys": "Tab"})]
    msgs = brain._build_messages(
        frames=[_img()], task="X",
        action_history=global_history, screen_size=(1280, 720),
        per_step_action_history=per_step,
    )
    text_blocks = [
        b["text"] for b in msgs[0]["content"]
        if b.get("type") == "text"
    ]
    joined = "\n".join(text_blocks)
    assert "Tab" in joined
    # No click x=500+ should leak through.
    for x in range(505, 512):
        assert str(x) not in joined


# ── brain_fara._build_messages ─────────────────────────────────────


def test_fara_build_messages_renders_retry_attempts_block(_has_fara) -> None:
    if not _has_fara:
        pytest.skip("brain_fara not importable")
    from mantis_agent.brain_fara import FaraBrain

    brain = FaraBrain.__new__(FaraBrain)
    brain.input_size = (1280, 720)
    msgs = brain._build_messages(
        frames=[_img()], task="Click Apply",
        action_history=None,
        retry_attempts=[
            {"x": 50, "y": 100, "matched_label": "Search",
             "kind": "wrong_target", "reason": ""},
        ],
    )
    text_blocks = [
        b["text"] for b in msgs[1]["content"]
        if b.get("type") == "text"
    ]
    joined = "\n".join(text_blocks)
    assert "Recent attempts on this sub-goal" in joined
    assert "Search" in joined


def test_fara_per_step_history_overrides_global(_has_fara) -> None:
    if not _has_fara:
        pytest.skip("brain_fara not importable")
    from mantis_agent.brain_fara import FaraBrain

    brain = FaraBrain.__new__(FaraBrain)
    brain.input_size = (1280, 720)
    global_history = [
        Action(ActionType.CLICK, {"x": 999, "y": 999, "button": "left"})
        for _ in range(6)
    ]
    per_step = [Action(ActionType.WAIT, {"seconds": 0.5})]
    msgs = brain._build_messages(
        frames=[_img()], task="X",
        action_history=global_history,
        per_step_action_history=per_step,
    )
    text_blocks = [
        b["text"] for b in msgs[1]["content"]
        if b.get("type") == "text"
    ]
    joined = "\n".join(text_blocks)
    assert "wait" in joined.lower()
    assert "999" not in joined
