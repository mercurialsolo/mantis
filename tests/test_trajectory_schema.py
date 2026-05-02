"""Tests for #120 step 1 — TrajectoryStep world-model schema landing."""

from __future__ import annotations

from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.runner import TrajectoryStep, _OBSERVED_STATE_KEYS, _observed_state


# ── Schema defaults ──────────────────────────────────────────────────────


def test_trajectory_step_defaults_all_world_model_fields_to_empty() -> None:
    """A step constructed with only the required fields must have safe
    defaults for every #120 field — no surprise required-arg shifts."""
    step = TrajectoryStep(
        step=1,
        action=Action(ActionType.CLICK, {"x": 1, "y": 1}),
        thinking="",
        reward=0.0,
        done=False,
        inference_time=0.0,
    )
    assert step.feedback == ""
    assert step.reward_components == {}
    assert step.frame_hash == ""
    assert step.observed_state == {}
    assert step.hypothesized_state == ""
    assert step.predicted_outcome == ""
    assert step.observed_outcome == ""


def test_trajectory_step_world_model_fields_are_independent_per_instance() -> None:
    a = TrajectoryStep(
        step=1, action=Action(ActionType.CLICK, {"x": 1, "y": 1}),
        thinking="", reward=0.0, done=False, inference_time=0.0,
    )
    b = TrajectoryStep(
        step=2, action=Action(ActionType.CLICK, {"x": 2, "y": 2}),
        thinking="", reward=0.0, done=False, inference_time=0.0,
    )
    a.observed_state["url"] = "https://x.test/a"
    a.reward_components["foo"] = 1.0
    # b's defaults must not leak across instances.
    assert b.observed_state == {}
    assert b.reward_components == {}


# ── _observed_state helper ───────────────────────────────────────────────


def test_observed_state_handles_none() -> None:
    assert _observed_state(None) == {}


def test_observed_state_handles_empty_dict() -> None:
    assert _observed_state({}) == {}


def test_observed_state_picks_only_whitelisted_keys() -> None:
    info = {
        "url": "https://x.test/p",
        "title": "Page",
        "focused_input": {"placeholder": "search"},
        # Excluded — high-cardinality / large blobs:
        "screenshot": b"\x89PNG\r\n...",
        "cookies": [{"name": "session"}],
        "raw_html": "<html>...</html>",
    }
    state = _observed_state(info)
    assert state == {
        "url": "https://x.test/p",
        "title": "Page",
        "focused_input": {"placeholder": "search"},
    }


def test_observed_state_includes_warning_and_backtracked() -> None:
    """Off-site backtrack signal must round-trip into the trajectory so
    rewards / SFT pipelines can train against it."""
    info = {
        "url": "https://x.test/p",
        "backtracked": True,
        "warning": "Off-site navigation",
    }
    state = _observed_state(info)
    assert state["backtracked"] is True
    assert state["warning"] == "Off-site navigation"


def test_observed_state_keys_constant_is_low_cardinality_set() -> None:
    """Sanity: the whitelisted-keys tuple is small and stable — adding a
    high-cardinality key here would inflate trajectory storage at scale."""
    assert len(_OBSERVED_STATE_KEYS) <= 10
    assert "url" in _OBSERVED_STATE_KEYS
    assert "focused_input" in _OBSERVED_STATE_KEYS


# ── Backward compat: positional construction still works ────────────────


def test_legacy_positional_construction_still_works() -> None:
    """Existing code constructs TrajectoryStep with positional args. The
    #120 fields are appended at the end with defaults so this can't break."""
    step = TrajectoryStep(
        1,
        Action(ActionType.CLICK, {"x": 1, "y": 1}),
        "thinking text",
        0.5,
        False,
        0.123,
    )
    assert step.step == 1
    assert step.thinking == "thinking text"
    assert step.reward == 0.5
    assert step.frame_hash == ""


# ── frame_hash via phash_64 ─────────────────────────────────────────────


def test_frame_hash_stable_for_identical_screenshots() -> None:
    """frame_hash uses the same phash_64 as the loop detector — identical
    images must produce equal hashes so two trajectories at the same logical
    state can be compared offline."""
    from mantis_agent.loop_detector import phash_64

    img = Image.new("RGB", (32, 32), (128, 64, 200))
    h1 = phash_64(img)
    h2 = phash_64(img.copy())
    assert h1 == h2
    assert h1 != ""
