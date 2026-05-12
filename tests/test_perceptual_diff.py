"""Tests for #293 — perceptual_diff helper.

Covers the high-risk classifier, the region/global hash comparison,
and the ablation toggle.
"""

from __future__ import annotations

import random

import pytest
from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.perceptual_diff import (
    action_had_effect,
    is_enabled,
    is_high_risk,
    region_hash,
)


def _noisy_img(seed: int, size: tuple[int, int] = (1280, 720)) -> Image.Image:
    """Image whose phash differs per seed — solid-color frames all share
    the same dHash regardless of color (see test_adaptive_settle)."""
    rng = random.Random(seed)
    img = Image.new("RGB", size, color=(0, 0, 0))
    pixels = img.load()
    for _ in range(2000):
        x, y = rng.randrange(size[0]), rng.randrange(size[1])
        pixels[x, y] = (255, 255, 255)
    return img


# ── is_enabled toggle ──────────────────────────────────────────────────


def test_is_enabled_default_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANTIS_PERCEPTUAL_VERIFY", raising=False)
    assert is_enabled() is True


def test_is_enabled_disabled_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_PERCEPTUAL_VERIFY", "disabled")
    assert is_enabled() is False


# ── is_high_risk classifier ────────────────────────────────────────────


def test_key_press_return_is_high_risk() -> None:
    assert is_high_risk(Action(ActionType.KEY_PRESS, {"keys": "Return"})) is True
    assert is_high_risk(Action(ActionType.KEY_PRESS, {"keys": "enter"})) is True
    assert is_high_risk(Action(ActionType.KEY_PRESS, {"keys": "Enter"})) is True
    assert is_high_risk(Action(ActionType.KEY_PRESS, {"keys": "ctrl+Return"})) is True


def test_key_press_other_keys_not_high_risk() -> None:
    assert is_high_risk(Action(ActionType.KEY_PRESS, {"keys": "Tab"})) is False
    assert is_high_risk(Action(ActionType.KEY_PRESS, {"keys": "Escape"})) is False
    assert is_high_risk(Action(ActionType.KEY_PRESS, {"keys": "Page_Down"})) is False


def test_click_with_submit_keyword_in_reasoning_is_high_risk() -> None:
    for keyword in [
        "Click the submit button.",
        "Confirm the order.",
        "Press the Buy button.",
        "Send the message now.",
        "Click delete on the row.",
        "Save the form.",
        "Sign in with credentials.",
        "Log in to staff-crm.",
        "Place order — final step.",
    ]:
        action = Action(ActionType.CLICK, {"x": 100, "y": 200}, reasoning=keyword)
        assert is_high_risk(action) is True, keyword


def test_click_without_keyword_not_high_risk() -> None:
    action = Action(
        ActionType.CLICK, {"x": 100, "y": 200},
        reasoning="Click on the next event card to read its details.",
    )
    assert is_high_risk(action) is False


def test_click_with_empty_reasoning_not_high_risk() -> None:
    assert is_high_risk(Action(ActionType.CLICK, {"x": 1, "y": 1})) is False


@pytest.mark.parametrize("kind", [
    ActionType.WAIT, ActionType.DONE, ActionType.SCROLL, ActionType.TYPE,
])
def test_non_actionable_types_never_high_risk(kind: ActionType) -> None:
    assert is_high_risk(Action(kind, {})) is False


# ── region_hash ────────────────────────────────────────────────────────


def test_region_hash_centers_on_coords() -> None:
    img = _noisy_img(seed=1)
    h = region_hash(img, 640, 360)
    assert h, "non-empty hash expected"


def test_region_hash_clamped_at_image_bounds() -> None:
    img = _noisy_img(seed=2)
    # Coord beyond bounds — should still produce a hash (clamped).
    assert region_hash(img, 9999, 9999)
    assert region_hash(img, -100, -100)


def test_region_hash_returns_empty_on_failure() -> None:
    """Robust to garbage input — never raises into the runner."""
    # ``None`` will crash inside _crop_around → return ""
    assert region_hash(None, 100, 100) == ""  # type: ignore[arg-type]


# ── action_had_effect — happy paths ────────────────────────────────────


def _click_submit() -> Action:
    return Action(
        ActionType.CLICK, {"x": 640, "y": 320},
        reasoning="Click the submit button.",
    )


def test_no_effect_when_frames_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANTIS_PERCEPTUAL_VERIFY", raising=False)
    frame = _noisy_img(seed=1)
    check = action_had_effect(frame, frame, _click_submit())
    assert check.effect_observed is False
    assert check.global_changed is False
    assert check.region_changed is False
    assert check.reason == "global_and_region_stable"


def test_effect_observed_when_global_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANTIS_PERCEPTUAL_VERIFY", raising=False)
    pre = _noisy_img(seed=1)
    post = _noisy_img(seed=2)
    check = action_had_effect(pre, post, _click_submit())
    assert check.effect_observed is True


# ── action_had_effect — skip conditions ────────────────────────────────


def test_skipped_when_toggle_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_PERCEPTUAL_VERIFY", "disabled")
    frame = _noisy_img(seed=1)
    check = action_had_effect(frame, frame, _click_submit())
    assert check.effect_observed is None
    assert check.reason == "toggle disabled"


def test_skipped_when_pre_frame_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANTIS_PERCEPTUAL_VERIFY", raising=False)
    check = action_had_effect(None, _noisy_img(1), _click_submit())
    assert check.effect_observed is None
    assert check.reason == "missing frame"


def test_skipped_when_post_frame_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANTIS_PERCEPTUAL_VERIFY", raising=False)
    check = action_had_effect(_noisy_img(1), None, _click_submit())
    assert check.effect_observed is None
    assert check.reason == "missing frame"


def test_skipped_when_action_not_high_risk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MANTIS_PERCEPTUAL_VERIFY", raising=False)
    benign_click = Action(
        ActionType.CLICK, {"x": 1, "y": 1},
        reasoning="Click the first listing card.",
    )
    pre = _noisy_img(1)
    post = _noisy_img(2)
    check = action_had_effect(pre, post, benign_click)
    assert check.effect_observed is None
    assert check.reason == "not high-risk"


# ── action_had_effect — region-only signal ─────────────────────────────


def test_region_change_alone_counts_as_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If only the action region changed (e.g. a button label flipped to
    a spinner) we should still count that as effect_observed."""
    monkeypatch.delenv("MANTIS_PERCEPTUAL_VERIFY", raising=False)

    # Build two large images that are identical except for a small patch
    # near (640, 360). Use seeded noise so phash_64 produces different
    # hashes for the cropped regions even when the global hash is close
    # to equal.
    pre = Image.new("RGB", (1280, 720), color=(50, 50, 50))
    post = pre.copy()
    rng = random.Random(7)
    px = post.load()
    for _ in range(800):
        x = 540 + rng.randrange(200)
        y = 260 + rng.randrange(200)
        px[x, y] = (255, 255, 255)

    check = action_had_effect(pre, post, _click_submit())
    assert check.effect_observed is True
    assert check.region_changed is True
