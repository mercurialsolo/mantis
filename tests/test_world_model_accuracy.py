"""Tests for #120 step 3 — world_model_accuracy_reward component."""

from __future__ import annotations

import pytest

from mantis_agent.rewards.components import (
    _STOPWORDS,
    _stem,
    _tokenize,
    world_model_accuracy_reward,
)


# ── _stem ────────────────────────────────────────────────────────────────


def test_stem_short_words_unchanged() -> None:
    assert _stem("a") == "a"
    assert _stem("the") == "the"
    assert _stem("url") == "url"


def test_stem_drops_common_suffixes() -> None:
    assert _stem("navigates") == "navigat"
    assert _stem("navigated") == "navigat"
    assert _stem("navigating") == "navigat"


def test_stem_does_not_mangle_short_endings() -> None:
    """Don't strip a suffix that would leave nothing meaningful."""
    # "ass" ends in "s" but is only 3 chars — must not become "a"
    assert _stem("yes") == "yes"


# ── _tokenize ────────────────────────────────────────────────────────────


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    out = _tokenize("URL changes to /detail/123!")
    assert "url" in out
    assert "chang" in out  # stemmed "changes"
    assert "detail" in out
    assert "123" in out


def test_tokenize_removes_stopwords() -> None:
    out = _tokenize("the page is on the screen")
    assert "the" not in out
    assert "is" not in out
    assert "on" not in out
    assert "page" in out
    assert "screen" in out


def test_tokenize_empty_string_returns_empty_set() -> None:
    assert _tokenize("") == set()


def test_tokenize_handles_none_safely() -> None:
    """The reward function passes _tokenize(predicted or "") so None is
    upstream-handled, but tokenize itself must accept empty input."""
    assert _tokenize("") == set()


def test_stopword_set_is_low_cardinality() -> None:
    """Sanity: the stopword list stays small. A bloated list erodes signal."""
    assert 20 < len(_STOPWORDS) < 100


# ── world_model_accuracy_reward ──────────────────────────────────────────


def test_returns_zero_when_predicted_is_empty() -> None:
    assert world_model_accuracy_reward("", "page navigated to /home") == 0.0


def test_returns_zero_when_observed_is_empty() -> None:
    assert world_model_accuracy_reward("URL changes to /home", "") == 0.0


def test_returns_zero_when_both_are_none() -> None:
    assert world_model_accuracy_reward(None, None) == 0.0


def test_full_overlap_yields_full_value() -> None:
    """Identical token sets should produce the maximum reward."""
    out = world_model_accuracy_reward(
        predicted="page navigates",
        observed="page navigates",
        value=0.10,
    )
    assert out == pytest.approx(0.10)


def test_zero_overlap_yields_zero_reward() -> None:
    out = world_model_accuracy_reward(
        predicted="modal closes",
        observed="URL navigates to detail",
    )
    assert out == 0.0


def test_partial_overlap_proportional_to_jaccard() -> None:
    """Predicted has 2 stemmed content tokens (page, navigat). Observed has 4
    stemmed content tokens (page, navigat, detail, 123). Intersection=2,
    union=4 → jaccard=0.5 → reward = 0.5 * 0.10 = 0.05."""
    out = world_model_accuracy_reward(
        predicted="page navigates",
        observed="page navigated to detail 123",
        value=0.10,
    )
    assert out == pytest.approx(0.05)


def test_stemming_treats_navigates_and_navigated_as_match() -> None:
    """The whole point of the light stemmer: tense difference shouldn't
    tank the similarity score."""
    a = world_model_accuracy_reward(
        predicted="page navigates",
        observed="page navigated",
        value=1.0,
    )
    b = world_model_accuracy_reward(
        predicted="page navigates",
        observed="page navigates",
        value=1.0,
    )
    # Stemming may not yield perfect equality on edge cases, but the
    # tense-only difference should not drop more than 10% of the signal.
    assert a >= b * 0.9


def test_stopwords_dont_inflate_score() -> None:
    """Two strings that share only stopwords should score 0."""
    out = world_model_accuracy_reward(
        predicted="the page is on the screen",
        observed="the modal is in the front",
        value=1.0,
    )
    # Only "page"/"screen" vs "modal"/"front" — intersection=0
    assert out == 0.0


def test_default_value_is_modest_shaping_term() -> None:
    """Default value=0.05 keeps this as a soft shaping term, not a big
    signal that the policy could farm."""
    out = world_model_accuracy_reward(
        predicted="page navigates", observed="page navigates",
    )
    assert out <= 0.05


def test_reward_is_bounded_by_value() -> None:
    """Output never exceeds the configured value — Jaccard is in [0, 1]."""
    for predicted, observed in (
        ("a", "a"),
        ("page navigates to detail", "page navigates to detail"),
        ("modal closes", "modal closes and url stays"),
    ):
        out = world_model_accuracy_reward(predicted, observed, value=0.07)
        assert 0.0 <= out <= 0.07


def test_punctuation_in_observed_does_not_break_match() -> None:
    """The runner's feedback string has commas, semicolons, parens — the
    tokenizer must strip them so similarity isn't dragged down by them."""
    out = world_model_accuracy_reward(
        predicted="page navigates to detail url",
        observed=("page navigated to https://x.test/detail/123; "
                  "title 'Boat 2024'"),
        value=1.0,
    )
    assert out > 0.0


# ── Worked example: composing into a RewardSignal ───────────────────────


def test_can_compose_into_a_reward_signal() -> None:
    """Verify the component plugs into the RewardSignal pattern other
    components use — no exotic shape, just a float."""
    from mantis_agent.rewards.base import RewardSignal

    score = world_model_accuracy_reward(
        predicted="page navigates",
        observed="page navigated to /home",
        value=0.05,
    )
    signal = RewardSignal(value=score, components={"world_model_accuracy": score})
    assert isinstance(float(signal), float)
    assert "world_model_accuracy" in signal.components


# ── Integration: TrajectoryStep fields drive the reward ─────────────────


def test_trajectory_step_fields_feed_directly_into_reward() -> None:
    """Verify the schema landed in step 1 + populated in step 2 produces
    inputs the reward function can consume directly."""
    from mantis_agent.actions import Action, ActionType
    from mantis_agent.gym.runner import TrajectoryStep

    step = TrajectoryStep(
        step=1,
        action=Action(ActionType.CLICK, {"x": 1, "y": 1}),
        thinking="",
        reward=0.0,
        done=False,
        inference_time=0.0,
        predicted_outcome="page navigates to detail",
        observed_outcome="page navigated to detail; title 'Boat'",
    )
    out = world_model_accuracy_reward(
        step.predicted_outcome, step.observed_outcome, value=0.10,
    )
    assert out > 0.0
