"""Unit tests for :func:`mantis_agent.rewards.components.oracle_step_reward`
and :class:`mantis_agent.recipes.marketplace_listings.rewards.SyntheticEnvReward`.

These are pure-function tests — no FastAPI / no env / no HTTP. The
``fetch_mutations`` HTTP client is exercised separately in
``test_oracle_client.py``.
"""

from __future__ import annotations

import pytest
from PIL import Image

# Import ``mantis_agent.rewards`` first to ensure the package
# ``__init__.py`` finishes loading before ``recipes.marketplace_listings.rewards``
# is touched. The deprecated ``BoatTraderReward`` alias in
# ``rewards/__init__.py`` subclasses ``MarketplaceListingReward`` at
# import time, creating a latent forward dependency that surfaces as a
# circular import if the recipe module is loaded first.
from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymObservation, GymResult
from mantis_agent.rewards.base import EpisodeState
from mantis_agent.rewards.components import oracle_step_reward
from mantis_agent.recipes.marketplace_listings.rewards import (  # noqa: I001
    DEFAULT_MARKETPLACE_OPS_BY_STEP_KIND,
    SyntheticEnvReward,
)


# ── oracle_step_reward — pure function ─────────────────────────────────


def test_no_mutations_returns_zero():
    assert oracle_step_reward([], {"lead_submitted"}) == 0.0


def test_empty_expected_ops_returns_zero():
    delta = [{"id": 1, "operation": "lead_submitted"}]
    assert oracle_step_reward(delta, set()) == 0.0


def test_single_matching_mutation_returns_value():
    delta = [{"id": 1, "operation": "lead_submitted", "target_id": "boat-123"}]
    assert oracle_step_reward(delta, {"lead_submitted"}, value=0.1) == 0.1


def test_multiple_matches_scale_linearly():
    delta = [
        {"id": 1, "operation": "lead_submitted"},
        {"id": 2, "operation": "lead_submitted"},
        {"id": 3, "operation": "lead_submitted"},
    ]
    # Three matches × 0.1 each. The docstring documents the linear scale
    # so the test pins it explicitly. ``pytest.approx`` handles
    # 0.1 * 3 != 0.3 IEEE-754 drift.
    assert oracle_step_reward(delta, {"lead_submitted"}, value=0.1) == pytest.approx(0.3)


def test_non_matching_mutations_return_zero():
    delta = [
        {"id": 1, "operation": "consent_set"},
        {"id": 2, "operation": "env_reset"},
    ]
    assert oracle_step_reward(delta, {"lead_submitted"}) == 0.0


def test_mixed_matching_and_non_matching():
    delta = [
        {"id": 1, "operation": "consent_set"},
        {"id": 2, "operation": "lead_submitted"},
        {"id": 3, "operation": "env_reset"},
    ]
    assert oracle_step_reward(delta, {"lead_submitted"}, value=0.1) == 0.1


def test_iterable_expected_ops_accepted():
    """Accepts a list / tuple / frozenset / generator, not just a set."""
    delta = [{"id": 1, "operation": "lead_submitted"}]
    assert oracle_step_reward(delta, ["lead_submitted"]) > 0
    assert oracle_step_reward(delta, ("lead_submitted",)) > 0
    assert oracle_step_reward(delta, frozenset({"lead_submitted"})) > 0


def test_malformed_mutation_entries_ignored():
    """Entries missing ``operation`` or non-dict shapes don't crash."""
    delta = [
        {"id": 1},  # missing operation
        "not a dict",  # malformed
        None,  # malformed
        {"id": 2, "operation": "lead_submitted"},
    ]
    assert oracle_step_reward(delta, {"lead_submitted"}, value=0.1) == 0.1


def test_custom_value_overrides_default():
    delta = [{"id": 1, "operation": "lead_submitted"}]
    assert oracle_step_reward(delta, {"lead_submitted"}, value=0.5) == 0.5


# ── SyntheticEnvReward — step path ─────────────────────────────────────


def _gym_result(info: dict) -> GymResult:
    """Build a minimal GymResult carrying the supplied info dict."""
    obs = GymObservation(screenshot=Image.new("RGB", (1, 1)))
    return GymResult(observation=obs, reward=0.0, done=False, info=info)


def _click_action() -> Action:
    return Action(ActionType.CLICK, {"x": 100, "y": 200})


def test_step_no_oracle_delta_returns_baseline():
    """Without ``oracle_mutations_delta`` in info, behaves exactly like
    the parent ``MarketplaceListingReward`` step (format + loop + off-site)."""
    reward = SyntheticEnvReward(allowed_domains=("example.test",))
    state = EpisodeState()
    result = reward.step(
        action=_click_action(),
        gym_result=_gym_result({}),
        state=state,
    )
    # No oracle_step component should appear.
    assert "oracle_step" not in result.components
    # But the format reward inherited from PlanAdherenceReward should.
    assert "format" in result.components


def test_step_with_matching_mutation_adds_oracle_step_component():
    reward = SyntheticEnvReward(allowed_domains=("example.test",))
    state = EpisodeState()
    info = {
        "oracle_mutations_delta": [
            {"id": 1, "operation": "lead_submitted", "target_id": "boat-1"},
        ],
        "oracle_step_kind": "submit_lead",
    }
    result = reward.step(
        action=_click_action(),
        gym_result=_gym_result(info),
        state=state,
    )
    assert result.components.get("oracle_step") == 0.1
    assert state.extras["oracle_step_total"] == 0.1


def test_step_with_unrelated_mutation_skips_oracle_component():
    """A consent_set mutation tagged step_kind=submit_lead doesn't match —
    expected_ops for ``submit_lead`` is ``{lead_submitted}``."""
    reward = SyntheticEnvReward(allowed_domains=("example.test",))
    state = EpisodeState()
    info = {
        "oracle_mutations_delta": [{"id": 1, "operation": "consent_set"}],
        "oracle_step_kind": "submit_lead",
    }
    result = reward.step(
        action=_click_action(),
        gym_result=_gym_result(info),
        state=state,
    )
    assert "oracle_step" not in result.components


def test_step_without_step_kind_falls_back_to_union_of_expected_ops():
    """Without a step_kind tag, ANY mutation in the union of expected_ops
    counts — back-compat for callers that don't tag steps yet."""
    reward = SyntheticEnvReward(allowed_domains=("example.test",))
    state = EpisodeState()
    info = {
        "oracle_mutations_delta": [
            {"id": 1, "operation": "lead_submitted"},
        ],
        # No oracle_step_kind.
    }
    result = reward.step(
        action=_click_action(),
        gym_result=_gym_result(info),
        state=state,
    )
    assert result.components.get("oracle_step") == 0.1


def test_step_kind_with_empty_expected_set_skips_oracle():
    """``filter`` kind maps to an empty op set (filters aren't stamped
    as mutations in boattrader) — no oracle_step component, but no crash."""
    reward = SyntheticEnvReward(allowed_domains=("example.test",))
    state = EpisodeState()
    info = {
        "oracle_mutations_delta": [
            {"id": 1, "operation": "lead_submitted"},
        ],
        "oracle_step_kind": "filter",
    }
    result = reward.step(
        action=_click_action(),
        gym_result=_gym_result(info),
        state=state,
    )
    assert "oracle_step" not in result.components


def test_step_oracle_total_accumulates_across_steps():
    reward = SyntheticEnvReward(allowed_domains=("example.test",))
    state = EpisodeState()
    info = {
        "oracle_mutations_delta": [{"id": 1, "operation": "lead_submitted"}],
        "oracle_step_kind": "submit_lead",
    }
    reward.step(action=_click_action(), gym_result=_gym_result(info), state=state)
    reward.step(
        action=_click_action(),
        gym_result=_gym_result({
            "oracle_mutations_delta": [{"id": 2, "operation": "lead_submitted"}],
            "oracle_step_kind": "submit_lead",
        }),
        state=state,
    )
    assert state.extras["oracle_step_total"] == pytest.approx(0.2)


# ── SyntheticEnvReward — terminal path ─────────────────────────────────


def _run_result_with_done(success: bool, summary: str):
    """Build a minimal RunResult-shaped object the reward.episode reads."""
    from dataclasses import dataclass

    @dataclass
    class _Step:
        action: Action

    @dataclass
    class _Result:
        trajectory: list

    done = Action(ActionType.DONE, {"success": success, "summary": summary})
    return _Result(trajectory=[_Step(action=done)])


def test_episode_without_oracle_terminal_falls_back_to_parent_gate():
    """When ``state.extras['oracle_terminal']`` is absent, reward uses
    the parent's done-summary parser exactly as MarketplaceListingReward."""
    reward = SyntheticEnvReward(allowed_domains=("example.test",))
    state = EpisodeState()
    rr = _run_result_with_done(
        success=True,
        summary="Year: 2022, Make: Sea Ray, Model: 230 SLX, Price: $50,700 url: https://example.test/boat/foo",
    )
    signal = reward.episode(run_result=rr, state=state, ground_truth=None)
    # Parent gate fires when all required fields parse and URL matches.
    assert signal.components.get("gate_passed") == 1.0


def test_episode_oracle_terminal_passed_overrides_gate():
    """When the oracle says ``passed=true`` with score=0.9, reward replaces
    the parent's done-summary gate with the oracle's F1 score plus the
    pass bonus."""
    reward = SyntheticEnvReward(allowed_domains=("example.test",))
    state = EpisodeState()
    state.extras["oracle_terminal"] = {
        "passed": True,
        "score": 0.9,
        "reasons": ["1 qualifying lead(s) submitted with no collateral"],
        "diff": {"hits": 1, "misses": 0},
    }
    rr = _run_result_with_done(success=False, summary="")  # parent gate would fail
    signal = reward.episode(run_result=rr, state=state, ground_truth=None)
    assert "oracle_score" in signal.components
    assert signal.components["oracle_score"] == 0.9  # default terminal_weight=1.0
    assert signal.components.get("oracle_passed_bonus") == 1.0
    # Parent's gate_failed should be wiped out.
    assert "gate_failed" not in signal.components
    assert state.extras["oracle_passed"] is True
    assert state.extras["oracle_score"] == 0.9


def test_episode_oracle_terminal_failed_keeps_zero_bonus():
    reward = SyntheticEnvReward(allowed_domains=("example.test",))
    state = EpisodeState()
    state.extras["oracle_terminal"] = {
        "passed": False,
        "score": 0.0,
        "reasons": ["no leads submitted"],
        "diff": {"hits": 0, "misses": 0},
    }
    rr = _run_result_with_done(success=True, summary="anything")
    signal = reward.episode(run_result=rr, state=state, ground_truth=None)
    assert signal.components.get("oracle_score") == 0.0
    assert "oracle_passed_bonus" not in signal.components


def test_episode_oracle_terminal_with_custom_weight():
    reward = SyntheticEnvReward(
        allowed_domains=("example.test",),
        oracle_terminal_weight=2.0,
    )
    state = EpisodeState()
    state.extras["oracle_terminal"] = {
        "passed": True,
        "score": 0.5,
        "reasons": [],
        "diff": {},
    }
    rr = _run_result_with_done(success=False, summary="")
    signal = reward.episode(run_result=rr, state=state, ground_truth=None)
    assert signal.components["oracle_score"] == 1.0  # 2.0 * 0.5
    assert signal.components["oracle_passed_bonus"] == 2.0


# ── Default config sanity ──────────────────────────────────────────────


def test_default_op_table_has_expected_kinds():
    assert "submit_lead" in DEFAULT_MARKETPLACE_OPS_BY_STEP_KIND
    assert "lead_submitted" in DEFAULT_MARKETPLACE_OPS_BY_STEP_KIND["submit_lead"]
    # Filter/navigate intentionally empty per docstring.
    assert DEFAULT_MARKETPLACE_OPS_BY_STEP_KIND["filter"] == frozenset()
    assert DEFAULT_MARKETPLACE_OPS_BY_STEP_KIND["navigate"] == frozenset()


def test_custom_op_table_overrides_default():
    """A recipe targeting a different sim env can pass its own table."""
    custom = {"submit_form": frozenset({"form_posted"})}
    reward = SyntheticEnvReward(
        allowed_domains=("example.test",),
        expected_ops_by_step_kind=custom,
    )
    state = EpisodeState()
    info = {
        "oracle_mutations_delta": [{"id": 1, "operation": "form_posted"}],
        "oracle_step_kind": "submit_form",
    }
    result = reward.step(
        action=_click_action(),
        gym_result=_gym_result(info),
        state=state,
    )
    assert result.components.get("oracle_step") == 0.1
