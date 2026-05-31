"""Tests for the S0 retrieval substrate.

The substrate wraps ``hint_memory.apply_hint_overlay``, so these exercise
the *real* overlay against an in-memory hint store rather than mocking it —
a stored anchor must actually land on the plan's step as a
``preferred_target_description`` hint.
"""

from __future__ import annotations

from types import SimpleNamespace

from mantis_agent.gym.hint_memory import (
    HintRecord,
    InMemoryHintStore,
    hint_key_for,
)
from mantis_agent.learning.substrates import RetrievalSubstrate
from mantis_agent.learning.substrates.base import (
    Durability,
    LearningSubstrate,
    SubstrateContext,
)

PLAN_SIG = "plansig12345"
URL = "https://www.boattrader.com/boat/1986-marine-trader/"


def _step(intent: str = "click Show More", step_type: str = "click"):
    # apply_hint_overlay reads .type / .intent and writes .hints in place.
    return SimpleNamespace(type=step_type, intent=intent, hints=None)


def _ctx(plan=None, *, plan_signature: str = PLAN_SIG, start_url: str = URL):
    extras = {"plan_signature": plan_signature, "start_url": start_url}
    if plan is not None:
        extras["plan"] = plan
    return SubstrateContext(task_id="BT01", cluster="knowledge", extras=extras)


def _store_with_anchor(step, *, anchor: str = "Show More") -> InMemoryHintStore:
    store = InMemoryHintStore()
    key = hint_key_for(PLAN_SIG, step, URL)
    store.add(key, HintRecord(anchor_text=anchor, confidence=1.0))
    return store


# ── protocol / cheap-path invariants ───────────────────────────────────


def test_conforms_to_substrate_protocol() -> None:
    assert isinstance(RetrievalSubstrate(), LearningSubstrate)


def test_durability_is_ephemeral() -> None:
    assert RetrievalSubstrate().durability is Durability.EPHEMERAL


def test_cost_estimate_is_free() -> None:
    # A store read costs nothing — the bandit must rank this rung at $0.
    assert RetrievalSubstrate().cost_estimate(_ctx()) == 0.0


def test_default_store_is_safe_to_construct() -> None:
    # No backend wired ⇒ NullHintStore ⇒ apply finds nothing, never raises.
    sub = RetrievalSubstrate()
    res = sub.apply(_ctx(plan=SimpleNamespace(steps=[_step()])))
    assert res.applied is False


# ── apply ───────────────────────────────────────────────────────────────


def test_apply_overlays_a_stored_anchor() -> None:
    step = _step()
    plan = SimpleNamespace(steps=[step])
    sub = RetrievalSubstrate(_store_with_anchor(step))

    res = sub.apply(_ctx(plan=plan))

    assert res.applied is True
    assert res.delta_artifacts["hints_applied"] == 1
    assert res.dollars_spent == 0.0
    # The overlay actually mutated the plan step.
    assert step.hints["preferred_target_description"] == "Show More"


def test_apply_without_plan_is_not_applied() -> None:
    # Pure cost-probe path: no live plan handle ⇒ nothing to overlay.
    res = RetrievalSubstrate(InMemoryHintStore()).apply(_ctx())
    assert res.applied is False
    assert "nothing to overlay" in res.notes


def test_apply_without_plan_signature_is_not_applied() -> None:
    plan = SimpleNamespace(steps=[_step()])
    res = RetrievalSubstrate(InMemoryHintStore()).apply(_ctx(plan=plan, plan_signature=""))
    assert res.applied is False


def test_apply_with_empty_store_is_not_applied() -> None:
    plan = SimpleNamespace(steps=[_step()])
    res = RetrievalSubstrate(InMemoryHintStore()).apply(_ctx(plan=plan))
    assert res.applied is False
    assert res.delta_artifacts["hints_applied"] == 0


def test_apply_does_not_overwrite_operator_hint() -> None:
    step = _step()
    step.hints = {"preferred_target_description": "operator choice"}
    plan = SimpleNamespace(steps=[step])
    sub = RetrievalSubstrate(_store_with_anchor(step, anchor="stored anchor"))

    sub.apply(_ctx(plan=plan))

    # Operator-authored hint wins; the store is a fallback, not an override.
    assert step.hints["preferred_target_description"] == "operator choice"


# ── observe ─────────────────────────────────────────────────────────────


def test_observe_is_noop() -> None:
    sub = RetrievalSubstrate()
    # Must not raise — recording happens in the run executor, not here.
    assert sub.observe(_ctx(), sub.apply(_ctx()), 1.0) is None
