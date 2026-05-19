"""Tests for the pre-dispatch preview gate primitive (#481).

Covers:

* gate is OFF by default (production parity until rolled out);
* IRREVERSIBLE actions get the verifier called; reversible / read-
  only actions skip the verifier (perf);
* PASS / FAIL / low-confidence / verifier-error / unknown-action
  outcomes all return distinct ``reason`` codes;
* the gate never raises — verifier exceptions are caught and
  downgraded to a fail-closed result;
* confidence threshold is configurable via env, clamped sensibly,
  defaults to 0.6.
"""

from __future__ import annotations

import pytest

from mantis_agent.gym.preview_gate import (
    PreviewResult,
    confidence_threshold,
    evaluate,
    is_enabled,
)


def _accept_verifier(**_kwargs) -> tuple[bool, float, str]:
    """Verifier that always says yes with high confidence."""
    return (True, 0.95, "looks like the right button")


def _reject_verifier(**_kwargs) -> tuple[bool, float, str]:
    return (False, 0.3, "highlighted region is a checkbox, not Submit")


def _low_conf_verifier(**_kwargs) -> tuple[bool, float, str]:
    return (True, 0.4, "ambiguous — could be Submit, could be Save")


def _raises_verifier(**_kwargs) -> tuple[bool, float, str]:
    raise RuntimeError("simulated verifier crash")


def _evaluate(verifier, *, action_type: str = "submit", monkeypatch=None):
    """Helper — enables the gate via env, runs evaluate with stub args."""
    if monkeypatch is not None:
        monkeypatch.setenv("MANTIS_PREVIEW_GATE", "on")
    return evaluate(
        action_type=action_type,
        intent="Click the Submit button",
        target_label="Submit",
        target_coordinates=(420, 568),
        screenshot=object(),  # opaque — verifier doesn't actually use it
        verifier=verifier,
    )


# ── Env gating ─────────────────────────────────────────────────────────


def test_is_enabled_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MANTIS_PREVIEW_GATE", raising=False)
    assert is_enabled() is False


@pytest.mark.parametrize("value", ["on", "1", "true", "yes", "ON", "True"])
def test_is_enabled_honours_truthy_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("MANTIS_PREVIEW_GATE", value)
    assert is_enabled() is True


@pytest.mark.parametrize("value", ["", "off", "0", "false", "no"])
def test_is_enabled_rejects_falsy_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("MANTIS_PREVIEW_GATE", value)
    assert is_enabled() is False


def test_evaluate_skips_when_gate_disabled(monkeypatch) -> None:
    """Default off — preserves production behaviour until rollout."""
    monkeypatch.delenv("MANTIS_PREVIEW_GATE", raising=False)
    out = evaluate(
        action_type="submit", intent="x", target_label="Submit",
        target_coordinates=(0, 0), screenshot=object(),
        verifier=_reject_verifier,  # would FAIL if it ran
    )
    assert out.passed is True
    assert out.reason == "skipped"


# ── Reversibility-based skip ───────────────────────────────────────────


@pytest.mark.parametrize(
    "action_type",
    ["click", "scroll", "navigate", "fill_field", "extract_data"],
)
def test_evaluate_skips_for_non_irreversible_actions(monkeypatch, action_type: str) -> None:
    """Reversible / read-only actions don't need the gate. Verifier
    is never called — confirmed by passing one that would raise."""
    monkeypatch.setenv("MANTIS_PREVIEW_GATE", "on")
    out = evaluate(
        action_type=action_type, intent="x", target_label="x",
        target_coordinates=(0, 0), screenshot=object(),
        verifier=_raises_verifier,
    )
    assert out.passed is True
    assert out.reason == "skipped"


def test_evaluate_fires_verifier_for_irreversible_actions(monkeypatch) -> None:
    """``submit`` is IRREVERSIBLE — must call the verifier and report
    its verdict."""
    out = _evaluate(_accept_verifier, monkeypatch=monkeypatch)
    assert out.passed is True
    assert out.reason == "verifier_accepted"
    assert out.confidence == 0.95
    assert "looks like the right button" in out.evidence


def test_evaluate_fires_verifier_for_launch_app(monkeypatch) -> None:
    """``launch_app`` is the other IRREVERSIBLE member of the
    ontology — gate it too."""
    out = _evaluate(_accept_verifier, action_type="launch_app", monkeypatch=monkeypatch)
    assert out.passed is True
    assert out.reason == "verifier_accepted"


# ── Verifier outcomes ──────────────────────────────────────────────────


def test_evaluate_returns_verifier_rejected_on_explicit_false(monkeypatch) -> None:
    out = _evaluate(_reject_verifier, monkeypatch=monkeypatch)
    assert out.passed is False
    assert out.reason == "verifier_rejected"
    assert "checkbox" in out.evidence


def test_evaluate_blocks_on_low_confidence_even_when_passed(monkeypatch) -> None:
    """Defensive against verifiers that report weak signals as
    positive — confidence floor enforced by the gate."""
    out = _evaluate(_low_conf_verifier, monkeypatch=monkeypatch)
    assert out.passed is False
    assert out.reason == "low_confidence"
    assert out.confidence == 0.4
    assert "ambiguous" in out.evidence


def test_evaluate_catches_verifier_exception(monkeypatch) -> None:
    """Verifier crashes never propagate — fail-closed instead."""
    out = _evaluate(_raises_verifier, monkeypatch=monkeypatch)
    assert out.passed is False
    assert out.reason == "verifier_error"
    assert "simulated verifier crash" in out.evidence


def test_evaluate_handles_non_float_confidence(monkeypatch) -> None:
    """Hand-rolled verifiers sometimes return strings — coerce
    defensively, fail-closed if uncoercible."""
    monkeypatch.setenv("MANTIS_PREVIEW_GATE", "on")

    def _bad_verifier(**_kw) -> tuple[bool, str, str]:
        return (True, "not-a-number", "evidence")

    out = evaluate(
        action_type="submit", intent="x", target_label="x",
        target_coordinates=(0, 0), screenshot=object(),
        verifier=_bad_verifier,
    )
    assert out.passed is False
    assert out.reason == "verifier_error"


def test_evaluate_fails_closed_on_unknown_action_type(monkeypatch) -> None:
    """Safety contract: an action_type the ontology doesn't recognise
    cannot pass the gate — we can't guarantee it isn't IRREVERSIBLE."""
    monkeypatch.setenv("MANTIS_PREVIEW_GATE", "on")
    out = evaluate(
        action_type="unknown_action", intent="x", target_label="x",
        target_coordinates=(0, 0), screenshot=object(),
        verifier=_accept_verifier,
    )
    assert out.passed is False
    assert out.reason == "verifier_error"
    assert "unknown action_type" in out.evidence


# ── Confidence threshold configuration ────────────────────────────────


def test_confidence_threshold_default(monkeypatch) -> None:
    monkeypatch.delenv("MANTIS_PREVIEW_CONFIDENCE_THRESHOLD", raising=False)
    assert confidence_threshold() == 0.6


def test_confidence_threshold_honours_env(monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_PREVIEW_CONFIDENCE_THRESHOLD", "0.9")
    assert confidence_threshold() == 0.9


def test_confidence_threshold_clamps_to_unit_interval(monkeypatch) -> None:
    """Out-of-range values are config bugs, not intent — clamp."""
    monkeypatch.setenv("MANTIS_PREVIEW_CONFIDENCE_THRESHOLD", "1.5")
    assert confidence_threshold() == 1.0
    monkeypatch.setenv("MANTIS_PREVIEW_CONFIDENCE_THRESHOLD", "-0.1")
    assert confidence_threshold() == 0.0


def test_confidence_threshold_falls_back_on_garbage(monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_PREVIEW_CONFIDENCE_THRESHOLD", "high")
    assert confidence_threshold() == 0.6  # default


def test_high_threshold_blocks_otherwise_passing_verifier(monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_PREVIEW_GATE", "on")
    monkeypatch.setenv("MANTIS_PREVIEW_CONFIDENCE_THRESHOLD", "0.99")
    out = evaluate(
        action_type="submit", intent="x", target_label="x",
        target_coordinates=(0, 0), screenshot=object(),
        verifier=_accept_verifier,  # returns 0.95 < 0.99
    )
    assert out.passed is False
    assert out.reason == "low_confidence"


# ── Result shape ───────────────────────────────────────────────────────


def test_preview_result_is_frozen() -> None:
    """Immutability — callers don't get to mutate the evidence after
    the gate emits it. Matches the rest of the cua_contracts shape."""
    out = PreviewResult(passed=True, confidence=1.0, reason="skipped")
    with pytest.raises(Exception):  # FrozenInstanceError
        out.passed = False  # type: ignore[misc]
