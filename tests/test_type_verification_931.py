"""#931 P0 — post-type read-back verification.

The CUA logged ``typed "…" (unverified)`` and reported success even when
the text never landed in the field (the report's #1 reliability gap:
"logs claim success but the video proves otherwise"). These tests pin the
new behavior:

* ``XdotoolGymEnv.step`` populates ``info['type_verified']`` after a TYPE
  action so the runner's action log can say verified / FAILED instead of
  the blanket ``(unverified)``.
* The verdict tolerates field auto-formatting and is fail-closed only when
  it can positively read the focused field (CDP off / no field → no
  verdict, preserving legacy behavior).
"""

from __future__ import annotations

import pytest

from mantis_agent.gym.xdotool_env import XdotoolGymEnv


def _env():
    """A bare env instance without running __init__ (no Xvfb / Chrome)."""
    return XdotoolGymEnv.__new__(XdotoolGymEnv)


# ── _type_matches: normalized, auto-format tolerant ─────────────────────


@pytest.mark.parametrize(
    "expected,actual,ok",
    [
        ("Hello world", "Hello world", True),
        ("  Hello   world ", "Hello world", True),          # whitespace-normalized
        ("5551234567", "(555) 123-4567", False),            # not a substring as-is
        ("123-4567", "555 123-4567", True),                 # expected contained in actual
        ("alice@x.io", "alice@x.io", True),
        ("hello", "", False),                                # nothing landed
        ("hello", "goodbye", False),
        ("", "anything", True),                              # empty expected always matches
    ],
)
def test_type_matches(expected, actual, ok):
    assert XdotoolGymEnv._type_matches(expected, actual) is ok


# ── _verify_typed_text: verdict shape + gating ──────────────────────────


def test_verify_typed_text_success(monkeypatch):
    env = _env()
    monkeypatch.delenv("MANTIS_VERIFY_TYPE", raising=False)
    monkeypatch.setattr(env, "_read_active_element_text", lambda: "alice@x.io", raising=False)
    v = env._verify_typed_text("alice@x.io")
    assert v == {"success": True, "expected": "alice@x.io", "actual": "alice@x.io"}


def test_verify_typed_text_failure(monkeypatch):
    env = _env()
    monkeypatch.delenv("MANTIS_VERIFY_TYPE", raising=False)
    monkeypatch.setattr(env, "_read_active_element_text", lambda: "", raising=False)
    v = env._verify_typed_text("hunter2")
    assert v is not None and v["success"] is False and v["actual"] == ""


def test_verify_typed_text_unverified_when_no_field(monkeypatch):
    """No focused text field (CDP returns None) → no verdict, so the
    handler keeps its legacy optimistic success (no new false failures)."""
    env = _env()
    monkeypatch.delenv("MANTIS_VERIFY_TYPE", raising=False)
    monkeypatch.setattr(env, "_read_active_element_text", lambda: None, raising=False)
    assert env._verify_typed_text("anything") is None


def test_verify_typed_text_disabled_by_env(monkeypatch):
    env = _env()
    monkeypatch.setenv("MANTIS_VERIFY_TYPE", "disabled")
    # Even with a readable field, the toggle short-circuits to unverified.
    monkeypatch.setattr(env, "_read_active_element_text", lambda: "x", raising=False)
    assert env._verify_typed_text("x") is None


# ── step() attaches the verdict to the TYPE GymResult ───────────────────


def _stub_step_env(monkeypatch, active_text):
    """An env whose _execute_action / _capture are stubbed so we can drive
    the real ``step`` body and assert on its ``info``."""
    env = _env()
    env._human_speed = False
    env._settle_time = 0.0
    monkeypatch.setenv("MANTIS_ADAPTIVE_SETTLE", "disabled")  # else-branch sleeps settle_time=0
    monkeypatch.setattr(env, "_execute_action", lambda a: None, raising=False)
    monkeypatch.setattr(env, "_capture", lambda: object(), raising=False)
    monkeypatch.setattr(env, "_read_active_element_text", lambda: active_text, raising=False)
    return env


def test_step_type_action_populates_type_verified(monkeypatch):
    from mantis_agent.actions import Action, ActionType

    monkeypatch.delenv("MANTIS_VERIFY_TYPE", raising=False)
    env = _stub_step_env(monkeypatch, active_text="hello")
    res = env.step(Action(action_type=ActionType.TYPE, params={"text": "hello"}))
    assert res.info.get("type_verified", {}).get("success") is True


def test_step_url_type_skips_verification(monkeypatch):
    """URL-shaped text drives the omnibox — there's no DOM field to read,
    so we must not emit a (false) verdict for it."""
    from mantis_agent.actions import Action, ActionType

    monkeypatch.delenv("MANTIS_VERIFY_TYPE", raising=False)
    env = _stub_step_env(monkeypatch, active_text="whatever")
    res = env.step(Action(action_type=ActionType.TYPE, params={"text": "https://x.io"}))
    assert "type_verified" not in res.info


def test_step_non_type_action_has_no_verdict(monkeypatch):
    from mantis_agent.actions import Action, ActionType

    env = _stub_step_env(monkeypatch, active_text="hello")
    res = env.step(Action(action_type=ActionType.CLICK, params={"x": 1, "y": 2}))
    assert "type_verified" not in res.info
