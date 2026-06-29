"""#931 P1 (+ LinkedIn follow-up) — contenteditable text entry.

LinkedIn's message box and Reddit's composer are ``contenteditable`` divs,
not ``<input>``s. The report showed both predict (``form_target_not_input:
SPAN``) and cua (``typed "…" (unverified)`` but the box stayed empty)
failing on them. These tests pin the three halves of the fix:

* ``XdotoolGymEnv.cdp_contenteditable_insert`` inserts via
  ``execCommand('insertText')`` (drives the ``beforeinput`` pipeline the
  editor's model listens to) and **verifies the text landed**, returning
  ``False`` (not a false-positive ``True``) when the box stayed empty.
* ``probe_element_tag_at`` finds the editor even when the grounded pixel
  hits a SIBLING placeholder overlay stacked on top of it (LinkedIn) —
  via ``elementsFromPoint`` / ``role="textbox"``, not ancestor-only
  ``closest()``.
"""

from __future__ import annotations

from mantis_agent.gym.som_dispatch import is_input_like, probe_element_tag_at
from mantis_agent.gym.xdotool_env import XdotoolGymEnv


def _env():
    return XdotoolGymEnv.__new__(XdotoolGymEnv)


# ── cdp_contenteditable_insert ──────────────────────────────────────────


def test_insert_success_when_execcommand_lands_and_readback_confirms():
    """execCommand inserts and the same call's read-back shows the text."""
    env = _env()
    env.cdp_evaluate = lambda js: {"ok": True, "inserted": True, "text": "hello"}
    env._cdp_insert_text = lambda t: True
    assert env.cdp_contenteditable_insert("hello") is True


def test_insert_falls_back_when_no_editable_host():
    env = _env()
    env.cdp_evaluate = lambda js: {"ok": False}   # no contenteditable host in focus
    env._cdp_insert_text = lambda t: True
    assert env.cdp_contenteditable_insert("hello") is False


def test_insert_execcommand_noop_falls_back_to_insertText_then_verifies():
    """When execCommand is a no-op, the Input.insertText fallback runs and
    the post-insert read-back confirms the text landed → True."""
    env = _env()

    def _eval(js):
        if "execCommand" in js:
            return {"ok": True, "inserted": False, "text": ""}  # execCommand no-op
        return "hello"                                          # readback after insertText
    env.cdp_evaluate = _eval
    env._cdp_insert_text = lambda t: True
    assert env.cdp_contenteditable_insert("hello") is True


def test_insert_returns_false_when_text_never_lands():
    """The core false-positive fix: execCommand AND insertText both leave the
    box empty → return False so the caller doesn't report a filled box (which
    sent the director looping on a disabled Send)."""
    env = _env()

    def _eval(js):
        if "execCommand" in js:
            return {"ok": True, "inserted": False, "text": ""}
        return ""                                              # readback still empty
    env.cdp_evaluate = _eval
    env._cdp_insert_text = lambda t: True
    assert env.cdp_contenteditable_insert("hello") is False


def test_insert_falls_back_when_cdp_insert_fails():
    env = _env()
    env.cdp_evaluate = lambda js: {"ok": True, "inserted": False, "text": ""}
    env._cdp_insert_text = lambda t: False        # Input.insertText didn't take
    assert env.cdp_contenteditable_insert("hello") is False


def test_insert_falls_back_when_no_cdp():
    env = _env()
    env.cdp_evaluate = None                        # env without CDP (other adapters)
    assert env.cdp_contenteditable_insert("hello") is False


# ── probe + guard accept contenteditable ────────────────────────────────


def test_probe_normalizes_contenteditable_true():
    env = _env()
    env.cdp_evaluate = lambda js: {"tag": "SPAN", "contentEditable": True}
    info = probe_element_tag_at(env, 100, 200)
    assert info == {"tag": "SPAN", "contentEditable": True}
    assert is_input_like(info) is True  # contenteditable span is fillable


def test_probe_span_without_editable_still_rejected():
    env = _env()
    env.cdp_evaluate = lambda js: {"tag": "SPAN", "contentEditable": False}
    info = probe_element_tag_at(env, 100, 200)
    assert is_input_like(info) is False


def test_probe_js_looks_through_sibling_overlay_stack():
    """LinkedIn's placeholder overlay is a sibling stacked on top of the
    editor, so ancestor-only ``closest()`` misses it. Pin (at the JS-string
    contract level) that the probe consults the full ``elementsFromPoint``
    z-stack and treats ``role="textbox"`` as editable."""
    captured = {}

    env = _env()

    def _eval(js):
        captured["js"] = js
        return {"tag": "SPAN", "contentEditable": True}

    env.cdp_evaluate = _eval
    probe_element_tag_at(env, 100, 200)
    assert "elementsFromPoint" in captured["js"]
    assert "textbox" in captured["js"]


# ── env.step(TYPE) contenteditable fallback (cua-issues 2026-06-29) ──────
#
# The /v1/cua Claude loop types via env.step(TYPE) with raw xdotool only.
# When the read-back shows the text didn't land (contenteditable editors,
# focus-stolen fields), step() now falls back to cdp_contenteditable_insert
# and re-verifies — porting the #934 recovery to every env.step(TYPE) caller.


def _type_step_env(monkeypatch):
    env = _env()
    env._human_speed = False
    env._settle_time = 0.0
    monkeypatch.setenv("MANTIS_ADAPTIVE_SETTLE", "disabled")
    monkeypatch.setattr(env, "_execute_action", lambda a: None, raising=False)
    monkeypatch.setattr(env, "_capture", lambda: object(), raising=False)
    return env


def test_step_type_falls_back_to_contenteditable_on_verify_fail(monkeypatch):
    from mantis_agent.actions import Action, ActionType

    env = _type_step_env(monkeypatch)
    # First read-back: empty (raw type didn't land). After the insert: landed.
    verdicts = iter([
        {"success": False, "expected": "hi", "actual": ""},
        {"success": True, "expected": "hi", "actual": "hi"},
    ])
    monkeypatch.setattr(env, "_verify_typed_text", lambda t: next(verdicts), raising=False)
    seen = {"inserted": False}
    monkeypatch.setattr(
        env, "cdp_contenteditable_insert",
        lambda t: seen.__setitem__("inserted", True) or True, raising=False,
    )
    res = env.step(Action(action_type=ActionType.TYPE, params={"text": "hi"}))
    assert seen["inserted"] is True
    assert res.info["type_verified"]["success"] is True


def test_step_type_no_fallback_when_first_verify_succeeds(monkeypatch):
    from mantis_agent.actions import Action, ActionType

    env = _type_step_env(monkeypatch)
    monkeypatch.setattr(
        env, "_verify_typed_text",
        lambda t: {"success": True, "expected": "x", "actual": "x"}, raising=False,
    )
    calls = {"n": 0}
    monkeypatch.setattr(
        env, "cdp_contenteditable_insert",
        lambda t: calls.__setitem__("n", calls["n"] + 1) or True, raising=False,
    )
    res = env.step(Action(action_type=ActionType.TYPE, params={"text": "x"}))
    assert calls["n"] == 0  # plain input that landed → no fallback
    assert res.info["type_verified"]["success"] is True


def test_step_type_fallback_keeps_failure_when_insert_cannot_land(monkeypatch):
    """Insert runs but the text still doesn't land → verdict stays failed
    (no false success — the honesty contract holds)."""
    from mantis_agent.actions import Action, ActionType

    env = _type_step_env(monkeypatch)
    monkeypatch.setattr(
        env, "_verify_typed_text",
        lambda t: {"success": False, "expected": "hi", "actual": ""}, raising=False,
    )
    monkeypatch.setattr(env, "cdp_contenteditable_insert", lambda t: False, raising=False)
    res = env.step(Action(action_type=ActionType.TYPE, params={"text": "hi"}))
    assert res.info["type_verified"]["success"] is False
