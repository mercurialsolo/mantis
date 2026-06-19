"""#931 P1 — contenteditable text entry.

LinkedIn's message box and Reddit's composer are ``contenteditable`` divs,
not ``<input>``s. The report showed both predict (``form_target_not_input:
SPAN``) and cua (``typed "…" (unverified)`` but the box stayed empty)
failing on them. These tests pin the two halves of the fix:

* ``XdotoolGymEnv.cdp_contenteditable_insert`` focuses the editable host,
  inserts via CDP, and reports success/fallback correctly.
* The ``probe_element_tag_at`` JS now walks up to the nearest editable
  host (asserted at the Python-wrapper contract level).
"""

from __future__ import annotations

from mantis_agent.gym.som_dispatch import is_input_like, probe_element_tag_at
from mantis_agent.gym.xdotool_env import XdotoolGymEnv


def _env():
    return XdotoolGymEnv.__new__(XdotoolGymEnv)


# ── cdp_contenteditable_insert ──────────────────────────────────────────


def test_insert_success_when_host_focused_and_insert_ok():
    env = _env()
    env.cdp_evaluate = lambda js: True            # focus host + event dispatch
    env._cdp_insert_text = lambda t: True
    assert env.cdp_contenteditable_insert("hello") is True


def test_insert_falls_back_when_no_editable_host():
    env = _env()
    env.cdp_evaluate = lambda js: False           # no contenteditable host in focus
    env._cdp_insert_text = lambda t: True
    assert env.cdp_contenteditable_insert("hello") is False


def test_insert_falls_back_when_cdp_insert_fails():
    env = _env()
    env.cdp_evaluate = lambda js: True
    env._cdp_insert_text = lambda t: False        # Input.insertText didn't take
    assert env.cdp_contenteditable_insert("hello") is False


def test_insert_falls_back_when_no_cdp():
    env = _env()
    env.cdp_evaluate = None                        # env without CDP (other adapters)
    assert env.cdp_contenteditable_insert("hello") is False


def test_insert_event_dispatch_failure_is_nonfatal():
    """If the post-insert input-event dispatch throws, the insert still
    counts as done (the text landed via insertText)."""
    env = _env()
    calls = {"n": 0}

    def _eval(js):
        calls["n"] += 1
        if calls["n"] == 1:
            return True          # focus succeeded
        raise RuntimeError("event dispatch blew up")

    env.cdp_evaluate = _eval
    env._cdp_insert_text = lambda t: True
    assert env.cdp_contenteditable_insert("hello") is True


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
