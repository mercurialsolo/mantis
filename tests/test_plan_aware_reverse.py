"""Tests for #121 step 2 — plan-aware reverse decisions.

We focus on the pure helper :meth:`MicroPlanRunner._reverse_decision_from_diff`
plus an integration test of the higher-level dispatcher
:meth:`MicroPlanRunner._plan_aware_reverse_actions`. The keystroke-firing
:meth:`_reverse_step` itself is one ``self.env.step`` per decision — that
path is exercised by the existing integration suite.
"""

from __future__ import annotations

from mantis_agent.gym.micro_runner import MicroPlanRunner
from mantis_agent.gym.step_snapshot import StepDiff, StepStateSnapshot
from mantis_agent.plan_decomposer import MicroIntent


def _step(step_type: str = "click", reverse: str = "") -> MicroIntent:
    """Tiny MicroIntent constructor with the fields _reverse_step touches."""
    return MicroIntent(
        intent="click first listing",
        type=step_type,
        reverse=reverse,
    )


# ── _reverse_decision_from_diff ─────────────────────────────────────────


def test_url_change_returns_alt_left() -> None:
    delta = StepDiff(url_changed=True, changed_fields=["url: a → b"])
    decision = MicroPlanRunner._reverse_decision_from_diff(_step("click"), delta)
    assert decision == [("key_press", "alt+Left")]


def test_extraction_added_skips_reverse() -> None:
    """Forward progress was made — reverting would destroy work."""
    delta = StepDiff(
        extraction_added=True,
        changed_fields=["last_extracted: https://x.test/a"],
    )
    assert MicroPlanRunner._reverse_decision_from_diff(_step("click"), delta) == []


def test_new_urls_seen_skips_reverse() -> None:
    """Even without an extracted-URL change, ``seen_urls +N`` means the
    step actually discovered something."""
    delta = StepDiff(new_urls_seen=True, changed_fields=["seen_urls +2"])
    assert MicroPlanRunner._reverse_decision_from_diff(_step("click"), delta) == []


def test_no_changes_skips_reverse() -> None:
    delta = StepDiff()  # has_changes is False
    assert MicroPlanRunner._reverse_decision_from_diff(_step("click"), delta) == []


def test_focus_only_change_returns_escape_only() -> None:
    """Modal trap pattern: focus moved (autocomplete opened) but URL,
    scroll, page all unchanged. Escape dismisses without alt+left."""
    delta = StepDiff(focus_changed=True, changed_fields=["focused_input"])
    decision = MicroPlanRunner._reverse_decision_from_diff(_step("type"), delta)
    assert decision == [("key_press", "Escape")]


def test_focus_plus_scroll_falls_back_to_legacy() -> None:
    """Mixed signal — diff is non-trivial but doesn't match a clean
    pattern. Returning None tells the dispatcher to use the legacy map."""
    delta = StepDiff(
        focus_changed=True,
        scroll_changed=True,
        changed_fields=["focused_input", "scroll_state"],
    )
    assert MicroPlanRunner._reverse_decision_from_diff(_step("click"), delta) is None


def test_viewport_change_alone_falls_back_to_legacy() -> None:
    delta = StepDiff(viewport_changed=True, changed_fields=["viewport_stage: 0 → 1"])
    assert MicroPlanRunner._reverse_decision_from_diff(_step("click"), delta) is None


def test_page_change_alone_falls_back_to_legacy() -> None:
    """Page advance without URL change shouldn't happen in normal flow,
    but if it does the diff isn't conclusive — defer to legacy."""
    delta = StepDiff(page_changed=True, changed_fields=["page: 1 → 2"])
    assert MicroPlanRunner._reverse_decision_from_diff(_step("click"), delta) is None


# ── _plan_aware_reverse_actions integration ─────────────────────────────


def _bare_runner() -> MicroPlanRunner:
    """Construct a MicroPlanRunner with just the attributes the reverse
    flow touches. Avoids the heavy __init__."""
    runner = MicroPlanRunner.__new__(MicroPlanRunner)
    runner._pre_step_snapshot = None
    runner._last_known_url = ""
    runner._current_page = 1
    runner._viewport_stage = 0
    runner._scroll_state = {}
    runner._last_extracted = {}
    runner._extracted_titles = []
    runner._seen_urls = set()
    runner.env = type("E", (), {})()
    return runner


def test_no_snapshot_falls_back_to_legacy_map() -> None:
    """Backward compat: callers that bypass __init__ (legacy or tests)
    have no snapshot — the static REVERSE_ACTIONS map kicks in."""
    runner = _bare_runner()
    actions = runner._plan_aware_reverse_actions(_step("click"))
    # Legacy click map: Escape then alt+Left.
    assert actions == [("key_press", "Escape"), ("key_press", "alt+Left")]


def test_no_snapshot_legacy_respects_step_reverse_hint() -> None:
    runner = _bare_runner()
    actions = runner._plan_aware_reverse_actions(_step("scroll", reverse="alt+left"))
    # Scroll legacy: Home + alt+Left appended (per step.reverse hint).
    assert ("key_press", "alt+Left") in actions


def test_snapshot_with_url_change_uses_diff_decision() -> None:
    runner = _bare_runner()
    runner._pre_step_snapshot = StepStateSnapshot(url="https://x.test/p1")
    runner._last_known_url = "https://x.test/p2"  # URL moved
    actions = runner._plan_aware_reverse_actions(_step("click"))
    assert actions == [("key_press", "alt+Left")]


def test_snapshot_with_no_change_skips_reverse_entirely() -> None:
    """Step "failed" but state is identical pre/post — nothing to undo."""
    runner = _bare_runner()
    runner._pre_step_snapshot = StepStateSnapshot(url="https://x.test/p")
    runner._last_known_url = "https://x.test/p"  # same
    actions = runner._plan_aware_reverse_actions(_step("click"))
    assert actions == []


def test_snapshot_with_extraction_added_preserves_work() -> None:
    """The form-fill partial-success scenario from the issue body."""
    runner = _bare_runner()
    runner._pre_step_snapshot = StepStateSnapshot(
        url="https://x.test/form", last_extracted_url="",
    )
    runner._last_known_url = "https://x.test/form"  # same URL
    runner._last_extracted = {"last_completed_url": "https://x.test/form/done"}
    actions = runner._plan_aware_reverse_actions(_step("click"))
    # Forward progress detected — preserve it, skip reverse.
    assert actions == []


def test_snapshot_focus_only_returns_escape_only() -> None:
    """Modal trap: only focus changed; Escape, no alt+Left."""
    runner = _bare_runner()
    pre = StepStateSnapshot(
        url="https://x.test/f",
        focused_input_signature="empty",
    )
    runner._pre_step_snapshot = pre
    runner._last_known_url = "https://x.test/f"  # same
    runner.env.last_focused_input = {"placeholder": "field5", "empty": True}
    actions = runner._plan_aware_reverse_actions(_step("type"))
    assert actions == [("key_press", "Escape")]


def test_snapshot_capture_failure_falls_back_to_legacy(monkeypatch) -> None:
    """If post-step capture explodes, the dispatcher must not crash —
    it falls back to the legacy map."""
    runner = _bare_runner()
    runner._pre_step_snapshot = StepStateSnapshot()
    # Force capture() to raise.
    import mantis_agent.gym.step_snapshot as mod
    monkeypatch.setattr(mod, "capture", lambda r: (_ for _ in ()).throw(RuntimeError("boom")))
    actions = runner._plan_aware_reverse_actions(_step("click"))
    # Legacy click map is still applied.
    assert actions == [("key_press", "Escape"), ("key_press", "alt+Left")]


def test_snapshot_diff_falls_back_when_decision_is_none() -> None:
    """A non-trivial but ambiguous diff (focus + scroll changed)
    should drop to the legacy map."""
    runner = _bare_runner()
    pre = StepStateSnapshot(focused_input_signature="empty", scroll_signature="empty")
    runner._pre_step_snapshot = pre
    runner.env.last_focused_input = {"placeholder": "f"}
    runner._scroll_state = {"page_downs": 2}
    actions = runner._plan_aware_reverse_actions(_step("click"))
    # Legacy click map.
    assert actions == [("key_press", "Escape"), ("key_press", "alt+Left")]
