"""Tests for #121 step 1 — StepStateSnapshot + diff helpers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from mantis_agent.gym.step_snapshot import (
    StepStateSnapshot,
    _hash_dict,
    capture,
    diff,
)


# ── _hash_dict ──────────────────────────────────────────────────────────


def test_hash_dict_empty_returns_sentinel() -> None:
    assert _hash_dict(None) == "empty"
    assert _hash_dict({}) == "empty"


def test_hash_dict_stable_across_calls() -> None:
    a = _hash_dict({"k": "v", "n": 1})
    b = _hash_dict({"k": "v", "n": 1})
    assert a == b


def test_hash_dict_order_independent() -> None:
    a = _hash_dict({"a": 1, "b": 2})
    b = _hash_dict({"b": 2, "a": 1})
    assert a == b


def test_hash_dict_differs_on_value_change() -> None:
    a = _hash_dict({"k": "old"})
    b = _hash_dict({"k": "new"})
    assert a != b


def test_hash_dict_handles_non_json_safely() -> None:
    """Anything that JSON can't serialize should fall through to a repr-based
    digest rather than raise."""

    class _Weird:
        pass

    h = _hash_dict({"obj": _Weird()})
    assert isinstance(h, str)
    assert len(h) == 12


# ── StepStateSnapshot ──────────────────────────────────────────────────


def test_snapshot_is_frozen() -> None:
    snap = StepStateSnapshot()
    with pytest.raises(FrozenInstanceError):
        snap.url = "https://x.test"  # type: ignore[misc]


def test_snapshot_to_dict_is_serializable() -> None:
    snap = StepStateSnapshot(
        url="https://x.test", current_page=2, viewport_stage=1,
        focused_input_signature="sig123", scroll_signature="scr456",
        last_extracted_url="https://x.test/a",
        extracted_titles_count=5, seen_urls_count=10,
    )
    d = snap.to_dict()
    assert d["url"] == "https://x.test"
    assert d["current_page"] == 2
    assert d["focused_input_sig"] == "sig123"


def test_snapshot_defaults_compare_equal() -> None:
    """Two no-state snapshots should be indistinguishable so the diff
    against a fresh runner is empty."""
    a = StepStateSnapshot()
    b = StepStateSnapshot()
    assert a == b


# ── capture ─────────────────────────────────────────────────────────────


class _FakeRunner:
    def __init__(self, **attrs: object) -> None:
        # Defaults match a freshly-init'd MicroPlanRunner.
        self._last_known_url = ""
        self._current_page = 1
        self._viewport_stage = 0
        self._scroll_state = {}
        self._last_extracted = {}
        self._extracted_titles = []
        self._seen_urls = set()
        self.env = type("E", (), {})()
        for k, v in attrs.items():
            setattr(self, k, v)


def test_capture_neutral_runner_yields_default_snapshot() -> None:
    runner = _FakeRunner()
    snap = capture(runner)  # type: ignore[arg-type]
    assert snap.url == ""
    assert snap.current_page == 1
    assert snap.viewport_stage == 0
    assert snap.focused_input_signature == "empty"
    assert snap.scroll_signature == "empty"


def test_capture_reflects_runtime_state() -> None:
    runner = _FakeRunner(
        _last_known_url="https://x.test/p2",
        _current_page=2,
        _viewport_stage=1,
        _scroll_state={"page_downs": 3, "context": "results"},
        _last_extracted={"last_completed_url": "https://x.test/detail/1"},
        _extracted_titles=["a", "b"],
        _seen_urls={"u1", "u2", "u3"},
    )
    snap = capture(runner)  # type: ignore[arg-type]
    assert snap.url == "https://x.test/p2"
    assert snap.current_page == 2
    assert snap.viewport_stage == 1
    assert snap.scroll_signature != "empty"
    assert snap.last_extracted_url == "https://x.test/detail/1"
    assert snap.extracted_titles_count == 2
    assert snap.seen_urls_count == 3


def test_capture_reads_focused_input_when_env_exposes_it() -> None:
    runner = _FakeRunner()
    runner.env.last_focused_input = {"placeholder": "search", "empty": True}
    snap = capture(runner)  # type: ignore[arg-type]
    assert snap.focused_input_signature != "empty"


def test_capture_handles_env_without_focused_input_attr() -> None:
    """Adapters that don't expose last_focused_input must not crash."""
    runner = _FakeRunner()
    # Default env has no last_focused_input attribute.
    snap = capture(runner)  # type: ignore[arg-type]
    assert snap.focused_input_signature == "empty"


# ── diff ─────────────────────────────────────────────────────────────────


def test_diff_equal_snapshots_has_no_changes() -> None:
    a = StepStateSnapshot(url="https://x.test", current_page=1)
    b = StepStateSnapshot(url="https://x.test", current_page=1)
    delta = diff(a, b)
    assert delta.has_changes is False
    assert delta.summary() == "no change"


def test_diff_url_change_flagged() -> None:
    a = StepStateSnapshot(url="https://x.test/p1")
    b = StepStateSnapshot(url="https://x.test/p2")
    delta = diff(a, b)
    assert delta.url_changed is True
    assert any("url:" in s for s in delta.changed_fields)


def test_diff_page_advance_flagged() -> None:
    a = StepStateSnapshot(current_page=1)
    b = StepStateSnapshot(current_page=2)
    delta = diff(a, b)
    assert delta.page_changed is True


def test_diff_focus_change_flagged_via_signature() -> None:
    a = StepStateSnapshot(focused_input_signature="empty")
    b = StepStateSnapshot(focused_input_signature="sigA")
    delta = diff(a, b)
    assert delta.focus_changed is True


def test_diff_scroll_change_flagged_via_signature() -> None:
    a = StepStateSnapshot(scroll_signature="empty")
    b = StepStateSnapshot(scroll_signature="scr1")
    delta = diff(a, b)
    assert delta.scroll_changed is True


def test_diff_viewport_advance_flagged() -> None:
    a = StepStateSnapshot(viewport_stage=0)
    b = StepStateSnapshot(viewport_stage=2)
    delta = diff(a, b)
    assert delta.viewport_changed is True


def test_diff_extraction_added_only_when_after_has_url() -> None:
    """The extracted-url change should fire only when ``after`` has a URL,
    not on the inverse (clearing the field)."""
    # Forward: empty → set  (extraction happened)
    a = StepStateSnapshot(last_extracted_url="")
    b = StepStateSnapshot(last_extracted_url="https://x.test/listing/1")
    delta_fwd = diff(a, b)
    assert delta_fwd.extraction_added is True

    # Inverse: set → empty  (treated as no extraction added)
    delta_rev = diff(b, a)
    assert delta_rev.extraction_added is False


def test_diff_seen_urls_increase_flagged() -> None:
    a = StepStateSnapshot(seen_urls_count=5)
    b = StepStateSnapshot(seen_urls_count=8)
    delta = diff(a, b)
    assert delta.new_urls_seen is True
    assert any("seen_urls +3" in s for s in delta.changed_fields)


def test_diff_seen_urls_decrease_not_flagged() -> None:
    """Counters only flag forward progress — running the same step twice
    should not produce a delta from a count drop (which shouldn't happen
    anyway, but defensive)."""
    a = StepStateSnapshot(seen_urls_count=8)
    b = StepStateSnapshot(seen_urls_count=5)
    delta = diff(a, b)
    assert delta.new_urls_seen is False


def test_diff_summary_combines_multiple_changes() -> None:
    a = StepStateSnapshot(
        url="https://x.test/p1", current_page=1, viewport_stage=0,
    )
    b = StepStateSnapshot(
        url="https://x.test/p2", current_page=2, viewport_stage=1,
    )
    summary = diff(a, b).summary()
    assert "url:" in summary
    assert "page:" in summary
    assert "viewport_stage:" in summary


def test_diff_does_not_mutate_inputs() -> None:
    """``diff()`` is pure — neither argument changes."""
    a = StepStateSnapshot(url="https://x.test")
    b = StepStateSnapshot(url="https://x.test/p2")
    a_repr = repr(a)
    b_repr = repr(b)
    diff(a, b)
    assert repr(a) == a_repr
    assert repr(b) == b_repr


# ── Worked example: the form-fill partial-success scenario ──────────────


def test_form_fill_partial_success_diff_signals_focus_only() -> None:
    """Scenario from #121 issue body: a form step that typed 3 of 5 fields.
    The diff should show focus changed but URL did not — telling the
    follow-on plan-aware reverse to NOT fire alt+left."""
    before = StepStateSnapshot(
        url="https://x.test/form",
        focused_input_signature="empty",
        scroll_signature="scr0",
    )
    after = StepStateSnapshot(
        url="https://x.test/form",  # same — type didn't navigate
        focused_input_signature="sig_field5",  # last typed field still focused
        scroll_signature="scr0",
    )
    delta = diff(before, after)
    assert delta.url_changed is False
    assert delta.focus_changed is True
    assert delta.has_changes is True
