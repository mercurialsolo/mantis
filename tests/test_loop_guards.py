"""Tests for loop guards — state-repeat + off-source drift (#782, PR 5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mantis_agent.gym.loop_guards import (
    LOOP_STUCK,
    OFF_SOURCE_DRIFT,
    LoopGuardHalt,
    LoopGuardSuite,
    OffSourceDriftGuard,
    StateRepeatGuard,
    fingerprint,
)


# ── fingerprint ──────────────────────────────────────────────────


def test_fingerprint_is_stable():
    a = fingerprint("https://x.com/", b"png-bytes", "Hello")
    b = fingerprint("https://x.com/", b"png-bytes", "Hello")
    assert a == b
    assert len(a) == 32


def test_fingerprint_changes_on_url():
    a = fingerprint("https://x.com/", b"png", "h")
    b = fingerprint("https://y.com/", b"png", "h")
    assert a != b


def test_fingerprint_changes_on_screenshot():
    a = fingerprint("https://x.com/", b"png-a", "h")
    b = fingerprint("https://x.com/", b"png-b", "h")
    assert a != b


def test_fingerprint_changes_on_text():
    a = fingerprint("https://x.com/", b"png", "Hello")
    b = fingerprint("https://x.com/", b"png", "World")
    assert a != b


# ── StateRepeatGuard ─────────────────────────────────────────────


def test_state_repeat_disabled_by_default():
    g = StateRepeatGuard()  # threshold=0
    # Identical observations forever — should never halt.
    for _ in range(10):
        assert g.observe(url="x", screenshot_bytes=b"a", visible_text="t") is None


def test_state_repeat_trips_at_threshold():
    g = StateRepeatGuard(threshold=3)
    # Iter 1: streak=1. Iter 2: streak=2. Iter 3: streak=3 → halt.
    assert g.observe(url="x", screenshot_bytes=b"a", visible_text="t") is None
    assert g.observe(url="x", screenshot_bytes=b"a", visible_text="t") is None
    halt = g.observe(url="x", screenshot_bytes=b"a", visible_text="t")
    assert halt is not None
    assert halt.halt_class == LOOP_STUCK
    assert halt.evidence["streak"] == 3


def test_state_repeat_resets_on_state_change():
    g = StateRepeatGuard(threshold=3)
    g.observe(url="x", screenshot_bytes=b"a", visible_text="t")
    g.observe(url="x", screenshot_bytes=b"a", visible_text="t")
    # Different screenshot resets streak.
    g.observe(url="x", screenshot_bytes=b"b", visible_text="t")
    # Two more identical to 'b' — should not halt because streak restarted.
    assert g.observe(url="x", screenshot_bytes=b"b", visible_text="t") is None


def test_state_repeat_reset_method():
    g = StateRepeatGuard(threshold=2)
    g.observe(url="x", screenshot_bytes=b"a", visible_text="t")
    g.reset()
    # After reset, single fresh observe is not enough to halt.
    assert g.observe(url="x", screenshot_bytes=b"a", visible_text="t") is None


# ── OffSourceDriftGuard ──────────────────────────────────────────


def test_off_source_disabled_by_default():
    g = OffSourceDriftGuard()  # no pattern, budget=0
    for _ in range(10):
        assert g.observe(url="https://random.com/") is None


def test_off_source_disabled_without_budget():
    g = OffSourceDriftGuard(pinned_pattern="https://x.com/*", step_budget=0)
    for _ in range(10):
        assert g.observe(url="https://off.com/") is None


def test_off_source_halts_at_budget():
    g = OffSourceDriftGuard(pinned_pattern="https://hn.com/*", step_budget=3)
    # Three consecutive off-pattern observations.
    g.observe(url="https://off.com/a")
    g.observe(url="https://off.com/b")
    halt = g.observe(url="https://off.com/c")
    assert halt is not None
    assert halt.halt_class == OFF_SOURCE_DRIFT
    assert halt.evidence["off_streak"] == 3
    assert halt.evidence["off_urls_trail"][-1] == "https://off.com/c"


def test_off_source_resets_on_return_to_pattern():
    g = OffSourceDriftGuard(pinned_pattern="https://hn.com/*", step_budget=3)
    g.observe(url="https://off.com/a")
    g.observe(url="https://off.com/b")
    # Return to pinned origin resets streak.
    g.observe(url="https://hn.com/news")
    # Two more off-pattern — should not halt.
    g.observe(url="https://off.com/c")
    halt = g.observe(url="https://off.com/d")
    assert halt is None


def test_off_source_glob_pattern():
    g = OffSourceDriftGuard(pinned_pattern="https://hn.com/*", step_budget=2)
    # Subpath of pin — should NOT count as off.
    assert g.observe(url="https://hn.com/news?p=1") is None
    assert g.observe(url="https://hn.com/item?id=42") is None


def test_off_source_exact_origin_pattern():
    g = OffSourceDriftGuard(pinned_pattern="https://hn.com", step_budget=2)
    assert g.observe(url="https://hn.com/") is None
    assert g.observe(url="https://hn.com/news") is None


def test_off_source_evidence_trail_is_bounded():
    g = OffSourceDriftGuard(pinned_pattern="https://hn.com/*", step_budget=10)
    for i in range(8):
        g.observe(url=f"https://off.com/{i}")
    halt = g.observe(url="https://off.com/8")
    # Trail capped at last 4 entries even after 9 off-pattern observations.
    assert halt is None  # haven't hit budget=10 yet
    assert len(g._off_urls) == 4
    assert g._off_urls[-1] == "https://off.com/8"


# ── LoopGuardSuite ───────────────────────────────────────────────


def test_loop_guard_suite_off_source_takes_precedence():
    suite = LoopGuardSuite(
        state_repeat=StateRepeatGuard(threshold=2),
        off_source=OffSourceDriftGuard(
            pinned_pattern="https://hn.com/*", step_budget=2
        ),
    )
    # Identical fingerprints AND off-pattern URLs — both guards would
    # trip. Suite picks off_source (more diagnostic).
    suite.observe(url="https://off.com/", screenshot_bytes=b"a", visible_text="t")
    halt = suite.observe(
        url="https://off.com/", screenshot_bytes=b"a", visible_text="t"
    )
    assert halt is not None
    assert halt.halt_class == OFF_SOURCE_DRIFT


def test_loop_guard_suite_state_repeat_when_no_off_source():
    suite = LoopGuardSuite(
        state_repeat=StateRepeatGuard(threshold=2),
        off_source=OffSourceDriftGuard(),  # disabled
    )
    suite.observe(url="https://x.com/", screenshot_bytes=b"a", visible_text="t")
    halt = suite.observe(
        url="https://x.com/", screenshot_bytes=b"a", visible_text="t"
    )
    assert halt is not None
    assert halt.halt_class == LOOP_STUCK


def test_loop_guard_suite_reset_clears_both():
    suite = LoopGuardSuite(
        state_repeat=StateRepeatGuard(threshold=10),
        off_source=OffSourceDriftGuard(
            pinned_pattern="https://hn.com/*", step_budget=10
        ),
    )
    suite.observe(url="https://off.com/", screenshot_bytes=b"a", visible_text="t")
    suite.reset()
    assert suite.state_repeat._streak == 0
    assert suite.off_source._off_streak == 0


# ── plan.schema.json wiring ──────────────────────────────────────


def test_schema_declares_loop_guard_runtime_fields():
    schema_path = (
        Path(__file__).parent.parent / "docs" / "reference" / "plan.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    runtime_props = schema["$defs"]["Runtime"]["properties"]
    assert "state_repeat_threshold" in runtime_props
    assert runtime_props["state_repeat_threshold"]["default"] == 0
    assert "pinned_source_url_pattern" in runtime_props
    assert "off_source_step_budget" in runtime_props
    assert runtime_props["off_source_step_budget"]["default"] == 0


# ── FailureCategory wiring ───────────────────────────────────────


def test_failure_category_has_loop_guard_classes():
    from mantis_agent.gym.workflow_runner import FailureCategory

    assert FailureCategory.LOOP_STUCK == "loop_stuck"
    assert FailureCategory.OFF_SOURCE_DRIFT == "off_source_drift"


def test_loop_config_carries_guard_fields():
    from mantis_agent.gym.workflow_runner import LoopConfig

    cfg = LoopConfig(
        iteration_intent="test",
        state_repeat_threshold=5,
        pinned_source_url_pattern="https://x.com/*",
        off_source_step_budget=3,
    )
    assert cfg.state_repeat_threshold == 5
    assert cfg.pinned_source_url_pattern == "https://x.com/*"
    assert cfg.off_source_step_budget == 3


def test_loop_config_defaults_guards_off():
    from mantis_agent.gym.workflow_runner import LoopConfig

    cfg = LoopConfig(iteration_intent="test")
    assert cfg.state_repeat_threshold == 0
    assert cfg.pinned_source_url_pattern == ""
    assert cfg.off_source_step_budget == 0
