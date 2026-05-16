"""Reasoning-trace module — structured runner decision stream.

Backs the ``action=reasoning_trace`` HTTP endpoint that viewer
overlays poll to render a timeline beside the MJPEG live feed.
The module reuses ``runner._healing_events`` for the in-memory
list so existing healing-event consumers keep working, and
mirrors events to a JSONL file when the runtime configures one.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from mantis_agent.gym import reasoning_trace


def _runner() -> SimpleNamespace:
    return SimpleNamespace(_healing_events=[])


# ── record ─────────────────────────────────────────────────────────


def test_record_appends_event_with_required_fields() -> None:
    """Every event carries ``ts``, ``layer``, ``kind``, ``summary``,
    ``step_index``, ``detail``, and ``category=reasoning``. The
    last lets healing-event consumers filter the timeline by source
    without rebuilding the list."""
    runner = _runner()
    reasoning_trace.record(
        runner, layer="critic-frontier", kind="fire",
        summary="gate passed on wrong_target", step_index=6,
        failure_count=2, failure_class="wrong_target",
    )
    assert len(runner._healing_events) == 1
    ev = runner._healing_events[0]
    assert ev["layer"] == "critic-frontier"
    assert ev["kind"] == "fire"
    assert ev["summary"] == "gate passed on wrong_target"
    assert ev["step_index"] == 6
    assert ev["category"] == "reasoning"
    assert ev["detail"]["failure_count"] == 2
    assert ev["detail"]["failure_class"] == "wrong_target"
    # ISO-8601 with TZ — sortable lexicographically across events.
    assert "T" in ev["ts"]
    assert ev["ts"].endswith(("+00:00", "Z"))


def test_record_silently_drops_when_runner_lacks_events_list() -> None:
    """A MagicMock runner auto-creates ``_healing_events`` as a Mock,
    not a list. The trace path must not crash production runs on
    weird runner shapes — silently drop the event."""
    runner = SimpleNamespace(_healing_events=None)  # not a list
    reasoning_trace.record(
        runner, layer="x", kind="y", summary="z",
    )
    # No crash; the non-list stays non-list.
    assert runner._healing_events is None


def test_record_clips_long_summary() -> None:
    """Summary is bounded to 300 chars so a sloppy caller (e.g.
    dumping a 5KB Claude response into ``summary``) doesn't bloat
    the timeline."""
    runner = _runner()
    reasoning_trace.record(
        runner, layer="x", kind="y", summary="x" * 500,
    )
    assert len(runner._healing_events[0]["summary"]) == 300


def test_record_drops_non_jsonable_detail_values() -> None:
    """Detail dict values that don't survive ``json.dumps`` (sockets,
    Mocks, lambdas) get stringified — never raised. Preserves the
    event even when one field is unhealthy."""
    runner = _runner()
    reasoning_trace.record(
        runner, layer="x", kind="y", summary="z",
        good=123, weird=lambda: None,
    )
    ev = runner._healing_events[0]
    assert ev["detail"]["good"] == 123
    # Weird key survives, value coerced to a string.
    assert "weird" in ev["detail"]


# ── disk stream ────────────────────────────────────────────────────


def test_configure_disk_stream_creates_parent_directory(tmp_path) -> None:
    """The configured path's parent is created on demand — keeps
    callers from having to mkdir for every nested run_dir."""
    runner = _runner()
    target = tmp_path / "tenants" / "x" / "runs" / "abc" / "reasoning.jsonl"
    reasoning_trace.configure_disk_stream(runner, target)
    assert target.parent.is_dir()
    assert runner._reasoning_jsonl_path == str(target)


def test_record_writes_to_configured_jsonl_path(tmp_path) -> None:
    """After ``configure_disk_stream``, every ``record`` call also
    appends a single JSONL line to the file. Live-tailable during
    a run."""
    runner = _runner()
    target = tmp_path / "reasoning.jsonl"
    reasoning_trace.configure_disk_stream(runner, target)
    reasoning_trace.record(
        runner, layer="critic-frontier", kind="fire",
        summary="gate passed", step_index=6,
    )
    reasoning_trace.record(
        runner, layer="critic-frontier", kind="result",
        summary="add_hint: click contacted text", step_index=6,
    )
    lines = [
        json.loads(line)
        for line in target.read_text().splitlines() if line.strip()
    ]
    assert len(lines) == 2
    assert lines[0]["summary"] == "gate passed"
    assert lines[1]["summary"] == "add_hint: click contacted text"


def test_record_does_not_crash_on_unwritable_path(tmp_path) -> None:
    """A misconfigured path (parent doesn't exist, permissions denied,
    disk full) must not propagate the OSError — observability isn't
    worth crashing the run for. Event still lands in the in-memory
    list."""
    runner = _runner()
    # configure_disk_stream is normally what creates the parent; here
    # we bypass it to simulate a stale path that disappeared.
    runner._reasoning_jsonl_path = "/this/path/does/not/exist/x.jsonl"
    reasoning_trace.record(
        runner, layer="x", kind="y", summary="z",
    )
    # Event still in memory.
    assert len(runner._healing_events) == 1


# ── read_jsonl ─────────────────────────────────────────────────────


def test_read_jsonl_returns_empty_on_missing_file(tmp_path) -> None:
    """The endpoint hits ``read_jsonl`` before the runner has fired
    any events — must return ``[]`` cleanly rather than raise."""
    assert reasoning_trace.read_jsonl(tmp_path / "nope.jsonl") == []


def test_read_jsonl_filters_by_since_ts(tmp_path) -> None:
    """The endpoint supports incremental polling — ``since_ts`` filters
    to events strictly more recent than the cursor. Lexicographic
    compare on ISO-8601 UTC timestamps is order-preserving."""
    runner = _runner()
    target = tmp_path / "reasoning.jsonl"
    reasoning_trace.configure_disk_stream(runner, target)
    reasoning_trace.record(runner, layer="x", kind="y", summary="first")
    first_ts = runner._healing_events[-1]["ts"]
    reasoning_trace.record(runner, layer="x", kind="y", summary="second")
    reasoning_trace.record(runner, layer="x", kind="y", summary="third")

    out = reasoning_trace.read_jsonl(target, since_ts=first_ts)
    summaries = [e["summary"] for e in out]
    assert "first" not in summaries
    assert summaries == ["second", "third"]


def test_read_jsonl_skips_malformed_lines(tmp_path) -> None:
    """A partially-written final line (the runner crashed mid-flush)
    or an operator hand-editing the file with a stray text line must
    not break the reader. Skip malformed lines silently and return
    the rest."""
    target = tmp_path / "reasoning.jsonl"
    target.write_text(
        '{"ts": "2026-05-16T19:00:00+00:00", "summary": "valid"}\n'
        'this is not json\n'
        '{"ts": "2026-05-16T19:01:00+00:00", "summary": "after garbage"}\n'
        '{"partial-line": "no-close-brace"\n'
    )
    out = reasoning_trace.read_jsonl(target)
    assert len(out) == 2
    assert out[0]["summary"] == "valid"
    assert out[1]["summary"] == "after garbage"
