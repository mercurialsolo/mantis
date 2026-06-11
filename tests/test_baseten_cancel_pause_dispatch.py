"""Tests for the Baseten runtime cancel/pause dispatch fix + terminal-sticky.

Bug report: ``action=cancel`` and ``action=pause`` fell through to the
``detached`` branch in ``BasetenCUARuntime.run`` because the dispatch
set only contained ``status/result/logs/resume``. That routed a cancel
onto ``_start_detached``, which either 400'd on an active run or
created a fresh run on a finished run_id (overwriting the record).

These tests drive ``MantisRuntime`` directly with an isolated data
root so we don't need Chrome / GPUs / Modal.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch):
    """Construct a ``BasetenCUARuntime`` against an isolated data dir."""
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MANTIS_BRAIN", "holo3")
    # Don't bring up Chrome / llama-cpp during tests.
    monkeypatch.setenv("MANTIS_SKIP_BRAIN_LOAD", "1")
    from mantis_agent.baseten_server.runtime import BasetenCUARuntime
    rt = BasetenCUARuntime()
    # Stub out the heavy ``load`` so the dispatch path doesn't try to
    # boot a brain. ``run`` only calls load() on the non-action path.
    rt.load = lambda: None  # type: ignore[assignment]
    return rt


def _seed_run(rt: Any, run_id: str, status: str) -> Path:
    """Materialize a status.json for ``run_id`` so the action
    handlers can read it back."""
    rt._write_detached_status(run_id, {
        "status": status,
        "mode": "detached",
        "model": "holo3",
    })
    return rt._run_path(run_id) / "status.json"


# ── Dispatch fix ──────────────────────────────────────────────────────


def test_cancel_dispatches_to_action_handler_not_detached(runtime) -> None:
    """The previous bug: cancel + detached=true fell through to
    ``_start_detached`` which started a new run on the run_id."""
    _seed_run(runtime, "r-cancel", "running")
    out = runtime.run({
        "action": "cancel", "run_id": "r-cancel", "detached": True,
    })
    # Must be the cancelled status — NOT a freshly-queued run.
    assert out["status"] == "cancelled"
    # And no new detached thread spun up.
    assert "r-cancel" not in runtime.detached_threads


def test_pause_dispatches_to_action_handler_not_detached(runtime) -> None:
    _seed_run(runtime, "r-pause", "running")
    out = runtime.run({
        "action": "pause", "run_id": "r-pause", "detached": True,
        "reason": "manual_takeover",
    })
    assert out["status"] == "paused"
    assert out["pause_reason"] == "manual_takeover"
    # Sentinel exists on disk for the worker to poll.
    sentinel = runtime._run_path("r-pause") / "pause_request.json"
    assert sentinel.exists()
    blob = json.loads(sentinel.read_text())
    assert blob["reason"] == "manual_takeover"


# ── Cancel on a finished run does NOT overwrite the record ───────────


def test_cancel_on_finished_run_does_not_overwrite(runtime) -> None:
    """A cancel on a run that already succeeded should be a no-op
    (return the existing status). The pre-fix behaviour created a
    NEW run on the same run_id and flipped the status to queued."""
    _seed_run(runtime, "r-done", "succeeded")
    out = runtime.run({
        "action": "cancel", "run_id": "r-done", "detached": True,
    })
    assert out["status"] == "succeeded", out
    # No new thread.
    assert "r-done" not in runtime.detached_threads


# ── Cancel on an unknown run_id returns 404 (not a new run) ─────────


def test_cancel_on_unknown_run_id_raises_404_style_error(runtime) -> None:
    with pytest.raises(FileNotFoundError, match="unknown run_id"):
        runtime.run({
            "action": "cancel", "run_id": "r-does-not-exist", "detached": True,
        })


def test_pause_on_unknown_run_id_raises_404_style_error(runtime) -> None:
    with pytest.raises(FileNotFoundError, match="unknown run_id"):
        runtime.run({
            "action": "pause", "run_id": "r-never-existed", "detached": True,
        })


# ── Pause requires a running run ─────────────────────────────────────


def test_pause_on_finished_run_400s(runtime) -> None:
    _seed_run(runtime, "r-pause-late", "succeeded")
    with pytest.raises(ValueError, match="requires a running run"):
        runtime.run({
            "action": "pause", "run_id": "r-pause-late", "detached": True,
        })


# ── Terminal-sticky guard ────────────────────────────────────────────


def test_terminal_status_blocks_overwrite_after_cancel(runtime) -> None:
    """After cancel writes cancelled, the worker's eventual terminal
    write (succeeded/failed/halted) must NOT clobber the cancelled
    record."""
    _seed_run(runtime, "r-stick", "running")
    runtime.run({"action": "cancel", "run_id": "r-stick", "detached": True})
    # Worker writes its terminal status from inside the thread.
    runtime._write_detached_status("r-stick", {"status": "succeeded"})
    final = runtime.run({"action": "status", "run_id": "r-stick", "detached": True})
    assert final["status"] == "cancelled", (
        "executor terminal write clobbered cancelled record"
    )


def test_terminal_sticky_allows_same_status_writes(runtime) -> None:
    """An idempotent re-write of cancelled is allowed (annotations may
    be appended). Only DIFFERENT status writes are blocked."""
    _seed_run(runtime, "r-same", "cancelled")
    out = runtime._write_detached_status(
        "r-same", {"status": "cancelled", "extra": "annotation"},
    )
    assert out["extra"] == "annotation"


def test_terminal_sticky_does_not_block_initial_status(runtime) -> None:
    """No prior status on disk → write goes through normally."""
    out = runtime._write_detached_status("r-fresh", {"status": "queued"})
    assert out["status"] == "queued"


# ── Cancel sentinel is written so worker thread can observe it ───────


def test_cancel_writes_sentinel_file(runtime) -> None:
    _seed_run(runtime, "r-sentinel", "running")
    runtime.run({"action": "cancel", "run_id": "r-sentinel", "detached": True})
    sentinel = runtime._run_path("r-sentinel") / "cancel_request.json"
    assert sentinel.exists()
    body = json.loads(sentinel.read_text())
    assert "requested_at" in body


# ── _start_detached unreached for action calls (regression guard) ───


def test_action_cancel_never_calls_start_detached(runtime, monkeypatch) -> None:
    """Explicitly assert ``_start_detached`` is NOT invoked when
    ``action=cancel`` is the request. Catches a re-introduction of
    the original bug."""
    _seed_run(runtime, "r-no-spawn", "running")
    sentinel = threading.Event()

    def _boom(payload):
        sentinel.set()
        raise AssertionError("_start_detached must not be called for action=cancel")

    monkeypatch.setattr(runtime, "_start_detached", _boom)
    runtime.run({
        "action": "cancel", "run_id": "r-no-spawn", "detached": True,
    })
    assert not sentinel.is_set()
