"""Unit tests for the hot-mode capture bracket.

The bracket (``freeze_chrome``) sends SIGSTOP + SIGCONT to a Chrome
process tree. These tests cover every branch that doesn't require a
real Chrome — the bracket's logic (PID resolution, signal dispatch,
SIGCONT-on-exception), not the writer's archive contents.

A separate integration test (deferred to a follow-up) will exercise
the writer's hot-mode branch against a real Chrome with a known
cookie write in flight, to prove the SQLite WAL replay reads the
captured pair correctly.
"""

from __future__ import annotations

import signal

import pytest

from mantis_agent.observability.chrome_hot_snapshot import (
    _pid_alive,
    _resolve_process_tree,
    _signal_pids,
    freeze_chrome,
    fsync_profile_dir,
)


# ── freeze_chrome short-circuits ──────────────────────────────────────


def test_freeze_chrome_noop_when_pid_zero() -> None:
    """``chrome_pid=0`` (caller doesn't know the PID) → no signals."""
    sent: list[tuple[int, int]] = []

    def _capture_kill(pid: int, sig: int) -> None:
        sent.append((pid, sig))

    import mantis_agent.observability.chrome_hot_snapshot as mod
    original = mod.os.kill
    mod.os.kill = _capture_kill
    try:
        with freeze_chrome(0):
            pass
    finally:
        mod.os.kill = original
    assert sent == []


def test_freeze_chrome_noop_when_pid_dead(monkeypatch) -> None:
    """``chrome_pid`` not alive → bracket short-circuits, no SIGSTOP."""
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot._pid_alive",
        lambda pid: False,
    )
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot.os.kill",
        lambda *a, **kw: sent.append(a),
    )
    with freeze_chrome(99999):
        pass
    assert sent == []


def test_freeze_chrome_sends_sigstop_then_sigcont(monkeypatch) -> None:
    """Happy path — SIGSTOP on enter, SIGCONT on exit, in order."""
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot._pid_alive",
        lambda pid: True,
    )
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot._resolve_process_tree",
        lambda pid: [pid, pid + 1, pid + 2],
    )
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot.os.kill",
        lambda pid, sig: sent.append((pid, sig)),
    )
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot.time.sleep",
        lambda *a, **kw: None,
    )
    with freeze_chrome(1234):
        pass
    # 3 SIGSTOP + 3 SIGCONT, in that order.
    assert sent[:3] == [(1234, signal.SIGSTOP), (1235, signal.SIGSTOP), (1236, signal.SIGSTOP)]
    assert sent[3:] == [(1234, signal.SIGCONT), (1235, signal.SIGCONT), (1236, signal.SIGCONT)]


def test_freeze_chrome_sends_sigcont_even_on_exception(monkeypatch) -> None:
    """If the protected region raises, SIGCONT MUST still fire."""
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot._pid_alive",
        lambda pid: True,
    )
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot._resolve_process_tree",
        lambda pid: [pid],
    )
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot.os.kill",
        lambda pid, sig: sent.append((pid, sig)),
    )
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot.time.sleep",
        lambda *a, **kw: None,
    )
    with pytest.raises(RuntimeError, match="simulated body failure"):
        with freeze_chrome(5678):
            raise RuntimeError("simulated body failure")
    assert (5678, signal.SIGSTOP) in sent
    assert (5678, signal.SIGCONT) in sent


def test_freeze_chrome_swallows_sigcont_failure(monkeypatch) -> None:
    """If SIGCONT raises (extremely unusual), the bracket must not
    propagate it — that would mask whatever raised in the body."""
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot._pid_alive",
        lambda pid: True,
    )
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot._resolve_process_tree",
        lambda pid: [pid],
    )
    call_count = {"n": 0}

    def _kill(pid: int, sig: int) -> None:
        call_count["n"] += 1
        if sig == signal.SIGCONT:
            raise OSError("simulated SIGCONT failure")

    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot.os.kill", _kill,
    )
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot.time.sleep",
        lambda *a, **kw: None,
    )
    # No exception bubbles out — SIGCONT failure is logged + swallowed.
    with freeze_chrome(42):
        pass


# ── helpers ───────────────────────────────────────────────────────────


def test_pid_alive_zero_is_false() -> None:
    assert _pid_alive(0) is False


def test_pid_alive_negative_is_false() -> None:
    assert _pid_alive(-1) is False


def test_pid_alive_current_process_is_true() -> None:
    import os as _os
    assert _pid_alive(_os.getpid()) is True


def test_resolve_process_tree_dead_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot._pid_alive",
        lambda pid: False,
    )
    assert _resolve_process_tree(99999) == []


def test_signal_pids_skips_dead(monkeypatch) -> None:
    sent: list[tuple[int, int]] = []

    def _kill(pid: int, sig: int) -> None:
        if pid == 200:
            raise ProcessLookupError("dead")
        sent.append((pid, sig))

    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot.os.kill", _kill,
    )
    _signal_pids([100, 200, 300], signal.SIGSTOP)
    # 200 was dead; 100 + 300 still get signaled.
    assert (100, signal.SIGSTOP) in sent
    assert (300, signal.SIGSTOP) in sent
    assert (200, signal.SIGSTOP) not in sent


# ── fsync_profile_dir ──────────────────────────────────────────────────


def test_fsync_profile_dir_handles_missing_dir(tmp_path) -> None:
    # Must not raise — best-effort.
    fsync_profile_dir(tmp_path / "does-not-exist")


def test_fsync_profile_dir_visits_each_file(tmp_path, monkeypatch) -> None:
    """The helper should issue ``fsync`` on each file under the dir."""
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b")
    fsync_calls: list[int] = []
    real_fsync = __import__("os").fsync

    def _track_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        try:
            real_fsync(fd)
        except OSError:
            pass

    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot.os.fsync",
        _track_fsync,
    )
    fsync_profile_dir(tmp_path)
    # At least the two files + the directory entry → ≥3 fsync calls.
    assert len(fsync_calls) >= 3


# ── snapshotter wiring (env-gated hot mode) ────────────────────────────


def test_capture_hot_raises_when_allow_hot_mode_off(tmp_path, monkeypatch) -> None:
    from tests.test_profile_snapshotter_loader import (
        _FakeS3, _make_profile_dir,
    )
    from mantis_agent.observability.profile_snapshotter import (
        ProfileSnapshotter,
    )
    snap = ProfileSnapshotter(
        bucket="b", chrome_major=131, s3_client=_FakeS3(),
        allow_hot_mode=False,
    )
    src = _make_profile_dir(tmp_path)
    with pytest.raises(NotImplementedError, match="allow_hot_mode"):
        snap.capture(
            tenant_id="acme", profile_id="user-1",
            source_profile_dir=src, mode="hot",
        )


def test_capture_hot_skips_freeze_when_no_pid(tmp_path, monkeypatch) -> None:
    """``allow_hot_mode=True`` + ``chrome_pid=0`` → capture succeeds
    but the freeze bracket is a no-op (cold-mode equivalent)."""
    from tests.test_profile_snapshotter_writer import _CASFakeS3
    from mantis_agent.observability.profile_snapshotter import (
        ProfileSnapshotter,
    )
    from tests.test_profile_snapshotter_loader import _make_profile_dir
    snap = ProfileSnapshotter(
        bucket="b", chrome_major=131, s3_client=_CASFakeS3(),
        allow_hot_mode=True,
    )
    src = _make_profile_dir(tmp_path)
    result = snap.capture(
        tenant_id="acme", profile_id="user-1",
        source_profile_dir=src, mode="hot",
        chrome_pid=0,
    )
    assert result.outcome == "captured"


def test_capture_hot_brackets_with_signals_when_pid_given(
    tmp_path, monkeypatch,
) -> None:
    """``allow_hot_mode=True`` + ``chrome_pid>0`` → signals fire
    around the archive step."""
    from tests.test_profile_snapshotter_writer import _CASFakeS3
    from mantis_agent.observability.profile_snapshotter import (
        ProfileSnapshotter,
    )
    from tests.test_profile_snapshotter_loader import _make_profile_dir

    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot._pid_alive",
        lambda pid: True,
    )
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot._resolve_process_tree",
        lambda pid: [pid],
    )
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot.os.kill",
        lambda pid, sig: sent.append((pid, sig)),
    )
    monkeypatch.setattr(
        "mantis_agent.observability.chrome_hot_snapshot.time.sleep",
        lambda *a, **kw: None,
    )

    snap = ProfileSnapshotter(
        bucket="b", chrome_major=131, s3_client=_CASFakeS3(),
        allow_hot_mode=True,
    )
    src = _make_profile_dir(tmp_path)
    result = snap.capture(
        tenant_id="acme", profile_id="user-1",
        source_profile_dir=src, mode="hot",
        chrome_pid=4242,
    )
    assert result.outcome == "captured"
    assert (4242, signal.SIGSTOP) in sent
    assert (4242, signal.SIGCONT) in sent
