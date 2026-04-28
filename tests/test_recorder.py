"""Tests for the screen recorder + /v1/runs/{run_id}/video download endpoint.

We don't actually run ffmpeg in CI — every test patches subprocess.Popen
with a mock that mimics the lifecycle (started, then exits cleanly on 'q'
sent to stdin) and writes a small placeholder file to disk.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mantis_agent.recorder import (
    RecorderResult,
    ScreenRecorder,
    _build_ffmpeg_cmd,
    content_type_for,
    ffmpeg_available,
)


# ── Command construction ────────────────────────────────────────────────────
def test_build_ffmpeg_cmd_mp4(tmp_path: Path):
    cmd = _build_ffmpeg_cmd(
        display=":99", output=tmp_path / "out.mp4",
        fps=5, fmt="mp4", width=1280, height=720,
    )
    assert cmd[0] == "ffmpeg"
    assert "x11grab" in cmd
    assert "-i" in cmd
    assert ":99" in cmd
    assert "libx264" in cmd
    assert str(tmp_path / "out.mp4") == cmd[-1]


def test_build_ffmpeg_cmd_webm(tmp_path: Path):
    cmd = _build_ffmpeg_cmd(
        display=":99", output=tmp_path / "out.webm",
        fps=10, fmt="webm", width=640, height=480,
    )
    assert "libvpx-vp9" in cmd


def test_build_ffmpeg_cmd_gif(tmp_path: Path):
    cmd = _build_ffmpeg_cmd(
        display=":99", output=tmp_path / "out.gif",
        fps=8, fmt="gif", width=640, height=360,
    )
    # Filtergraph with palettegen/paletteuse
    vf = cmd[cmd.index("-vf") + 1]
    assert "palettegen" in vf and "paletteuse" in vf


def test_build_ffmpeg_cmd_rejects_unknown_format(tmp_path: Path):
    with pytest.raises(ValueError):
        _build_ffmpeg_cmd(
            display=":99", output=tmp_path / "out.foo",
            fps=5, fmt="foo",  # type: ignore[arg-type]
            width=1280, height=720,
        )


def test_content_type_for_each_format():
    assert content_type_for("mp4") == "video/mp4"
    assert content_type_for("webm") == "video/webm"
    assert content_type_for("gif") == "image/gif"


# ── ScreenRecorder lifecycle (mocked subprocess) ────────────────────────────
class _FakePopen:
    """Mocks subprocess.Popen for the recorder lifecycle:
       - poll() returns None until 'q' is written to stdin
       - then poll()/wait() returns 0
       - the 'output' file is created on stop with placeholder bytes.
    """

    def __init__(self, output_path: Path, output_bytes: bytes = b"FAKEMP4"):
        self.stdin = io.BytesIO()
        self.stderr = io.BytesIO()
        self.returncode: int | None = None
        self._output_path = output_path
        self._output_bytes = output_bytes
        self._stopped = False

    def poll(self):
        return self.returncode

    def wait(self, timeout: float | None = None):
        # Simulate clean shutdown only after 'q' was written.
        if self._stdin_has_q():
            self._finalize()
            return 0
        # Simulate ffmpeg still running
        if not self._stopped:
            raise __import__("subprocess").TimeoutExpired(cmd="ffmpeg", timeout=timeout or 1)
        self._finalize()
        return 0

    def send_signal(self, *_args):
        self._stopped = True

    def kill(self):
        self._stopped = True
        self.returncode = -9

    def _stdin_has_q(self) -> bool:
        try:
            return b"q" in self.stdin.getvalue()
        except Exception:
            return False

    def _finalize(self):
        if self.returncode is None:
            self.returncode = 0
        if self._output_bytes and not self._output_path.exists():
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            self._output_path.write_bytes(self._output_bytes)


def test_recorder_no_ffmpeg_returns_clean_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("mantis_agent.recorder.ffmpeg_available", lambda: False)
    rec = ScreenRecorder(tmp_path / "vid.mp4")
    started = rec.start()
    assert not started
    assert rec.result is not None
    assert rec.result.error == "ffmpeg-not-installed"
    assert not rec.result.succeeded


def test_recorder_happy_path(tmp_path: Path):
    out = tmp_path / "vid.mp4"
    fake = _FakePopen(out, output_bytes=b"x" * 4096)
    with patch("mantis_agent.recorder.ffmpeg_available", return_value=True), \
         patch("mantis_agent.recorder.subprocess.Popen", return_value=fake):
        rec = ScreenRecorder(out, fps=5)
        ok = rec.start()
        assert ok
        time.sleep(0.05)  # let _started_at advance
        result = rec.stop()
    assert result.succeeded
    assert result.output_path == out
    assert result.bytes_written == 4096
    assert result.duration_seconds > 0
    assert result.error is None


def test_recorder_idempotent_stop(tmp_path: Path):
    out = tmp_path / "vid.mp4"
    fake = _FakePopen(out, output_bytes=b"abc")
    with patch("mantis_agent.recorder.ffmpeg_available", return_value=True), \
         patch("mantis_agent.recorder.subprocess.Popen", return_value=fake):
        rec = ScreenRecorder(out)
        rec.start()
        r1 = rec.stop()
        r2 = rec.stop()
    assert r1 is r2  # second stop returns the same RecorderResult


def test_recorder_empty_output_reports_error(tmp_path: Path):
    out = tmp_path / "vid.mp4"
    # Empty bytes -> file never gets bytes_written
    fake = _FakePopen(out, output_bytes=b"")
    with patch("mantis_agent.recorder.ffmpeg_available", return_value=True), \
         patch("mantis_agent.recorder.subprocess.Popen", return_value=fake):
        rec = ScreenRecorder(out)
        rec.start()
        result = rec.stop()
    assert not result.succeeded
    assert result.error == "empty-output"


def test_recorder_startup_crash_captured(tmp_path: Path):
    out = tmp_path / "vid.mp4"
    fake = MagicMock()
    fake.poll.return_value = 1  # exited with code 1 immediately
    fake.stderr = io.BytesIO(b"ffmpeg: cannot open display :99\n")
    fake.stdin = io.BytesIO()
    with patch("mantis_agent.recorder.ffmpeg_available", return_value=True), \
         patch("mantis_agent.recorder.subprocess.Popen", return_value=fake):
        rec = ScreenRecorder(out)
        ok = rec.start()
    assert not ok
    assert rec.result is not None
    assert rec.result.error.startswith("ffmpeg-startup-failed")


def test_recorder_context_manager(tmp_path: Path):
    out = tmp_path / "vid.mp4"
    fake = _FakePopen(out, output_bytes=b"data")
    with patch("mantis_agent.recorder.ffmpeg_available", return_value=True), \
         patch("mantis_agent.recorder.subprocess.Popen", return_value=fake):
        with ScreenRecorder(out) as rec:
            pass
        assert rec.result is not None
        assert rec.result.succeeded


def test_ffmpeg_available_real(monkeypatch):
    # Without monkeypatch, just verifies the helper returns a bool.
    assert isinstance(ffmpeg_available(), bool)
