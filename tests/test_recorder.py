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
    ScreenRecorder,
    _build_ffmpeg_cmd,
    content_type_for,
    ffmpeg_available,
    x_display_alive,
)


# The existing happy-path / lifecycle tests in this file pre-date the
# X-display probe ``ScreenRecorder.start`` now does before spawning
# ffmpeg. On Modal images (and most linux dev boxes) ``xdpyinfo`` IS on
# PATH and the probe would correctly report "no display" — which would
# fail the lifecycle tests for reasons unrelated to what they pin. Stub
# the probe to "alive" by default so those tests keep covering ffmpeg
# lifecycle; the new probe-failure path is exercised explicitly below
# (the inner ``with patch(...)`` overrides this fixture for the
# probe-failure case).
@pytest.fixture(autouse=True)
def _stub_x_display_alive():
    with patch("mantis_agent.recorder.x_display_alive", return_value=True):
        yield


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


def test_recorder_no_ffmpeg_returns_clean_failure(tmp_path: Path):
    """When ffmpeg_available() returns False, ScreenRecorder.start() must
    short-circuit with a clean error rather than attempting to spawn ffmpeg.

    Uses ``with patch(...)`` (matching every other test in this file)
    rather than the monkeypatch fixture, because monkeypatch interacts
    poorly with later tests that ``import ScreenRecorder`` at module load
    time and snapshot the ``ffmpeg_available`` reference inside the closure.
    """
    with patch("mantis_agent.recorder.ffmpeg_available", return_value=False):
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


# ── X-display probe (the new layer for the integrator error) ────────────────


def test_x_display_alive_returns_true_when_xdpyinfo_missing():
    """On hosts without ``xdpyinfo`` (most CI runners), the probe must
    return True so we don't false-fail recorder.start()."""
    with patch("mantis_agent.recorder.shutil.which", return_value=None):
        assert x_display_alive(":99") is True


def test_x_display_alive_returns_true_when_probe_succeeds(tmp_path: Path):
    """``xdpyinfo`` exits 0 → display is alive."""
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    with patch("mantis_agent.recorder.shutil.which", return_value="/usr/bin/xdpyinfo"), \
         patch("mantis_agent.recorder.subprocess.run", return_value=fake_proc):
        assert x_display_alive(":99") is True


def test_x_display_alive_returns_false_when_probe_fails():
    """``xdpyinfo`` exits non-zero → display is not alive."""
    fake_proc = MagicMock()
    fake_proc.returncode = 1
    with patch("mantis_agent.recorder.shutil.which", return_value="/usr/bin/xdpyinfo"), \
         patch("mantis_agent.recorder.subprocess.run", return_value=fake_proc):
        assert x_display_alive(":99") is False


def test_x_display_alive_returns_false_on_exception():
    """``xdpyinfo`` raises (timeout, missing display, …) → False."""
    with patch("mantis_agent.recorder.shutil.which", return_value="/usr/bin/xdpyinfo"), \
         patch(
             "mantis_agent.recorder.subprocess.run",
             side_effect=__import__("subprocess").TimeoutExpired("xdpyinfo", 3),
         ):
        assert x_display_alive(":99") is False


def test_recorder_refuses_to_spawn_when_display_not_ready(tmp_path: Path):
    """When the X display isn't alive, ``start()`` must fail with a
    precise ``x-display-not-ready:<display>`` envelope BEFORE
    spawning ffmpeg. This is the actionable replacement for the
    cryptic ``Cannot open display :99, error 1`` integrators saw.

    Patches the module-level ``x_display_alive`` to False (overriding
    the autouse fixture). The subprocess.Popen patch is asserted
    *not* to be called — proving we short-circuit before ffmpeg fires.
    """
    out = tmp_path / "vid.mp4"
    popen_mock = MagicMock()
    with patch("mantis_agent.recorder.ffmpeg_available", return_value=True), \
         patch("mantis_agent.recorder.x_display_alive", return_value=False), \
         patch("mantis_agent.recorder.subprocess.Popen", popen_mock):
        rec = ScreenRecorder(out, display=":99")
        ok = rec.start()

    assert ok is False
    assert rec.result is not None
    assert rec.result.error == "x-display-not-ready::99"
    assert not rec.result.succeeded
    # Most important assertion: ffmpeg never got spawned.
    popen_mock.assert_not_called()
