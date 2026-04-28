"""Screen recorder for the Mantis CUA service.

Captures the Xvfb display (the same one Chrome + xdotool drive) while a run
is in flight, and produces an MP4/WebM/GIF screencast saved to the per-run
data dir. Implementation: ``ffmpeg -f x11grab`` subprocess started before the
run and signaled at run completion.

Why x11grab:
  • The agent loop is already painting the Xvfb display — recording it gives
    a faithful reproduction of what the agent saw.
  • Lower overhead than per-frame Pillow encoding (~5–10% CPU at 5 fps mp4).
  • Output format choice flows directly to the client (mp4 for sharing,
    webm for embedding, gif for Slack/docs).

This module is import-safe even when ffmpeg isn't installed; failures are
caught and surfaced through ``RecorderResult.error`` so the run itself is
never blocked by recording trouble.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger("mantis_agent.recorder")

VideoFormat = Literal["mp4", "webm", "gif"]


@dataclasses.dataclass(frozen=True)
class RecorderResult:
    """Outcome of a recording session."""

    output_path: Optional[Path]
    duration_seconds: float
    bytes_written: int
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.output_path is not None and self.bytes_written > 0


def ffmpeg_available() -> bool:
    """Check whether the ffmpeg binary is on PATH."""
    return shutil.which("ffmpeg") is not None


def _build_ffmpeg_cmd(
    display: str,
    output: Path,
    fps: int,
    fmt: VideoFormat,
    width: int,
    height: int,
) -> list[str]:
    """Build the ffmpeg argv for the chosen format.

    For GIF we transcode through libx264-grabbed frames using a
    palette-based filtergraph (best quality/size tradeoff for a screencast).
    For MP4 / WebM we use a fast preset (CPU-cheap; fine for 1280x720@5fps).
    """
    common = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",                           # overwrite if the file exists
        "-f", "x11grab",
        "-framerate", str(fps),
        "-video_size", f"{width}x{height}",
        "-i", display,
    ]
    if fmt == "mp4":
        return [
            *common,
            "-vcodec", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",       # broad player compatibility
            str(output),
        ]
    if fmt == "webm":
        return [
            *common,
            "-vcodec", "libvpx-vp9",
            "-row-mt", "1",
            "-cpu-used", "5",            # fast encode
            "-crf", "32",
            "-b:v", "0",
            str(output),
        ]
    if fmt == "gif":
        # palettegen + paletteuse for a sharp gif at moderate size
        vf = (
            f"fps={fps},scale={width}:-1:flags=lanczos,"
            "split[a][b];[a]palettegen[p];[b][p]paletteuse"
        )
        return [
            *common,
            "-vf", vf,
            "-loop", "0",
            str(output),
        ]
    raise ValueError(f"unsupported video format: {fmt!r}")


class ScreenRecorder:
    """Spawn ffmpeg in the background to record the Xvfb display.

    Use as a context manager so cleanup is guaranteed:

        with ScreenRecorder(output=Path(...)) as rec:
            ...  # run the agent loop; ffmpeg captures concurrently
        # rec.result is populated on exit
    """

    DEFAULT_FPS = 5
    DEFAULT_WIDTH = 1280
    DEFAULT_HEIGHT = 720
    SHUTDOWN_TIMEOUT_S = 15

    def __init__(
        self,
        output: Path,
        *,
        display: Optional[str] = None,
        fps: int = DEFAULT_FPS,
        fmt: VideoFormat = "mp4",
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> None:
        self._output = Path(output)
        self._display = display or os.environ.get("DISPLAY", ":99")
        self._fps = max(1, min(fps, 30))
        self._fmt = fmt
        self._width = width
        self._height = height
        self._proc: Optional[subprocess.Popen] = None
        self._started_at: float = 0.0
        self._stop_lock = threading.Lock()
        self._stopped = False
        self.result: Optional[RecorderResult] = None

    # ── Lifecycle ───────────────────────────────────────────────────────
    def start(self) -> bool:
        """Spawn the ffmpeg subprocess. Returns False if recording can't start
        (binary missing, display not exported, etc.); the caller should
        proceed without recording rather than fail the run."""
        if not ffmpeg_available():
            logger.warning("ffmpeg not on PATH; screen recording disabled")
            self.result = RecorderResult(
                output_path=None, duration_seconds=0.0, bytes_written=0,
                error="ffmpeg-not-installed",
            )
            return False
        self._output.parent.mkdir(parents=True, exist_ok=True)
        cmd = _build_ffmpeg_cmd(
            display=self._display,
            output=self._output,
            fps=self._fps,
            fmt=self._fmt,
            width=self._width,
            height=self._height,
        )
        logger.info("recorder starting display=%s -> %s", self._display, self._output)
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,            # so we can send 'q' to stop cleanly
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            logger.warning("failed to spawn ffmpeg: %s", exc)
            self.result = RecorderResult(
                output_path=None, duration_seconds=0.0, bytes_written=0,
                error=f"spawn-failed:{exc}",
            )
            return False
        # Give ffmpeg a moment to attach to the display; if it died immediately
        # capture the stderr for diagnostics.
        time.sleep(0.3)
        if self._proc.poll() is not None:
            stderr = (self._proc.stderr.read() or b"").decode(errors="replace") if self._proc.stderr else ""
            logger.warning("ffmpeg died on startup: %s", stderr[:500])
            self.result = RecorderResult(
                output_path=None, duration_seconds=0.0, bytes_written=0,
                error=f"ffmpeg-startup-failed:{stderr[:200]}",
            )
            self._proc = None
            return False
        self._started_at = time.time()
        return True

    def stop(self) -> RecorderResult:
        """Tell ffmpeg to flush + finalize the file, wait for it to exit,
        and populate ``self.result``."""
        with self._stop_lock:
            if self._stopped:
                assert self.result is not None
                return self.result
            self._stopped = True

        proc = self._proc
        if proc is None:
            assert self.result is not None
            return self.result

        duration = time.time() - self._started_at
        try:
            # 'q' on stdin → ffmpeg cleanly closes the file
            if proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.write(b"q")
                    proc.stdin.flush()
                except (OSError, BrokenPipeError):
                    pass
            try:
                proc.wait(timeout=self.SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                logger.warning("ffmpeg did not exit cleanly; sending SIGTERM")
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error("ffmpeg ignored SIGTERM; sending SIGKILL")
                    proc.kill()
                    proc.wait(timeout=2)
        except Exception as exc:  # pragma: no cover
            logger.exception("recorder stop failed: %s", exc)

        bytes_written = 0
        if self._output.exists():
            try:
                bytes_written = self._output.stat().st_size
            except OSError:
                bytes_written = 0

        error: Optional[str] = None
        if bytes_written == 0:
            error = "empty-output"
        elif proc.returncode and proc.returncode != 0:
            # ffmpeg returns non-zero when it gets 'q'; only report a problem
            # if the file is empty.
            pass

        self.result = RecorderResult(
            output_path=self._output if bytes_written > 0 else None,
            duration_seconds=duration,
            bytes_written=bytes_written,
            error=error,
        )
        logger.info(
            "recorder stopped duration=%.1fs bytes=%d output=%s err=%s",
            duration, bytes_written,
            self._output if bytes_written > 0 else "(no file)",
            error,
        )
        return self.result

    # ── Context-manager sugar ───────────────────────────────────────────
    def __enter__(self) -> "ScreenRecorder":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def content_type_for(fmt: VideoFormat) -> str:
    return {
        "mp4": "video/mp4",
        "webm": "video/webm",
        "gif": "image/gif",
    }[fmt]
