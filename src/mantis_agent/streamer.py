"""ScreenStreamer — continuous frame capture as a rolling video buffer.

Instead of discrete screenshots, this captures the screen continuously so the
model always has fresh temporal context. It sees what changed between frames,
which is critical for understanding animations, loading states, and the
consequences of its own actions.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass

import mss
from PIL import Image


@dataclass
class Frame:
    """A single captured frame with metadata."""

    image: Image.Image
    timestamp: float
    index: int


class ScreenStreamer:
    """Continuously captures the screen into a rolling frame buffer.

    The buffer acts as a short-term visual memory — the model receives the
    last N frames as temporal context, letting it observe how the screen
    changed in response to its actions.

    Args:
        fps: Capture rate. 2-5 FPS is the sweet spot — fast enough to catch
             transitions, slow enough to not waste compute.
        buffer_size: Number of frames to retain. At 3 FPS with buffer=15,
                     the model gets ~5 seconds of visual history.
        monitor: Which monitor to capture (0 = all, 1 = primary, etc.)
        scale: Downscale factor for captured frames. 0.5 = half resolution.
               Lower resolution = faster inference + lower token cost.
    """

    def __init__(
        self,
        fps: float = 3.0,
        buffer_size: int = 15,
        monitor: int = 1,
        scale: float = 1.0,
    ):
        self.fps = fps
        self.buffer_size = buffer_size
        self.monitor = monitor
        self.scale = scale
        self._buffer: deque[Frame] = deque(maxlen=buffer_size)
        self._frame_count = 0
        self._running = False
        self._capture_task: asyncio.Task | None = None
        self._screen_size: tuple[int, int] = (0, 0)

    @property
    def frames(self) -> list[Frame]:
        """Get all frames currently in the buffer, oldest first."""
        return list(self._buffer)

    @property
    def latest(self) -> Frame | None:
        """Get the most recent frame."""
        return self._buffer[-1] if self._buffer else None

    @property
    def screen_size(self) -> tuple[int, int]:
        """Screen dimensions (width, height) in actual pixels."""
        return self._screen_size

    def capture_once(self) -> Frame:
        """Capture a single frame synchronously. Useful for one-shot queries."""
        with mss.mss() as sct:
            mon = sct.monitors[self.monitor]
            self._screen_size = (mon["width"], mon["height"])
            raw = sct.grab(mon)
            image = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

        if self.scale != 1.0:
            new_size = (int(image.width * self.scale), int(image.height * self.scale))
            image = image.resize(new_size, Image.LANCZOS)

        frame = Frame(image=image, timestamp=time.time(), index=self._frame_count)
        self._frame_count += 1
        self._buffer.append(frame)
        return frame

    async def start(self) -> None:
        """Start continuous background capture."""
        if self._running:
            return
        self._running = True
        self._capture_task = asyncio.create_task(self._capture_loop())

    async def stop(self) -> None:
        """Stop background capture."""
        self._running = False
        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
            self._capture_task = None

    async def _capture_loop(self) -> None:
        """Main capture loop — runs in background, fills the frame buffer."""
        interval = 1.0 / self.fps
        while self._running:
            t0 = time.monotonic()
            # Run mss capture in executor to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.capture_once)
            elapsed = time.monotonic() - t0
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def get_recent_frames(self, count: int | None = None) -> list[Image.Image]:
        """Get the N most recent frames as PIL Images.

        This is what gets fed to Gemma4 as the visual context — a sequence
        of images representing recent screen state.

        Args:
            count: Number of frames to return. None = all buffered frames.
        """
        frames = self.frames
        if count is not None:
            frames = frames[-count:]
        return [f.image for f in frames]

    def get_frame_timestamps(self, count: int | None = None) -> list[float]:
        """Get timestamps for recent frames (useful for temporal reasoning)."""
        frames = self.frames
        if count is not None:
            frames = frames[-count:]
        return [f.timestamp for f in frames]

    def clear(self) -> None:
        """Clear the frame buffer."""
        self._buffer.clear()
        self._frame_count = 0
