"""`LocalXdotoolImpl` — Phase 0 `ComputerClient` backed by the in-process
`XdotoolGymEnv`.

Subclass-only — adds latency instrumentation around `_screenshot()` and
`_xdotool()` so we can read p50/p95/p99 baselines before the Phase 1
HTTPS hop is added. Everything else is inherited unchanged.

The baseline numbers feed Phase 1's go/no-go gate (#697 acceptance
criteria #4): abort the Phase 1 migration if intra-Modal HTTPS RT pushes
p50 above the captured local p50 + 20 ms.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from PIL import Image

from .computer_client import ComputerClient
from .xdotool_env import XdotoolGymEnv


class LatencyTracker:
    """Bounded-capacity ring buffer with on-demand percentile summary.

    Threadsafe because xdotool / screenshot calls can come from the
    runner thread and the recorder thread concurrently.
    """

    __slots__ = ("name", "_samples_ms", "_lock")

    def __init__(self, name: str, capacity: int = 1024):
        self.name = name
        self._samples_ms: deque[float] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record_ms(self, dt_ms: float) -> None:
        with self._lock:
            self._samples_ms.append(dt_ms)

    def summary(self) -> dict[str, float | int]:
        with self._lock:
            data = sorted(self._samples_ms)
        n = len(data)
        if n == 0:
            return {"name": self.name, "count": 0}

        def _pct(p: float) -> float:
            idx = max(0, min(n - 1, int(n * p)))
            return round(data[idx], 3)

        return {
            "name": self.name,
            "count": n,
            "p50_ms": _pct(0.50),
            "p95_ms": _pct(0.95),
            "p99_ms": _pct(0.99),
            "mean_ms": round(sum(data) / n, 3),
            "max_ms": round(data[-1], 3),
        }


class LocalXdotoolImpl(XdotoolGymEnv, ComputerClient):
    """In-process `ComputerClient` backed by `XdotoolGymEnv`.

    Wraps `_screenshot()` and `_xdotool()` with `time.perf_counter()` so
    we can read p50/p95/p99 distributions for Phase 1's go/no-go gate.

    Inherits the full `XdotoolGymEnv` surface (reset/step/close, CDP
    helpers, `current_url`, `capture_browser_state`, ...). Multiple
    inheritance gives us `isinstance(env, ComputerClient)` for free.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.screenshot_latency = LatencyTracker("screenshot")
        self.xdotool_latency = LatencyTracker("xdotool")

    def _screenshot(self) -> Image.Image:
        t0 = time.perf_counter()
        try:
            return super()._screenshot()
        finally:
            self.screenshot_latency.record_ms((time.perf_counter() - t0) * 1000.0)

    def _xdotool(self, *args: str) -> None:
        t0 = time.perf_counter()
        try:
            super()._xdotool(*args)
        finally:
            self.xdotool_latency.record_ms((time.perf_counter() - t0) * 1000.0)

    def latency_report(self) -> dict[str, dict[str, float | int]]:
        """Return {screenshot, xdotool} latency distributions.

        Call once at end-of-run to log a Phase 0 baseline; Phase 1's
        `RemoteComputerImpl` will emit the same shape so the comparison
        is mechanical.
        """
        return {
            "screenshot": self.screenshot_latency.summary(),
            "xdotool": self.xdotool_latency.summary(),
        }
