"""Pre-warm pool of booted Morph microVMs — kill cold-start latency.

A run that provisions a Morph instance on demand pays the full boot cost
(snapshot start + Chrome up + ``/health``). This pool keeps ``size`` instances
**already booted and health-checked**, so :meth:`acquire` returns one instantly
on the hot path; it then refills in the background. This is the legitimate
latency/reliability win that motivated the Morph backend — it has nothing to do
with, and does nothing for, bot-detection.

Lifecycle: an operator builds one long-lived pool (e.g. at server startup),
calls :meth:`top_up` to fill it, and hands :meth:`acquire`'d instances to
:class:`morph_impl.MorphComputerImpl` in *claim* mode. Each instance is
single-use — a CUA run mutates VM state — so :meth:`release` permanently stops
it and the pool boots a replacement.

The Morph SDK is reused from :class:`MorphComputerImpl` (same isolated,
injectable helpers), so tests drive the whole pool against a fake SDK with no
real API calls. ``refill_async=False`` makes refills synchronous for
deterministic tests.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .morph_impl import MorphComputerImpl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WarmInstance:
    """A booted, health-checked Morph instance ready to claim."""

    instance: Any
    base_url: str


class MorphPrewarmPool:
    """Maintains ``size`` ready Morph instances; ``acquire`` pops one instantly."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        snapshot_id: str = "",
        size: int = 2,
        port: int = 8000,
        service_name: str = "cua",
        startup_timeout_seconds: float = 120.0,
        sdk_module: Any = None,
        refill_async: bool = True,
    ) -> None:
        self._size = max(0, int(size))
        self._port = port
        self._service_name = service_name
        self._startup_timeout = startup_timeout_seconds
        self._refill_async = refill_async

        self._lock = threading.Lock()
        self._ready: deque[WarmInstance] = deque()
        self._closed = False

        api_key = api_key or os.environ.get("MORPH_API_KEY") or ""
        if not api_key:
            raise ValueError(
                "MorphPrewarmPool requires a Morph API key — pass api_key=..., "
                "or set MORPH_API_KEY in the env"
            )
        snapshot_id = snapshot_id or os.environ.get("MANTIS_MORPH_SNAPSHOT") or ""
        if not snapshot_id:
            raise ValueError(
                "MorphPrewarmPool requires a snapshot_id — pass snapshot_id=..., "
                "or set MANTIS_MORPH_SNAPSHOT in the env"
            )
        self._snapshot_id = snapshot_id
        sdk = sdk_module or MorphComputerImpl._import_sdk()
        # Reuse the impl's client resolver so module/class/client all work.
        self._client = MorphComputerImpl._make_client(sdk, api_key)

    # ── public surface ──

    @property
    def ready_count(self) -> int:
        with self._lock:
            return len(self._ready)

    def top_up(self) -> int:
        """Boot ready instances until the pool reaches ``size``.

        Returns the number booted this call. Boots happen OUTSIDE the lock (slow
        network) and are discarded if the pool filled or closed in the meantime.
        """
        booted = 0
        while True:
            with self._lock:
                if self._closed or len(self._ready) >= self._size:
                    return booted
            warm = self._boot_one()
            with self._lock:
                if self._closed or len(self._ready) >= self._size:
                    self._stop_quietly(warm.instance)
                    return booted
                self._ready.append(warm)
                booted += 1

    def acquire(self) -> WarmInstance:
        """Claim a ready instance (hot path) or boot one (cold miss).

        Triggers a background refill when a pre-warmed instance was taken so the
        pool trends back to ``size``.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("MorphPrewarmPool is closed")
            warm = self._ready.popleft() if self._ready else None
        if warm is None:
            # Cold miss — boot synchronously so the caller still gets an instance.
            logger.warning("MorphPrewarmPool: cold miss — booting on demand")
            return self._boot_one()
        self._schedule_refill()
        return warm

    def release(self, instance: Any) -> None:
        """Permanently stop a claimed instance (single-use) + schedule a refill."""
        self._stop_quietly(instance)
        self._schedule_refill()

    def drain(self) -> None:
        """Close the pool and stop every ready instance."""
        with self._lock:
            self._closed = True
            pending = list(self._ready)
            self._ready.clear()
        for warm in pending:
            self._stop_quietly(warm.instance)

    # ── internals ──

    def _boot_one(self) -> WarmInstance:
        instance = self._client.instances.start(snapshot_id=self._snapshot_id)
        try:
            base_url = MorphComputerImpl._resolve_base_url(
                instance, self._service_name, self._port,
            )
            self._await_ready(base_url)
        except Exception:
            self._stop_quietly(instance)
            raise
        return WarmInstance(instance=instance, base_url=base_url)

    def _await_ready(self, base_url: str) -> None:
        import requests

        deadline = time.monotonic() + self._startup_timeout
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            try:
                resp = requests.get(f"{base_url}/health", timeout=5)
                if resp.status_code == 200:
                    return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
            time.sleep(1.0)
        raise TimeoutError(
            f"MorphPrewarmPool: instance /health not ready within "
            f"{self._startup_timeout}s (last error: {last_exc})"
        )

    def _schedule_refill(self) -> None:
        if self._closed:
            return
        if self._refill_async:
            threading.Thread(target=self._safe_top_up, daemon=True).start()
        # refill_async=False: caller drives top_up() (deterministic tests).

    def _safe_top_up(self) -> None:
        try:
            self.top_up()
        except Exception as exc:  # noqa: BLE001 — background refill, never fatal
            logger.warning("MorphPrewarmPool: background top_up raised: %s", exc)

    @staticmethod
    def _stop_quietly(instance: Any) -> None:
        if instance is None:
            return
        try:
            stop = (
                getattr(instance, "stop", None)
                or getattr(instance, "shutdown", None)
                or getattr(instance, "close", None)
            )
            if stop is not None:
                stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MorphPrewarmPool: stop raised: %s", exc)


__all__ = ["MorphPrewarmPool", "WarmInstance"]
