"""Morph (https://morph.so) microVM backend for the computer plane.

Morph provisions snapshot-booted microVMs with a public HTTPS service URL.
Its defining feature is **instant snapshots**: boot a VM once (Xvfb + Chrome +
the screenshot/xdotool service), snapshot it, then `instances.start(snapshot_id)`
spins a fresh copy in seconds. That makes it a strong fit for the CUA runner —
and for a **pre-warm pool** (see :mod:`morph_prewarm_pool`) that keeps N booted
copies ready so a run claims one with ~0 cold-start.

This impl mirrors :class:`E2BComputerImpl` exactly: it provisions on
construction, delegates the screenshot+xdotool wire contract to
:class:`RemoteComputerImpl`, and tears the instance down at ``close()``. The
Morph-SDK-specific calls (start / expose-service / stop) are isolated in small
``getattr``-guarded helpers and the SDK is injectable (``sdk_module=``) so tests
never hit the real API. When claimed from a pre-warm pool, pass the already-booted
``instance`` + ``base_url`` to skip provisioning entirely.

The snapshot must boot a service on ``port`` that exposes ``GET /health`` (200
when ready) plus the screenshot/xdotool RPC surface RemoteComputerImpl speaks —
the same image contract E2B/Daytona use. Set ``MANTIS_MORPH_SNAPSHOT`` to the
snapshot id and ``MORPH_API_KEY`` to the API key.

NOTE: the exact morphcloud method names (``instances.start``,
``expose_http_service``, ``instance.stop``) follow Morph's documented SDK; the
helpers below ``getattr``-probe alternates, but verify against the installed
``morphcloud`` version before the first live deploy.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from .remote_computer_impl import RemoteComputerImpl

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8000
_DEFAULT_STARTUP_TIMEOUT = 120.0
_DEFAULT_SERVICE_NAME = "cua"


class MorphComputerImpl(RemoteComputerImpl):
    """Morph-microVM-backed computer plane.

    Two construction modes:

    * **Provision** (default) — boot a fresh instance from ``snapshot_id``,
      expose its HTTP service, await ``/health``, then serve.
    * **Claim** — pass a pre-booted ``instance`` + ``base_url`` (from a
      :class:`morph_prewarm_pool.MorphPrewarmPool`) to skip provisioning. By
      default a claimed instance is NOT torn down on ``close()`` (the pool owns
      its lifecycle); pass ``owns_instance=True`` to override.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        snapshot_id: str = "",
        startup_timeout_seconds: float = _DEFAULT_STARTUP_TIMEOUT,
        port: int = _DEFAULT_PORT,
        service_name: str = _DEFAULT_SERVICE_NAME,
        sdk_module: Any = None,
        # Claim mode: a pre-booted instance + its resolved base_url.
        instance: Any = None,
        base_url: str | None = None,
        owns_instance: bool | None = None,
        **remote_kwargs: Any,
    ) -> None:
        self._instance = instance
        # Provisioned instances are owned (torn down on close); claimed ones
        # are not, unless the caller explicitly asks.
        self._owns_instance = (
            owns_instance if owns_instance is not None else (instance is None)
        )

        if base_url is None:
            sdk = sdk_module or self._import_sdk()
            api_key = api_key or os.environ.get("MORPH_API_KEY") or ""
            if not api_key:
                raise ValueError(
                    "MorphComputerImpl requires a Morph API key — pass "
                    "api_key=..., or set MORPH_API_KEY in the env"
                )
            snapshot_id = (
                snapshot_id or os.environ.get("MANTIS_MORPH_SNAPSHOT") or ""
            )
            if not snapshot_id:
                raise ValueError(
                    "MorphComputerImpl requires a snapshot_id — pass "
                    "snapshot_id=..., or set MANTIS_MORPH_SNAPSHOT in the env"
                )
            logger.warning(
                "MorphComputerImpl: starting instance snapshot=%s port=%d",
                snapshot_id, port,
            )
            client = self._make_client(sdk, api_key)
            try:
                self._instance = client.instances.start(snapshot_id=snapshot_id)
            except Exception as exc:
                raise RuntimeError(
                    f"MorphComputerImpl: instance start failed: {exc}"
                ) from exc
            try:
                base_url = self._resolve_base_url(
                    self._instance, service_name, port,
                )
                self._await_ready(base_url, startup_timeout_seconds)
            except Exception:
                self._teardown_quietly()
                raise

        logger.warning("MorphComputerImpl: ready base_url=%s", base_url)
        super().__init__(base_url=base_url, **remote_kwargs)

    # ── lifecycle ──

    def close(self) -> None:
        """Close the upstream session; tear down the instance iff we own it."""
        try:
            super().close()
        finally:
            if self._owns_instance:
                self._teardown_quietly()

    def shutdown(self) -> None:
        """Parity with the Phase 1 computer_session contract."""
        self.close()

    # ── Morph-SDK-specific helpers (isolated + getattr-guarded) ──

    @staticmethod
    def _import_sdk() -> Any:
        try:
            import morphcloud  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "MorphComputerImpl requires the morphcloud package — "
                "`pip install morphcloud`"
            ) from exc
        return morphcloud

    @staticmethod
    def _make_client(sdk: Any, api_key: str) -> Any:
        """Build a MorphCloudClient from the SDK module (or a passed client).

        Accepts either the ``morphcloud`` module (resolves
        ``morphcloud.api.MorphCloudClient`` or ``morphcloud.MorphCloudClient``),
        a class, or an already-built client (tests inject the latter two).
        """
        # Already a client instance with an ``instances`` attribute? use it.
        if hasattr(sdk, "instances"):
            return sdk
        client_cls = (
            getattr(getattr(sdk, "api", None), "MorphCloudClient", None)
            or getattr(sdk, "MorphCloudClient", None)
        )
        if client_cls is None:
            raise RuntimeError(
                "MorphComputerImpl: could not locate MorphCloudClient on the "
                f"morphcloud SDK ({sdk!r})"
            )
        return client_cls(api_key=api_key)

    @staticmethod
    def _resolve_base_url(instance: Any, service_name: str, port: int) -> str:
        """Resolve the public HTTPS URL for the instance's CUA service.

        Prefers ``instance.expose_http_service(name, port)`` (Morph returns the
        URL); falls back to scanning already-exposed services on the instance's
        networking block.
        """
        expose = getattr(instance, "expose_http_service", None)
        if callable(expose):
            try:
                url = expose(name=service_name, port=port)
            except TypeError:
                url = expose(service_name, port)
            if isinstance(url, str) and url:
                return url
            # Some SDK versions mutate the instance and return None.
        # Fallback: read exposed http services off the instance.
        url = MorphComputerImpl._scan_exposed_url(instance, service_name, port)
        if url:
            return url
        raise RuntimeError(
            "MorphComputerImpl: could not resolve an HTTP service URL for "
            f"service={service_name!r} port={port}"
        )

    @staticmethod
    def _scan_exposed_url(instance: Any, service_name: str, port: int) -> str:
        networking = getattr(instance, "networking", None)
        services = getattr(networking, "http_services", None) or getattr(
            instance, "http_services", None,
        )
        for svc in services or []:
            svc_name = getattr(svc, "name", None) or (
                svc.get("name") if isinstance(svc, dict) else None
            )
            svc_port = getattr(svc, "port", None) or (
                svc.get("port") if isinstance(svc, dict) else None
            )
            svc_url = getattr(svc, "url", None) or (
                svc.get("url") if isinstance(svc, dict) else None
            )
            if svc_url and (svc_name == service_name or svc_port == port):
                return str(svc_url)
        return ""

    def _await_ready(self, base_url: str, deadline_seconds: float) -> None:
        """Poll ``GET {base_url}/health`` until 200 or deadline."""
        import requests

        deadline = time.monotonic() + deadline_seconds
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
            f"MorphComputerImpl: instance /health not ready within "
            f"{deadline_seconds}s (last error: {last_exc})"
        )

    def _teardown_quietly(self) -> None:
        if self._instance is None:
            return
        try:
            stop = (
                getattr(self._instance, "stop", None)
                or getattr(self._instance, "shutdown", None)
                or getattr(self._instance, "close", None)
            )
            if stop is not None:
                stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MorphComputerImpl: teardown raised: %s", exc)
        finally:
            self._instance = None


__all__ = ["MorphComputerImpl"]
