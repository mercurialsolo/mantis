"""E2B sandbox backend for the computer plane (#699 Phase 2).

E2B (https://e2b.dev/) provides per-call container sandboxes with a
public HTTPS URL. This impl provisions a sandbox on construction,
waits for the computer-plane HTTP server inside it to come up, then
delegates every wire-contract call to :class:`RemoteComputerImpl`
talking to the sandbox's URL.

Operator preconditions
======================

The sandbox image must:

1. Pre-install ``xvfb``, ``xdotool``, ``google-chrome-stable``, and
   the mantis_agent package.
2. Boot ``uvicorn mantis_agent.server.computer_agent:app`` on a
   known port (default 8000) at container start.
3. Expose the chosen port via E2B's port-forwarding so the brain can
   reach it over HTTPS.

The image SHA is configured via ``MANTIS_E2B_TEMPLATE`` (default:
``mantis-computer-plane-v1``); the brain's image-build pipeline
publishes the same SHA to both E2B and Daytona.

Construction failure modes
==========================

* **SDK missing.** The ``e2b`` package isn't installed → raises
  ``ImportError`` at construction. Operator-actionable; we don't
  silently fall back to local because the operator explicitly asked
  for the E2B backend.
* **Template not found.** E2B rejects the template SHA → raises
  ``RuntimeError`` carrying the E2B-side error.
* **Port not ready.** Server doesn't answer ``/health`` within
  ``startup_timeout_seconds`` → raises ``TimeoutError`` AND tears
  down the partially-provisioned sandbox to avoid leaking quota.

Per-run lifecycle
=================

* ``__init__`` — provisions the sandbox, waits for ``/health``, hands
  off to RemoteComputerImpl with the sandbox's tunnel URL.
* ``close()`` — closes the upstream session, then terminates the
  sandbox so the operator isn't billed for idle compute.
* ``shutdown()`` — alias for close() that matches the Phase 1
  computer_session contract.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from .remote_computer_impl import RemoteComputerImpl

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8000
_DEFAULT_STARTUP_TIMEOUT = 60.0
_DEFAULT_TEMPLATE = "mantis-computer-plane-v1"


class E2BComputerImpl(RemoteComputerImpl):
    """E2B-sandbox-backed computer plane.

    Provisions the sandbox at construction; tears it down at
    ``close()``. The rest of the wire contract delegates to
    :class:`RemoteComputerImpl`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        template: str = "",
        startup_timeout_seconds: float = _DEFAULT_STARTUP_TIMEOUT,
        port: int = _DEFAULT_PORT,
        # Optional injection for tests — wires in a fake SDK module so
        # we don't have to mock at import-time.
        sdk_module: Any = None,
        **remote_kwargs: Any,
    ) -> None:
        self._sandbox = None  # set on successful boot
        sdk = sdk_module or self._import_sdk()
        api_key = api_key or os.environ.get("E2B_API_KEY") or ""
        if not api_key:
            raise ValueError(
                "E2BComputerImpl requires an E2B API key — pass "
                "api_key=..., or set E2B_API_KEY in the env"
            )
        template = (
            template
            or os.environ.get("MANTIS_E2B_TEMPLATE")
            or _DEFAULT_TEMPLATE
        )
        logger.warning(
            "E2BComputerImpl: provisioning sandbox template=%s port=%d",
            template, port,
        )
        try:
            self._sandbox = sdk.Sandbox(template=template, api_key=api_key)
        except Exception as exc:
            raise RuntimeError(
                f"E2BComputerImpl: sandbox provisioning failed: {exc}"
            ) from exc

        try:
            base_url = self._resolve_base_url(self._sandbox, port)
            self._await_ready(base_url, startup_timeout_seconds)
        except Exception:
            # Tear down partial state so we don't leak quota.
            self._teardown_quietly()
            raise

        logger.warning(
            "E2BComputerImpl: sandbox ready base_url=%s", base_url,
        )
        super().__init__(base_url=base_url, **remote_kwargs)

    # ── lifecycle ──

    def close(self) -> None:
        """Close the upstream session + tear down the sandbox."""
        try:
            super().close()
        finally:
            self._teardown_quietly()

    def shutdown(self) -> None:
        """Parity with the Phase 1 computer_session contract."""
        self.close()

    # ── helpers ──

    @staticmethod
    def _import_sdk() -> Any:
        try:
            import e2b  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "E2BComputerImpl requires the e2b package — "
                "`pip install mantis-agent[e2b]` (or `pip install e2b`)"
            ) from exc
        return e2b

    @staticmethod
    def _resolve_base_url(sandbox: Any, port: int) -> str:
        """Resolve the public tunnel URL for ``port`` on ``sandbox``.

        E2B's SDK exposes ``Sandbox.get_host(port)`` returning the
        public hostname; we wrap with ``https://``. Tests inject a
        fake sandbox with the same method.
        """
        host = sandbox.get_host(port)
        if not host:
            raise RuntimeError(
                f"E2BComputerImpl: sandbox returned no host for port {port}"
            )
        return f"https://{host}"

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
            f"E2BComputerImpl: sandbox /health not ready within "
            f"{deadline_seconds:.0f}s (last_exc={last_exc!r})"
        )

    def _teardown_quietly(self) -> None:
        if self._sandbox is None:
            return
        try:
            close = getattr(self._sandbox, "close", None) or getattr(
                self._sandbox, "kill", None,
            )
            if close is not None:
                close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("E2BComputerImpl: teardown raised: %s", exc)
        finally:
            self._sandbox = None


__all__ = ["E2BComputerImpl"]
