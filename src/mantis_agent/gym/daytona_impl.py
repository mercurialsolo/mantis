"""Daytona sandbox backend for the computer plane (#699 Phase 2).

Daytona (https://daytona.io/) provides on-demand workspace sandboxes
with a public preview URL. Same shape as :class:`E2BComputerImpl`:
provision sandbox → wait for ``/health`` → delegate the wire
contract to :class:`RemoteComputerImpl`.

The operator-side image preconditions are the same as E2B's
(``xvfb`` + ``xdotool`` + Chrome + the ``mantis_agent`` package + a
boot-time ``uvicorn`` for ``computer_agent:app``). See
``deploy/sim_envs/`` for an existing Daytona-based image pattern;
the computer-plane image extends the same recipe with the wire
server.

Configuration
=============

* ``api_key`` — passed in, or ``DAYTONA_API_KEY`` env var. Required.
* ``server_url`` — Daytona control-plane URL; defaults to
  ``https://app.daytona.io`` matching the public SaaS.
* ``snapshot`` — Daytona snapshot id (their term for a pre-baked
  image) for the computer-plane container. Defaults to
  ``MANTIS_DAYTONA_SNAPSHOT`` env var.
* ``port`` — port the computer-plane HTTP server listens on inside
  the sandbox. Defaults to 8000.
* ``startup_timeout_seconds`` — how long to wait for ``/health``
  after sandbox boot. Daytona cold-builds can take 60s+, so default
  is 120s.

The Daytona preview URL has an interstitial that needs to be bypassed
via the ``X-Daytona-Skip-Preview-Warning: true`` header
(``feedback_daytona_preview_warning_bypass.md``). This impl injects
that header on every wire call via the inherited
``extra_http_headers`` kwarg.
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
_DAYTONA_SKIP_PREVIEW_HEADER = "X-Daytona-Skip-Preview-Warning"
# Preview URLs on daytonaproxy01.net are auth0-gated. The skip-warning
# header bypasses the cookie consent interstitial; the preview token
# header is the real auth bypass — every wire call needs it.
_DAYTONA_PREVIEW_TOKEN_HEADER = "X-Daytona-Preview-Token"


class DaytonaComputerImpl(RemoteComputerImpl):
    """Daytona-sandbox-backed computer plane.

    Provisions the sandbox at construction; tears it down at
    ``close()``. The rest of the wire contract delegates to
    :class:`RemoteComputerImpl` over the sandbox's preview URL.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        server_url: str = "",
        snapshot: str = "",
        sandbox_id: str = "",
        startup_timeout_seconds: float = _DEFAULT_STARTUP_TIMEOUT,
        port: int = _DEFAULT_PORT,
        sdk_module: Any = None,
        extra_http_headers: dict[str, str] | None = None,
        **remote_kwargs: Any,
    ) -> None:
        # ``self._owns_sandbox`` controls whether close() tears the
        # sandbox down. We own the sandbox iff we created it; picking
        # up an existing one (sandbox_id= / MANTIS_DAYTONA_SANDBOX_ID)
        # leaves teardown to the operator who provisioned it.
        self._sandbox = None
        self._owns_sandbox = False
        sdk = sdk_module or self._import_sdk()
        api_key = api_key or os.environ.get("DAYTONA_API_KEY") or ""
        if not api_key:
            raise ValueError(
                "DaytonaComputerImpl requires a Daytona API key — "
                "pass api_key=..., or set DAYTONA_API_KEY in the env"
            )
        snapshot = snapshot or os.environ.get("MANTIS_DAYTONA_SNAPSHOT") or ""
        sandbox_id = (
            sandbox_id
            or os.environ.get("MANTIS_DAYTONA_SANDBOX_ID")
            or ""
        )
        if not snapshot and not sandbox_id:
            raise ValueError(
                "DaytonaComputerImpl requires either snapshot= (or "
                "MANTIS_DAYTONA_SNAPSHOT) or sandbox_id= (or "
                "MANTIS_DAYTONA_SANDBOX_ID) — neither was provided"
            )
        # Don't override server_url when the caller didn't pass one —
        # the SDK's default chooses the right API endpoint
        # (forcing https://app.daytona.io routes the SDK at the
        # dashboard's auth0 wall instead of the API).
        config_kwargs = {"api_key": api_key}
        if server_url:
            config_kwargs["server_url"] = server_url

        try:
            client = sdk.Daytona(
                config=sdk.DaytonaConfig(**config_kwargs),
            )
        except Exception as exc:
            raise RuntimeError(
                f"DaytonaComputerImpl: Daytona client init failed: {exc}"
            ) from exc

        if sandbox_id:
            # Fast path — pick up an existing running sandbox by id.
            # Skips the 5-minute cold provision; the operator is
            # responsible for keeping the sandbox alive and tearing
            # it down later.
            logger.warning(
                "DaytonaComputerImpl: picking up existing sandbox id=%s "
                "port=%d (operator owns lifecycle)",
                sandbox_id, port,
            )
            try:
                self._sandbox = client.get(sandbox_id)
            except Exception as exc:
                raise RuntimeError(
                    f"DaytonaComputerImpl: sandbox lookup failed "
                    f"id={sandbox_id!r}: {exc}"
                ) from exc
            # Auto-restart STOPPED sandboxes. Daytona auto-stops idle
            # sandboxes (15-min default —
            # feedback_daytona_autostop_default_15min). Without this
            # the /health probe below would hang for the full
            # startup_timeout_seconds against a never-booting URL.
            state = str(getattr(self._sandbox, "state", "") or "")
            if "STOPPED" in state or "ARCHIVED" in state:
                logger.warning(
                    "DaytonaComputerImpl: sandbox id=%s state=%s — "
                    "calling start() before /health probe",
                    sandbox_id, state,
                )
                try:
                    self._sandbox.start(timeout=startup_timeout_seconds)
                except Exception as exc:
                    raise RuntimeError(
                        f"DaytonaComputerImpl: sandbox.start() failed "
                        f"id={sandbox_id!r}: {exc}"
                    ) from exc
            # Don't tear it down on close() — we didn't create it.
            self._owns_sandbox = False
        else:
            # Slow path — provision a fresh sandbox from a saved
            # snapshot. ``CreateSandboxFromSnapshotParams`` is the
            # Daytona SDK shape for snapshot-based create; raw
            # ``snapshot=`` kwarg on ``client.create()`` is rejected.
            logger.warning(
                "DaytonaComputerImpl: provisioning sandbox "
                "snapshot=%s port=%d (we own lifecycle)",
                snapshot, port,
            )
            try:
                self._sandbox = client.create(
                    sdk.CreateSandboxFromSnapshotParams(snapshot=snapshot),
                    timeout=startup_timeout_seconds,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"DaytonaComputerImpl: sandbox provisioning "
                    f"failed: {exc}"
                ) from exc
            self._owns_sandbox = True

        # Daytona's preview URLs require a skip-preview header to
        # bypass the interstitial (see feedback_daytona_preview_warning_bypass)
        # AND a per-sandbox token header for the auth0 wall on
        # daytonaproxy01.net.
        merged_headers = dict(extra_http_headers or {})
        merged_headers.setdefault(_DAYTONA_SKIP_PREVIEW_HEADER, "true")

        try:
            base_url, preview_token = self._resolve_base_url(
                self._sandbox, port,
            )
            if preview_token:
                merged_headers.setdefault(
                    _DAYTONA_PREVIEW_TOKEN_HEADER, preview_token,
                )
            self._await_ready(
                base_url, startup_timeout_seconds, merged_headers,
            )
        except Exception:
            self._teardown_quietly()
            raise

        logger.warning(
            "DaytonaComputerImpl: sandbox ready base_url=%s "
            "preview_token_present=%s",
            base_url, bool(preview_token),
        )
        super().__init__(
            base_url=base_url,
            extra_http_headers=merged_headers,
            **remote_kwargs,
        )

    # ── lifecycle ──

    def close(self) -> None:
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
            import daytona  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "DaytonaComputerImpl requires the daytona SDK — "
                "`pip install mantis-agent[daytona]` (or "
                "`pip install daytona`)"
            ) from exc
        return daytona

    @staticmethod
    def _resolve_base_url(sandbox: Any, port: int) -> tuple[str, str]:
        """Resolve the sandbox's preview URL + auth token for ``port``.

        Daytona's SDK exposes ``get_preview_link(port)`` returning a
        ``PreviewLink`` with ``.url`` AND ``.token`` — the token is the
        auth0 bypass for the preview proxy and must be sent on every
        request as ``X-Daytona-Preview-Token``.
        """
        link = sandbox.get_preview_link(port)
        url = getattr(link, "url", "") or ""
        token = getattr(link, "token", "") or ""
        if not url:
            raise RuntimeError(
                f"DaytonaComputerImpl: sandbox returned no preview URL "
                f"for port {port}"
            )
        return url.rstrip("/"), token

    def _await_ready(
        self,
        base_url: str,
        deadline_seconds: float,
        headers: dict[str, str],
    ) -> None:
        """Poll ``GET {base_url}/health`` until 200 or deadline.

        Forwards the skip-preview header so Daytona's interstitial
        doesn't masquerade as a server-not-ready response.
        """
        import requests

        deadline = time.monotonic() + deadline_seconds
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            try:
                resp = requests.get(
                    f"{base_url}/health", timeout=5, headers=headers,
                )
                if resp.status_code == 200:
                    return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
            time.sleep(2.0)
        raise TimeoutError(
            f"DaytonaComputerImpl: sandbox /health not ready within "
            f"{deadline_seconds:.0f}s (last_exc={last_exc!r})"
        )

    def _teardown_quietly(self) -> None:
        if self._sandbox is None:
            return
        if not self._owns_sandbox:
            # Picked up an existing sandbox; the operator owns its
            # lifecycle. Just drop our handle.
            logger.warning(
                "DaytonaComputerImpl: dropping non-owned sandbox handle"
            )
            self._sandbox = None
            return
        try:
            # Daytona SDK exposes ``.delete()`` on the Sandbox.
            delete = getattr(self._sandbox, "delete", None) or getattr(
                self._sandbox, "stop", None,
            )
            if delete is not None:
                delete()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DaytonaComputerImpl: teardown raised: %s", exc,
            )
        finally:
            self._sandbox = None


__all__ = ["DaytonaComputerImpl"]
