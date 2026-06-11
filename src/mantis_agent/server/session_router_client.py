"""HTTPS client for the session-router endpoints (Phase 1.5, #846, PR 3).

The brain plane uses this to mint a dedicated computer-plane session
at run start (``POST /v1/computer_sessions``) and tear it down at
end (``DELETE /v1/computer_sessions/{session_id}``). Pure-HTTPS, no
Modal client dependency — works the same from inside or outside a
Modal container.

Talks to the router defined in
``deploy/modal/modal_cua_server.build_api_app``; wire-contract from
``mantis_agent.session_wire``.

Failure semantics — quiet by default:

* Connection / 5xx → ``SessionRouterError`` with the upstream detail.
  Caller decides whether to fall back to the pinned ``computer_plane``
  ASGI app (the current Phase 1 behaviour).
* 4xx → raised with the upstream detail; not retried.
* ``close_session`` failures are best-effort by default — closing a
  session that the reaper already terminated should not surface as
  an error to the brain.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from ..session_wire import (
    SessionCloseResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionNotFoundError,
    SessionQuotaExceededError,
    SessionRouterError,
    SessionUnreachableError,
)

logger = logging.getLogger(__name__)


# Defaults tuned for the cold-start + tunnel-publish budget on the
# orchestrator side. The router itself caps the wait at 120s; we add
# a small margin so the HTTPS layer doesn't timeout first.
_DEFAULT_CREATE_TIMEOUT_SECONDS = 150.0
_DEFAULT_CLOSE_TIMEOUT_SECONDS = 15.0


class SessionRouterClient:
    """Tiny HTTPS shim around ``/v1/computer_sessions``.

    Stateless — constructed once per executor, used to create the
    session and (later) close it.
    """

    def __init__(
        self,
        *,
        router_url: str,
        auth_token: str,
        create_timeout_seconds: float = _DEFAULT_CREATE_TIMEOUT_SECONDS,
        close_timeout_seconds: float = _DEFAULT_CLOSE_TIMEOUT_SECONDS,
        # DI seam so tests can swap out the HTTP layer without
        # monkeypatching ``requests`` globally.
        session: Any | None = None,
    ) -> None:
        if not router_url:
            raise ValueError("SessionRouterClient requires a non-empty router_url")
        if not auth_token:
            raise ValueError("SessionRouterClient requires a non-empty auth_token")
        self._router_url = router_url.rstrip("/")
        self._auth_token = auth_token
        self._create_timeout = float(create_timeout_seconds)
        self._close_timeout = float(close_timeout_seconds)
        self._http = session or requests

    def _headers(self) -> dict[str, str]:
        return {
            "X-Mantis-Token": self._auth_token,
            "Content-Type": "application/json",
        }

    def _raise_for_status(self, resp: Any, op: str) -> None:
        if resp.status_code < 400:
            return
        try:
            detail = resp.json().get("detail") or resp.text
        except ValueError:
            detail = resp.text
        if resp.status_code == 404:
            raise SessionNotFoundError(f"{op}: {detail}")
        if resp.status_code == 429:
            raise SessionQuotaExceededError(f"{op}: {detail}")
        if resp.status_code == 504:
            raise SessionUnreachableError(f"{op}: {detail}")
        # Catch-all — preserves the upstream status code in the message.
        raise SessionRouterError(f"{op} HTTP {resp.status_code}: {detail}")

    # ── Create ────────────────────────────────────────────────────────

    def create_session(self, req: SessionCreateRequest) -> SessionCreateResponse:
        """Mint a session via the router. Blocks up to the
        ``create_timeout_seconds`` budget while the orchestrator does
        its cold-start + ``modal.forward`` dance."""
        try:
            resp = self._http.post(
                self._router_url + "/v1/computer_sessions",
                json=req.model_dump(mode="json"),
                headers=self._headers(),
                timeout=self._create_timeout,
            )
        except requests.RequestException as exc:
            raise SessionUnreachableError(
                f"create_session: router unreachable ({exc})"
            ) from exc
        self._raise_for_status(resp, "create_session")
        return SessionCreateResponse.model_validate(resp.json())

    # ── Close ─────────────────────────────────────────────────────────

    def close_session(
        self, session_id: str, *, reason: str = "brain_closed",
        quiet: bool = True,
    ) -> SessionCloseResponse | None:
        """Best-effort by default — a 404 here means the reaper got
        there first, and the brain shouldn't surface that. Set
        ``quiet=False`` to bubble errors when callers care.
        """
        url = f"{self._router_url}/v1/computer_sessions/{session_id}"
        try:
            resp = self._http.delete(
                url,
                headers={**self._headers(), "X-Mantis-Reason": reason},
                timeout=self._close_timeout,
            )
        except requests.RequestException as exc:
            if quiet:
                logger.warning(
                    "close_session: router unreachable session=%s (%s)",
                    session_id, exc,
                )
                return None
            raise SessionUnreachableError(
                f"close_session: router unreachable ({exc})"
            ) from exc
        try:
            self._raise_for_status(resp, "close_session")
        except SessionNotFoundError:
            if quiet:
                return None
            raise
        except SessionRouterError:
            if quiet:
                logger.warning(
                    "close_session: router error session=%s status=%s",
                    session_id, resp.status_code,
                )
                return None
            raise
        return SessionCloseResponse.model_validate(resp.json())
