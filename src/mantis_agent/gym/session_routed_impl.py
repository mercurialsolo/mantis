"""``SessionRoutedComputerImpl`` — RemoteComputerImpl backed by the
Phase 1.5 session router (#846).

On construction this client calls the router's
``POST /v1/computer_sessions`` to mint a dedicated per-session
container, then constructs the underlying ``RemoteComputerImpl``
against the per-session tunnel URL (with the pre-minted session token
short-circuiting the ``/session/init`` hop).

On ``shutdown()`` it tears the session down via the router's
``DELETE /v1/computer_sessions/{session_id}`` before delegating to
the superclass's session-close path. Close is best-effort by default
— a 404 from the router (reaper already terminated) is swallowed so
the brain's teardown doesn't surface noise.
"""

from __future__ import annotations

import logging
from typing import Any

from ..server.session_router_client import SessionRouterClient
from .remote_computer_impl import RemoteComputerImpl
from ..session_wire import SessionCreateRequest

logger = logging.getLogger(__name__)


class SessionRoutedComputerImpl(RemoteComputerImpl):
    """Construct-then-delegate wrapper around the per-session router."""

    def __init__(
        self,
        *,
        router_url: str,
        auth_token: str,
        # Forwarded into SessionCreateRequest:
        tenant_id: str,
        profile_id: str,
        run_id: str,
        start_url: str = "about:blank",
        viewport: tuple[int, int] = (1280, 720),
        proxy_server: str = "",
        profile_dir: str | None = None,
        extra_http_headers: dict[str, str] | None = None,
        enable_cdp: bool = False,
        ttl_seconds: int = 3600,
        # Optional DI for tests:
        router_client: SessionRouterClient | None = None,
        # Everything else (chrome_flags, settle_time, …) is forwarded
        # straight through to ``RemoteComputerImpl.__init__`` for
        # XdotoolGymEnv kwarg parity.
        **remote_kwargs: Any,
    ) -> None:
        client = router_client or SessionRouterClient(
            router_url=router_url, auth_token=auth_token,
        )
        req = SessionCreateRequest(
            tenant_id=tenant_id,
            profile_id=profile_id,
            run_id=run_id,
            start_url=start_url,
            viewport=viewport,
            proxy_server=proxy_server,
            profile_dir=profile_dir,
            extra_http_headers=extra_http_headers,
            enable_cdp=enable_cdp,
            ttl_seconds=ttl_seconds,
        )
        # Bubble SessionRouterError up unchanged — make_computer_client
        # callers decide whether to fall back to the pinned
        # ``computer_plane`` ASGI URL.
        response = client.create_session(req)

        self._router_client = client
        self._session_id = response.session_id
        self._session_expires_at_ms = response.expires_at_ms
        self._session_sandbox_id = response.sandbox_id

        super().__init__(
            base_url=response.base_url,
            auth_token=None,  # tunnel URLs are unauthenticated; session
                              # token is the only credential the
                              # ComputerAgent enforces (X-Mantis-Session).
            tenant_id=tenant_id,
            profile_id=profile_id,
            run_id=run_id,
            start_url=start_url,
            viewport=viewport,
            proxy_server=proxy_server,
            profile_dir=profile_dir,
            extra_http_headers=extra_http_headers,
            enable_cdp=enable_cdp,
            pre_minted_session_token=response.session_token,
            **remote_kwargs,
        )

    # ── lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Tear the session down via the router, then delegate.

        Best-effort: a router-side 404 (already reaped) is swallowed so
        the brain doesn't see noise when shutting down a session the
        reaper already cleaned up.
        """
        try:
            self._router_client.close_session(
                self._session_id, reason="brain_closed", quiet=True,
            )
        except Exception as exc:  # noqa: BLE001 — never block teardown
            logger.warning(
                "SessionRoutedComputerImpl: router close raised (%s); "
                "delegating to super().close() anyway",
                exc,
            )
        super().close()

    # ── observability ─────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        """Router-minted session id; useful for log correlation."""
        return self._session_id

    @property
    def sandbox_id(self) -> str:
        """Modal FunctionCall id for the per-session container."""
        return self._session_sandbox_id

    @property
    def session_expires_at_ms(self) -> int:
        """When the reaper will consider this session orphaned."""
        return self._session_expires_at_ms
