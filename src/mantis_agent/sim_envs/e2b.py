"""E2B runtime backend — reserved stub for the high-throughput batch path.

E2B Firecracker microVMs offer ~200ms cold starts versus Modal's 5-15s,
which only matters at thousands-of-plans-per-day batch volume (see #336
§Hosting). At v1 we are nowhere near that volume, so this file exists
solely to reserve the ``--runtime e2b`` flag so a future PR can wire
the real backend without touching CLI, plans, or env images.

Calling :meth:`E2BBackend.start` raises ``NotImplementedError`` with a
message pointing at the relevant issue. Tests assert on the specific
exception so we catch silent regressions if someone adds a half-baked
implementation here.
"""

from __future__ import annotations

from typing import Any

from .runtime import RuntimeHandle


class E2BBackend:
    name = "e2b"

    def start(
        self,
        env_name: str,
        *,
        seed: int = 42,
        now: str = "2026-01-15T09:00:00Z",
        admin_token: str | None = None,
    ) -> RuntimeHandle:
        raise NotImplementedError(
            "E2B backend not implemented in v1. See #336 §Hosting — the flag "
            "is reserved so the backend can be added without touching plans "
            "or CLI. Use --runtime local for dev and --runtime modal for CI."
        )

    def wait_healthy(self, handle: RuntimeHandle, *, timeout_s: float = 30.0) -> None:
        raise NotImplementedError("E2B backend not implemented in v1.")

    def get_url(self, handle: RuntimeHandle) -> str:
        raise NotImplementedError("E2B backend not implemented in v1.")

    def fetch_events(
        self,
        handle: RuntimeHandle,
        *,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("E2B backend not implemented in v1.")

    def stop(self, handle: RuntimeHandle) -> None:
        raise NotImplementedError("E2B backend not implemented in v1.")
