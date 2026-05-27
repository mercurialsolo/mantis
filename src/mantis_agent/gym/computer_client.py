"""`ComputerClient` — single seam for everything that talks to "the computer".

Phase 0 introduces this abstraction in-process with no behavior change:
`make_computer_client(cfg)` returns a `LocalXdotoolImpl` (a `XdotoolGymEnv`
subclass with added latency instrumentation). Phase 1 will add the
`"modal"` backend (HTTPS to a separate Modal function); Phase 2 will add
`"e2b"` and `"daytona"`.

The factory is the only thing the brain plane should call — direct
construction of `XdotoolGymEnv` outside of this module is the seam-leak
to avoid.

See `docs/reference/computer-plane.md` for the full spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .base import GymEnvironment

ComputerPlaneBackend = Literal["local", "modal", "e2b", "daytona"]


@dataclass
class ComputerPlaneConfig:
    """Configuration for the computer plane backend.

    `backend="local"` is Phase 0's default and matches today's production:
    Xvfb + Chrome + xdotool run inside the same process as the brain.

    Per-executor overrides allow rolling out Phase 1 / Phase 2 backends
    incrementally — e.g. `run_claude_cua` migrates first, GPU executors
    stay on `"local"` longer.
    """

    backend: ComputerPlaneBackend = "local"

    # Phase 1+ only: HTTPS base URL for the remote ComputerAgent.
    remote_base_url: str | None = None

    # Phase 1+ only: auth token sent as `Authorization: Bearer <token>`.
    remote_auth_token: str | None = None

    # Opt-in CDP escape hatch. Defaults off per `feedback_cua_no_dom_access`.
    enable_cdp: bool = False

    # Per-executor overrides — `{"run_claude_cua": "modal"}`.
    per_executor_overrides: dict[str, ComputerPlaneBackend] = field(default_factory=dict)

    def resolve_for_executor(self, executor_name: str | None) -> "ComputerPlaneConfig":
        """Return a copy of this config with `backend` swapped per executor.

        Lets a single global config carry rollout overrides without each
        call site needing to know about them.
        """
        if not executor_name or executor_name not in self.per_executor_overrides:
            return self
        override = self.per_executor_overrides[executor_name]
        if override == self.backend:
            return self
        return ComputerPlaneConfig(
            backend=override,
            remote_base_url=self.remote_base_url,
            remote_auth_token=self.remote_auth_token,
            enable_cdp=self.enable_cdp,
            per_executor_overrides=self.per_executor_overrides,
        )


class ComputerClient(GymEnvironment):
    """Marker base. All impls are `GymEnvironment` subclasses.

    The marker lets the brain code type-check that it received a
    computer-plane-aware env (one that respects the wire contract) versus
    an arbitrary `GymEnvironment` like `GymAnythingEnv` or `PlaywrightGymEnv`.
    """


def make_computer_client(
    cfg: ComputerPlaneConfig | None = None,
    /,
    **env_kwargs: Any,
) -> ComputerClient:
    """Factory: build a `ComputerClient` for the given backend.

    `env_kwargs` are forwarded to the underlying impl — for `"local"`
    that's the existing `XdotoolGymEnv` keyword set (`start_url`,
    `viewport`, `proxy_server`, ...). Phase 1+ impls translate the same
    kwargs into wire-contract `SessionInitRequest` fields.

    Keeping the kwarg surface identical means call sites only need to
    swap `XdotoolGymEnv(**kw)` → `make_computer_client(cfg, **kw)` with
    no other edits.
    """
    cfg = cfg or ComputerPlaneConfig()
    backend = cfg.backend

    if backend == "local":
        from .local_xdotool_impl import LocalXdotoolImpl

        return LocalXdotoolImpl(**env_kwargs)

    if backend == "modal":
        if not cfg.remote_base_url:
            raise ValueError(
                "ComputerPlaneConfig.backend='modal' requires remote_base_url"
            )
        from .remote_computer_impl import RemoteComputerImpl

        return RemoteComputerImpl(
            base_url=cfg.remote_base_url,
            auth_token=cfg.remote_auth_token,
            enable_cdp=cfg.enable_cdp,
            **env_kwargs,
        )

    if backend in ("e2b", "daytona"):
        raise NotImplementedError(
            f"ComputerPlaneConfig.backend={backend!r} lands in Phase 2 (#699)"
        )

    raise ValueError(f"unknown ComputerPlaneConfig.backend: {backend!r}")
