"""Unified `ComputeClient` contract — umbrella over both compute planes.

Mantis runs two compute planes (#785):

- **Computer Plane** (#696) — Xvfb + Chrome + xdotool; CUA-pure. The historical
  Mantis runtime; today's `ComputerClient` lives here. `dom_aware=False`.
- **Browser-Use Plane** — Chrome under Playwright/CDP-native control;
  DOM-aware. `dom_aware=True`.

Both planes implement the **base surface** (session_init, screenshot,
dispatch, health). DOM verbs (`state.*`, `tabs.*`, `links.*`) ship only on
Browser-Use Plane and are **capability-gated** behind the `dom_aware`
capability advertised at `session_init`.

The brain plane reads the advertised `Capabilities` from session_init and
cross-checks against a per-executor `CapabilityAllowlist`. Consuming a
capability not on the allowlist raises `CapabilityNotAllowed` — this is the
enforcement seam that prevents pure-CUA executors (Holo3, Claude vision)
from silently consuming DOM-aware reads even when wired to Browser-Use
Plane (`feedback_cua_no_dom_access.md`).

This module defines the contract types only. PR 2 wires `compute_backend`
plumbing through the executor and adds the Browser-Use Plane client. See
`docs/reference/compute-client.md` for the umbrella spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ComputeBackend(str, Enum):
    """Which plane a `ComputeClient` instance targets."""

    COMPUTER_PLANE = "computer_plane"
    BROWSER_USE_PLANE = "browser_use_plane"


@dataclass(frozen=True)
class Capabilities:
    """Capabilities a `ComputeClient` advertises at `session_init`.

    Defaults match Computer Plane (the historical Mantis runtime): no DOM
    reads, stealth-capable via Xvfb + xdotool, CDP off unless explicitly
    enabled.

    Browser-Use Plane returns `dom_aware=True, stealth=False` at v1 — the
    epic (#785) explicitly defers CF/Turnstile parity on that plane.
    """

    dom_aware: bool = False
    stealth: bool = True
    supports_cdp: bool = False
    backend: ComputeBackend = ComputeBackend.COMPUTER_PLANE

    def as_dict(self) -> dict[str, Any]:
        return {
            "dom_aware": self.dom_aware,
            "stealth": self.stealth,
            "supports_cdp": self.supports_cdp,
            "backend": self.backend.value,
        }

    @classmethod
    def for_computer_plane(cls, *, enable_cdp: bool = False) -> "Capabilities":
        return cls(
            dom_aware=False,
            stealth=True,
            supports_cdp=enable_cdp,
            backend=ComputeBackend.COMPUTER_PLANE,
        )

    @classmethod
    def for_browser_use_plane(cls) -> "Capabilities":
        # v1: Browser-Use Plane uses Playwright/CDP-native control. Stealth
        # parity on CF-protected sites is an explicit non-goal at v1
        # (#785). supports_cdp=True because Playwright IS CDP under the
        # hood — but the capability matters for handlers that gate on it.
        return cls(
            dom_aware=True,
            stealth=False,
            supports_cdp=True,
            backend=ComputeBackend.BROWSER_USE_PLANE,
        )


class CapabilityNotAllowed(RuntimeError):
    """Raised when an executor consumes a capability not on its allowlist.

    The capability allowlist is configured per-executor at startup; this
    prevents a pure-CUA executor wired to Browser-Use Plane from silently
    starting to consume DOM-aware reads. The fail-loud behavior is
    deliberate — quiet degradation is what `feedback_cua_no_dom_access`
    explicitly warns against.
    """

    def __init__(self, capability: str, executor: str | None = None) -> None:
        self.capability = capability
        self.executor = executor
        msg = f"capability {capability!r} not allowed"
        if executor:
            msg += f" for executor {executor!r}"
        super().__init__(msg)


@dataclass(frozen=True)
class CapabilityAllowlist:
    """Per-executor capability allowlist.

    Used by handlers and the runner to gate consumption of advertised
    capabilities. Configured at executor startup; immutable for the
    lifetime of a run.

    `dom_aware` is the load-bearing entry — it is the one that, if leaked
    into a pure-CUA executor, silently changes the grounding model.
    """

    allowed: frozenset[str] = field(default_factory=frozenset)
    executor: str | None = None

    def enforce(self, capability: str) -> None:
        """Raise `CapabilityNotAllowed` if `capability` is not on the list.

        Call this BEFORE consuming a capability-gated extension verb. The
        caller's job is to either degrade (skip the verb) or halt (let the
        exception propagate up to the runner).
        """
        if capability not in self.allowed:
            raise CapabilityNotAllowed(capability, self.executor)

    def allows(self, capability: str) -> bool:
        """Non-raising query — for handlers that want to degrade silently."""
        return capability in self.allowed

    def with_added(self, *capabilities: str) -> "CapabilityAllowlist":
        return replace(self, allowed=self.allowed | frozenset(capabilities))

    @classmethod
    def pure_cua(cls, executor: str | None = None) -> "CapabilityAllowlist":
        """Allowlist for pure-CUA executors: NO DOM-aware extensions.

        This is the canonical pure-CUA posture. Handlers consuming DOM
        verbs against this allowlist will raise.
        """
        return cls(allowed=frozenset(), executor=executor)

    @classmethod
    def browser_use(cls, executor: str | None = None) -> "CapabilityAllowlist":
        """Allowlist for browser-use executors: DOM-aware extensions OK."""
        return cls(
            allowed=frozenset({"dom_aware", "supports_cdp"}),
            executor=executor,
        )


# Extension protocols — implemented by Browser-Use Plane only. Capability-
# gated behind `dom_aware`. Handlers consuming these should runtime-check
# with `isinstance(client, SupportsBrowserState)` AND enforce the
# allowlist BEFORE the call.


@runtime_checkable
class SupportsBrowserState(Protocol):
    """`state.*` extensions — browser-state reads (#778).

    Implemented by Browser-Use Plane. Capability-gated behind `dom_aware`.
    """

    def state_current_url(self) -> str: ...
    def state_tabs(self) -> list[dict[str, Any]]: ...
    def state_focused_element(self) -> dict[str, Any] | None: ...
    def state_clipboard(self) -> str: ...
    def state_page_load(self) -> str: ...


@runtime_checkable
class SupportsTabs(Protocol):
    """`tabs.*` extensions — tab management (#779)."""

    def tabs_open_in_new(self, url: str | None = None) -> str: ...
    def tabs_close(self, tab_id: str) -> None: ...
    def tabs_activate(self, tab_id: str) -> None: ...


@runtime_checkable
class SupportsLinkPeek(Protocol):
    """`links.*` extensions — read anchor href without committing the click (#780)."""

    def links_peek_target(self, selector_or_bbox: Any) -> str | None: ...


__all__ = [
    "Capabilities",
    "CapabilityAllowlist",
    "CapabilityNotAllowed",
    "ComputeBackend",
    "SupportsBrowserState",
    "SupportsLinkPeek",
    "SupportsTabs",
]
