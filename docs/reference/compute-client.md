# Compute Client — unified contract across both planes

**Status:** Foundation (PR 1 of #785). PR 2 wires the toggle through the executor; PRs 3-4 add the extension surface on Browser-Use Plane.
**Owners:** TBD
**Tracks issue:** #785 (browser-use epic)

## Summary

Mantis runs **two compute planes**:

- **Computer Plane** (#696) — Xvfb + Chrome + xdotool. CUA-pure: screenshot + key/mouse only. Stealth-capable.
- **Browser-Use Plane** (#785) — Chrome under Playwright/CDP-native control. DOM-aware: `state.*`, `tabs.*`, `links.*` extensions.

Both planes implement the **same `ComputeClient` base contract**. Plane selection is a runtime flag (`compute_backend`), not a fork in the brain plane or handler set. DOM verbs are **capability-gated extensions** advertised at `session_init` and enforced via a per-executor `CapabilityAllowlist`.

This document is the umbrella spec. Plane-specific docs:

- `docs/reference/computer-plane.md` — Computer Plane (existing).
- `docs/reference/browser-use-plane.md` — Browser-Use Plane (lands with PR 2).

## Goals

1. **Single contract.** A plan + handler set drives either plane. Switching is `compute_backend: computer_plane | browser_use_plane` at submit time.
2. **Capability gating.** DOM-aware extensions are advertised at `session_init` and only consumed by handlers whose executor allowlist permits them. Pure-CUA executors fail loud if they consume an extension — quiet degradation is explicitly avoided (`feedback_cua_no_dom_access.md`).
3. **Uniform profile + proxy contracts.** Same `(tenant_id, profile_id)` identity on both planes. Same `ProxyConfig` accepted at `session_init`. Storage is per-plane.

## Non-goals

- **Cross-plane profile handoff.** v1 ships per-plane profiles; cross-plane handoff (package → release → mount) is a deferred follow-up gated on real demand (#785).
- **Stealth parity across planes.** Browser-Use Plane on CF-protected sites is not a v1 requirement; pure-CUA on Computer Plane stays the path for stealth-sensitive harvesting.
- **Shared persistent volume across planes.** Forces same-region/provider; ruled out unless mid-plan plane-switching becomes a hot path.

## Two-plane layout

```
                  ┌─────────────────────────┐
                  │      Brain Plane        │
                  │  (executors + handlers) │
                  └────────┬────────────────┘
                           │  one ComputeClient contract
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
   ┌──────────────────────┐   ┌──────────────────────┐
   │   Computer Plane     │   │  Browser-Use Plane   │
   │   (CUA-pure)         │   │  (DOM-aware)         │
   │                      │   │                      │
   │  Xvfb + Chrome +     │   │  Playwright / CDP-   │
   │  xdotool             │   │  native Chrome       │
   │                      │   │                      │
   │  capabilities:       │   │  capabilities:       │
   │    dom_aware=False   │   │    dom_aware=True    │
   │    stealth=True      │   │    stealth=False(v1) │
   └──────────────────────┘   └──────────────────────┘
```

## Base surface (both planes implement)

| Method | Notes |
|---|---|
| `session_init(profile_id, proxy_config)` → `(token, Capabilities)` | Bind tenant + profile + run. Advertises capabilities. |
| `session_close(token)` | Tear down. |
| `screenshot()` → `(png, viewport)` | PNG + scroll_y + captured_at_ms. |
| `dispatch(action)` | Uniform action verb — click(x,y) / key / type / scroll. Each plane translates to its native primitive (xdotool on Computer Plane, Playwright `page.mouse`/`keyboard` on Browser-Use Plane). |
| `health()` | Liveness + last-action timestamp. |

The base surface is enforced by the umbrella. Adding a Computer-Plane-only verb that isn't a no-op stub on Browser-Use Plane breaks the toggle promise — file a separate issue if a divergence is genuinely needed.

## Extension surface (Browser-Use Plane only, capability-gated)

| Verb | Issue | Capability |
|---|---|---|
| `state.current_url()`, `state.tabs()`, `state.focused_element()`, `state.clipboard()`, `state.page_load()` | #778 | `dom_aware` |
| `tabs.open_in_new()`, `tabs.close()`, `tabs.activate()` | #779 | `dom_aware` |
| `links.peek_target(selector)` | #780 | `dom_aware` |

Computer Plane's client refuses these methods at the contract level. Handlers gate on `isinstance(client, SupportsBrowserState)` AND `allowlist.enforce("dom_aware")` BEFORE each call.

## Capabilities

`Capabilities` (`src/mantis_agent/gym/compute_contract.py`) is a frozen dataclass advertised at `session_init`:

```python
@dataclass(frozen=True)
class Capabilities:
    dom_aware: bool = False
    stealth: bool = True
    supports_cdp: bool = False
    backend: ComputeBackend = ComputeBackend.COMPUTER_PLANE
```

Computer Plane returns `Capabilities.for_computer_plane(enable_cdp=...)`. Browser-Use Plane returns `Capabilities.for_browser_use_plane()`. On the wire, `Capabilities` is serialized as a dict in `SessionInitResponse.capabilities` (Phase-0/Phase-1 servers that don't populate the field are interpreted as Computer-Plane CUA-pure via `SessionInitResponse.resolved_capabilities()`).

## CapabilityAllowlist

`CapabilityAllowlist` is the enforcement seam. It is **per-executor**, configured at startup, and immutable for the lifetime of a run.

```python
# Pure-CUA executor: NO DOM-aware extensions.
allowlist = CapabilityAllowlist.pure_cua(executor="run_holo3")

# Browser-use executor: DOM-aware extensions OK.
allowlist = CapabilityAllowlist.browser_use(executor="run_browser_use_claude")

# Inside a handler:
allowlist.enforce("dom_aware")  # raises CapabilityNotAllowed for pure-CUA executors
```

Consuming a non-allowed capability raises `CapabilityNotAllowed`. Fail-loud is deliberate — quiet degradation is what `feedback_cua_no_dom_access` warns against.

## `compute_backend` selection

A plan declares which plane it runs on under its `runtime` block:

```yaml
runtime:
  compute_backend: computer_plane     # default — Xvfb + Chrome + xdotool, CUA-pure
  # or
  compute_backend: browser_use_plane  # Playwright/CDP-native, DOM-aware extensions
```

The same field is also accepted as a submission-time argument (HTTP body / CLI flag). Precedence — highest wins:

1. **Plan `runtime.compute_backend`** (the plan author's choice — most local; HN URL-harvest plans set `browser_use_plane`).
2. **Submission-time `compute_backend`** (operator override — e.g. running an existing plan on the other plane for A/B).
3. **Global default — `computer_plane`** (Xvfb + xdotool stays the path for stealth-sensitive harvesting; pure-CUA is the safe default).

The wiring through executor + client factory lands in PR 2. At session start, the executor's `CapabilityAllowlist` is checked against the advertised `Capabilities`; mismatched runs fail fast at that boundary (not mid-plan).

### When to pick which

| Pick `computer_plane` if … | Pick `browser_use_plane` if … |
|---|---|
| Target site is CF-protected / Turnstile-fronted | Plan needs DOM-aware reads (`state.current_url`, tab management) |
| Plan works with screenshot + key/mouse only | Plan needs to read anchor href before click |
| You don't know — leave it out (default is `computer_plane`) | List page has visually-similar links (semantic role disambiguation) |

## What lands in PR 1 vs later

| PR | What |
|---|---|
| **PR 1 (this)** | Contract types — `Capabilities`, `ComputeBackend`, `CapabilityAllowlist`, extension `Protocol`s, `CapabilityNotAllowed`. Wire-model `SessionInitResponse.capabilities` field (additive). Umbrella docs. **No behavior change.** |
| PR 2 | Browser-Use Plane scaffold — Playwright host + `BrowserUsePlaneClient` base impl + `compute_backend` toggle wired through executor + allowlist enforcement. |
| PR 3 | `state.*` extensions on Browser-Use Plane (#778). |
| PR 4 | `tabs.*` + `links.*` + `target_role` (#779, #780, #781). |

## References

- Epic: #785 (browser-use support)
- Sibling spec: `docs/reference/computer-plane.md` (#696)
- Memory anchors: `feedback_cua_no_dom_access.md`, `feedback_browser_infra_rpc_surface.md`, `project_user_feedback_hn_url_collection_2026_06_07.md`
