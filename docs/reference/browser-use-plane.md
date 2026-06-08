# Browser-Use Plane — DOM-aware companion to Computer Plane

**Status:** Scaffold landed (PR 2 of #785). PR 3-4 add the DOM-aware extension surface (`state.*`, `tabs.*`, `links.*`).
**Owners:** TBD
**Tracks issue:** #785

## Summary

Browser-Use Plane is the **second** Mantis compute plane — the DOM-aware companion to Computer Plane. Chrome runs under **Playwright / CDP-native control** (no Xvfb, no xdotool). Both planes implement the same base `ComputeClient` contract; what differs is the dispatch primitives and the extension verbs Browser-Use Plane admits.

| | Computer Plane | Browser-Use Plane |
|---|---|---|
| Driver | Xvfb + xdotool | Playwright (headless Chromium by default) |
| Capabilities | `dom_aware=False, stealth=True` | `dom_aware=True, stealth=False (v1)` |
| Dispatch | raw `xdotool argv` | structured action verbs (`click`/`key`/`type`/`scroll`) |
| Profile storage | Chrome `--user-data-dir` blob | Playwright `userDataDir` + `storageState` |
| CF / Turnstile parity | yes | **non-goal at v1** |

Pick `computer_plane` (the default) for stealth-sensitive harvesting; pick `browser_use_plane` when the plan needs DOM-aware reads (tab management, anchor `href` peek, semantic click role disambiguation).

## Wire contract (PR 2 — base surface only)

Mirrors `docs/reference/computer-plane.md` in shape; differs in the dispatch verb.

| Method | Path | Required | Notes |
|---|---|---|---|
| `POST` | `/session/init` | yes | Bind `(tenant_id, profile_id, run_id)`. Advertises `Capabilities` (PR 1). Idempotent on `run_id`. |
| `POST` | `/session/close` | yes | Tear down browser context + Playwright runtime. |
| `POST` | `/screenshot` | yes | PNG base64 + viewport metadata. |
| `POST` | `/dispatch` | yes | Structured action verb (`click`/`key`/`type`/`scroll`) + `step_id`. Server keeps a TTL-bounded LRU and returns `deduplicated=true` on retry. |
| `GET` | `/health` | yes | Liveness + last-action timestamp. |

DOM-aware extensions (`/state/*`, `/tabs/*`, `/links/peek`) land in PR 3-4 — they are explicitly NOT in the base surface and NOT supported by Computer Plane.

### Pydantic wire models

Defined in `src/mantis_agent/gym/browser_use_wire.py`:

- `BrowserUseSessionInitRequest` / `BrowserUseSessionInitResponse`
- `BrowserUseSessionCloseRequest` / `BrowserUseSessionCloseResponse`
- `BrowserUseScreenshotResponse`
- `DispatchActionRequest` / `DispatchActionResponse`
- `BrowserUseHealthResponse`

### Capabilities

`session/init` returns `Capabilities.for_browser_use_plane()`:

```python
Capabilities(
    dom_aware=True,
    stealth=False,           # explicit non-goal at v1
    supports_cdp=True,       # Playwright IS CDP under the hood
    backend=ComputeBackend.BROWSER_USE_PLANE,
)
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Modal app "mantis-browser-use"  (NEW, this PR)      │
│                                                       │
│  ┌──────────── BROWSER-USE PLANE ────────────┐       │
│  │ Browser-Use Agent FastAPI:                │       │
│  │   POST /session/init                      │       │
│  │   POST /session/close                     │       │
│  │   POST /screenshot                        │       │
│  │   POST /dispatch                          │       │
│  │   GET  /health                            │       │
│  │ Playwright + Chromium (headless)          │       │
│  └───────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────┘
                       ▲
                       │ HTTPS
                       │
┌──────────────────────────────────────────────────────┐
│  Modal app "mantis-cua-server"  (UNCHANGED)          │
│                                                       │
│  Brain plane → run_browser_use executor →            │
│    BrowserUsePlaneClient ───────────────┘            │
└──────────────────────────────────────────────────────┘
```

Two Modal apps. Independent deploy cadence — redeploying the brain doesn't redeploy Browser-Use Plane and vice versa.

## Image (Modal)

`mcr.microsoft.com/playwright-python:v1.49.0-jammy` — bundles Chromium + the Playwright Python SDK pinned to the same version. CPU-only (no GPU, no Xvfb, no xdotool). ~2 GB.

Locale + TZ matched to Computer Plane so screenshot-comparison and date-format-sensitive sites behave identically across planes.

## Deploy

```bash
modal deploy deploy/modal/browser_use_plane.py
```

Reads the function URL after deploy:

```python
import modal
url = modal.Function.from_name("mantis-browser-use", "browser_use").get_web_url()
```

## Brain-plane integration

`src/mantis_agent/run_browser_use.py` exports `run_browser_use_executor(...)` — the executor entry point. PR 2 leaves it **un-wired** from `modal_cua_server.py`'s `cua_model` dispatch table; PR 3 wires it once the DOM-aware extensions make the surface useful for plan authors.

Configure plan-level via `runtime.compute_backend: browser_use_plane`:

```yaml
runtime:
  compute_backend: browser_use_plane

steps:
  - intent: "Open Hacker News"
    type: navigate
    url: https://news.ycombinator.com
```

The resolver in `src/mantis_agent/gym/compute_backend_resolver.py` reads this; the factory in `src/mantis_agent/gym/compute_factory.py` dispatches to the right client. Default remains `computer_plane`.

## Capability enforcement

`run_browser_use` configures `CapabilityAllowlist.browser_use(executor="run_browser_use")` — admits `dom_aware` + `supports_cdp`. Pure-CUA executors (`run_claude_cua`, `run_holo3`, etc.) use `CapabilityAllowlist.pure_cua()` and will raise `CapabilityNotAllowed` if a handler tries to consume DOM-aware extensions against them even when the client speaks Browser-Use Plane.

The mismatch check runs at session start (`run_browser_use._validate_executor_compat`) — failures here happen before any browser action, not mid-plan.

## Profile + proxy at v1

Profiles are **per-plane**. The same `(tenant_id, profile_id)` identity exists on both planes but storage is independent. Layout (this plane):

```
/data/browser-use-profile/<tenant>__<profile_id>/   ← Playwright userDataDir blob
```

Computer Plane keeps its existing layout at `/data/chrome-profile/...`. Both volumes mounted; no shared bytes. Cross-plane profile handoff is the deferred follow-up gated on real demand (#785).

Proxy is passed at `session/init` and forwarded to Playwright's `launch({proxy: {server}})`. Same `PrivateProxy` creds as Computer Plane.

## Non-goals (v1)

- Stealth on CF-protected sites — use Computer Plane for those.
- Cross-plane profile handoff — deferred follow-up.
- Concurrent sessions per container — single-session pinned at v1, matching Computer Plane's posture.

## Open questions

1. **Async vs sync Playwright.** v1 uses `sync_playwright` for simplicity. Switching to `async_api` is a non-trivial refactor — defer until concurrency pressure makes it worth it.
2. **Session pool sizing.** Same as Computer Plane: pin to one container at v1, revisit after a real workload.

## References

- Umbrella contract: `docs/reference/compute-client.md`
- Sibling plane spec: `docs/reference/computer-plane.md`
- Epic: #785
- PR 1 (foundation): #786
