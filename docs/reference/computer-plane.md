# Computer Plane — modular split between Modal, E2B, and Daytona

**Status:** Proposed
**Owners:** TBD
**Tracks issue:** TBD (this doc is the canonical spec; the GitHub issue points at it)

> Computer Plane is one of two compute planes under the unified
> `ComputeClient` contract — see `docs/reference/compute-client.md`
> (#785). It advertises `Capabilities(dom_aware=False, stealth=True)`
> at `session_init`. The DOM-aware extensions (`state.*`, `tabs.*`,
> `links.*`) ship on Browser-Use Plane only; Computer Plane refuses
> them at the contract level. Pure-CUA — by design.

## Summary

Today Mantis runs the brain (Holo3 / Claude / OpenCUA / Fara / Gemma4), the task loop, the step handlers, Xvfb, and Chrome **inside one Modal function invocation**. The Chrome profile lives on `modal.Volume("osworld-data")` mounted at `/data`. Brain ↔ environment calls are in-process Python.

This document specifies the split of that monolith into two planes connected by a narrow RPC:

- **Brain plane** — keeps the executor pool, brain, task loop, step handlers, and Augur emission. Unchanged code, lighter image.
- **Computer plane** — runs Xvfb + Chrome + xdotool behind a small FastAPI service (`ComputerAgent`). Exposes `screenshot` + `xdotool` + opt-in `cdp`.

The seam between them is the existing `GymEnvironment` ABC, with `ComputerClient` as a new factory-built implementation. Step handlers do not change.

The split is delivered in three phases. Phase 0 introduces the seam in-process (no behavior change). Phase 1 lifts the computer plane out as a separate Modal function (still inside the same Modal app, sharing the same volume). Phase 2 makes the computer plane host pluggable so an E2B Desktop or Daytona sandbox can replace the Modal computer-plane function via a deploy-config change.

## Goals

1. Single seam (`ComputerClient`) for everything that talks to "the computer," replacing direct `XdotoolGymEnv` instantiation in the executor lifecycle.
2. Wire contract that is **CUA-pure**: `screenshot` + `xdotool(argv)` are required; `cdp_evaluate` / `cdp_click_at_point` are opt-in. No DOM-aware verbs.
3. Phase 1 ships zero new infra: brain and computer planes are two Modal functions in the same app, both mounting `osworld-data`. Profile bytes stay where they are. No tar.zst, no S3.
4. Phase 2 makes E2B Desktop and Daytona pluggable behind the same `ComputerClient` interface via a `RemoteComputerImpl` and a per-provider adapter.
5. Per-profile lock semantics, Augur emission, plan format, and HTTP API are unchanged.

## Non-goals

- Moving the brain. Brain stays on Modal in all phases.
- Profile snapshotting / S3 / `ProfileSnapshotter`. Only required when leaving Modal Volume — fully specified in [`computer-plane-profile-snapshots.md`](computer-plane-profile-snapshots.md). The snapshot doc is the gating doc that has to land before any Phase 2 host backend can ship.
- Replacing the GPU executor pool. Holo3 / OpenCUA / Fara stay on their current Modal GPU tiers.
- Multi-region, geo-pinning, sandbox identity management. Out of scope here.

## Architecture overview

```
┌──────────────────────────────────────────────────────────────┐
│  Modal app "mantis-cua-server"                               │
│                                                              │
│  ┌─────────── BRAIN PLANE ───────────┐                       │
│  │ FastAPI  →  Executor pool  →      │                       │
│  │ Brain    →  Task Loop      →      │                       │
│  │ Step Handlers  →                  │                       │
│  │   ComputerClient (NEW)            │                       │
│  │     ├── LocalXdotoolImpl  (Phase 0)                       │
│  │     ├── RemoteModalImpl    (Phase 1)                      │
│  │     ├── RemoteE2BImpl      (Phase 2)                      │
│  │     └── RemoteDaytonaImpl  (Phase 2)                      │
│  └───────────────────────────────────┘                       │
│                       │ HTTPS (Phase 1+)                     │
│                       ▼                                      │
│  ┌────────── COMPUTER PLANE (NEW Modal fn, Phase 1) ──────┐  │
│  │ ComputerAgent FastAPI:                                 │  │
│  │   POST /screenshot                                     │  │
│  │   POST /xdotool                                        │  │
│  │   POST /cdp        (opt-in)                            │  │
│  │   POST /session    (init: bind tenant + profile)       │  │
│  │ Xvfb + Chrome + xdotool                                │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
              │ both mount /data
              ▼
┌──────────────────────────────────────────────────────┐
│ modal.Volume "osworld-data"  — UNCHANGED              │
│   /data/chrome-profile/<tenant>__<profile_id>/        │
│   /data/results/, /data/models/, /data/training/      │
└──────────────────────────────────────────────────────┘
```

Phase 2 adds non-Modal backends, connected over the same HTTPS surface. Profile snapshotting only enters the picture when the backend cannot mount `osworld-data`.

## Wire contract

The contract is the **only** thing all backends must implement identically. Anything beyond this surface is implementation detail of a specific backend.

### Endpoints

| Method | Path | Required | Notes |
|---|---|---|---|
| `POST` | `/session/init` | yes | Bind this `ComputerAgent` instance to a `(tenant_id, profile_id, run_id)` triple. Returns session token; subsequent calls carry it as a header. |
| `POST` | `/session/close` | yes | Tear down Chrome, flush state, release. |
| `POST` | `/screenshot` | yes | Returns PNG bytes + viewport metadata. |
| `POST` | `/xdotool` | yes | Executes `xdotool argv` against the bound display. Returns stdout + rc. |
| `POST` | `/cdp` | **no** (opt-in) | Executes a CDP `Runtime.evaluate` / `Input.*` against the bound Chrome target. Disabled unless `executor.cdp_enabled=true`. |
| `GET` | `/health` | yes | Liveness + last-action timestamp. |

### Pydantic models (canonical)

```python
# src/mantis_agent/gym/computer_wire.py  (NEW)

from pydantic import BaseModel, Field
from typing import Literal


class SessionInitRequest(BaseModel):
    tenant_id: str
    profile_id: str
    run_id: str
    proxy_server: str | None = None           # e.g. "http://user:pass@host:port"
    chrome_flags: list[str] = Field(default_factory=list)
    enable_cdp: bool = False                  # opt-in escape hatch
    viewport: tuple[int, int] = (1280, 720)


class SessionInitResponse(BaseModel):
    session_token: str                        # carry as X-Mantis-Session header
    chrome_pid: int | None = None
    xvfb_display: str                         # e.g. ":99"


class ScreenshotRequest(BaseModel):
    # Empty body for v1. Reserved for future params (region, format).
    format: Literal["png"] = "png"


class ScreenshotResponse(BaseModel):
    image_b64: str                            # PNG bytes, base64-encoded
    width: int
    height: int
    scroll_y: int                             # read from window.scrollY via CDP
    captured_at_ms: int                       # unix ms


class XdotoolRequest(BaseModel):
    argv: list[str]                           # e.g. ["mousemove", "842", "317", "click", "1"]
    step_id: str                              # client-generated; agent dedupes within window
    timeout_ms: int = 5000


class XdotoolResponse(BaseModel):
    stdout: str
    stderr: str
    returncode: int
    deduplicated: bool = False                # true if step_id already saw a successful execution


class CDPRequest(BaseModel):
    expression: str                           # JS expression (action-dispatch only by convention)
    await_promise: bool = False
    step_id: str


class CDPResponse(BaseModel):
    result_json: str                          # JSON-stringified CDP result
    returncode: int = 0
```

### Idempotency

`xdotool` is not naturally idempotent — sending a click twice can navigate twice. The wire requires:

- `step_id` on every `xdotool` and `cdp` call, client-generated, unique per logical step.
- `ComputerAgent` maintains a TTL-bounded LRU of `step_id → response` (default TTL = 30s, capacity = 1000). A repeated `step_id` returns the cached response with `deduplicated=true`.
- Retries by the client use the **same** `step_id`. New logical steps use a new `step_id`.

### Session lifecycle

```
brain plane                          computer plane
───────────                          ──────────────
POST /session/init {tenant,...}  →
                                 ←   SessionInitResponse {token, ...}
                                     [Xvfb + Chrome launched, profile mounted]

POST /screenshot                 →
POST /xdotool {argv, step_id}    →
   ... 50–200 iterations of the loop ...

POST /session/close              →
                                     [Chrome SIGTERM, Xvfb stopped]
```

`/session/init` is idempotent on `run_id`: a second init for the same `run_id` returns the existing session token. Different `run_id` against an active session is a 409.

## `ComputerClient` factory

```python
# src/mantis_agent/gym/computer_client.py  (NEW)

from src.mantis_agent.gym.base import GymEnvironment

class ComputerClient(GymEnvironment):
    """Marker base. All impls are GymEnvironment subclasses."""

def make_computer_client(cfg: ComputerPlaneConfig) -> ComputerClient:
    match cfg.backend:
        case "local":   return LocalXdotoolImpl(cfg)              # Phase 0
        case "modal":   return RemoteComputerImpl(cfg, ...)       # Phase 1
        case "e2b":     return RemoteE2BImpl(cfg, ...)            # Phase 2
        case "daytona": return RemoteDaytonaImpl(cfg, ...)        # Phase 2
        case _: raise ValueError(...)
```

`ComputerPlaneConfig` lives in `src/mantis_agent/config.py` with a `backend: Literal[...]` field, default `"local"`. Per-executor overrides allowed (e.g. `run_claude_cua` migrates first; the GPU executors stay on `"local"` longer).

## Phase 0 — seam refactor (zero behavior change)

**Goal:** prove the abstraction is faithful by routing today's production through it with no observable change.

### File changes

| File | Change |
|---|---|
| `src/mantis_agent/gym/computer_client.py` | **NEW.** Defines `ComputerClient` marker + `make_computer_client(cfg)` factory. |
| `src/mantis_agent/gym/local_xdotool_impl.py` | **NEW.** Thin wrapper around the existing `XdotoolGymEnv`. Implements `ComputerClient`. |
| `src/mantis_agent/gym/computer_wire.py` | **NEW.** Pydantic models above (used by Phase 1 client; defined now to lock the contract). |
| `src/mantis_agent/config.py` | Add `ComputerPlaneConfig`, default `backend="local"`. |
| `src/mantis_agent/task_loop.py` | Replace direct `XdotoolGymEnv(...)` construction with `make_computer_client(cfg)`. |
| `deploy/modal/modal_cua_server.py` | Pass `cfg.computer_plane` into the executor lifecycle. Default `"local"`. |
| `tests/gym/test_computer_client_factory.py` | **NEW.** Verifies `backend="local"` returns a working `LocalXdotoolImpl` and the existing test suite passes through it. |

### Acceptance

- All existing `XdotoolGymEnv` tests pass via the factory.
- A boattrader smoke plan rerun on Modal completes identically (lead count, cost-per-lead within ±5% of the baseline).
- Production unchanged: no Modal redeploy required beyond the seam PR.

## Phase 1 — Modal computer-plane function

**Goal:** lift Xvfb + Chrome + xdotool into a separate Modal function, accessed via `RemoteComputerImpl`. Same Modal app, same volume.

### File changes

| File | Change |
|---|---|
| `deploy/modal/computer_plane.py` | **NEW.** `@app.function(volumes={"/data": data_volume}, image=computer_image, cpu=2.0)` exposing the `ComputerAgent` FastAPI via `@modal.fastapi_endpoint`. Implements all six endpoints. Manages Xvfb (lazy start on first request, lifetime = session). |
| `deploy/modal/Dockerfile` or `modal.Image` | Brain plane image **loses** Xvfb / Chrome / xdotool layers (they move to `computer_image`). |
| `src/mantis_agent/gym/remote_computer_impl.py` | **NEW.** Implements `ComputerClient` over HTTPS. Generates `step_id`s. Honors timeouts. Catches transient 5xx, retries with the same `step_id`. ~150–250 LOC. |
| `src/mantis_agent/gym/computer_client.py` | Factory dispatches `backend="modal"` → `RemoteComputerImpl(base_url=cfg.modal_computer_plane_url)`. |
| `deploy/modal/modal_cua_server.py` | Env var `COMPUTER_PLANE_URL` resolved per-region from a Modal Dict lookup; passed into `RemoteComputerImpl`. |
| `tests/gym/test_remote_computer_impl.py` | **NEW.** Mock-server tests for retry/dedup/timeout. |
| `tests/integration/test_phase1_e2e.py` | **NEW.** Spins up both Modal functions in a staging app; runs a 5-step plan. |
| `docs/hosting/modal.md` | Update: brain plane vs computer plane, image split, env vars. |

### Migration order (per-executor)

Roll out per-executor behind `ComputerPlaneConfig.per_executor_overrides`:

1. `run_claude_cua` (0 GPU, easiest, biggest cost win from a clean split)
2. `run_gemma4_cua` (T4 planner)
3. `run_fara` and `run_holo3` once the above two are stable
4. EvoCUA / OpenCUA tiers last

Roll back is a config flip; no redeploy required.

### Acceptance

- Per-step p50 latency increase ≤ 20 ms (target: 6–16 ms RT same-region).
- Boattrader plan: leads / cost / runtime within ±10% of pre-split baseline.
- Chrome crash (intentional `kill -9` in the computer-plane container) recoverable without restarting the brain executor.
- Modal logs split cleanly: `modal app logs mantis-cua-server` shows brain; `modal app logs mantis-cua-server --function computer_plane` shows browser.

## Phase 2 — E2B + Daytona backends

**Goal:** make the computer-plane host pluggable. No code changes to step handlers or the brain.

### File changes

| File | Change |
|---|---|
| `deploy/computer-plane/Dockerfile` | **NEW.** Same `ComputerAgent` code as Phase 1, packaged as a generic Linux image (not `modal.Image`). |
| `deploy/computer-plane/e2b.toml` | **NEW.** E2B Desktop sandbox spec referencing the image. |
| `deploy/computer-plane/daytona.toml` | **NEW.** Daytona workspace spec. |
| `src/mantis_agent/gym/remote_e2b_impl.py` | **NEW.** `ComputerClient` impl that drives an E2B Desktop sandbox. Wraps the E2B SDK's mouse/keyboard/screenshot primitives. ~200 LOC. |
| `src/mantis_agent/gym/remote_daytona_impl.py` | **NEW.** `ComputerClient` impl that talks HTTPS to a `ComputerAgent` running inside a Daytona workspace. Mostly the Phase 1 client with a different provisioning path. |
| `src/mantis_agent/observability/profile_snapshotter.py` | **NEW** (gated on Phase 2 actually shipping). Implements tar.zst snapshot + S3 canonical store, with the per-profile lock TTL/renewal semantics. **Only relevant for non-Modal backends.** |
| [`docs/reference/computer-plane-profile-snapshots.md`](computer-plane-profile-snapshots.md) | **LANDED.** Dedicated spec for the snapshot pipeline — gating doc for any Phase 2 host backend that can't mount `osworld-data`. |

### Acceptance

- A boattrader plan family can be routed to E2B via config alone. No code in `src/mantis_agent/gym/step_handlers/` or `src/mantis_agent/brain_*.py` changes.
- A second plan family routed to Daytona simultaneously. Brain dispatches per executor based on `ComputerPlaneConfig.per_plan_overrides`.

## Reliability notes

- **Idempotency.** Discussed under wire contract. `step_id` everywhere. Augur events carry `step_id` so post-hoc reconciliation is possible.
- **Lock TTL.** Phase 1 inherits today's in-process per-profile lock. Phase 2 must move to a TTL'd lock with renewal pings (Modal Dict) because the sandbox can outlive the executor invocation. Spec for that lock lives in the Phase 2 follow-up doc.
- **Observability.** Augur emission stays brain-side and is unaffected. The computer plane emits its own WARNING-level logs (per `feedback_warning_level_for_modal_observability.md`) for any state the brain cannot see (Chrome crash, Xvfb segfault, proxy DNS failure).
- **CUA-purity boundary.** The wire is the enforcement point. CDP endpoints are config-gated per executor (`enable_cdp` defaults false) and never used to derive grounding — `feedback_browser_infra_rpc_surface.md`.

## Open questions

1. **Per-profile lock placement in Phase 2.** Modal Dict vs. Redis vs. file-based. Defer until Phase 2 scope starts.
2. **Screenshot transport.** Base64-in-JSON is simple but ~33% overhead. Multipart binary is faster but uglier on the client. Defer; profile first.
3. **Computer-plane container warm pool.** Should computer-plane functions stay warm pinned to a profile (faster) or be ephemeral per-session (cheaper)? Production data needed.
4. **Cross-region.** Out of scope until a customer asks for it.

## Risks

| Risk | Mitigation |
|---|---|
| Per-step latency tax larger than projected | Phase 1 is rollback-as-config. Benchmark first executor, abort migration if p50 > 20 ms. |
| `step_id` dedup mishandles legit retries | TTL + capacity tuning; surface `deduplicated=true` in Augur events. |
| Brain image regression from removing Chrome layers | Phase 1 builds and validates both images side-by-side before flipping `COMPUTER_PLANE_URL`. |
| Augur event timing skews because env-side is no longer in-process | Add a brain-side `xdotool_dispatched_at_ms` event; reconcile in Augur. |

## References

- `feedback_browser_infra_rpc_surface.md` — CUA-pure RPC surface; computer naming.
- `feedback_cua_no_dom_access.md` — no DOM-derived grounding.
- `feedback_modal_warm_container_caveat.md` — 10-min stale-code window; Phase 1 decouples per side.
- `feedback_modal_new_app_id_per_deploy.md` — log resolution by app name, not id.
- `project_sim_envs_route_ordering.md` — FastAPI route ordering pitfall for `ComputerAgent` `/session/*` vs catch-alls.
