# Computer Plane as a Product — strategic spec

**Status:** Proposal / exploration (NOT an implementation commitment)
**Audience:** Internal product + engineering leadership
**Not a GitHub issue.** This is a strategy document; tracking via issue would imply build-intent that hasn't been authorized.

## One-line thesis

There is real, growing market demand for a **CUA-pure remote computer** that bundles compute + a programmable surface + browser + xdotool primitives + live streaming + human takeover + agent-shaped observability — and current vendors each address only some of those pillars. The combined product has a viable wedge against incumbents (E2B Desktop, Scrapybara, Daytona) and against browser-cloud players (Browserbase, Steel, Hyperbrowser) that are stretching upward into computer-use territory.

This document spells out what such a product would actually be, why none of the existing players are doing exactly this, and what would be required to validate that an external buyer exists before any build commitment.

---

## Why now

CUA-style agents are exiting the "demo" phase. OpenAI Operator, Claude Computer Use, Anthropic's agents, Manus, Devin-style coding agents, and a long tail of vertical products (RPA replacement, customer-support automation, QA testing, lead generation) all need a remote computer to drive. The buyer profile is consolidating:

1. **Agent builders** who don't want to operate Xvfb / Chrome / proxy stealth themselves.
2. **Enterprise CUA adopters** who need per-end-user identity, audit trails, and human-in-the-loop fallbacks.
3. **Researchers and evals teams** who run CUA benchmarks (OSWorld, WebArena, VWA, ScreenSpot) and need reproducible computer environments.

The first wave of vendors (E2B, Scrapybara) treat computer-use as an extension of their existing platform (E2B's code interpreter; Scrapybara's Anthropic-CU compatibility play). Neither is purpose-built for the bundle described below. Browser-cloud incumbents (Browserbase, Steel) are shape-mismatched — they expose CDP / DOM and bill against a browser session, not a computer.

The window where a purpose-built entrant can establish a position is the next 12–18 months. After that, expect consolidation (E2B-as-Lambda, Scrapybara-or-successor-as-Vercel) and category lock-in.

---

## Market gap — pillar-by-pillar

| Pillar | E2B Desktop | Scrapybara | Daytona | Browserbase/Steel | Gap? |
|---|---|---|---|---|---|
| Compute primitive (sandbox + display) | yes (Firecracker) | yes (Ubuntu) | yes (general) | browser only | covered |
| xdotool-shaped action API | yes | yes (polymorphic) | bring-your-own | no (CDP) | covered |
| Streaming live view | yes (VNC) | yes (recording) | partial | yes | covered |
| Built-in browser bundled | bring your own | yes | bring your own | the product | mostly covered |
| **First-class stealth** (CF / Turnstile / PerimeterX survivability) | "bring your own" | provider-managed (residential proxies) | none | provider-managed | **open** |
| **Per-end-user identity / profile fleet** | filesystem snapshots | auth-state save/restore (cookies only) | bring your own | "contexts" (browser only) | **open** |
| **Agent-shaped observability** (intent ↔ outcome, per-step diff, off-rails moment) | dashboard + logs | session replay | logs | session replay | **open** |
| **Human takeover with state continuity** | possible via stream | hinted | possible | yes (Browserbase has it for browsers) | **partly open** |
| **Suspend / resume warm state** | filesystem snapshot | session-level pause | none | session-level | **partly open** |
| **CUA-purity (no DOM API by design)** | n/a (general) | optional | n/a | inverse — DOM is the product | **open as positioning** |
| **GPU-attached desktops** | no | no | possible if you build | no | **open** |
| **Multi-app / multi-screen choreography** | possible | Chrome-centric | possible | no | **partly open** |

The "open" rows define the wedge. A product that closes three or more of those gaps coherently is differentiated; closing one is a feature, not a product.

---

## Product principles

The following are not nice-to-haves — they are the bet.

### 1. CUA-pure abstraction is the contract.

The API exposes `screenshot` + `xdotool(argv)` as the required surface, with `cdp_evaluate` / `cdp_click_at_point` as opt-in escape hatches. No `click_by_selector`, no `query_dom`, no `read_text`. This is the same constraint Mantis has internally; making it the marketed boundary is a counter-positioning move against Stagehand / Anchor / agent-flavored browser products that have been burned by brittle DOM tricks.

**Why it matters strategically:** the customer cohort that has already lost time to DOM-coupled automation is large and growing. "Our API does not let your agent cheat" is a sticky promise.

### 2. Stealth posture is a product surface, not a side-effect.

Most vendors say "we have stealth." None compete on it. Mantis has accumulated enough field experience (`feedback_headless_vs_xvfb.md`, `feedback_proxy_provider.md`, `feedback_proxy_ip_blanket_ban_diagnosis.md`, `project_stealth_parity_pr550_verification.md`) to make this a real position.

Concretely: shipped product would expose stealth posture as configurable knobs (residential vs DC proxy, headed-Xvfb fingerprint vs canonical-Chrome, geo-pinning per-profile, Turnstile-survivability tier) and publish a periodic pass-rate dashboard against a portfolio of CF/Turnstile/PerimeterX test sites. The dashboard becomes the marketing artifact.

### 3. Identity is a first-class object, not a folder.

Every other vendor models "session state" or "auth state." The right abstraction is `Profile` — a long-lived identity bound to a real-or-synthetic end user, with cookies + IndexedDB + localStorage + browser history + extension state, plus metadata (geo, language, persona, last-used-at, trust-score), plus a per-profile lock that prevents concurrent use, plus snapshot history and restore.

**Why it matters:** enterprise CUA at scale runs not 1 profile but 10K — one per end-customer. Identity fleet management at that scale is what nobody has shipped well.

### 4. Observability is event-shaped and agent-aware.

The product emits, per action: `intent` (what the agent meant to do, supplied by the caller), `dispatch` (the xdotool argv issued), `outcome` (screen diff, scrollY change, URL change, error class), `latency`, `dedup_status`, and a `step_id` correlation key. These are queryable as a time-series and replayable as a video alongside the agent's reasoning trace.

This is what Augur is internally. The product surface is "Augur for the rest of you" — minus mantis-specific schemas.

### 5. Human takeover is a first-class flow with state continuity.

A live customer is mid-purchase, hits a 3DS challenge, the agent invokes `escalate_to_human(reason="3ds_challenge", payload=...)`. The platform routes the session to an operator UI; the operator completes the challenge; the platform returns control to the agent with the original task context restored.

State continuity here means: the same Chrome, the same `step_id` thread, the same Augur correlation, no profile mutation surprises. Built once, it's a moat — switching costs are high.

### 6. Suspend / resume is the cost story.

Agents are bursty. Active for 30s, idle for 30 minutes, active for 5s. Per-second or per-hour billing is wrong for both ends. The platform snapshots the X session + browser memory image + profile delta on idle, resumes in <1s on next activity, bills only for active CPU-time + storage. Firecracker memory snapshotting is the primitive; nobody has productized it for desktops yet.

This is the only pillar where E2B has a substrate advantage. Catching up means either licensing/partnering on Firecracker or owning a comparable virtualization layer.

### 7. CUA-purity does not preclude programmability.

The product exposes:
- **Computer API** — screenshot + xdotool + opt-in CDP (the runtime surface).
- **Provisioning API** — create profile, snapshot, restore, list, lock, geo-pin (the identity surface).
- **Observability API** — query events, fetch replays, subscribe to streams (the post-hoc surface).
- **Operator UI** — for human-takeover, debugging replays, dashboard reading.

These are four distinct surfaces. Conflating them — as Anchor and Stagehand do with "the agent surface" — is what makes those products brittle.

### 8. Multi-app is the long-term position.

Today: Chrome. Tomorrow: Chrome + a file manager + a terminal + a native client app the customer ships into the image. The X-display abstraction makes this free if you commit to it early. Vendors that bind tight to "Chrome" foreclose this future.

---

## Architecture sketch

```
                         Customers (agent builders)
                                    │
                       Computer API + Provisioning + Observability + Operator UI
                                    │
                  ┌─────────────────┴─────────────────┐
                  │            Control plane           │
                  │  ──────────────────────────────    │
                  │   Profile registry + lock service  │
                  │   Session scheduler + region pin   │
                  │   Stealth-tier router              │
                  │   Observability ingest (Augur-like)│
                  │   Operator UI (takeover)           │
                  │   Billing meter (active-CPU)       │
                  └─────────────────┬─────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │            Compute plane           │
                  │   Firecracker/KVM micro-VMs        │
                  │   Per-tenant network namespaces    │
                  │   Xvfb + headed Chrome + xdotool   │
                  │   Optional GPU attach              │
                  │   Snapshot/resume engine           │
                  │   Per-profile mount (block dev)    │
                  └─────────────────┬─────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │            Data plane              │
                  │   Profile object store (canonical) │
                  │   Snapshot CAS (content-addressed) │
                  │   Event store (time-series, ~30d)  │
                  │   Replay video store (~7d, tiered) │
                  └────────────────────────────────────┘
```

### Key technical choices

- **Substrate:** Firecracker microVMs (or comparable lightweight virtualization). Heavier than containers; necessary for fast snapshot/resume and tenant isolation.
- **Profile storage:** content-addressed block snapshots (tar.zst + sha256 + `latest` pointer protocol from the internal computer-plane spec, generalized). Profiles are tenanted at the API level, isolated at the network and FS level.
- **Stealth fingerprint:** the compute-plane image enforces canonical Xvfb + headed Chrome + window-size canonicalization + WebGL / canvas / font fingerprint discipline. Proxy injection is mandatory; geo-pin is a profile attribute.
- **Network egress:** through a managed proxy fleet (multi-provider — PrivateProxy class for residential, datacenter pool for low-cost). Stealth tier = proxy class + browser fingerprint set + behavioral pacing.
- **Observability ingest:** event schema borrowed from Augur, generalized. ClickHouse or comparable time-series. Replay videos rendered from screenshots + dispatch events on demand (cheaper than streaming full video to storage).
- **Human takeover:** noVNC + websockify (mirrors the path in #695). Operator UI is a thin SPA that connects to the live session, exposes the intent context, and provides "release back to agent" with a state-continuity token.

---

## What this product is NOT

These are explicit anti-scope statements to keep the wedge sharp.

- **Not a browser API.** Anyone wanting `page.click("#search-button")` should use Browserbase / Steel. The product refuses to ship DOM verbs.
- **Not an agent.** No bundled LLM, no "give it a goal and it does the thing." The product is the operating-system-shaped substrate; the agent is the customer's IP.
- **Not a code-execution sandbox.** E2B owns that category. Code execution is an opt-in side-feature, not a positioning pillar.
- **Not a VDI.** Cloud workstations for human knowledge workers is a different buyer with a different SLA. Optimizing for ephemeral agent sessions makes the product worse for VDI buyers and vice versa.
- **Not Anthropic-CU-flavored.** Compatibility with Anthropic's reference API is a bonus, not a constraint. The wire is `screenshot + xdotool(argv)` — Anthropic-CU translates trivially; vice versa works too. The product is not anchored to any one LLM provider.

---

## GTM hypothesis

**Initial wedge:** developer-product motion targeting other CUA agent builders. Free tier with metered usage; teams plan adds identity-fleet, stealth tiers, and operator UI.

**Buyer profile (12-month):**
- ICP1: small-to-mid agent companies (10–100 person engineering) currently self-hosting Anthropic's reference image, getting beaten by CF challenges, and spending engineering time on profile orchestration.
- ICP2: enterprise CUA pilots — internal tools / RPA-replacement projects at Fortune-500 companies — that need audit logs and human-takeover for compliance reasons.
- Not ICP yet: pure scraping operations (price-conscious, sophisticated stealth ops in-house); pure benchmark / eval users (already on OSWorld stacks).

**Pricing model directional:**
- Per-second active CPU + per-GB-month profile storage + per-event observability ingestion.
- Stealth tier (residential proxy class) is a multiplier on per-second rate.
- Human-takeover sessions billed separately (operator-seat tier).
- Suspend/resume amortizes idle-time cost — the cost story is "you pay for actions, not for waiting."

**Distribution:**
- SDK in Python + TypeScript with one-call session start. Imitate E2B's developer onboarding because that is the bar.
- Reference integrations with: Anthropic Claude (computer use), OpenAI (operator), open Holo3-class models, an "import an agent loop" tutorial.
- Stealth pass-rate dashboard as public marketing artifact.

---

## Minimum viable product (12-week scope, hypothetical)

To validate buyer demand without committing to the full platform:

1. **Weeks 1–3:** single-region compute-plane (Modal-hosted or Firecracker on a small fleet). One profile per session, no fleet yet. Wire = `screenshot` + `xdotool` + opt-in `cdp`. Python SDK only.
2. **Weeks 4–6:** profile registry + per-profile lock + snapshot/restore via content-addressed object store. No fleet operations UI yet.
3. **Weeks 7–9:** observability event ingest (Augur-shaped). Streaming live view via noVNC. Operator UI shell (read-only).
4. **Weeks 10–12:** human takeover (state-continuity, full RFB input). Stealth tier flag (canonical headed-Xvfb vs. canonical Chrome). Public pass-rate dashboard against a small portfolio (3 target sites).

**Acceptance criteria for "buyer demand validated":**
- 5+ external paying design partners using the API in production.
- Stealth pass-rate ≥ Mantis's internal numbers on the same site portfolio.
- Per-session p95 latency within parity of Scrapybara on a 30-step Anthropic-CU benchmark task.
- At least one customer who voluntarily writes a public case study about identity-fleet management.

---

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Bandwidth split with Mantis agent product | **high** | Treat as a separate company-shaped bet from day one. Do not staff with Mantis core engineering until the MVP customer signal is real. |
| E2B or Scrapybara closes the gap faster | medium | Identity + stealth + observability is a three-pillar bundle. Single-pillar response from incumbents leaves the rest of the wedge open. |
| Stealth is a cat-and-mouse cost center | medium | Productize as a moat, not a deliverable. Bake it into pricing tiers; never sell "100% pass rate" as a guarantee. |
| Firecracker / virtualization is the wrong substrate | medium | Defer the substrate bet until MVP customer data justifies it. MVP can run on Modal or Daytona-shaped infra. |
| Compliance / SOC2 / multi-tenant security is expensive | high | Required for enterprise ICP2 buyer; ICP1 buyer can use without it. Sequence: ICP1 first, SOC2 after revenue validates the bet. |
| The "CUA-pure" marketing position confuses customers used to high-level browser verbs | low | Counter-positioning by definition splits the market. Write the docs against the confusion. |
| Buyer education cost is high (agent companies don't yet know they need identity fleet management) | medium | This is the wedge — being early. Mitigate via design-partner program: 5 customers shaping the product, paid for usage. |

---

## Open strategic questions

1. **Build inside Mason or spin out.** A separate company-shaped bet implies a fundraise; building inside Mason means Mantis pays the bandwidth tax. The right answer depends on whether the product accelerates Mantis (because shared infra is shared learning) or competes with it (because external customers will ask for features Mantis won't prioritize internally).
2. **Anthropic relationship.** The product is Anthropic-CU-compatible by construction but agnostic. Strategically, is Anthropic a partner (resell channel, joint customers) or an eventual competitor (they could ship a managed compute plane themselves)?
3. **Open-source seed vs. closed core.** Open-sourcing the wire contract + a reference `ComputerAgent` image lowers buyer-adoption friction and constrains competitors' shapes (helps lock in the screenshot+xdotool contract as the category standard). Open-sourcing the control plane gives the moat away.
4. **The 12-month verticalization question.** If buyer demand concentrates in one vertical (e.g. customer support automation, or e-commerce QA), should the product verticalize or stay horizontal? Horizontal is harder; vertical is faster revenue but ceilings sooner.

---

## How to use this document

This is **strategy**, not a build plan. If the answer to "should we build this?" is yes, the next artifact is:

1. A design-partner program brief (3 ICP1 buyers, 6-week commitment, paid for usage during MVP).
2. A technical hiring plan distinct from Mantis core team.
3. A revised version of this document with `Status: Approved`, an owner, and a budget.

Until those exist, this document is a position paper, not a roadmap.

## References

- Internal architecture: `docs/reference/computer-plane.md` (the Mantis-internal split spec that lays the groundwork; this product proposal generalizes that surface).
- CUA-pure RPC constraint: `feedback_browser_infra_rpc_surface.md` (auto-memory).
- Stealth posture experience base: `feedback_headless_vs_xvfb.md`, `project_stealth_parity_pr550_verification.md`, `feedback_proxy_ip_blanket_ban_diagnosis.md`.
- Observability schema base: `src/mantis_agent/observability/augur.py`.
- Live-viewer / takeover groundwork: GitHub issue #695 (noVNC + websockify).
