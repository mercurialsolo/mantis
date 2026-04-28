# Integration: vision_claude → Mantis (Path C, orchestrated)

How vision_claude consumes Mantis without giving up the Xvfb + mounted-browser
desktop it owns today. Side-by-side architecture, sequence flow, deployment
topology, and migration cost.

---

## A. Today — vision_claude with Claude CUA (in-process)

```
       ┌──────────────────────────────────────────────────────────────┐
       │ 3p caller (orchestrator / tool_runner / scheduled job)       │
       └──────────────────────────┬───────────────────────────────────┘
                                  │  invoke VisionClaudeTool
                                  ▼
   ╔════════════════════════════════════════════════════════════════════╗
   ║                       vision_claude pod (ECS)                      ║
   ║                                                                    ║
   ║   ┌────────────────────────────────────────────────────────────┐   ║
   ║   │ vision_claude_server (FastAPI / RPC entry)                 │   ║
   ║   └────────────────┬───────────────────────────────────────────┘   ║
   ║                    │                                               ║
   ║                    ▼                                               ║
   ║   ┌────────────────────────────────────────────────────────────┐   ║
   ║   │ ClaudeCUABackend.run_loop()                                │   ║
   ║   │   • full plan as a single Claude prompt                    │   ║
   ║   │   • Anthropic Computer-Use API does perception+reasoning   │   ║
   ║   └─────┬───────────────────────────┬──────────────────────────┘   ║
   ║         │ pyautogui / xdotool       │ HTTPS / image_url blocks     ║
   ║         ▼                           ▼                              ║
   ║   ┌──────────────────┐    ╔═════════════════════════╗              ║
   ║   │  ActionExecutor  │    ║  Anthropic API          ║              ║
   ║   │  ScreenStreamer  │    ║  (claude-sonnet-4)      ║              ║
   ║   └────────┬─────────┘    ╚═════════════════════════╝              ║
   ║            │                                                       ║
   ║            ▼                                                       ║
   ║   ┌────────────────────────────────────────────────────────────┐   ║
   ║   │  Xvfb display  ← Chrome (mounted profile, cookies, sessions)  ║
   ║   └────────────────────────────────────────────────────────────┘   ║
   ║            │                                                       ║
   ║            ▼                                                       ║
   ║   ┌──────────────────┐                                             ║
   ║   │  EFS volume:     │                                             ║
   ║   │  Chrome profile  │                                             ║
   ║   └──────────────────┘                                             ║
   ╚════════════════════════════════════════════════════════════════════╝
                                  │
                                  ▼
                         Target site (StaffAI CRM, …)
```

**Cost shape:** every screen pixel goes to Anthropic. Claude reasons through the
whole plan ("login → find lead → edit industry"). Pricey per task, but
reliable on multi-step plans because Claude has the reasoning depth.

---

## B. Path C — orchestrated Mantis (browser stays in vision_claude)

```
       ┌──────────────────────────────────────────────────────────────┐
       │ 3p caller (unchanged)                                        │
       └──────────────────────────┬───────────────────────────────────┘
                                  │  invoke VisionClaudeTool
                                  ▼
   ╔════════════════════════════════════════════════════════════════════╗
   ║                vision_claude pod (ECS, no GPU needed)              ║
   ║                                                                    ║
   ║   ┌────────────────────────────────────────────────────────────┐   ║
   ║   │ vision_claude_server                                       │   ║
   ║   └────────────────┬───────────────────────────────────────────┘   ║
   ║                    ▼                                               ║
   ║   ┌────────────────────────────────────────────────────────────┐   ║
   ║   │ MantisOrchestratedBackend.run_loop()  (rewrite of          │   ║
   ║   │  mantis_backend.py — uses MicroPlanRunner from the         │   ║
   ║   │  imported `mantis-agent` library)                          │   ║
   ║   └────────────────┬───────────────────────────────────────────┘   ║
   ║                    ▼                                               ║
   ║   ┌────────────────────────────────────────────────────────────┐   ║
   ║   │  MicroPlanRunner (imported, runs in-process)               │   ║
   ║   │   • sections / gates / loops / scroll-fail-fallback        │   ║
   ║   │   • per-step checkpoint                                    │   ║
   ║   │                                                            │   ║
   ║   │   ┌──────────┐  ┌──────────────┐  ┌────────────────────┐   │   ║
   ║   │   │ BrainHolo3│  │ ClaudeExtractor│ ClaudeGrounding    │   │   ║
   ║   │   │ Remote    │  │ (extract data) │ (refine click x,y) │   │   ║
   ║   │   └─────┬─────┘  └──────┬───────┘ └──────┬─────────────┘   │   ║
   ║   └─────────┼────────────────┼────────────────┼─────────────────┘  ║
   ║             │ HTTPS          │ HTTPS          │ HTTPS              ║
   ║             ▼                ▼                ▼                    ║
   ║      (see Mantis svc)   ╔═══════════════════════════════╗          ║
   ║                         ║ Anthropic API (gates,         ║          ║
   ║                         ║   extract, grounding only —   ║          ║
   ║                         ║   not perception)             ║          ║
   ║                         ╚═══════════════════════════════╝          ║
   ║                                                                    ║
   ║   ┌────────────────────────────────────────────────────────────┐   ║
   ║   │ VisionClaudeGymEnv  (new ~120 LoC adapter implementing     │   ║
   ║   │   mantis_agent.gym.GymEnvironment over the existing        │   ║
   ║   │   desktop.py / computer_tool.py)                           │   ║
   ║   └────────────────┬───────────────────────────────────────────┘   ║
   ║                    │ pyautogui / xdotool / screenshots              ║
   ║                    ▼                                               ║
   ║   ┌────────────────────────────────────────────────────────────┐   ║
   ║   │  Xvfb display  ← Chrome (mounted profile, cookies, sessions)  ║
   ║   └────────────────────────────────────────────────────────────┘   ║
   ║            │                                                       ║
   ║            ▼                                                       ║
   ║   ┌──────────────────┐                                             ║
   ║   │  EFS volume:     │                                             ║
   ║   │  Chrome profile  │ ← unchanged                                 ║
   ║   └──────────────────┘                                             ║
   ╚════════════════════════════════════════════════════════════════════╝

           ╔════════════════════════════════════════════════════════╗
           ║         Mantis Holo3 service (Baseten / EKS / Modal)   ║
           ║                                                        ║
           ║  ┌────────────────────────────────────────────────┐    ║
           ║  │ FastAPI (baseten_server.py)                    │    ║
           ║  │  • /predict             — full orchestrator    │    ║
           ║  │  • /v1/chat/completions — Holo3 inference proxy│    ║
           ║  │  • /v1/models           — list models          │    ║
           ║  │  • auth: X-Mantis-Token + Api-Key gateway      │    ║
           ║  └────────────────────┬───────────────────────────┘    ║
           ║                       ▼                                ║
           ║  ┌────────────────────────────────────────────────┐    ║
           ║  │ llama.cpp server (port :18080, internal only)  │    ║
           ║  │  • Holo3-35B-A3B GGUF Q8_0                     │    ║
           ║  │  • CUDA on H100 / A100 / L40S                  │    ║
           ║  └────────────────────────────────────────────────┘    ║
           ╚════════════════════════════════════════════════════════╝
```

**Cost shape:** Holo3 (cheap GPU) does click/scroll/type. Claude does only
gate-verify + structured extraction + click grounding. Browser stays in
vision_claude, so the mounted profile and pause/resume machinery are
untouched.

---

## C. One micro-step — sequence diagram

```
  vision_claude pod                         Mantis service       Anthropic
  ─────────────────                         ──────────────       ─────────
                                                  │                  │
  MicroPlanRunner.next_step()                     │                  │
        │                                         │                  │
        │ env.screenshot() ──────────────┐        │                  │
        │                                ▼        │                  │
        │                       Xvfb + Chrome     │                  │
        │                                ▲        │                  │
        │ ◄──────────────  PIL.Image  ───┘        │                  │
        │                                         │                  │
   step.type == "click"  (grounding=True)         │                  │
        │                                         │                  │
        │ BrainHolo3Remote.think(frame, intent) ──►──────────────────┐
        │                                         │  POST /v1/chat   │
        │                                         │    /completions  │
        │                                         ▼                  │
        │                                 llama.cpp(Holo3) ──────────►
        │                                         │   action proposal│
        │ ◄────────  Action(CLICK, {x:982, y:183}) ◄─────────────────┘
        │                                                            │
        │                                                            │
        │ ClaudeGrounding.refine(frame, action) ─────────────────────►
        │                                              POST /messages│
        │ ◄────────  Action(CLICK, {x:976, y:188})  ◄────────────────┘
        │                                                            │
        │ env.step(action) ──────────────┐                           │
        │                                ▼                           │
        │                        xdotool click 976 188               │
        │                       (in vision_claude pod)               │
        │                                                            │
                                                                     │
   step.type == "extract_data"  (claude_only=True)                   │
        │                                                            │
        │ env.screenshot() → frame                                   │
        │                                                            │
        │ ClaudeExtractor.extract(frame, schema) ────────────────────►
        │                                              POST /messages│
        │ ◄──── ExtractionResult(year, make, phone, …) ◄─────────────┘
        │                                                            │
        │ runner accumulates lead → checkpoint → next step           │
```

Note the screenshot bytes flow:
- **Holo3 path:** screenshot bytes → Mantis service (own Baseten/EKS).
  Image leaves vision_claude only to your own deployment.
- **Claude path:** screenshot bytes → Anthropic. Same as today.
- **Mantis service never sees** the Anthropic key, the StaffAI cookies,
  the EFS profile, or any per-tenant secrets.

---

## D. Reliability comparison

| Workflow | Claude-only CUA today           | Path C (Mantis orchestrated)                |
|----------|---------------------------------|---------------------------------------------|
| 1-step task ("logout") | ✅ very reliable      | ✅ reliable (single click step)              |
| Multi-step ("login → filter → export") | ✅ Claude reasons through | ✅ MicroPlanRunner enforces section+gate semantics |
| Form-heavy ("update 12 fields, submit")| ⚠️ drift on long context | ✅ each field is its own claude_only extract step |
| Pagination loop                | ⚠️ Claude can lose count    | ✅ explicit `loop_count` + `paginate` step types |
| OTP / human-in-the-loop pause  | ✅ first-class               | ✅ unchanged — loop is local, pause hooks intact |

The reliability win is the **structured plan + per-step verification**, not
the model. Holo3 alone fails on multi-step plans; Holo3 inside MicroPlanRunner
matches Claude's success rate at a fraction of the cost.

---

## E. Deployment topology

```
                  ┌────────────────────────────────────────────────┐
                  │  AWS account (or GCP project)                  │
                  │                                                │
                  │  ┌──────────────────────────────────────────┐  │
                  │  │  vision_claude task                      │  │
                  │  │  • c6i.2xlarge (no GPU)                  │  │
                  │  │  • Xvfb + Chrome + persisted profile     │  │
                  │  │  • runs MicroPlanRunner in-process       │  │
                  │  │  • per-tenant isolation                  │  │
                  │  └──────────────┬───────────────────────────┘  │
                  │                 │ HTTPS                         │
                  │                 ▼                               │
                  │  ┌──────────────────────────────────────────┐  │
                  │  │  Mantis Holo3 service                    │  │
                  │  │  • Baseten H100  (already deployed)      │  │
                  │  │  • OR EKS g6e.2xlarge / GKE a2-highgpu   │  │
                  │  │  • autoscale 0..N                        │  │
                  │  │  • shared across tenants                 │  │
                  │  │  • only sees screenshot bytes            │  │
                  │  └──────────────────────────────────────────┘  │
                  │                                                │
                  └────────────────────────────────────────────────┘

                                 │ HTTPS
                                 ▼
                       Anthropic API (gates, extract, grounding)
```

**vision_claude task scales per tenant** (or per concurrent session).
**Mantis Holo3 scales by inference throughput** — one shared deployment
serves all tenants. GPU cost amortizes across customers.

---

## F. What blocks adoption

| Blocker | Resolution |
|---------|------------|
| `mantis_agent` import surface today is monolithic (drags in xdotool, playwright, etc. via __init__.py) | Add `[orchestrator]` extras in `pyproject.toml` that pin only the runner + brain + extraction + grounding deps. ~30 LoC. |
| `BrainHolo3` doesn't carry the `X-Mantis-Token` + `Api-Key` headers natively | New `BrainHolo3Remote(extra_headers=…)` subclass. ~30 LoC. |
| Plan management — vision_claude callers send Claude messages today, not micro-plans | Two on-ramps:  (a) **plan_decomposer** auto-decomposes plain English into a micro-plan ($0.10 each, cached);  (b) hand-author a JSON micro-plan per recurring StaffAI workflow. |
| `VisionClaudeGymEnv` doesn't exist yet | New ~120 LoC adapter wrapping `desktop.py` + `computer_tool.py` to satisfy `mantis_agent.gym.GymEnvironment`. |

Total work to ship: ~290 LoC in cua-agent, ~335 LoC in vision_claude, plus
~150 LoC of docs (this file).

---

## G. Migration phases

```
Phase 1 — cua-agent ships
  ├─ /v1/chat/completions proxy on Baseten endpoint
  ├─ BrainHolo3Remote (auth-aware client)
  ├─ [orchestrator] extras
  └─ docs/integration-vision_claude.md  ← this file

Phase 2 — vision_claude integrates (canary)
  ├─ VisionClaudeGymEnv
  ├─ MantisOrchestratedBackend
  ├─ env-flag gate: VISION_CLAUDE_CUA_BACKEND=mantis-orchestrated
  └─ canary on one tenant / one workflow

Phase 3 — fleet rollout
  ├─ tenant-by-tenant flip
  ├─ track {success_rate, latency, cost_per_task} vs Claude baseline
  └─ default-flip when 30-day metrics match-or-beat

Phase 4 — decommission
  ├─ remove ClaudeCUABackend if no fallback callers
  ├─ drop GPU-backed vision_claude task spec
  └─ retire agent_loop.py + Anthropic Computer Use plumbing
```

Each phase is independently revertable — env-var flip restores Claude
backend with zero data migration.
