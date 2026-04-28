# Integration: vision_claude → Mantis (Path C, orchestrated)

How vision_claude consumes Mantis without giving up the Xvfb + mounted-browser
desktop it owns today. Side-by-side architecture, sequence flow, deployment
topology, and migration cost.

---

## A. Today — vision_claude with Claude CUA (in-process)

Detailed view of what runs in production today. `agent_loop.py` calls Anthropic Computer Use, the tool dispatch covers `computer_tool` / `bash_tool` / `edit_tool` / `user_input_tool` / `correspondent_message_tool`, and the desktop stack is Xvfb → Chrome → mounted EFS profile. `ExecutionObserver` handles pause / resume / OTP flows.

![Current state — vision_claude with Claude CUA](diagrams/current-state.png)

[Edit in FigJam](https://www.figma.com/board/1MoQ7KJVM0a5GJQCWUTYpH)

**Cost shape:** every screen pixel goes to Anthropic. Claude reasons through the whole plan ("login → find lead → edit industry"). Pricey per task, but reliable on multi-step plans because Claude has the reasoning depth.

---

## B. Path C — orchestrated Mantis (browser stays in vision_claude)

![Proposed state — Path C orchestrated Mantis](diagrams/proposed-state.png)

[Edit in FigJam](https://www.figma.com/board/zp00qbT3Be58laMFlHWomA)

The skeleton matches the current-state diagram one-for-one — the same caller, server, handler, observer, desktop, Xvfb, Chrome, profile, and site sit in the same positions. **Green** highlights what's new or changed in Path C: `MantisOrchestratedBackend`, `MicroPlanRunner`, the three brain/Claude workers, the `VisionClaudeGymEnv` adapter, and the Mantis Holo3 service.

**Cost shape:** Holo3 (cheap GPU) does click/scroll/type. Claude does only
gate-verify + structured extraction + click grounding. Browser stays in
vision_claude, so the mounted profile and pause/resume machinery are
untouched.

---

## C. One micro-step — sequence diagram

![One micro-step — sequence flow (click + extract)](diagrams/sequence-one-step.png)

[Edit in FigJam](https://www.figma.com/board/RDnyMSecbwTtu7W7CEdnZs)

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
