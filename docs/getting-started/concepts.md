# Concepts

Once you've done the [Quickstart](quickstart.md), this page tells you what every component actually does. Read it once before designing your own plan.

## The runtime

```
Caller                /v1/predict                 MicroPlanRunner
─────                 ───────────                 ───────────────
                                                  ┌─ BrainHolo3   (Holo3 GPU inference)
plan, tenant ──HTTP──► validate, clamp ──────────►├─ ClaudeGrounding (refine clicks)
                       caps, namespace             ├─ ClaudeExtractor (read structured data)
                       state_key                   ├─ DynamicPlanVerifier
                                                   └─ XdotoolGymEnv (Xvfb + Chrome)
                                                          │
                                                          ▼
                                                     Target site
```

| Component | Purpose | Where it runs |
|---|---|---|
| **Holo3-35B-A3B** (GGUF) | Tactical click / scroll / type / drag actions | Single GPU per Mantis pod |
| **Claude (Anthropic API)** | Strategic gate verification, structured data extraction, click coordinate refinement | Anthropic cloud, called per surgical step |
| **MicroPlanRunner** | Section/gate/loop state machine, per-step retry, checkpointing | Mantis pod, in-process |
| **XdotoolGymEnv** | Real Chrome inside Xvfb + xdotool for fingerprint-free clicks | Mantis pod |
| **IPRoyal proxy** | Residential, geo-targeted egress for sites with bot detection | Mantis pod (sticky session per run) |

## Plans

A plan is a structured description of what the agent should do. Three shapes are supported, in priority order on `/v1/predict`:

| Shape | Field | When to use |
|---|---|---|
| Inline task suite | `task_suite: { ... }` | You have arbitrary task data and don't want to bake it into the container image |
| Pre-baked path | `task_file: "tasks/crm/crm_tasks.json"` | The plan ships in the container image |
| Micro-plan | `micro: "plans/boattrader/...json"` (path) or inline list | High-reliability extraction with sections / gates / loops |
| Plain text | `plan_text: "Extract the first 3 boat listings from BoatTrader Miami"` | One-shot ad-hoc; server decomposes via Claude (cached after first call) |

See [Plan formats](plan-formats.md) for full schemas and examples.

## Step types (micro-plan shape)

Each step in a micro-plan is a JSON object with a `type` and an `intent`:

| `type` | What the runner does |
|---|---|
| `navigate` | Loads `intent`'s URL via env.reset(); waits for Cloudflare; sets the proxy |
| `click` | Fresh Holo3 inference loop with `budget` actions; optionally refined by `ClaudeGrounding` |
| `scroll` | Holo3 scrolls until intent satisfied; `scroll-fail-as-success` fallback |
| `extract_url` | Reads the address bar via Claude — no Holo3 |
| `extract_data` | Claude reads the screenshot and emits structured fields per the schema |
| `navigate_back` | Alt+Left + verify URL change |
| `paginate` | URL-based or grounded click on the Next button |
| `loop` | Jumps back to step `loop_target` up to `loop_count` times |
| `filter` | Claude finds the filter checkbox and clicks it |

Useful per-step modifiers:

| Field | Effect |
|---|---|
| `section` | One of `setup`, `extraction`, `pagination`. Used by retry/halt logic. |
| `required` | If true: retry on fail, then halt the whole run. |
| `gate` | Claude verifies a condition; halt the run on fail. |
| `verify` | Free-text condition Claude checks. |
| `claude_only` | Skip Holo3 entirely; Claude does the perception. Use for extract / gate steps. |
| `grounding` | Refine Holo3's click coordinates with `ClaudeGrounding`. |
| `budget` | Max actions Holo3 can take in this step (default 8). |
| `loop_target` | Step index to jump back to (only on `loop` steps). |
| `loop_count` | Max loop iterations; clamped to `MANTIS_MAX_LOOP_ITERATIONS`. |

## Tenants and tokens

Mantis is multi-tenant from Tier 1. A *tenant* is just a record in the operator's keys file mapping an `X-Mantis-Token` to:

- `tenant_id` — the namespace prefix used in `state_key` and on the data volume
- `scopes` — which actions this token can do (`run`, `status`, `result`, `logs`)
- `max_concurrent_runs`, `max_cost_per_run`, `max_time_minutes_per_run`, `rate_limit_per_minute` — caps the server enforces in addition to the global hard caps
- `anthropic_secret_name` — which Anthropic key this tenant's runs use (each tenant can bring its own billing)
- `allowed_domains` — wildcards matched against `navigate` URLs in submitted plans
- `webhook_url`, `webhook_secret_name` — optional run-completion callback

Plans submitted by tenant A cannot read tenant B's checkpoints, profiles, or recordings — `state_key` is server-prefixed with the tenant id and the data volume is namespaced.

## state_key — the resume primitive

`state_key` is the most important per-run field after the plan itself. It controls:

| | Behavior |
|---|---|
| Browser profile | A Chrome profile dir at `tenants/<tenant_id>/chrome-profile/<state_key>/` is created or reused. Cookies + sessions persist across runs with the same key. |
| Checkpoint resume | The runner saves progress to `tenants/<tenant_id>/checkpoints/<state_key>.json`. Pass `resume_state: true` to pick up where the last run left off. |
| Idempotency | (Not the same as `Idempotency-Key` header) — `state_key` is the *workflow* identity, the header is the *request* identity. |

Pick `state_key` to match the conceptual workflow: `boattrader-miami-private-v1`, `crm-prod`, `customer-12345-onboarding`. Reuse the same key across runs of the same workflow; pick a new key when the workflow definition changes.

## The cost meter

Every run produces a cost breakdown:

```jsonc
{
  "summary": {
    "cost_total": 0.42,
    "cost_breakdown": {
      "gpu":    0.12,    // Holo3 GPU minutes
      "claude": 0.12,    // Anthropic API tokens
      "proxy":  0.18     // IPRoyal residential proxy bandwidth
    }
  }
}
```

Caps are enforced server-side: `max_cost` (default $25, env-overridable to global, per-tenant clamped) is the wall before the runner halts. `max_time_minutes` is the wall-clock cap.

## Polished recording

When `record_video: true`, every run produces both a raw screencast and a polished walkthrough composed by ffmpeg:

```
title card · 3s   →   captioned run · per-step + per-action overlays   →   outro card · 5s
```

The polished version is what `GET /v1/runs/<id>/video` returns by default; pass `?raw=1` for the screen capture without overlays.

Action overlays include click ripples, keyboard chord badges (`CTRL + S`), scroll arrows, type captions, and drag trails. They render the same regardless of *what* the agent clicks (browser, file manager, terminal, dialog) because all communication is in pixels.

## What's NOT in scope

- **Browser automation via DevTools / CDP** — Mantis uses Xvfb + xdotool specifically to avoid the fingerprints CDP leaves. If you need DOM-level access, a different tool is the right fit.
- **Headless mode** — the agent watches a real Xvfb display so the model sees what a human would.
- **Audio / video stream as input** — Mantis takes screenshots, not arbitrary media.

## Next

- [Plan formats](plan-formats.md) — write your first plan
- [Hosting](../hosting/index.md) — deploy your own instance
- [Client](../client/index.md) — integrate from your app
