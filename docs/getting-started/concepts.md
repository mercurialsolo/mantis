# Concepts

Once you've done the [Quickstart](quickstart.md), this page tells you what every component actually does. Read it once before designing your own plan.

## The runtime

```
Caller                /v1/predict                 MicroPlanRunner
─────                 ───────────                 ───────────────
                                                  ┌─ BrainHolo3   (Holo3 GPU inference)
plan, tenant ──HTTP──► validate, clamp ──────────►├─ ClaudeGrounding (refine clicks)
                       caps, namespace             ├─ ClaudeExtractor (read structured data)
                       profile_id, workflow_id     ├─ DynamicPlanVerifier
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
| Micro-plan | `micro: "plans/example/...json"` (path) or inline list | High-reliability extraction with sections / gates / loops |
| Plain text | `plan_text: "Extract the first 3 product listings from example.com"` | One-shot ad-hoc; server decomposes via Claude (cached after first call) |

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
| `loop` | Jumps back to step `loop_target` up to `loop_count` times. Optional `stop_var` reads a runner state variable and exits the loop early when truthy. |
| `if_else` | Branches on `runner._state_vars[condition_var]` to `then_target` (truthy) or `else_target` (falsy/missing). Composes with `detect_visible`. Missing var / out-of-range target falls through to the next step. ([#820](https://github.com/mercurialsolo/mantis/pull/820)) |
| `detect_visible` | One Claude/Holo3 yes/no vision call ("is the cookie banner visible?"); writes a bool to `runner._state_vars[out_var]`. Pairs with `if_else` and step-level `guard`. |
| `extract_rows` | Multi-row extraction in one Claude call (top-N from a list page). Same handler as `extract_data`; `extract_data` also takes the multi-row branch when its schema has `max_items > 1`. ([#820](https://github.com/mercurialsolo/mantis/pull/820)) |
| `filter` | Claude finds the filter checkbox and clicks it |
| `fill_field` | Claude finds the labelled input (`params.label`), clears it, types `params.value` |
| `submit` | Claude finds the labelled button / nav-link / row-link (`params.label` + `params.kind`) and left-clicks it |
| `select_option` | Opens a labelled dropdown then picks the named option (`params.dropdown_label` + `params.option_label`) |
| `right_click` | Claude finds the labelled element (`params.label`) and right-clicks to open the native context menu — use for "Open Link in New Tab" / "Copy Link" / app-defined context menus on table rows or grid cells ([#373](https://github.com/mercurialsolo/mantis/issues/373)) |

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
| `stop_var` | (`loop` only) Name of a runner state variable; when truthy, exit the loop early instead of running remaining iterations. |
| `condition_var` | (`if_else` only) Name of a state variable read for the branch decision. |
| `then_target` / `else_target` | (`if_else` only) Absolute step indices to jump to. -1 = fall through to next step. |
| `out_var` | (`detect_visible` only) Name of the state variable that receives the boolean answer. |
| `guard` | Name of a state variable; when falsy, the step is skipped entirely (no vision call, no env action). |

## Tenants and tokens

Mantis is multi-tenant from Tier 1. A *tenant* is just a record in the operator's keys file mapping an `X-Mantis-Token` to:

- `tenant_id` — the namespace prefix used in `profile_id` / `workflow_id` / legacy `state_key` and on the data volume
- `scopes` — which actions this token can do (`run`, `status`, `result`, `logs`)
- `max_concurrent_runs`, `max_cost_per_run`, `max_time_minutes_per_run`, `rate_limit_per_minute` — caps the server enforces in addition to the global hard caps
- `anthropic_secret_name` — which Anthropic key this tenant's runs use (each tenant can bring its own billing)
- `allowed_domains` — wildcards matched against `navigate` URLs in submitted plans
- `webhook_url`, `webhook_secret_name` — optional run-completion callback

Plans submitted by tenant A cannot read tenant B's checkpoints, profiles, or recordings — `profile_id` / `workflow_id` (and legacy `state_key`) are all server-prefixed with the tenant id and the data volume is namespaced.

## `profile_id` + `workflow_id` — the resume primitives (#341)

The two most important per-run fields after the plan itself. They were one field — `state_key` — until [#341](https://github.com/mercurialsolo/mantis/issues/341) split them because they have opposite rotation lifetimes.

| | Field | Behavior |
|---|---|---|
| Browser profile | `profile_id` | A Chrome user-data-dir at `tenants/<tenant_id>/chrome-profile/<profile_id>/` is created or reused. Cookies + logged-in sessions persist across runs. **Sticky** — keep this stable so you don't have to log back in every time the plan changes. |
| Checkpoint | `workflow_id` | The runner saves progress to `tenants/<tenant_id>/checkpoints/<workflow_id>.json`. Pass `resume_state: true` to pick up where the last run with this id left off. **Rotate** when the plan definition changes meaningfully — resuming step `N/12` of an old plan against a new layout is incoherent. |
| Pause-time browser snapshot | `browser_state` | Captured automatically on `PauseRequested` and round-tripped through `PauseState`: current URL, scroll offset, viewport size, and unsubmitted form input. On `action=resume` the agent re-lands on the exact pixel + repopulates half-filled forms. Passwords are masked; missing selectors on the resumed DOM are silently skipped. See [api.md → What's captured at pause time](../api.md#whats-captured-at-pause-time) for the full table. (Epic [#358](https://github.com/mercurialsolo/mantis/issues/358).) |
| Idempotency | (not these) | The `Idempotency-Key` header is the *request* identity (24h dedup); `workflow_id` is the *workflow* identity. |

Pick `profile_id` to match the *account or persona* — e.g. `alice-prod`, `customer-12345`. Pick `workflow_id` to match the *plan revision* — e.g. `marketplace-miami-listings-v3`, the default is `plan_signature[:12]`.

Same account running 5 different workflows in parallel? Pass one `profile_id` and five distinct `workflow_id`s. (Note: Chrome serializes those runs because two processes cannot share a user-data-dir at the same time — distinct `profile_id`s are required for true parallelism. See [#342](https://github.com/mercurialsolo/mantis/issues/342) for the Modal HTTP endpoint that surfaces this as a 409 instead of silent corruption.)

### Legacy `state_key`

The single-field `state_key` still works — when set alone, the server routes it to both `profile_id` and `workflow_id` for back-compat. New code should set the two fields independently. The result envelope echoes all three fields so callers can grep either.

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
