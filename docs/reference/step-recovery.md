# Step recovery

When a required step fails after its retry budget is exhausted, Mantis runs a multi-layer recovery chain before halting. This page documents the layers, the gates that bound them, and the env vars that tune them.

The recovery surface sits inside `MicroPlanRunner` — it doesn't apply to legacy `GymRunner.run()` (OSWorld-style benchmarks, `cua_model=claude` direct path). Plan-driven runs (`/v1/predict` with `micro: …`, `task_suite: {_micro_plan: …}`, or `plan_text: …`) all flow through this chain.

## Layers, in firing order

```
step fails (required:true, retry budget = max_retries=2 exhausted)
   │
   ├─ Layer 1: per-step handler escalation
   │   (form → Holo3StepHandler when ≥2 no_state_change demotes)
   │
   ├─ Layer 2: intent_rewriter (Opus, same-step rewrite)
   │   gated on failure_class ∈ {brain_loop_exhausted, wrong_target,
   │   no_state_change}; budget 1 rewrite per step
   │
   └─ Layer 3: agentic_recovery (Haiku → Opus on validation-blocked
       submits) — picks one of:
           add_hint | edit_step | insert_steps | halt
       budget: 2 per step, 5 per run
       on no_state_change submit: tab-blur traversal before screenshot
       on insert_steps: inherits parent's hints.region, cascade-depth
       capped at ≤2
```

Each layer is independent; failure at one layer falls through to the next. The chain ends in either a `RecoveryOutcome(halt=False, …)` (run continues, possibly at a different step) or `RecoveryOutcome(halt=True, …)` (legacy HALT path).

## Layer 1 — Handler escalation

When a step type has multiple handlers registered, the form handler's submit / fill_field path tracks `_step_failure_history` per index. After 2 same-kind failures (canonical: `no_state_change` demotes), `_maybe_set_handler_override` flips the per-step override:

```
runner._step_handler_override[index] = "holo3"
```

The next dispatch routes through `Holo3StepHandler`, which spawns a fresh `GymRunner` with the Holo3 brain (`step_handlers/holo3.py`). The brain receives the augmented intent prose including a `CONTEXT: previous attempts clicked on …` block built from `_step_failure_history`. Budget bumps to `max(step.budget, 25)` for the brain-grounded loop.

This layer fires before `intent_rewriter` and `agentic_recovery` get a turn. Many staff-crm halts get unblocked here without ever reaching layers 2 or 3.

## Layer 2 — `intent_rewriter` (epic #377 Phase B)

Module: `src/mantis_agent/gym/intent_rewriter.py`. Model: `claude-opus-4-7`.

Fires from `RunExecutor._maybe_rewrite_intent_for_retry` after `_handle_failure` decides retry (not halt). Asks Opus to propose a more mechanical / specific intent for the next retry. Returns either:

- A new intent string → stashed in `runner._step_intent_overrides[step_index]`, next retry uses it
- `KEEP` → no rewrite; retry uses the original intent

**Pre-step filter** (issue #428 Part B): pre-step-shaped rewrites like `"First fix the invalid X field"` or `"Before clicking, set Y"` are recognized via `_looks_like_pre_step_rewrite` and dropped to `None`. Those describe a different verb that should be a Layer-3 `insert_steps`, not a same-step rewrite. The rewriter prompt also forbids them upstream so they don't reach the filter in the common case.

**Triggering classes**: `REWRITE_TRIGGERING_CLASSES = {brain_loop_exhausted, wrong_target, no_state_change}`. Anything else short-circuits at `should_attempt_rewrite()`.

**Budget**: `max_attempts=1` per step (`_step_rewrite_attempts[step_index]`). One rewrite per step per run. The first rewrite either lands (next retry takes the new intent) or no-changes (retry uses original); a second call short-circuits with `[N] rewriter_skipped: per-step budget exhausted`.

**Env**: `ANTHROPIC_API_KEY` required. No opt-out env var today; the budget cap is the throttle.

## Layer 3 — `agentic_recovery` (issue #224)

Module: `src/mantis_agent/agentic_recovery.py` (the Claude call) + `step_recovery.StepRecoveryPolicy._try_agentic_recovery` (the runner integration). Default model: `claude-haiku-4-5-20251001`. **Upgraded to `claude-opus-4-7` for `no_state_change` failures on `submit` steps** (issue #432) — Haiku's vision frequently misses red field-validation indicators, which is the canonical "silent submit rejection" pattern.

Fires after layers 1 and 2 have given up — i.e., the per-step retry budget is exhausted and the step is still failing. Returns a `RecoveryDecision` with one of four modes:

| Mode | What it does | When to use (per the prompt) |
|---|---|---|
| `add_hint` | Append a clarifying instruction to the next retry's search prompt | Right element visible, just need to tell the searcher to pick it |
| `edit_step` | Mutate `step.intent` / `step.type` / `step.params` in place | Labels in params are wrong, or step type genuinely needs to change |
| `insert_steps` | Splice helper sub-flow before the failed step | Precondition missing — modal blocking, section collapsed, scroll needed, field value invalid |
| `halt` | Surface the failure to the operator | Target genuinely missing, anti-bot block, page state contradicts step premise |

**Tab-blur traversal**: when `failure_class == "no_state_change"` AND `step.type == "submit"`, the policy walks `Tab × 12` through the form *before* capturing the screenshot. Most React forms render `:invalid` / `aria-invalid` styles only after focus leaves a bad field — the post-submit-fail screenshot is visually clean even when the submit handler short-circuited on a validation error. Tab traversal forces the rendering so Claude can see what to insert a normalize step for. Disable via `MANTIS_RECOVERY_TAB_BLUR=disabled`; count via `MANTIS_RECOVERY_TAB_BLUR_COUNT` (default 12).

**Region hint inheritance**: when `insert_steps` splices new steps before the failed one, the new steps inherit the parent step's `hints.region` so `find_form_target` runs with the same scoping the parent had. The submit-only hints (`visual`, `position`) aren't carried — they don't translate to the inserted step.

**Cascade cap**: if an inserted step itself fails and triggers another `_try_agentic_recovery` that wants `insert_steps`, the runner halts cleanly at cascade depth ≥ 2 instead of looping `insert_steps` on the same root cause (which would burn the per-run recovery budget on cascading no-effect retries — observed pattern: $0.92 / 32-step halt before the cap was added).

**Budgets**: `DEFAULT_MAX_RECOVERIES_PER_STEP = 2`, `DEFAULT_MAX_RECOVERIES_PER_RUN = 5`. Both are hard caps in `agentic_recovery.py`. The per-step counter sits on `runner._recovery_attempts_per_step`; per-run on `runner._total_recovery_attempts`. Cascade tracking on `runner._recovery_inserted_steps` (set) + `runner._recovery_cascade_depth` (dict).

## Failure-class taxonomy

`StepResult.failure_class` is the discriminator across the chain. Common values:

| `failure_class` | Set by | Meaning |
|---|---|---|
| `""` (empty) | default | Generic failure, no specific diagnosis |
| `no_state_change` | `_maybe_demote_form_no_change` / `_maybe_demote_click_no_change` | Action reported success but the page didn't transition |
| `brain_loop_exhausted` | `Holo3StepHandler` | Inner GymRunner hit `max_steps` or the loop detector tripped |
| `wrong_target` | various click handlers | Click landed on a non-action element (status badge, label) |
| `selector_miss` | form `right_click` handler | `find_form_target` returned None on a structured-label call |

The rewriter only triggers on the first three; the recovery analyser sees all of them.

## Visibility

Issue #431 normalized every silent-skip path in the chain to log at WARNING. Grep markers:

| Marker | What it means |
|---|---|
| `[N] rewriter_skipped: …` | Rewriter call site exited without producing a rewrite |
| `[N] rewriter_no_change: Claude returned no actionable rewrite` | Rewriter ran, Claude returned KEEP / empty / identical |
| `[rewriter] step N intent rewritten from … → …` | Rewriter applied a new intent |
| `[N] recovery_skipped: per-step budget exhausted` | Layer 3 hit per-step cap |
| `[N] recovery_skipped: per-run budget exhausted` | Layer 3 hit per-run cap |
| `[N] recovery_skipped: inserted-step cascade depth N exceeded` | Layer 3 cascade cap fired |
| `[N] recovery: tab-blur traversal × 12 …` | Layer 3 ran the blur-driven validation render before screenshot |
| `[N] agentic recovery: mode=… — …` | Layer 3 applied a decision |
| `agentic_recovery_skipped: 200 OK but no record_recovery tool_use block` | Anthropic returned text-only instead of the forced tool |

Each of these surfaces enough state to debug from Modal logs alone — issue #433's diagnosis chain depended on every one of them firing reliably.

## Env knobs

| Env var | Default | Effect |
|---|---|---|
| `MANTIS_RECOVERY_TAB_BLUR` | (unset) | Set to `"disabled"` to skip Tab traversal before the recovery screenshot. Useful for bisecting whether the traversal helps or harms a specific plan. |
| `MANTIS_RECOVERY_TAB_BLUR_COUNT` | `12` | Override the Tab count for forms with more inputs. Each Tab is ~50ms; 12 → ~0.6s overhead per recovery call. |
| `ANTHROPIC_API_KEY` | (unset) | Required for layers 2 and 3. If missing, both fall through to the legacy retry-then-halt path with a WARNING log. |

## What's NOT here (open follow-ups)

* **Runtime sub-goal decomposition by the frontier model** (epic #435 item 8). Today the planner is plan-time only; mid-run replanning lives in `agentic_recovery.insert_steps` which only fires on failure. A continuous planner that observes step outcomes and emits directives mid-run would be a different layer.
* **CDP-readonly probe** (option 3 in #432). Inspect `input.validity` / `aria-invalid` directly via DOM injection instead of relying on Claude vision. Deterministic, but breaks the zero-fingerprint architecture invariant — opt-in flag at minimum.

See [`docs/cua_notes.md`](../cua_notes.md) for the design rationale behind the planner-executor split and the "history mostly text, not images" pattern that frames how this chain is built.
