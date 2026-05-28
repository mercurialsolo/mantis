# Brain Context Optimization — borrow patterns from `claude_stateful`

**Status:** Proposed
**Owners:** TBD
**Tracks issue:** TBD (this doc is the canonical spec; the GitHub issue points at it)

## Summary

Two patterns from an upstream `claude_stateful` integration backend solve cost / context problems Mantis still pays on every run:

1. **Prompt-cache split** — partition the brain prompt into a stable section (`system + tools + plan + recipe + site_config`) and a mutable section (`state + screenshot + recent_history`). The stable section caches across turns; the mutable section is re-sent each turn. Net: ~30-50% reduction in Claude grounding cost on long runs.
2. **LLM-owned digest** — let the brain write one structured observation line per step (`digest_line`). The framework persists it in a ring buffer and re-injects it on subsequent steps. The brain amortizes its own reasoning instead of re-deriving "what page am I on, what have I collected" from screenshot pixels every turn.

Both patterns share a single insight: **don't accumulate transcripts; structure what stays vs. what changes.** claude_stateful does this at the prompt-cache boundary; Mantis already does it at the brain-protocol boundary (stateless per step). The patterns are complementary — cache the stable boundary, let the LLM write the mutable boundary.

This document specs the migration in two tracks. Track A (prompt-cache split) is the bigger cost win and lands first. Track B (digest) needs Track A in place so the digest belongs in the cached-prompt structure.

## Goals

1. **Track A — measurable cost reduction**: 30-50% drop in Claude grounding + extraction cost per lead, measured against the current $0.18-0.36/lead baseline on boattrader. No quality regression.
2. **Track B — close the running-narrative gap**: brain-emitted digest lines persist across steps; auto-digest fires when N steps run without progress. Surfaces in Augur trajectories for free.
3. **Compose with existing memory layers**: digest lines should sit cleanly alongside `hint_memory` (per-step anchors) and `plan_evolution` (per-plan URL rewrites). No overlap, no replacement.
4. **CUA-purity preserved**: the digest is brain-side reasoning, not DOM-derived state. See `feedback_cua_no_dom_access.md`.

## Non-goals

- **Replacing the brain's stateless-per-step protocol.** Mantis brains still see a fresh screenshot per turn; they don't accumulate transcripts. The digest is a *sidecar*, not a transcript.
- **Postgres-backed state.** Mantis's volume-on-disk pattern (used by plan_evolution + hint_memory) is simpler, tenant-scoped, lower-latency at our run sizes. claude_stateful's `cua_run_states` is operationally heavier than we need.
- **Optimistic-locking on shared mutable state.** Useful when the LLM writes; we own writes framework-side already.
- **LLM-callable arbitrary state tool** (`set / increment / append_to / remove_from / checkpoint`). Mantis's structural memory (`hint_memory`, `plan_evolution_store`, `BrowserState`) covers these without giving the LLM tokens to reason about.
- **Generic prompt caching outside the brain.** ClaudeExtractor + ClaudeGrounding caches separately if at all; out of scope here.

## Architecture overview

```
┌──────────────────────────────────────────────────────────────────────┐
│ Brain prompt (per step)                                              │
│                                                                      │
│  ┌─ STABLE (cache_control: ephemeral) ─────────────────────────┐    │
│  │  system instructions                                         │    │
│  │  tool definitions                                            │    │
│  │  plan / sub-goal template                                    │    │
│  │  recipe + site_config                                        │    │
│  │  brain-budget caps                                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─ MUTABLE (re-sent each turn) ──────────────────────────────┐    │
│  │  current page state (url, scroll, viewport)                  │    │
│  │  per-step hints (preferred_target, region, near, ...)        │    │
│  │  recovery hints (from agentic_recovery)                      │    │
│  │  digest ring buffer (Track B) ← 10 most-recent LLM observations │ │
│  │  recent failure history (last 3)                             │    │
│  │  screenshot                                                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                       Claude API / llama.cpp
                              │
                              ▼
              ┌──────────────────────────────────┐
              │ Brain response                   │
              │   action: <click / type / ...>   │
              │   digest: <one line>  (Track B)  │
              │   reasoning: <thinking blob>     │
              └──────────────────────────────────┘
                              │
                              ▼
                  Runner appends digest to ring buffer
                  Runner records hint_memory + plan_evolution
                  Runner emits Augur step + digest
```

## Phases

### Phase 0 — `brain_claude` prompt-cache split (Track A)

**Goal:** measurable cost reduction on every Claude call from the runner — grounding, recovery analysis, extractor.

#### File changes

| File | Change |
|---|---|
| `src/mantis_agent/brain_claude.py` | Restructure prompt assembly into stable + mutable sections. Add `cache_control: {type: "ephemeral"}` to the stable section's last content block. Confirms via response headers (`anthropic-cache-creation-input-tokens`, `anthropic-cache-read-input-tokens`). |
| `src/mantis_agent/grounding.py` (`ClaudeGrounding`) | Same shape — system + tools stable, screenshot + target hints mutable. |
| `src/mantis_agent/extraction.py` (`ClaudeExtractor`) | Same shape — schema + system stable, screenshot + region hints mutable. |
| `src/mantis_agent/agentic_recovery.py` | Same shape — system + tool schema stable, failure_data + screenshot mutable. |
| `tests/test_brain_claude_cache.py` | **NEW.** Assert the stable section ends with a `cache_control` block; assert mutable section comes after. |
| `tests/test_grounding_cache.py` / `tests/test_extraction_cache.py` | **NEW.** Same shape. |

#### Acceptance

- Boattrader smoke run shows `anthropic-cache-read-input-tokens` > 0 in Augur trajectories after the first step.
- **Cost-per-lead drops 30-50% vs the pre-PR baseline** on a 30-min boattrader run with warm CF clearance. Measured against runs `20260528_042856` (baseline: $0.34/lead) and `20260528_150245` (post-hint-memory: $0.23/lead).
- No quality regression — leads / step count / failure-class distribution within ±10% of baseline.
- All Claude calls in the runner pass the same cache boundary check (a single shared helper to assemble stable/mutable sections).

#### Out of scope

- Cache strategies for non-runner calls (PlanDecomposer's one-shot Claude call has nothing to cache against).

### Phase 1 — `brain_holo3` prompt-cache split (Track A)

**Goal:** apply the same partition to the Holo3 (llama.cpp) prompt builder. llama.cpp's prompt-cache mechanism is different from Claude's `cache_control` — it caches by token-prefix matching across calls. Separating stable / mutable still cuts prefix re-tokenization on every call.

#### File changes

| File | Change |
|---|---|
| `src/mantis_agent/gym/step_handlers/holo3.py` | Refactor `_build_scoped_task` to return a typed `(stable_prefix, mutable_suffix)` tuple. The handler emits them concatenated; future llama.cpp work can read `stable_prefix` separately for prefix-cache control. |
| `src/mantis_agent/brain_holo3.py` | If the llama.cpp endpoint supports a `cache_prompt` parameter (recent llama.cpp `server` builds do), pass `cache_prompt=True` + the stable prefix length so subsequent calls skip re-tokenization. |
| `tests/test_holo3_prompt_partition.py` | **NEW.** Lock the section ordering: stable section ends before any screenshot or hint content. |

#### Acceptance

- Holo3 inference wall-time drops measurably (10-20% target) on a boattrader run because llama.cpp's KV-cache hits the stable prefix.
- Holo3 prompt builder exposes the boundary publicly (a typed return value); downstream callers can introspect.
- Same brain output quality — verified against a tape-replay test where the same screenshots produce the same actions.

### Phase 2 — `digest` action + ring buffer (Track B)

**Goal:** brain emits one structured observation line per turn; runner persists it in a ring buffer and re-injects it on subsequent turns.

#### File changes

| File | Change |
|---|---|
| `src/mantis_agent/actions.py` | Add an optional `digest: str = ""` field to `Action` (alongside `memorize_fact`). Capped at 200 chars by the runner. Tools that emit this: Holo3 (via extended `act` schema), Claude (via tool_use response field). |
| `src/mantis_agent/brain_protocol.py` | Document the digest field as part of the brain response contract. |
| `src/mantis_agent/brain_claude.py` | Surface the digest in the tool-use schema; parse it into `Action.digest`. |
| `src/mantis_agent/brain_holo3.py` | Same — add `digest` to the function-calling schema. Brains that don't emit it produce an empty string (no behavior change). |
| `src/mantis_agent/gym/micro_runner.py` | Maintain `_digest_ring: deque[str]` of the 10 most recent non-empty digest lines. Inject as a "Running observations" mutable section in the next step's prompt. |
| `src/mantis_agent/gym/step_handlers/holo3.py` | Read the runner's `_digest_ring`, format as a section in the mutable suffix from Phase 1. |
| `src/mantis_agent/observability/augur.py` | Emit each digest line as a structured reasoning event so trajectories show the running narrative. |
| `tests/test_digest_ring.py` | **NEW.** Ring eviction at cap, empty-digest skip, prompt-injection ordering. |

#### Acceptance

- A boattrader run produces a coherent running narrative in the Augur trajectory: "page 1 loaded, 5 listings visible", "clicked Meridian, on detail page", "extracted 5 leads, going back", ...
- The runner's mutable prompt section grows by at most ~2 KB (10 × 200 char cap).
- Brain-loop-exhausted incidence drops on tall-page extraction phases (target: 30% reduction on boattrader detail pages). Measured by counting `brain_loop_exhausted` in 5 runs before / 5 after.

#### Out of scope

- Persisting digest across runs (the digest is per-run; cross-run learning belongs to `hint_memory` + `plan_evolution`).
- LLM-callable mutation of the digest (the LLM only appends; runner owns eviction).

### Phase 3 — auto-digest gate (Track B)

**Goal:** catch brain drift earlier by forcing a digest when N consecutive steps complete without observable progress.

#### File changes

| File | Change |
|---|---|
| `src/mantis_agent/gym/micro_runner.py` | Track `_no_progress_streak` — increments each step where `current_url`, `scroll_y`, `failure_class`, and leads-collected count are unchanged. When the streak hits N (default 3), force a reflection turn: prompt the brain with `"Pause and digest: what's happening on this page?"` and store the response as a digest entry. |
| `src/mantis_agent/gym/failure_class.py` | Add `auto_digest_triggered` as a structured metric (not a failure_class). |
| `tests/test_auto_digest_gate.py` | **NEW.** Gate fires after N steps without progress; doesn't fire when leads are accumulating. |

#### Acceptance

- A run that today halts with `brain_loop_exhausted` at step ~14 instead halts at step ~10 with a structured `auto_digest_triggered` event preceding the halt — telling the operator the brain noticed it was stuck before burning more budget.
- Auto-digest fires no more than once per N-step window (no thrashing).

#### Out of scope

- Routing the auto-digest into `agentic_recovery` automatically. That's a separate composition once Track B is stable in production.

## Reliability notes

- **Cache invalidation.** Claude's prompt cache TTL is 5 minutes. A run with sparse Claude calls (extraction every 10s) may fall outside the window. Phase 0 measures cache hit rate via response headers and surfaces it in Augur — operators can see when cache-miss rate climbs.
- **Stable section drift.** If the stable section changes mid-run (recipe swap, brain budget change), the cache is invalidated. The Phase 0 assembler validates that stable content is computed once at runner-init and frozen for the run's lifetime.
- **Digest budget.** 10 lines × 200 chars = 2 KB. Even with the cache miss this is negligible vs the screenshot payload (~80-300 KB). No per-token cost concern.
- **Brain compatibility.** Phase 2's `digest` field is optional. Brains that don't emit it leave it empty; the runner skips empty entries. Fara and OpenCUA brains keep working without changes.

## Open questions

1. **Cache TTL renewal.** Claude's prompt cache renews on access. On a sparse run, should the runner emit a "keepalive" lightweight Claude call to renew the cache? Defer until Phase 0 telemetry shows the miss rate.
2. **Digest at what granularity?** One line per step is the spec default. For multi-action steps (a `scroll` that emits 3 wheel events), should the brain emit one digest or three? Defer; observe production.
3. **Digest schema.** Free-form string vs structured fields (`status`, `next_action`, `confidence`)? Free-form is simpler; structured is more analyzable. Start free-form, revisit if Augur queries get unwieldy.

## Risks

| Risk | Mitigation |
|---|---|
| Cache-control marker placement wrong → no cache benefit | Telemetry in Augur from day one. If `cache_read_tokens / total_tokens < 0.3` after warmup, surface a WARNING. |
| Stable section accidentally includes mutable content | The Phase 0 assembler is a pure function with a typed return; tests lock the boundary. |
| Digest tokens drift from accurate page state | The brain re-derives state from the screenshot anyway; digest is a *summary*, not authoritative. Augur surfaces both so operators can audit drift. |
| Auto-digest gate fires on slow-but-progressing pages | Gate is conditioned on URL / scroll / leads-count being all unchanged — not just URL. Tunable per recipe via `auto_digest_no_progress_window`. |
| Brain refuses to emit digest (older models) | Field is optional; runner-side ring is no-op when every entry is empty. No regression for non-emitting brains. |

## References

- `claude_stateful` design — upstream integration-side pattern (a `stateful_loop` module with a `stateful_prompts` cache-split helper and a `state_adapter` tool surface). Pattern names referenced abstractly to satisfy this repo's customer-name isolation policy.
- Anthropic prompt caching docs — `https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching`
- `docs/reference/plan-evolution.md` — same phase-by-phase shape; per-tenant storage approach informs how digest persistence (if ever added) would scope.
- `docs/reference/computer-plane.md` — same docstyle.
- `feedback_warning_level_for_modal_observability.md` — cache-hit-rate logs must be WARNING-level to survive Modal's INFO suppression.
- `feedback_cua_no_dom_access.md` — digest is brain-side reasoning, never DOM-derived.
