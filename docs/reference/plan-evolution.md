# Plan Evolution — continuous-memory plan healing

**Status:** Proposed
**Owners:** TBD
**Tracks issue:** TBD (this doc is the canonical spec; the GitHub issue points at it)

## Summary

Plans in Mantis are authored once and re-run many times. Real-world failure modes drift between author-time and run-time — URLs slug-change, button labels rename, form fields move, sites redesign. Today every run hits the drift fresh, burns budget on the agentic recovery loop, and forgets the fix when it terminates. The next run repeats the same recovery work.

This document specs a **plan evolution layer** that closes the loop: detect, recover, persist, re-apply.

```
detect drift  →  recover in-flight  →  persist the rewrite  →  apply on next run
   [Phase 0]      [Phase 1]              [Phase 2]              [Phase 2]
```

URLs are Phase 1's beachhead because the failure signal is unambiguous (`chrome-error://`, HTTP 404, wrong-domain redirect) and the recovery cost is cheap (web_search tool call or pattern transform). Once the loop is proven on URLs, Phase 3 generalises to selector labels, button aliases, and form-field names — anywhere `agentic_recovery` already produces a step rewrite.

## Goals

1. **First-class URL-failure detection** — a structured `failure_class='bad_url'` with subclasses (`dns`, `not_found`, `wrong_domain`) that hooks into the existing recovery loop instead of being inferred from pixels.
2. **URL-aware recovery actions** — pattern transform → page-of-links discovery → web-search, in cheapness order.
3. **Persistent plan evolution store** — successful rewrites recorded per-plan with a promotion gate so single one-off successes don't pollute the next run.
4. **Pre-flight overlay** — at submit time, promoted rewrites are applied to the plan *before* dispatch; the brain never sees the stale URL.
5. **Scoping** — workflow vs tenant vs site, so a fix learned on one customer's CRM doesn't leak to another.
6. **Drift invalidation** — when a promoted rewrite starts failing, demote and re-explore. Catches "site changed back" without manual intervention.

## Non-goals

- **Replacing agentic_recovery's brain reasoning.** This layer adds *persistence* and *structured detection* on top of the four existing recovery modes (`add_hint`, `edit_step`, `insert_steps`, `halt`). The Claude analyser still does the hard thinking when patterns + heuristics don't suffice.
- **Cross-plan generalisation.** A rewrite learned on plan A doesn't automatically apply to plan B even on the same site. Phase 3 may add `scope=site` rewrites; until then everything is plan-scoped.
- **Plan authorship UX.** Operators edit the authored plan; the overlay is a separate file and never mutates the source.
- **Replacing sim envs / oracles.** Pre-production grading via `/__env__/oracle` remains the right place to catch structural plan issues before they ship.
- **General-purpose plan rewriting.** This is a *healing* layer for drift, not a *generator*. Wholesale plan-restructuring (different recipe, different vertical) is out of scope.

## Architecture overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  Submit time                                                          │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ PlanDecomposer / dispatcher                                  │    │
│  │   └─→ apply_plan_overlay(plan_hash) → MicroPlan              │    │
│  │         ↑                                                    │    │
│  │  ┌──────┴──────────────────┐                                 │    │
│  │  │ Plan Evolution Store     │  (Phase 2)                    │    │
│  │  │ /data/plan_evolution/    │                                │    │
│  │  │ {workflow_id}/           │                                │    │
│  │  │ {plan_hash}.json         │                                │    │
│  │  └──────┬──────────────────┘                                 │    │
│  └─────────┼────────────────────────────────────────────────────┘    │
│            ▼                                                          │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │  Run                                                       │      │
│  │  ┌──────────────────────────────────────────────────────┐  │      │
│  │  │ step_handlers/navigate.py                            │  │      │
│  │  │   ├── env.step(navigate)                             │  │      │
│  │  │   └── url_health.classify(current_url, screenshot)   │  │      │
│  │  │         ├─ ok          → continue                    │  │      │
│  │  │         └─ bad_url     → failure_class='bad_url'    │  │      │
│  │  └──────────────────────────────────────────────────────┘  │      │
│  │                       │                                    │      │
│  │                       ▼                                    │      │
│  │  ┌──────────────────────────────────────────────────────┐  │      │
│  │  │ agentic_recovery.analyse_failure_and_recover        │  │      │
│  │  │   New mode: rewrite_url(step_index, new_url, src)    │  │      │
│  │  │   Sources: pattern_transform → page_links →          │  │      │
│  │  │            web_search (Claude tool_use)              │  │      │
│  │  └──────────────────────────────────────────────────────┘  │      │
│  │                       │                                    │      │
│  │                       ▼                                    │      │
│  │  ┌──────────────────────────────────────────────────────┐  │      │
│  │  │ rewrite_logger.record(plan_hash, step_idx, rewrite,  │  │      │
│  │  │                       outcome, source, confidence)   │  │      │
│  │  └──────────────────────────────────────────────────────┘  │      │
│  └──────────────────────┼─────────────────────────────────────┘      │
│                         ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Plan Evolution Store (Phase 2)                              │    │
│  │  Promotion gate: 3 consecutive successes → promoted          │    │
│  │  Drift gate:     2 consecutive failures while promoted →     │    │
│  │                  demoted, re-explore                         │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

## Data shape

### Rewrite record

```python
# src/mantis_agent/recipes/plan_evolution_store.py  (NEW)

from typing import Literal
from pydantic import BaseModel, Field

RewriteScope = Literal["workflow", "tenant", "site"]
RewriteStatus = Literal["candidate", "promoted", "demoted", "cold"]
RewriteSource = Literal[
    "pattern_transform",
    "page_links",
    "web_search",
    "brain_proposal",
    "manual",
]


class StepRewrite(BaseModel):
    step_index: int
    original: dict           # the original step body as authored
    rewritten: dict          # the in-place replacement
    scope: RewriteScope = "workflow"
    status: RewriteStatus = "candidate"
    source: RewriteSource
    confidence: float        # 0.0–1.0 from the source heuristic
    first_seen: str          # ISO-8601
    last_seen: str           # ISO-8601
    successful_runs: int = 0
    failed_runs: int = 0
    # When a rewrite has been promoted then demoted, this captures
    # the demotion reason so the next exploration knows what NOT to
    # try again immediately.
    demotion_reason: str | None = None


class PlanEvolution(BaseModel):
    plan_hash: str           # sha256 of the canonical plan text
    workflow_id: str
    tenant_id: str | None    # None when scope=site
    rewrites: list[StepRewrite] = Field(default_factory=list)
```

### Storage

```
/data/plan_evolution/
├── workflow/
│   └── <workflow_id>/
│       └── <plan_hash>.json     # workflow-scoped overlays
├── tenant/
│   └── <tenant_id>/
│       └── <site>/<plan_hash>.json
└── site/
    └── <site>/
        └── <plan_hash>.json     # site-scoped (shared across tenants)
```

JSON-on-volume is simple, atomic-via-tmpfile-rename, and doesn't require a new dependency. If multi-writer contention becomes a problem the store can move to SQLite on the same volume without changing the rewrite shape.

## URL health classifier

```python
# src/mantis_agent/gym/url_health.py  (NEW)

from typing import Literal

UrlHealthSubclass = Literal[
    "ok",
    "dns",            # chrome-error://, net::ERR_NAME_NOT_RESOLVED
    "not_found",      # HTTP 404 page detected via heuristics
    "wrong_domain",   # navigated to a domain not in the plan's expected set
    "soft_404",       # 200 OK but page content matches "not found" patterns
    "blocked",        # Cloudflare / WAF challenge — handled elsewhere
]


def classify(
    *,
    current_url: str,
    expected_url: str,
    expected_domains: set[str],
    screenshot_phash: str | None = None,
) -> UrlHealthSubclass:
    """Decide whether the post-navigate state is OK or a URL failure."""
```

Detection signals:

- `current_url.startswith("chrome-error://")` → `dns`
- `current_url == ""` after a navigate step → `dns` (Chrome failed to navigate)
- `urlparse(current_url).netloc not in expected_domains` → `wrong_domain`
- Page screenshot pHash matches known 404 templates → `not_found` (heuristic; per-site)
- CDP `Network.loadingFailed` event observed during navigate → `dns` / `not_found`
- Existing `external_pause` / `cf_challenge` paths → `blocked` (not handled here; remains in `external_pause.py`)

## Recovery sources

In cheapness order, each emits a `rewrite_url` proposal with a confidence score:

### 1. `pattern_transform` (~free)

Slug normalisation, trailing-slash handling, common path mutations:

- `/boats/state-fl/` ↔ `/boats/state/fl/`
- `/boats/by-owner` ↔ `/boats/by-owner/`
- `/leads/123` ↔ `/leads/123/edit`
- `?param=value` ↔ `?param=Value` (case folding)

A pure-Python rule set. Try each transform, only emit when the transformed URL passes a HEAD-200 check via the proxy. Confidence 0.6 — pattern matches but doesn't validate semantic correctness.

### 2. `page_links` (~$0.001)

Navigate back to the plan's last known good page (or the site's homepage). Use a CDP `document.querySelectorAll('a[href]')` evaluation to enumerate links. Score links by text overlap with the original step's intent (anchor text + URL path tokens). Top match becomes the candidate. Confidence 0.7 — page-derived but heuristic.

### 3. `web_search` (~$0.005)

Last resort. Claude tool call with `web_search` + the failed URL's domain + the step's intent (`"site:boattrader.com state fl by owner"`). Top result that matches `expected_domains` becomes the candidate. Confidence 0.85 — semantic match by Claude.

### 4. `brain_proposal` (~free, already-paid)

The existing `agentic_recovery.edit_step` brain output, when Claude proposes a navigate-step rewrite without an explicit recovery source. Captured for memory; confidence 0.5 (no validation beyond the brain's reasoning).

### Promotion gate

A rewrite proposed on run N is `status='candidate'` until **3 consecutive successful runs** of the same plan produce the same rewrite (within ±10% URL diff to allow tracker params). On the 3rd success it transitions to `promoted` and is applied pre-flight on subsequent runs.

A `promoted` rewrite that fails 2 consecutive runs flips to `demoted`. Demoted rewrites stay in the store for diagnostic purposes but aren't applied. The original step runs again, the recovery loop fires fresh, a new candidate emerges.

A rewrite unused for 30 days flips to `cold`. Cold rewrites must re-promote (3 fresh successes) before re-applying — protects against "site changed back to the original URL" without manual intervention.

## Phase 0 — URL health detection (zero recovery, observability only)

**Goal:** ship the classifier + structured `failure_class='bad_url'` emission so the existing recovery loop sees a clean signal. No new recovery action yet. Useful on its own: surfaces URL-drift incidence in Augur for sizing the actual fix.

### File changes

| File | Change |
|---|---|
| `src/mantis_agent/gym/url_health.py` | **NEW.** Classifier + subclass enum. |
| `src/mantis_agent/gym/step_handlers/navigate.py` | Hook `url_health.classify` after `env.step(navigate)`. On non-`ok` subclass, return a `StepResult` with `failure_class='bad_url'` + `failure_subclass=<value>`. |
| `src/mantis_agent/gym/failure_class.py` | Add `bad_url` to the enum. |
| `src/mantis_agent/agentic_recovery.py` | Recognise `bad_url` in the recovery-trigger set; for Phase 0 just routes to `halt` with a structured reason. |
| `tests/test_url_health.py` | **NEW.** Pure-Python classifier tests covering each subclass. |
| `docs/reference/plan-evolution.md` | This doc. |

### Acceptance

- Every Modal run emits Augur events tagged with `url_health_subclass` for every navigate step.
- The fresh-profile boattrader run that today halts with generic `claude-click ... blocked` instead halts with `failure_class='bad_url' subclass='blocked'` (existing CF path) or `subclass='wrong_domain'` if the CF detour redirects.
- Test suite covers `dns`, `not_found`, `wrong_domain`, `soft_404`, `ok`.
- Zero behaviour change for runs that have a working URL.

## Phase 1 — URL-rewrite recovery action (in-memory only)

**Goal:** add the `rewrite_url` recovery mode wired to the three sources. Rewrites apply within the current run; nothing is persisted yet. Validates the recovery quality before paying for the storage layer.

### File changes

| File | Change |
|---|---|
| `src/mantis_agent/gym/url_recovery.py` | **NEW.** Implements the three sources (`pattern_transform`, `page_links`, `web_search`) behind a `propose_url_rewrites(failed_step, page_state, brain) -> list[StepRewrite]` API. |
| `src/mantis_agent/agentic_recovery.py` | Add `rewrite_url` to `RecoveryDecision` modes. When `failure_class='bad_url'`, call `url_recovery.propose_url_rewrites` *before* the Claude analyser; if a high-confidence candidate exists, dispatch it directly; otherwise hand off to Claude with the candidate set as context. |
| `src/mantis_agent/gym/runner.py` | Apply a `rewrite_url` decision in-place on the running plan; subsequent steps reference the new URL. |
| `tests/test_url_recovery.py` | **NEW.** Mock-server tests for each source. |
| `tests/test_agentic_recovery_url.py` | **NEW.** End-to-end recovery path with a fake `bad_url` failure. |

### Acceptance

- A boattrader plan with a deliberately munged URL (`/boats/state-fl-typo/by-owner/`) recovers via `pattern_transform` within one extra step on the first run.
- A plan pointing at a redirected URL recovers via `page_links` within ~3 extra steps.
- A plan pointing at a completely renamed slug (no rule + no link on the page) recovers via `web_search` within ~5 extra steps + 1 web_search tool call (~$0.005).
- Cost ceiling: URL recovery adds ≤ $0.05 per affected step.

## Phase 2 — Plan evolution store + pre-flight overlay

**Goal:** persist rewrites with the promotion gate; apply promoted rewrites at submit time before dispatch.

### File changes

| File | Change |
|---|---|
| `src/mantis_agent/recipes/plan_evolution_store.py` | **NEW.** JSON-on-volume store keyed by `(scope, scope_id, plan_hash)`. Atomic writes via tmpfile rename. |
| `src/mantis_agent/observability/rewrite_logger.py` | **NEW.** Records every rewrite-attempt + outcome. Applies promotion / demotion logic. Emits structured Augur events. |
| `src/mantis_agent/plan_decomposer.py` | After decomposition, call `apply_plan_overlay(plan_hash, workflow_id, tenant_id)` → returns the plan with promoted rewrites applied. Log `[plan-overlay]` lines listing what changed. |
| `src/mantis_agent/gym/runner.py` | On `rewrite_url` success, call `rewrite_logger.record(...)`. |
| `tests/test_plan_evolution_store.py` | **NEW.** Promotion gate, demotion, cold transition, scope resolution. |
| `tests/test_plan_overlay.py` | **NEW.** Decomposer applies promoted rewrites; ignores candidates. |

### Acceptance

- Run plan P three times with a stable rewrite → rewrite reaches `promoted`; the 4th run never dispatches the original URL.
- Run plan P after the site changes back → promoted rewrite fails twice → `demoted`; the 3rd run dispatches the original URL again.
- Two tenants running the same plan against the same site each get independent `workflow`-scoped stores. (Phase 2 ships with scope=`workflow` only; site-scope is Phase 3.)
- Promotion gate counters survive across container restarts (volume-backed).

## Phase 3 — Generalise beyond URLs + site-scope rewrites

**Goal:** the same store + overlay machinery applied to selector labels, button aliases, form fields. Site-scope rewrites for site-wide drifts.

### File changes

| File | Change |
|---|---|
| `src/mantis_agent/gym/agentic_recovery.py` | Every successful `add_hint` / `edit_step` / `insert_steps` emits via `rewrite_logger` (today they're in-memory only). |
| `src/mantis_agent/recipes/plan_evolution_store.py` | Site-scope resolution: a workflow-scoped rewrite promoted on N different plans for the same site auto-promotes to a site-scope rewrite (default N=5). |
| `src/mantis_agent/plan_decomposer.py` | Overlay resolves in scope order: workflow → tenant → site. First match wins; explicit demotions block higher-scope matches. |
| `tests/test_site_scope_promotion.py` | **NEW.** N-plan promotion logic. |
| `docs/reference/plan-evolution.md` | This doc — update with Phase 3 details once shape is concrete. |

### Acceptance

- A "Save" → "Update Lead" label rewrite learned on the staff-crm plans persists and re-applies on subsequent runs.
- A site-wide URL slug change on BoatTrader auto-promotes from workflow to site after the 5th independent confirmation.
- Demoting a site-scope rewrite cascades to all workflows that were inheriting it.

## Reliability notes

- **Atomic store writes.** All `PlanEvolution` updates go through tmpfile + atomic rename. Concurrent runs of the same plan will race; the rewrite-counter increments are last-writer-wins. Acceptable because the promotion gate uses *consecutive* successes — a missed increment delays promotion by one run.
- **Volume mount ordering.** The store lives under `/data/plan_evolution/`; brain and computer-plane both mount `/data`. Plan-overlay application runs in the API container (which also mounts `/data`), so no extra plumbing.
- **Scope isolation.** Workflow- and tenant-scoped rewrites never apply to a different tenant. Site-scoped rewrites are explicitly opt-in via per-tenant `allow_site_overlay=true` in their config (defaults to true; can be disabled for high-isolation tenants).
- **Replay determinism.** Replay env (`replay_env.py`) ignores the overlay — replays should reproduce the run exactly as captured. The overlay is a forward-only optimization, not a hindsight rewrite of trajectories.
- **Augur emission.** Every rewrite attempt (proposed / accepted / promoted / demoted / cold) emits a structured event. This becomes the training signal for a learned rewrite policy in a future phase.

## Open questions

1. **Web-search tool dependency.** Phase 1 introduces a Claude `web_search` tool call. Does this require a separate API path / billing line? Confirm cost rule-of-thumb before Phase 1 ships.
2. **Site-scope auto-promotion threshold.** N=5 is a guess. Defer concrete number until Phase 2 produces a few weeks of usage data.
3. **Cross-region store consistency.** If two regions run the same plan concurrently and each generates a rewrite candidate, the store has two competing partial rewrites. Phase 2 ships as single-region only; cross-region is a Phase 4 concern.
4. **Manual operator override.** Should operators be able to `mantis plan pin --workflow=X --step=2 --url=Y` to force a rewrite that bypasses the promotion gate? Useful for known-good operator interventions. Defer to Phase 2 follow-up.

## Risks

| Risk | Mitigation |
|---|---|
| Auto-rewrites silently break correctness without operators noticing | `[plan-overlay]` log lines + Augur events on every applied overlay. Status payloads include the applied-overlay summary so operators see it in the run dashboard. |
| Promotion gate too aggressive — rewrites lock in before they're truly stable | N=3 consecutive successes within a configurable window (default 7 days); demote at 2 consecutive failures. Tune from Phase 2 data. |
| Promotion gate too conservative — stable rewrites never promote, every run pays recovery cost | Per-plan dashboard surfaces `candidate` rewrites with success counts; operators can manually promote when confident. |
| Site changes back to original URL, demote loop oscillates | Cold-transition after 30 days idle requires fresh promotion. Oscillation across cold-transitions is bounded by the 3-run promotion gate. |
| Store grows unbounded with one-off rewrites | LRU cap per `(scope, scope_id)` — default 100 rewrites per plan; oldest demoted entries evicted first. Promoted entries are never evicted. |
| Cross-tenant leak via site-scope rewrites | Site-scope rewrites only apply when the rewrite's `scope='site'` was reached via N independent tenant-confirmations (Phase 3). Per-tenant opt-out always available. |

## References

- `agentic_recovery.py` — the four existing recovery modes this layer extends
- `hint_memory.py` — the closest existing memory artifact; eventually subsumed by this store
- `site_config.py` / `SiteProber` — URL pattern detection; `page_links` source builds on these
- `feedback_pre_deploy_infra_ping.md` — manual practice this automates
- `docs/reference/computer-plane.md` — same phase / acceptance / non-goals structure
- `project_session_retrospective_2026_05_17.md` — example of a multi-day debugging session driven by per-step plan drift; the per-step diffs would have been captured automatically with this layer
