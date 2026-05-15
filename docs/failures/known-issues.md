# Known failure modes

This is the operator-facing catalog of observed failure modes. The
authoritative taxonomy lives in
[`src/mantis_agent/gym/failure_class.py`](https://github.com/mercurialsolo/mantis/blob/main/src/mantis_agent/gym/failure_class.py) —
this page maps each class to concrete instances seen in production and
the mitigation that's wired up today.

For the loop-recovery and self-healing machinery that fires off these
classifications, see
[reference/loop-recovery.md](../reference/loop-recovery.md) and the
self-healing epic ([#377](https://github.com/mercurialsolo/mantis/issues/377)).

## The mitigation ladder

Every failure gets routed to one of four tiers. The earlier the tier,
the cheaper the fix.

| Tier | Locus | When to use | Examples |
|---|---|---|---|
| **T1 — Runtime** | within a single run | Model emits a parseable but stuck action | Loop recovery, intent rewriter, force-fill / force-submit, claude-director, recovery hints, brain ladder, adaptive settle |
| **T2 — Plan** | between steps / on plan shape | Class of failure is plan-author solvable | PlanDecomposer, Done acceptance gate ([#303](https://github.com/mercurialsolo/mantis/issues/303)), removing "Done when" verbiage, schema validation |
| **T3 — Tooling / dispatcher** | new primitive in the action surface | Model identified the right thing; dispatcher fumbled | CDP `Input.insertText` (vs xdotool typing), `right_click` ([#391](https://github.com/mercurialsolo/mantis/pull/391)), `key_press(Return)` for submit |
| **T4 — Training (SFT/DPO)** | continual-fine-tuning loop | Prompts + runtime can't shift the metric — capability ceiling | Holo3 distillation ([#179](https://github.com/mercurialsolo/mantis/issues/179)); promoted via [scorecard](../experiments/scorecard-history.md) |

Each row in the table below names its **primary** tier. Many real
fixes touch two — e.g. CDP insertText (T3) was rolled out alongside a
runtime fallback chain (T1) so the new primitive could be staged.

---

## In-taxonomy modes

These map 1:1 to a string `failure_class.classify()` may emit.

### `cf_challenge`

Cloudflare or anti-bot interstitial. Detected from HTTP 403 plus the
canonical CF page titles ("just a moment", "verify you are human",
"attention required", …).

| Field | Notes |
|---|---|
| Signature | `result.json.failure_class == "cf_challenge"`; page title matches a CF marker |
| Root cause | Headless detection, IP reputation, missing `--remote-allow-origins` |
| Mitigation | Run with Xvfb (NOT headless) — see memory note on the absolute-first-check rule. Use PrivateProxy via `PRIVATEPROXY_*` env vars; verify with `diagnose_proxy`. |
| Tier | T3 (infra/tooling — browser flags + proxy) |
| Don't | Chase IP reputation before retrying `--no-headless`; that's the high-yield first move. |

### `nav_timeout`

Playwright / CDP navigation timeout. Origin is reachable but the page
never finished loading within the configured timeout.

| Field | Notes |
|---|---|
| Signature | `data` contains `timeout`, `timed out`, `navigation timeout` |
| Root cause | Slow SPA bootstrap, blocked third-party request, transient network |
| Mitigation | Use the `adaptive-settle` policy ([reference/adaptive-settle.md](../reference/adaptive-settle.md)) — extends settle window on slow first-paints. |
| Tier | T1 (runtime — adaptive settle) |

### `http_4xx`

Origin returned 401 / 404 / 410. The agent reached the page but the
server says it's not available.

| Field | Notes |
|---|---|
| Signature | `data` contains `error 404`, `error 401`, etc. |
| Common cause | Stale URL in plan, expired session, deep link to deleted resource |
| Mitigation | Re-derive the URL with a `navigate` step from a known-good entry point; don't hard-code deep links across runs. |
| Tier | T2 (plan — re-derive URLs from entry points) |

### `http_5xx`

Origin returned 5xx. The site is broken or rate-limiting; not a plan
problem.

| Field | Notes |
|---|---|
| Signature | `data` contains `error 5`, `502`, `503`, `504`, `internal server error` |
| Mitigation | Retry with `MicroPlanRunner`'s built-in backoff. The Anthropic transient-error retry ([#404](https://github.com/mercurialsolo/mantis/pull/404)) covers extractor 5xx separately. |
| Tier | T1 (runtime — built-in backoff) |

### `selector_miss`

A `click`, `fill`, `submit`, or `select` step couldn't locate its
target. Stamped by individual step handlers as `click_error`,
`fill_error`, `select_error`, or `filters_not_applied`.

| Field | Notes |
|---|---|
| Signature | `data` contains `*_error`, `not found`, `no element`, `element not visible`, `filters_not_applied` |
| Root cause | Selector drift, late-rendering element, modal stacking, ad overlay |
| Observed instances | [#385](https://github.com/mercurialsolo/mantis/issues/385) (rewritten intents on lu.ma SPA grid still don't land); [#411](https://github.com/mercurialsolo/mantis/issues/411) (Claude vision mis-grounds the Title field on lu.ma 'Your Info' modal); [#412](https://github.com/mercurialsolo/mantis/pull/412) (tag-guard refusals re-fed as recovery hints) |
| Mitigation | `FormTargetProvider` protocol ([#407](https://github.com/mercurialsolo/mantis/pull/407)) lets a non-Claude VLM handle form grounding; `IntentRewriter` ([#379](https://github.com/mercurialsolo/mantis/pull/379)) rewrites stuck intents. |
| Tier | T1 primary (intent rewriter + form-target swap). T4 if a site class keeps reappearing → SFT target. |

### `no_state_change`

The handler reported success but the runner-state snapshot saw no URL /
page / scroll / focus change — i.e. the action did nothing observable.
Self-healing demotion signal for `click` / `submit` / `navigate_back`.

| Field | Notes |
|---|---|
| Signature | `data` ends with `:no_state_change`; emitted by self-healing demotion path ([#378](https://github.com/mercurialsolo/mantis/pull/378)) |
| Root cause | Handler over-reported success (e.g. clicked but DOM didn't react), or a React-controlled input didn't accept the synthesized event |
| Observed instances | Login forms across React SPAs (full chronology in [reports/INDEX.md](https://github.com/mercurialsolo/mantis/blob/main/reports/INDEX.md) runs 014–033 — xdotool typing → xclip paste → CDP `Input.insertText`) |
| Mitigation | `[#383]` click-demotion now reads `env.current_url` instead of handler-mutated runner state. `[#380]` preserves the handler-stamped `failure_class` so demotion can branch on it. |
| Tier | T3 (dispatcher — CDP `Input.insertText` over xdotool) + T1 (runtime — demotion signal) |

### `brain_loop_exhausted`

Inner GymRunner exited at its step budget or with a loop-detector trip,
without success. Means the intent was goal-shaped, not mechanical.

| Field | Notes |
|---|---|
| Signature | `data` contains `brain_loop_exhausted`, `max_steps`, `loop_terminated`; stamped by `RunExecutor` |
| Root cause | Step describes an outcome ("complete the form") rather than a verb ("fill_field name=…") |
| Observed instances | [#382](https://github.com/mercurialsolo/mantis/issues/382) (scroll burned brain budget without the stamp — fixed); [#302](https://github.com/mercurialsolo/mantis/issues/302) (loop recovery policy that decomposes the stuck class) |
| Mitigation | `IntentRewriter` ([#379](https://github.com/mercurialsolo/mantis/pull/379) Phase B); loop-recovery forced-class transitions ([reference/loop-recovery.md](../reference/loop-recovery.md)). |
| Tier | T1 (runtime — forced-class transition + intent rewriter) |

### `wrong_target`

`verify_post_click_navigation` decided the click DID navigate but to
the wrong destination — category card instead of an event detail,
login wall, ad, off-site link, …

| Field | Notes |
|---|---|
| Signature | `data` contains `wrong_target`; stamped by `ClaudeGuidedClickHandler` |
| Root cause | Card-vs-link disambiguation failure on dense grids; ad / interstitial caught the click |
| Mitigation | Intent rewriting ([#379](https://github.com/mercurialsolo/mantis/pull/379)) — issuing the same coords again won't help; the plan needs a different shaped intent. |
| Tier | T1 primary (intent rewriter). T4 candidate when concentrated on a site class — click-fixation is on the SFT queue via [#179](https://github.com/mercurialsolo/mantis/issues/179). |

### `extractor_error`

Claude extractor (or whichever extractor is wired up) failed or
returned empty.

| Field | Notes |
|---|---|
| Signature | `data` contains `scan_error`, `extract_error`, `extractor`, `scrape` |
| Common cause | Anthropic API transient error, prompt regression, screenshot too noisy |
| Mitigation | Transient retry ([#404](https://github.com/mercurialsolo/mantis/pull/404)); FAILURE_DATA legend in `recovery_analysis` prompt ([#415](https://github.com/mercurialsolo/mantis/pull/415)) so recovery sees the real error. |
| Tier | T1 (runtime — retry + recovery-prompt context) |

### `budget_exceeded`

Cost, time, or context budget tripped. Set by `ContextBudget` /
`CostMeter` ceilings.

| Field | Notes |
|---|---|
| Signature | `data` contains `budget_exceeded`, `max_cost`, `max_time`, `listing_budget_exceeded` |
| Mitigation | Reduce step count on the plan or raise the ceiling intentionally; never silently bump caps to mask runaway loops. |
| Tier | T2 (plan — shorten or decompose) |

### `unknown`

No rule matched. Caller should still surface `data` verbatim.

| Field | Notes |
|---|---|
| Action when seen | Read the raw `data` string from `result.json`; if it points at a real class, add it to `_DATA_RULES` in `failure_class.py` in the same PR as the fix. |

---

## Modes not yet in the taxonomy

These show up in production but the classifier rolls them into
`unknown` today. Adding them to `failure_class._DATA_RULES` is a
follow-up issue per row.

### Empty-summary `done(success=True)` on overlong plans

| Field | Notes |
|---|---|
| Trigger | Plans >~25k tokens; the model never engages with the workflow body |
| Signature | `result.json.success == true` AND `result.json.summary == ""` AND step count < 3 |
| Reference | `reports/INDEX.md` run 009 (`boattrader_scrape_bench`) is the canonical example |
| Mitigation | Decompose long plans with `PlanDecomposer` (`--decompose` flag); see [`#303`](https://github.com/mercurialsolo/mantis/issues/303) Done acceptance gate. |
| Follow-up | [#423](https://github.com/mercurialsolo/mantis/issues/423) — extend taxonomy with `false_success_empty_summary`. |

### Premature `done()` from per-step "Done when" verbiage

| Field | Notes |
|---|---|
| Trigger | Plans with per-step `Done when:` clauses; Holo3 satisfies the *first* and emits `done(success=True)` for the entire task |
| Signature | `summary` describes the current screen state ("login form is loaded") instead of the workflow outcome |
| Reference | `reports/INDEX.md` runs 010–011 — the DONE-CONDITIONS prompt rule fix landed at fix4 |
| Mitigation | Tighten `done()` semantics in the system prompt; replace per-step "Done when" with "proceed to Step N+1." |
| Follow-up | [#424](https://github.com/mercurialsolo/mantis/issues/424) — extend taxonomy with `premature_done_per_step_clause`. |

### Modal stacking / form re-mount between steps

| Field | Notes |
|---|---|
| Trigger | SPA modal closes between two `fill_field` calls — the second one finds no target |
| Observed | [#408](https://github.com/mercurialsolo/mantis/issues/408) lu.ma registration modal closes between `fill_field 'LinkedIn'` and `fill_field 'Title'` |
| Mitigation | Use `adaptive-settle` polling between fills; collapse adjacent fills into one micro-intent when possible. |
| Follow-up | [#425](https://github.com/mercurialsolo/mantis/issues/425) — extend taxonomy with `modal_remount_between_steps`; needs a programmatic detector. |

### Cross-product session drift on long chains

| Field | Notes |
|---|---|
| Trigger | A plan that hops sites (login on A → search on B → checkout on C) loses session cookies, scroll, or focus across hops |
| Today | `BrowserState` ([#386](https://github.com/mercurialsolo/mantis/pull/386), [#388](https://github.com/mercurialsolo/mantis/pull/388)) checkpoints URL + scroll + viewport + unsubmitted form input; `CheckpointManager` validates resume eligibility via SHA-256 plan signature |
| Open gap | [#361](https://github.com/mercurialsolo/mantis/issues/361) Phase C — full DOM / JS-state snapshot is still being evaluated. Until then, cross-site auth flows can drop session state between products. |
| Mitigation | Use `profile_id` / `workflow_id` ([#341](https://github.com/mercurialsolo/mantis/pull/341)) to keep the same Chrome profile across runs; pause/resume only works on `cua_model=holo3` over Modal HTTP today ([#347](https://github.com/mercurialsolo/mantis/issues/347)). |

### React-controlled input rejects synthesized keystrokes

| Field | Notes |
|---|---|
| Trigger | Login or registration forms on React SPAs; xdotool types the right characters but React's `onChange` never fires |
| Signature | Field appears empty after typing; subsequent submit fails with `AUTH_FAIL` or no state change |
| Reference | `reports/INDEX.md` runs 014–033 — chronology of xdotool delay → xclip paste → CDP `Input.insertText` |
| Mitigation | Today: CDP `Input.insertText` with `--remote-allow-origins=*` on Chrome launch flags. |

### Modal cold container serves stale code for ~10 min after deploy

| Field | Notes |
|---|---|
| Trigger | Verifying a fix on Modal immediately after `modal deploy` |
| Signature | The new code path doesn't fire on the next request because a warm container with the old code answers |
| Reference | [#393](https://github.com/mercurialsolo/mantis/pull/393); memory: `feedback_modal_warm_container_caveat.md` |
| Mitigation | `modal app stop <app> --yes` before redeploy when verifying a fix. |

### CDP screen-vs-viewport coordinate offset

| Field | Notes |
|---|---|
| Trigger | `elementFromPoint` lookups using screen Y instead of viewport Y |
| Reference | [#413](https://github.com/mercurialsolo/mantis/issues/413) / [#414](https://github.com/mercurialsolo/mantis/pull/414) |
| Mitigation | Already fixed — translation lives in the click handler. See [reference/coordinate-spaces.md](../reference/coordinate-spaces.md). |

---

## How a failure becomes a fix

The catalog above tells you *what* the mitigation is. This section
tells you *how* a new failure travels from a stuck production run to
a landed fix.

### Step 1 — Observe

Every run with `MANTIS_TRACE_EXPORT_DIR` set writes
`{run_id}.json` plus PNG screenshots
(`gym/trace_exporter.py:64`). The JSON carries:

* `failure_class` — the classifier's verdict (`gym/failure_class.py`).
* `healing_events[]` — every runtime intervention that fired:
  intent rewrites, click demotions, handler escalations,
  ExecutionCritic `insert_step` directives
  (`gym/healing_events.py:42–139`).
* Force-fill / force-submit / claude-director firings.
* `last_action.reasoning` — the brain's chain-of-thought on the final
  step.

**Gap:** the per-step `thinking` (Holo3 `<think>` blocks /
Claude extended-thinking blocks) is computed by the brain but only
the *last* action's reasoning persists into `result.json`. Open
follow-up: stamp `reasoning` onto every `StepResult` so triage can
diff "what the model said" vs "what the dispatcher did" on every
step, not just the terminal one.

### Step 2 — Triage: grounding vs tool vs plan

The taxonomy already encodes the axis most operators need:

| Class | What broke |
|---|---|
| `selector_miss` | Grounding — model couldn't identify the right pixel/element |
| `wrong_target` | Grounding — model picked the wrong element on a dense grid |
| `no_state_change` | Tool — dispatcher fired the action; the page didn't react |
| `brain_loop_exhausted` | Reasoning — intent was outcome-shaped, not verb-shaped |
| `extractor_error` | Reasoning — extractor brain failed |
| `cf_challenge` / `nav_timeout` / `http_*` | Infra — outside the agent loop |

What's **missing** is a per-step audit triple — `{reasoning,
predicted_outcome, observed_outcome}` — that would let you say
definitively "the model intended X, the dispatcher executed Y, the
page reacted Z." Today operators reconstruct this by hand from the
trace JSON + screenshots.

### Step 3 — Pick the tier

The mitigation ladder (top of this file) is ordered by cost-to-ship:

1. **T1 first** — can a runtime substitution / hint / forced
   action-class transition catch this? Cheap and shipping in the
   same PR as the bug fix.
2. **T2 if Tier 1 chases its tail** — if you're adding a third
   runtime override for the same class, the plan shape is wrong.
3. **T3 when a class of failures concentrates on one substrate** —
   if many runs trip on the same dispatcher gap (xdotool ↔ React
   keystrokes, missing `right_click`, no batched `fill_form`), add
   a primitive. T3 wins are usually larger than T1 wins per
   engineering hour.
4. **T4 only when prompts + runtime + tooling can't shift the
   metric.** The canonical signal: `reports/INDEX.md` run 013 —
   "14 nudges fired across steps 5–23. Model emitted 22 clicks, 1
   wait, **0 type_text**. Capability ceiling, not prompt or runtime
   issue." That's the kind of failure that goes onto the SFT queue.

### Step 4 — If Tier 4, feed continual learning

The pipeline ships end-to-end
([continual-finetuning.md](../operations/continual-finetuning.md)):

```
trace export → mantis trace label → convert_labelled_traces
  → train_holo3_distill → eval_harness → promotion_scorecard
  → ShadowRouter at 5% → full rollout
```

Today's state: every script is runnable; smoke tests exist; cadence
is **manual**. Shadow routing is wired with `candidate_pct=0` —
ready to ramp. The smallest concrete next step to close the loop is
a weekly job firing steps 2→5 against the prior week's
`MANTIS_TRACE_EXPORT_DIR` and appending the result to
[experiments/SCORECARD_HISTORY.jsonl](../experiments/scorecard-history.md).

### Step 5 — Measure

Append a row to
[`SCORECARD_HISTORY.jsonl`](../experiments/scorecard-history.md)
with `linked_pr` + `hypothesis` set so the diff against the prior
row tells you whether the fix moved any named gate. No measurement,
no learning.

## Updating this catalog

* **Ship docs with code.** A PR that fixes a failure mode adds (or
  edits) the row in this file. Self-check before opening.
* **One row per concrete instance**, not per class. Multiple instances
  of `selector_miss` get multiple rows.
* **Link the issue AND the PR.** Issues capture the report; PRs
  capture the fix. Future-you wants both.
* **Promote unknown→known.** Anything ending up in **Modes not yet in
  the taxonomy** should also have a follow-up issue tagged
  `failure-mode-taxonomy` to fold into
  `gym/failure_class.py:_DATA_RULES`.
* **Name the tier.** Every entry's `Tier` line answers "where does
  this get absorbed?" — that's how the catalog stays a triage tool
  and not just a list of bugs.
