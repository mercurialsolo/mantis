# mantis-holdout-v1 — typed holdout eval set

The frozen task set the slow-loop **champion/challenger gate** evaluates against
(mantis #894 / mantis-trainer). It is organized by **real-world capability TYPE**,
not by site — a holdout's job is to measure whether a new policy is better at the
*kinds of things agents do*, so a single site (e.g. boattrader) is just one
instance of `scrape`, not the eval itself.

`eval_set.json` is the manifest. Every task is grounded in a **real sim-env
oracle** (`/__env__/oracle?task_id=…`), so grading is deterministic ground-truth
(DB/mutation snapshots), never transcript-judged.

## Capability types & coverage

| Type | What it exercises | Holdout tasks (env · oracle) |
|---|---|---|
| `login` | sign-in across methods | auth·T01_password_login, auth·T07_email_otp, auth·T08_passkey, auth·T02_oauth_google |
| `scrape` | read structured data off detail pages | boattrader·BT02_spec_lookup_engine, boattrader·BT03_byowner_phone_reveal |
| `search` | query + filter + act on a result | boattrader·BT01_lead_capture_filtered_search, indeed·t01_search_save_remote |
| `form_fill` | fill + submit a multi-field form | indeed·t02_easy_apply, shopify·t04_create_support_ticket, mercor·t01_apply_to_ml_engineer |
| `crud_create` / `add_details` | create a record / add detail | crm·T04_add_meeting_note, shop·T03_create_coupon, linkedin·t02_post_text_update, fiverr·t03_leave_5star_review |
| `crud_edit` / `update` | modify an existing record | crm·T02_merge_acme_dupes, shop·T02_refund_line_item, shopify·t05_update_business_email |
| `export` | produce an export / side-effect | shopify·t03_export_payouts_csv, shop·T05_inventory_adjust |
| `navigate` | reach a specific section/record | shopify·t11_view_store_detail, indeed·t03_employer_review_applicant |

**24 tasks · 8 capability types · 9 envs.** Split: `visible` (seed 42, trainable)
vs `sealed` (seed 7, gate-only — never trained on). ~14 sealed / ~10 visible.

## Known gap: `upload`

There is **no true file-upload oracle** in any sim env today. The nearest proxies
are `easy_apply` (selects a `resume_id` — a *reference*, not a file upload) and
the CSV `export` tasks. So `upload` (drag/drop a file, attach a document, set an
avatar) is **uncovered** and the holdout can't measure it yet. **Recommended
follow-up:** add an upload oracle — e.g. avatar/document upload to `mantis_auth`
(profile) or a doc attach to `mantis_helpdesk` — then add it here under `upload`.

## How it's consumed

- **Trainer gate (direct):** loadable by `mantis_trainer.gate` / `eval_harness` —
  each entry is an `EvalTask` (`task_id`, `task_text`, `url`, `criteria`,
  `metadata`). The gate runs the challenger and champion over the **sealed** split
  and compares win-rate.
- **Per task:** the runner stands up `metadata.env` (Daytona/Modal), seeds it for
  the split, drives `task_text` from `metadata.start_path`, then grades via
  `GET /__env__/oracle?task_id=metadata.oracle_task_id` → `criteria.task_success`.

## Generating GRPO sibling rollouts (`run_rollout_sweep.py`)

The slow loop needs **sibling groups**: N rollouts of the *same* task instance
(template × env-seed) that share a `group_id`, with reward variance across
siblings so the group-relative advantage is non-zero.
`run_rollout_sweep.py` turns `SeedSweepGenerator` specs into real graded runs —
each sibling submits the template's micro-plan at `temperature > 0` (divergent
trajectories), forces Holo3 grounding (per-token logprobs for the GRPO ratio),
and is graded by the env oracle (#906).

```
python experiments/holdout/run_rollout_sweep.py indeed.t01_search_save_remote \
    --seed 42 --siblings 3 --temperature 0.7 --max-parallel 4
```

**THIS SPENDS** — N sibling Modal GPU runs per `(template, seed)`.

### Per-state-key parallelism (`state_key_dispatcher.py`)

Mantis **cannot** service two concurrent calls that share a `state_key`: the key
resolves (`server_utils.resolve_ids`) to a Chrome `profile_id` (user-data-dir)
*and* a `workflow_id` (checkpoint), so two in-flight runs on one key race on the
same profile + checkpoint. **Distinct keys are safe to run in parallel** (Modal
scales out containers via `@modal.concurrent`).

`StateKeyDispatcher` encodes that rule with the collision policy chosen **per
call**:

| Mode | `session=` | Behavior | Used by |
|---|---|---|---|
| **independent** | `False` (default) | auto-allocate a *fresh unique* `state_key`, run in parallel up to `max_parallel` | the sweep — siblings each get a distinct `sweep-<spec_id>` key |
| **session** | `True` | reuse a caller-supplied `state_key`, queued **FIFO** behind other session calls on that key (one at a time); distinct session keys still run in parallel | logged-in profiles / resumable checkpoints |

Waiting session calls don't occupy a worker slot (the real work is submitted to
the pool only once its same-key predecessor finishes), so a same-key backlog
can't deadlock the pool.

```python
from state_key_dispatcher import Call, StateKeyDispatcher

with StateKeyDispatcher(max_parallel=4) as d:
    # independent: 4 rollouts in parallel, each its own fresh key
    results = d.run_all([Call(make_work(spec)) for spec in specs])

    # session: two calls that must share one logged-in profile → serialized FIFO
    f1 = d.submit(step_a, state_key="acme-session", session=True)
    f2 = d.submit(step_b, state_key="acme-session", session=True)  # waits for f1
```

The sweep fans out siblings with `--max-parallel` (default 4; `1` = the old
sequential behavior). `variance-seek` stays sequential (decide, then add one) but
routes through the dispatcher too. Pure-Python/threads (no network) → unit-tested
in `tests/test_state_key_dispatcher.py`.

> **Throughput caveat:** client-side parallelism only becomes wall-clock speedup
> if the deployed CUA server actually runs the distinct-key calls on separate
> Modal containers. `MANTIS_RUNTIME_CONCURRENCY` defaults to 1 *per container*, so
> real parallelism comes from scale-out (`@modal.concurrent` min/max containers) —
> size the autoscaler max to match your `--max-parallel`.

## Running the promotion gate against a challenger (`run_gate_eval.py`, #916)

The trainer's gate evaluates a **challenger** (`base + LoRA adapter`, #911) vs the
**champion** (`base`) over the holdout. The catch: holdout tasks carry only
`env` + `task_id` + oracle — **no plan** — while `/v1/predict` requires a built
`task_suite`/`_micro_plan`. `run_gate_eval.py` is the missing execution link: it
**generates** the suite for each task (from `sealed_plans`, via `build_micro_suite`
— not raw steps, which silently score 0/0/0), runs both arms, oracle-grades, and
feeds the per-arm results to `promotion_gate.evaluate` → a `GateVerdict`.

```
# champion (base) vs challenger (adapter) over the 3 mantis-holdout-v1 tasks
python experiments/holdout/run_gate_eval.py \
    --task indeed.t01_search_save_remote \
    --task indeed.t03_employer_review_applicant \
    --task linkedin.t02_post_text_update \
    --lora-adapter mantis-trainer-vol:/checkpoints/sft-c3e0d799f432
```

Each (task, arm) gets a distinct `profile_id` (`gate-<arm>-<task>`), so the two
arms never collide on one Chrome profile (the per-`profile_id` 409 rule) and all
runs fan out in parallel via `StateKeyDispatcher` (`--max-parallel`, #912).
Omitting `--lora-adapter` runs champion == challenger (a plumbing sanity run;
expect `delta≈0`, `promote=False`). **This spends** — 2 × N Modal GPU runs.

**Equivalent path:** `--emit-tasks <path>` writes an `eval_harness`-shaped
`--tasks` JSON with the generated `micro_plan`s, so `training/eval_harness.py
run --lora-adapter … / compare` can consume holdout tasks that otherwise have no
plan. The suite-gen + result→`ArmResult` mapping are pure (unit-tested in
`tests/test_gate_eval_916.py`); the live submit arm is spend-gated.

## Freezing into an Augur eval-version (the official holdout)

The producer pipeline (mantis #901/#902) emits a `task_spec` + a `mark_for_eval`
candidate on each eval-worthy run. `experiments/holdout/freeze_eval_version.py`
turns that live candidate pool into the frozen version:

1. Run each `sealed` task through Mantis (real plan per env). Successes auto-flag
   as `source:producer` eval candidates (augur#178). Inspect what's freezable:
   `python experiments/holdout/freeze_eval_version.py --list` (read-only).
2. Freeze the sealed split into an immutable Augur **eval-version**, passing the
   explicit `task_spec_ids` (augur#181 honors the filter):
   `POST /eval-versions {name, task_spec_ids:[…]}` (operator session) — or
   `AUGUR_SESSION_COOKIE='session=…' python experiments/holdout/freeze_eval_version.py
   --name mantis-holdout-v1 --task-spec-id micro_indeed.t01_search_save_remote.v1 …`
3. `python -m mantis_trainer.holdout --version mantis-holdout-v1 --out tasks/eval_set.json`
   materializes the runnable holdout.

**Status (2026-06-14): `mantis-holdout-v1` is frozen — CLEAN** (3 distinct
oracle-verified task_specs, no probe noise): `micro_indeed.t01_search_save_remote.v1`
(search+save), `micro_indeed.t03_employer_review_applicant.v1` (navigate),
`micro_linkedin.t02_post_text_update.v1` (crud_create) — 2 envs, 3 capability
types. Each was driven through the deployed Modal CUA server against its live
Daytona env via `run_sealed_task.py` and graded `passed` by its env oracle.

**Two auth scopes (live-verified).** *Reads* (candidates/versions) accept the
producer key as `Authorization: Bearer <AUGUR_API_KEY>`. The *freeze write*
(`POST /eval-versions`) is **operator-session gated** — the producer key is
rejected (401). Done from a logged-in Augur tab via an in-page
`fetch('/api/v1/eval-versions', {method:'POST', credentials:'include'})` (carries
the session cookie; never exposed). `freeze_eval_version.py` does the same with
`AUGUR_SESSION_COOKIE` when headless.

**Curation (augur#181 — RESOLVED).** `POST /eval-versions` now honors a
`task_spec_ids` / `run_ids` filter **and** the candidate merge now excludes
archived runs — so a version freezes to exactly the specs you select. `v1` is
frozen with the explicit 3 `task_spec_ids` and contains no probe noise. (Before
#181: freeze snapshotted the whole pool; `demote-from-eval` no-op'd producer tags
and `bulk/archive`'s `archived_at` was ignored — the v0 superset that motivated
the issue.)

**Keying note.** Producer candidates are keyed `micro_<env>.<plan_name>.v1`
(`build_micro_suite` prefixes `micro_`), e.g. `micro_indeed.t01_search_save_remote.v1`
— cosmetically different from this manifest's `indeed.t01_search_save_remote` anchor.
The freeze takes whatever ids the runs produced; the consumer maps by suffix.

Until those runs land, `eval_set.json` here is the **curated definition** (the
source of truth for *what* the holdout contains); the Augur eval-version is the
frozen, run-backed instance.

## Why these tasks

- **Type coverage over site coverage** — every capability an agent needs is
  represented at least twice (different envs) so the gate isn't gameable by
  overfitting one site.
- **Deterministic oracles** — no LLM-judge in the gate path; every task has a
  DB/mutation grader with collateral (precision) guards.
- **Held-out split** — `sealed` (seed 7) is structurally distinct from the
  `visible` training seed, so a gate win reflects generalization, not memorization.
