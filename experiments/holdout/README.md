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

## Freezing into an Augur eval-version (the official holdout)

The producer pipeline (mantis #901/#902) emits a `task_spec` + a `mark_for_eval`
candidate on each eval-worthy run. `experiments/holdout/freeze_eval_version.py`
turns that live candidate pool into the frozen version:

1. Run each `sealed` task through Mantis (real plan per env). Successes auto-flag
   as `source:producer` eval candidates (augur#178). Inspect what's freezable:
   `python experiments/holdout/freeze_eval_version.py --list` (read-only).
2. Freeze the sealed split into an immutable Augur **eval-version**:
   `AUGUR_SESSION_COOKIE='session=…' python experiments/holdout/freeze_eval_version.py
   --name mantis-holdout-v1 --select-prefix boattrader.`
3. `python -m mantis_trainer.holdout --version mantis-holdout-v1 --out tasks/eval_set.json`
   materializes the runnable holdout.

**Status (2026-06-14): `mantis-holdout-v1` is frozen** — run-backed, containing 3
oracle-verified producer candidates: `micro_indeed.t01_search_save_remote.v1`
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

**Curation gap (augur#181) — the version is a SUPERSET.** `POST /eval-versions`
snapshots the *entire merged candidate pool* (`{name, description}` only; no
`task_spec_ids` filter). Producer `mark_for_eval` tags are sticky: `demote-from-eval`
returns 200 but is a no-op for them, and `bulk/archive` sets `archived_at` but the
merge ignores it. So `mantis-holdout-v1` also carries 6 probe candidates from
`#901/#902` verification runs (`mantis:probe` env, `micro_example_com`). **Consumer
workaround:** the gate/materializer filters to specs that map to a manifest
`oracle_task_id` (the probe specs have no env/oracle mapping → skipped). Fixed
cleanly once augur#181 lands (exclude archived from merge, or honor `task_spec_ids`).

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
