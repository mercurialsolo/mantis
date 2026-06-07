# The Learning Allocator on Mantis — Phase-2 Results

A second-vertical port of *"The Learning Allocator: Continual Agent Improvement
as Budget-Constrained Substrate Selection"*. See `PLAN.md` for the full design;
this document is the runnable, dated record of where the empirical work
currently stands. Numbers cite the on-disk artifacts that produced them.

**Status (as of this commit).** Phase 2 (two live substrates + allocator) is
closed on the **knowledge** cluster, the discriminator for the **policy**
cluster is built and committed but its live result is gated on fresh spend
authorization, and the **capability** cluster remains held out of the live set
by design. Phase 4 (S3 / weight consolidation) is not addressed by this writeup
— the full 4-rung claim is explicitly gated on that loop running end-to-end.

---

## 1. What "closing the loop" means here

The paper's headline claim is that an allocator beats *any* fixed substrate by
learning a per-cluster winner. PLAN §3 makes the heterogeneity testable by
engineering three failure clusters in the white-box `mantis_boattrader` sim
env, each oracle-graded:

- **BT01 — capability gap** (multi-step filter+form macro). Expected winner: S2.
- **BT02 — knowledge gap** (Caterpillar engine-spec lookup). Expected winner: S0.
- **BT03 — policy gap** (by-owner phone reveal). Expected winner: S1.

"Closing the loop" was: (a) all three oracles graded against the live env, (b)
each cluster's discriminator producing a behaviorally separable frozen-vs-rung
gap, (c) the matrix runner streaming results into `experiments/learning_allocator/eval/`,
and (d) the Phase-2 Table 1 / Fig 1 PLAN §5 names being assembleable from those
files. (a) and (c) landed earlier; (b) and (d) are what this PR finishes.

---

## 2. BT02 — knowledge cluster (VALIDATED)

**Mechanism.** The sim env (`seed._apply_deep_caterpillar`, gated by the
default-on `BT02_DEEP_CATERPILLAR`) buries all 60 Caterpillar-powered boats and
re-pins the two lowest-id ones as recommended-sort ranks 5 and 10. A frozen
agent crawls listings in natural order and budget-caps before reaching the
buried target. The S0 retrieval substrate stamps a `preferred_target_description`
hint onto the listing-title click step; `ClaudeGuidedClickHandler` reorders the
scanned cards so the rank-5 boat is iterated first.

**Pre-seeded anchor.** The Modal Dict `la-bt02-spec-lookup-hints-v5` carries
the rank-5 Viking 68 anchor under `plan_sig bf03055b920b / intent_hash 6700cf1d43c5`
with `url_pattern=""` and `confidence=0.9`. Re-seed is **free** and must be
re-applied after any plan edit (plan_signature is a content hash; an edit
orphans every prior anchor — see project memory `plan-edit-churns-hint-signature`).

**Result (live, 2026-06-01, 16 runs, $0.89 total — well under the $3–5
estimate).** Files:

- `experiments/learning_allocator/eval/bt02_matrix_results.tsv` — per-run rows.
- `experiments/learning_allocator/eval/bt02_matrix_table1.tsv` — aggregate.
- `experiments/learning_allocator/eval/bt02_matrix_fig1.tsv` — running oracle vs cumulative $.

| policy             | sealed | visible | vis−seal | score/$ | dollars | n |
| ------------------ | -----: | ------: | -------: | ------: | ------: | -: |
| frozen             |  0.00  |  0.00   |  0.0000  |   0.00  | 0.4355  | 8 |
| S0_only            |  1.00  |  1.00   |  0.0000  |  21.86  | 0.3660  | 8 |
| oracle_allocator   |  1.00  |  1.00   |  0.0000  |  22.50  | 0.0889  | 2 |

The discriminator is **also a cost asymmetry**: S0 succeeds cheaper than frozen
fails — frozen burns budget on the wrong cards before capping; S0 clicks the
rank-5 Viking on iteration 1. `vis−seal = 0.0000` for every policy because the
S0 hint key is `(plan_sig, intent_hash, url_pattern)` — seed-agnostic — so the
sealed seed-7 task hits the same anchor. (PLAN §4 overfitting check passes by
construction; the row is reported anyway for the table.)

The framework changes that made this work shipped in PR #765
(`feat(la): BT02 reach-asymmetry discriminator + CUA primitives`):
`MicroIntent.stop_var` for loop early-exit, the `preferred_target` reorder in
`ClaudeGuidedClickHandler`, a generic `fraction` knob on the CDP scroll path so
the spec block can be screenshotted without overshooting, and the
`_apply_deep_caterpillar` re-seed in the sim env.

---

## 3. BT03 — policy cluster (BUILT, LIVE GATED)

**The pincer the discriminator had to escape.** A layout-drift fixture seems
like the natural BT03 discriminator: relabel the "Show Phone Number" control,
expect S1 (exemplar replay) to outperform frozen. Empirically this fails — see
project memory `bt03-s1-discriminator-mismatch-2026-06-02` for the full session
log. The two-arm trap:

- **Relabel the target** (hard to find). Frozen fails — but S1 fails too,
  because the exemplar surfaces *what worked* (a click reveal), not *where to
  click* — that lever is S0's. Both 0.
- **Verbatim target + teaser** (easy to find). S1 CAN ground it — but so can
  frozen, so the frozen arm passes. No gap.

The single-screen reveal has no middle: any drift strong enough to make frozen
omit also removes the groundable target S1's replay depends on.

**The escape (gated-reveal, PR #773 MERGED 2026-06-03 squash `93727d0`).**
Drop the target confound entirely and move the difficulty into the *action
sequence*. The new env mode `BT03_REVEAL_GATE=1` (`seed._apply_byowner_reveal_gate`
plus a new `POST /boat/{slug}/contact-start` route) refuses the `phone_revealed`
mutation until a non-obvious prerequisite — "Start contact request" — has been
sent. The reveal control keeps its verbatim "Show Phone Number" label and stays
groundable. Frozen runs the shared decomposed plan (`plans/bt03_gated_reveal.json`,
tracked via a `.gitignore` negation per memory `gitignored-plan-fixture-fails-ci`)
as-authored — the reveal is a server-side no-op. S1 injects the missing
prerequisite step before the reveal via `apply_exemplar_overlay` (the
`bt03_gated_inject_exemplar.json` exemplar carries an `inject_before` flag);
the gate is satisfied and the unchanged reveal fires.

The injection is **loop-safe**: `_apply_injections` renumbers any `loop_target
>= insert_index` so the per-boat prerequisite re-runs each loop iteration. This
mirrors `agentic_recovery.splice_inserted_steps`; the contract is locked by
three new tests in `tests/learning/test_bt03_gated_discriminator.py`.

**Status.**
- Static discriminator: proven in CI. 28 overlay tests green; 10 env tests
  green; 6 end-to-end tests load the shipped plan+exemplar through the
  `live_runner` path and assert frozen omits / S1 injects exactly one prereq
  before the reveal / loop_target survives.
- Live behavioral separation: **MEASURED, NO LIVE SEPARATION YET**. Two
  authorized smoke runs against a freshly-booted Daytona env with
  `BT03_REVEAL_GATE=1`:

  | run | --max-cost | frozen oracle / $ | S1_only oracle / $ | env mutations beyond reset |
  | --- | ---: | --- | --- | --- |
  | minimal smoke (2026-06-06) | 0.40 | 0.00 / $0.072 | 0.00 / $0.143 | none |
  | higher-budget rerun (2026-06-07) | 1.20 | 0.00 / $0.000* | 0.00 / $0.138 | none |

  *The frozen arm of the higher-budget rerun hit a submit-time cost-meter
  miss — `[pull_cost] WARNING: cost file never appeared after 45s` — so the
  recorded $0.00 understates actual spend. Likely small (~$0.05–0.10), since
  the run aborted very quickly.

  Both runs are durable artifacts under
  `experiments/learning_allocator/eval/bt03_gated_matrix{_,_hi_}{results,table1,fig1}.tsv`.

  **What the data says:** the env's `/__env__/mutations` log shows ONLY the
  orchestrator's `env_reset` after either run — neither arm fired a
  `phone_revealed` or even a `contact-start`. The cap-quintupling between
  runs didn't change S1's spend per arm (~$0.14 in both), which means the
  halt isn't actually `--max-cost`-bound: the agent halts on an earlier
  step's brain budget — the listing-title click step, the largest in the
  plan. The agent never gets past navigation + listing-click into the
  reveal step where the gate / inject discriminator would actually fire.
  Static binary by construction; live `0 vs 0` because both halt upstream.

  **What this isn't:** it is NOT the layout-drift pincer (which was
  structural — the gated mechanism *would* separate if the agent reached
  the reveal step). The gated discriminator's mechanism is intact; the
  bottleneck is upstream framework behavior on the early plan steps.

  **What remains to close the live row cleanly:** either (a) widen the
  listing-click step's brain budget so the agent has more attempts to land
  the click before halting, (b) split the plan so the discriminating
  ACTION sequence lives in a deterministic-first step the agent can reach
  cheaply, or (c) cache the listing-click via S0 so it succeeds on
  iteration 1 (the BT02 reach-asymmetry recipe applied to BT03 — same
  `preferred_target_description` reorder path). All three are framework /
  plan-shape changes, not new spend; none are in scope for this PR.

---

## 4. BT01 — capability cluster (OFFLINE-ONLY)

BT01's oracle grades a clean *dealer* lead. The live set runs under the
no-dealer-lead policy, so the cluster carries `plan: null` and is intentionally
not a live Table 1 row. It stays gradeable offline and is the natural target
for the S2 skill/macro substrate (PLAN §3), which Phase 3 will land — not in
scope for this writeup.

---

## 5. Reproducing the BT02 result and running the BT03 smoke

Pre-flight everything first (free):

    uv run python -m experiments.learning_allocator.live_runner \
        --plan plans/bt03_gated_reveal.json \
        --policies frozen,S1_only \
        --exemplars experiments/learning_allocator/eval/bt03_gated_inject_exemplar.json \
        --dry-run

The dry-run validates the `--policies` tokens against `POLICY_SUBSTRATES`
(closes the loss vector in feedback memory `la-policy-vs-substrate-namespace`
where a `S1_exemplar` typo stranded $0.91 of valid earlier-arm spend), decomposes
the plan, and pings `/__env__/oracle` to confirm the Daytona preview token is
correct. Drop `--dry-run` to spend.

Then consolidate:

    uv run python -m experiments.learning_allocator.consolidate \
        --out experiments/learning_allocator/eval/phase2_table1.tsv

Re-runs are idempotent; the table refreshes the moment the BT03 matrix lands.

---

## 6. Honest gating (the PLAN §7 follow-through)

- **S3 is still the only rung never run end-to-end in this repo.** The full
  4-substrate claim is gated on Phase 4 actually running the distillation loop;
  Phase 2 alone is a partial result, not the headline.
- **BT03 live separation is measured 0 vs 0 at both `--max-cost 0.40` and
  `--max-cost 1.20`.** The CI-proven static discriminator is binary by
  construction, but the agent halts on the listing-click step before either
  arm reaches the reveal step. The bottleneck is plan-shape / framework, not
  the discriminator design or run budget — the listing-click step's brain
  budget caps the spend per arm at ~$0.14 regardless of `--max-cost`. The
  next data point is a plan-shape change (widen the listing-click budget /
  add a `preferred_target` S0 anchor / restructure the discriminating
  sequence to a deterministic-first step), not more spend.
- **Second-vertical caveat.** This program strengthens the paper as a second
  vertical (visual CUA); it does not validate the original voice/ASR headline.
- **Bandit ignores sequencing.** Most importantly, *when to consolidate* — wait
  for enough exemplars before promoting to S3 — is a bandit→POMDP gap, not
  hidden.

---

## 7. Artifacts (one place to find everything)

- Plan & overview: `experiments/learning_allocator/PLAN.md`.
- Live runner (Phase-2, spends): `experiments/learning_allocator/live_runner.py`.
- Offline runner / orchestrator harness: `experiments/learning_allocator/runner.py`.
- Phase-2 Table 1 consolidator: `experiments/learning_allocator/consolidate.py`.
- BT02 matrix (validated):
  `experiments/learning_allocator/eval/bt02_matrix_{results,table1,fig1}.tsv`.
- BT03 gated discriminator artifacts:
  - Plan: `plans/bt03_gated_reveal.json`.
  - Exemplar: `experiments/learning_allocator/eval/bt03_gated_inject_exemplar.json`.
  - Tests: `tests/learning/test_bt03_gated_discriminator.py`.
  - Live smoke (2026-06-06): `experiments/learning_allocator/eval/bt03_gated_matrix_{results,table1,fig1}.tsv`.
- Consolidated Phase-2 Table 1: `experiments/learning_allocator/eval/phase2_table1.tsv`.
- Cluster manifest: `experiments/learning_allocator/eval/clusters.json`.
- PR topology: `#749` phase-0 → `#750` phase-0b → `#751` substrates → `#752`
  orchestrator → `#755` oracles → `#765` BT02 discriminator → `#773` BT03
  gated-reveal discriminator. All squash-merged to `main`.
