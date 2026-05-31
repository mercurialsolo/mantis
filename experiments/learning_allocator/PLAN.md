# The Learning Allocator on Mantis — Experiment Plan

Adaptation of *"The Learning Allocator: Continual Agent Improvement as
Budget-Constrained Substrate Selection"* to the Mantis CUA testbed.

Status: draft v0.1. Tracked by the `[EPIC] Learning Allocator on Mantis`
task and its children. Not peer reviewed.

---

## 0. Thesis — what we adopt, what we adapt

**Adopt (the paper's durable core):**

- Continual improvement = **substrate selection** under a compute budget,
  not a single fixed learning mechanism.
- A cost/durability-ordered **substrate ladder** S0 → S3.
- Reward `R = Δscore − λ·cost`; hard budget `B`.
- The two properties that make the problem *meta*: **partial observability**
  (true competence is never observed, only a noisy estimate) and
  **non-stationarity** (the value of each substrate shifts as the agent matures).
- The tractable instance: a **myopic contextual-bandit** allocator.

**Adapt for Mantis (the deltas that make this a worthwhile testbed):**

1. **Domain is visual CUA, not voice.** The noisy-attribution channel is
   *vision misperception + layout drift + LLM-verifier self-disagreement*
   instead of ASR error. Mantis therefore tests the paper's **general** claim
   as a **second vertical** — it does *not* test the voice/ASR headline.
2. **Dual reward with ground truth — the key upgrade.** Mantis ships BOTH an
   expensive accurate **oracle** (the sim-env mutation-log grader) AND a cheap
   noisy **proxy** (the LLM verifier). The voice domain has a noisy reward but
   *no ground truth to measure the noise against*. Here we can (a) **quantify**
   the partial observability the paper can only assume (§3.2), and (b) **train
   the PRM the paper leaves stubbed** (§3.4). This is the single biggest reason
   to run the experiment on Mantis.
3. **S2 is reframed.** Mantis synthesizes *plans / skills / playbooks*, not
   Python tools. S2 = **skill/macro synthesis** (GraphLearner + playbook
   auto-generation), not tool synthesis. We state this reframing explicitly.

---

## 1. Substrate ladder mapped to Mantis

| Rung | Paper substrate | Mantis module(s) | Status today |
|---|---|---|---|
| **S0** | Retrieval / context | `gym/hint_memory.py`, `grounding_cache.py`, `curriculum/` | **LIVE** (called from `run_executor.py`) |
| **S1** | Exemplar | `gym/trace_exporter.py` → `gym/trace_labeller.py` → exemplar store | collection **LIVE**, inference-time replay **TO BUILD** |
| **S2** | Tool synth → **skill/macro synth** | `graph/learner.py`, `recipes/`, `verification/playbook.py` | **PARTIAL** (synthesizes plans, not tools — reframe) |
| **S3** | Weight consolidation | `training/rollout_collector.py` → `training/train_holo3_distill.py` | **SCAFFOLDED, NEVER RUN** (zero checkpoints, placeholder model names) |

Ordering by cost is **measured**, not assumed: hint cache (≈free, reversible)
< exemplar replay (cheap) < skill synthesis (moderate Claude cost) <
distillation (expensive, durable). Cost per rung comes from `gym/cost_meter.py`
+ `observability/claude_cost_meter.py`.

---

## 2. Reward model (the Mantis advantage)

- **Sealed oracle benchmark (ground truth):** `/__env__/oracle` + per-task
  mutation-log graders in `deploy/sim_envs/*/app/oracles/`. Reads server DB
  state, not the agent transcript.
- **Cheap online proxy (noisy):** the LLM verifier —
  `extraction/extractor.verify_gate`, `verification/step_verifier.py`,
  `gym/failure_class.py`.
- **Reward:** `R = Δ(oracle score) − λ·cost`, cost in dollars from the meters
  above.
- **Partial observability is literally measurable:** log oracle verdict vs
  proxy verdict per step → false-pass / false-fail rates → PRM training data.

---

## 3. Heterogeneous eval, by construction

The central claim (allocator beats any fixed substrate) is only meaningful if
the best substrate genuinely varies. We construct three failure clusters in the
white-box envs — we control the fixtures, so the heterogeneity is engineered,
not hoped for:

- **Knowledge gap** — the task answer is an indexable fact (a help-doc or
  catalog row the agent must look up). **S0 should win.**
- **Capability gap** — the task needs a multi-step macro the base agent fumbles
  but a synthesized skill nails. **S2 should win.**
- **Policy gap** — a recurring mis-handling (e.g. wrong-target click on a
  drifting layout). **S1 early, S3 once frequent.**

Envs: `mantis_crm` + `mantis_shop` for Phase 2–3; add `mantis_helpdesk` for a
generalization check. Sealed/visible split via seeded DB; rotate the sealed set
periodically. `failure_class.py` provides a starting *mechanical* signal, but
clusters are defined at the **task** level, not by mechanical class.

---

## 4. Baselines & protocol

- **Frozen** agent (lower reference).
- **Fixed-substrate** policies: S0-only, S1-only, S2-only, S3-only.
- **Learning Allocator** (ours).
- **Oracle-allocator** — best substrate per signal (upper reference / headroom).

All learning policies run under an **identical compute budget** `B`; report
mean ± variance across seeds. Report the **visible − sealed** gap as the
overfitting check.

---

## 5. Metrics & artifacts (Mantis-ified paper tables)

- **Table 1** — sealed score, score/compute, visible−sealed, per policy.
- **Table 2** — improvement-per-dollar, substrate × failure-cluster
  (off-diagonal structure = premise holds; a flat matrix falsifies it).
- **Fig 1** — oracle score vs **cumulative dollars** (not steps); allocator
  tracking the oracle above all fixed substrates. *Headline.*
- **Fig 2** — allocator selection distribution over time (shift
  exemplar→skill→consolidation = the non-stationarity justification).
- **Fig 3** — Table 2 heatmap.
- **Fig 4** — budget frontier (final sealed score vs `B`).
- **Fig 5** — visible vs sealed paired bars (overfitting check).
- **Fig P (beyond the paper)** — proxy-vs-oracle calibration / attribution
  noise per cluster; the PRM result.

**Minimum reportable set** if compute compresses: Fig 1 + Fig 3 + Table 1
(sealed column) + Fig P.

---

## 6. Phased plan (de-risked, cheap-first)

| Phase | Goal | Why this order |
|---|---|---|
| **0 Foundations** | uniform substrate interface; failure-cluster eval; dual-reward wiring | nothing runs without the action space + a heterogeneous eval |
| **1 Reward/PRM grounding** | measure oracle-vs-proxy noise; fit the PRM | cheapest, highest value-per-$, de-risks every later reward estimate |
| **2 Two live substrates + allocator MVP** | S0 + S1 behind the interface; myopic bandit; partial Table 1 + Fig 1 | first real result using only LIVE substrates |
| **3 S2 skill-synthesis substrate** | add S2; Table 2 off-diagonal | proves the heterogeneity premise |
| **4 S3 weight consolidation** | **actually run** the distillation loop; full Table 1/2 + Fig 2 | the repo's long-deferred piece; the only path to the full 4-rung claim |
| **5 Analysis & write-up** | overfitting/frontier/variance; second-vertical write-up | turns runs into the paper section |

---

## 7. Honest gating & risks

- **S3 is the only rung never run in this repo** (zero checkpoints, placeholder
  model names). Phases 0–3 produce a real **partial** result without it; the
  **full** 4-substrate claim is gated on Phase 4 actually succeeding.
- **Still single-domain.** This strengthens the paper as a *second* vertical;
  it does not validate the voice/ASR claim.
- **"Builds its own benchmark"** risk (paper §6) is mitigated by the sealed
  split + oracle upper bound + determinism already tested in
  `tests/sim_envs/*/test_oracle_determinism.py`.
- The myopic bandit ignores **sequencing** — most importantly *when to
  consolidate* (wait for enough exemplars before promoting to S3). Named as the
  bandit→full-POMDP gap, not hidden.

---

## 8. File targets (where code lands)

```
src/mantis_agent/learning/
  allocator.py                 # the myopic contextual bandit
  reward.py                    # oracle + proxy reward, PRM
  substrates/
    base.py                    # LearningSubstrate protocol + SubstrateResult
    retrieval.py               # S0  (wraps hint_memory / grounding_cache)
    exemplar.py                # S1  (wraps trace_exporter + replay)
    skill.py                   # S2  (wraps graph/learner + playbook)
    consolidation.py           # S3  (wraps training/rollout_collector + distill)
experiments/learning_allocator/
  eval/                        # failure-cluster manifests + sealed split
  results/                     # results.tsv, figures
  WRITEUP.md                   # second-vertical paper section
tests/learning/                # interface + bandit unit tests
```
