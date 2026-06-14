# Rollout Generator — synthetic rollout/plan generation for the CUA flywheel

Status: proposal (epic [#894](https://github.com/mercurialsolo/mantis/issues/894))
Scope: Mantis only. Consumes Augur (read-only); **no Augur changes** (see "Augur-side notes").

## Why

The continuous-improvement flywheel can already *execute* rollouts (Holo3/Claude
executors), *score* them (sim-env oracles), and *improve* non-parametrically
(recipes/hints/curriculum). The missing upstream piece is a **source of diverse
rollouts**. Today the task set is a static, hand-authored manifest
(`experiments/learning_allocator/eval/clusters.json`). Without a generator, the
loop can only re-run the same fixed tasks — it can't explore, can't target its
own weaknesses, and can't scale ground-truth-labeled data for the RL (slow) loop.

The lever Daytona unlocks: the sim envs are **seed-parameterizable** (`POST
/__env__/reset {seed}`) and **oracle-graded** (`GET /__env__/oracle?task_id`). So:

> **one task template × N env seeds = N distinct, ground-truth-labeled rollouts** — synthetic RL data with zero human labeling.

## Grounding (existing contracts the generator builds on)

| Surface | Where | Use |
|---|---|---|
| `POST /__env__/reset {seed}` | every sim env (`modal_stub.py:105`, boattrader/shopify/...) | mint a fresh env instance per seed |
| `GET /__env__/oracle?task_id` | sim envs | ground-truth reward (`learning/reward.py:oracle_channel`) |
| `PlanDecomposer.decompose_text()` | `plan_decomposer.py:1070` | free-text template → MicroPlan |
| `build_micro_suite()` | `server_utils.py:1177` | MicroPlan → runnable suite |
| `group_id` + `open_orchestrator_session` | `gym/fanout_runner.py`, Augur | GRPO sibling grouping for RL diversity |
| `list_failure_clusters` | Augur MCP/API (read-only) | bias generation toward weak spots |
| `Phase2Orchestrator` | `learning/orchestrator.py` | consumes (task, substrate) → run → reward → observe |

## Design

A `RolloutGenerator` emits `RolloutSpec`s; the orchestrator turns each into a
graded run. Two concrete generators cover the two data needs (volume + targeting).

```
TaskTemplate ──┐
               ├─► RolloutGenerator.generate() ─► Iterator[RolloutSpec]
env seeds   ───┘                                      │
                                                      ▼
        for each spec:  POST /__env__/reset {seed}  (mint instance)
                        decompose_text(template) → build_micro_suite
                        execute (Holo3/Claude) under group_id (siblings)
                        GET /__env__/oracle?task_id  → reward
                        → Augur bundle (one per rollout, grouped by group_id)
```

### Types
- `TaskTemplate(template_id, cluster, plan_text | plan_steps, oracle_task_id)` — a parameterizable task. `oracle_task_id` ties the rollout to its grader.
- `RolloutSpec(spec_id, template, env_seed, group_id, sibling_index)` — one concrete graded run to execute.

### Generators
1. **`SeedSweepGenerator`** — `templates × seeds × siblings_per_instance`. The volume engine: deterministic, exhaustive over a seed range. Each `(template, seed)` is a distinct env instance; `siblings_per_instance` ≥ 2 produces GRPO siblings sharing a `group_id`.
2. **`FailureBiasedGenerator`** — allocates a fixed instance budget across clusters **proportional to failure share** (failure counts passed in as plain data, read from Augur `list_failure_clusters` by the caller — the generator never imports Augur). Generates where the agent is weakest.

Both are pure/deterministic given an RNG seed → reproducible rollout sets, replayable for eval.

## Integration (next step, not in the scaffold)

A thin `RolloutRunner` adapter (mirrors `experiments/learning_allocator/live_runner.py`):
for each `RolloutSpec` → reset env to `seed` (with Daytona preview headers) →
`decompose_text` + `build_micro_suite` (fresh `workflow_id`, carry `group_id`) →
submit → poll → `reward_from_run` → record. Feeds `Phase2Orchestrator` a *stream*
instead of a static manifest.

## Phasing
- **P0** — types + `SeedSweepGenerator` + `FailureBiasedGenerator` (this scaffold) + tests. No execution wiring.
- **P1** — `RolloutRunner` adapter: spec → Daytona reset → execute → oracle reward → Augur. Wire into the orchestrator.
- **P2** — close the explore loop: pull `list_failure_clusters` each round → re-bias → generate → run → re-cluster.

## Augur-side notes (for a separate Augur PR — do NOT change here)
- **Required: none.** Reading `list_failure_clusters` + per-run bundles via the existing MCP/HTTP API is sufficient.
- **Optional convenience (nice-to-have, file separately):** a "register a rollout *group* as a dataset slice" helper so a seed-sweep's N sibling bundles can be pulled as one training shard by `group_id` (today: query bundles and filter by `group_id` tag client-side — works, just less ergonomic).
