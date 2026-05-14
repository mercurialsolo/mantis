# Wall-time aggregation

`summary.wall_time_breakdown` (epic [#362](https://github.com/mercurialsolo/mantis/issues/362)) reports where each run's seconds went. This page describes the **durable per-run log** the runtime writes on terminal status and the `mantis runs stats` CLI that aggregates p50/p95/p99 across runs.

The triad it replaces:

- Prometheus is for **live alerting** — cardinality kills it for per-workflow historical analysis.
- `result.json` is per-run — useful for one post-mortem, not for comparing across a week's worth of runs.
- The JSONL log + `mantis runs stats` answers "across the last 100 runs of workflow X, what's the p95 latency per bucket, and which bucket regressed since last month."

## Where the log lives

```
$MANTIS_DATA_DIR/runs_log/<YYYY-MM>.jsonl
```

Sharded by month so a single file stays bounded over time and old months can be archived independently. Defaults to `/workspace/mantis-data/runs_log/...` on Baseten, `/data/mantis-runs/runs_log/...` on Modal.

The runtime appends **one line per terminal run** (succeeded / failed). Paused runs are not logged — they'll land on the next terminal transition after `action=resume`.

## Row schema

Each line is a JSON object with stable keys. Permissive — extra keys are allowed so future fields don't break readers.

```jsonc
{
  "schema_version": 1,
  "run_id":            "20260513_180527_abc12345",
  "tenant_id":         "default",
  "profile_id":        "default__alice-prod",
  "workflow_id":       "default__marketplace-listings-v1",
  "plan_signature":    "a1b2c3d4e5f6",
  "model":             "Hcompany/Holo3-35B-A3B",
  "status":            "succeeded",          // succeeded | failed
  "created_at":        "2026-05-13T18:05:27Z",
  "finished_at":       "2026-05-13T18:09:34Z",
  "total_time_s":      247,
  "wall_time_breakdown": { ...9-bucket dict from Phase B... },
  "cost_breakdown":     { ...same shape as summary.cost_breakdown... },
  "steps_executed":    12,
  "viable":             3,
  "error":              null
}
```

## Append semantics

- **Atomic** between concurrent writers on one pod — files opened in append mode get POSIX `O_APPEND` guarantees for small writes, and a JSONL line easily fits under `PIPE_BUF` (4096 B+ on every modern OS). The writer issues one `write()` per row + `fsync` for durability.
- **Best-effort** — bookkeeping failures (full disk, EIO, …) are logged and swallowed. The runtime never breaks a finishing run because the log couldn't be written.
- **No external dependency** — pure stdlib. DuckDB / Postgres can replace the query side later when row counts demand it; the writer doesn't need to change.

## Querying with `mantis runs stats`

```
mantis runs stats <workflow_id> [--last 100] [--bucket NAME ...] [--since 7d] [--status STATUS] [--json]
```

Default rendering — bucket × percentile table, ordered by `p50` descending so the worst offender lands first:

```
$ mantis runs stats default__marketplace-listings-v1

Across last 100 succeeded runs of default__marketplace-listings-v1:

bucket                  p50        p95        p99
──────────────────────────────────────────────────
think                  92.0s     145.0s     201.0s
claude_extract         67.0s     128.0s     156.0s
settle                 28.0s      44.0s      55.0s
claude_ground          18.0s      31.0s      42.0s
perceive               11.0s      18.0s      22.0s
load                   10.0s      19.0s      28.0s
act                     6.0s       9.0s      12.0s
claude_verify           4.0s       7.0s       9.0s
overhead                2.0s       4.0s       5.0s
──────────────────────────────────────────────────
total_time_s          238.0s     383.0s     478.0s
```

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--last N` | `100` | Newest-first cap on rows considered. Truncation is deterministic — most recent terminal runs first. |
| `--bucket NAME` | — (all) | Restrict to a subset of wall-time buckets. Repeat to select multiple (`--bucket think --bucket claude_extract`). `total_time_s` is always included as the headline aggregate. |
| `--since 7d` / `24h` / `30m` | — (no bound) | Time-window filter: drops rows whose `finished_at` is older than `now - duration`. Combine with `--last` for a sliding window. |
| `--status STATUS` | `succeeded` | Filter by terminal status. Pass `--status ''` to include every terminal status (useful for failure-mode analysis). |
| `--json` | off | Emit a JSON object instead of the table. Field shape: `{workflow_id, run_count, percentiles: {bucket: {p50, p95, p99}}}`. |

### Percentile method

Linear interpolation between the closest ranks — matches numpy's default `linear` method. `p50` of 2 values is their midpoint; `p50` of an odd-length set is the middle element.

## When to use what

- **Live alerting** → Prometheus + Grafana over [`mantis_step_latency_seconds`](metrics.md). Real-time, low cardinality, no per-workflow drill-down.
- **One run's post-mortem** → `summary.wall_time_breakdown` + `summary.step_details[i].time_breakdown` on the `/v1/predict` `action=result` response. See [HTTP API / Wall-time breakdown](../api.md#wall-time-breakdown).
- **Cross-run analytics** → `mantis runs stats` over the JSONL log. Workflow-grouped, percentile-based, retained as long as you keep the files.

## When the JSONL outgrows Python

The current query path streams the JSONL with pure Python. It comfortably handles ~100k rows per month. If you need cross-month aggregation over millions of rows, swap the query path for DuckDB or a Postgres copy — the JSONL row schema is stable, so the read-side migration is a one-pass ETL. The writer never needs to change.

## See also

- [HTTP API / Wall-time breakdown](../api.md#wall-time-breakdown) — single-run schema.
- [Metrics](metrics.md) — Prometheus surface.
- [Cost model](cost.md) — the existing dollar twin of this seconds-side surface.
