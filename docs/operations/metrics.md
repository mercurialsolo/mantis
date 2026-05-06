# Metrics (Prometheus)

`GET /metrics` returns Prometheus text format. No auth required (standard scrape pattern); per-tenant labels are on every metric so isolation is preserved.

## Available metrics

| Metric | Type | Labels | Notes |
|---|---|---|---|
| `mantis_predict_requests_total` | counter | `tenant_id`, `mode`, `outcome` | mode = `run\|status\|result\|logs\|cancel`; outcome = `ok\|bad_request\|rate_limited\|denied_allowlist\|idempotent_hit\|error` |
| `mantis_chat_completions_total` | counter | `tenant_id`, `outcome` | outcome = `ok\|status_4xx\|status_5xx\|upstream_error` |
| `mantis_run_duration_seconds` | histogram | `tenant_id`, `model`, `status` | Buckets: 10, 30, 60, 120, 300, 600, 1200, 1800, 3600 |
| `mantis_run_cost_usd` | histogram | `tenant_id`, `model`, `status` | Buckets: $0.01, $0.05, $0.10, $0.25, $0.50, $1, $2.5, $5, $10, $25 |
| `mantis_concurrent_runs` | gauge | `tenant_id` | Currently in-flight runs |
| `mantis_rate_limit_rejections_total` | counter | `tenant_id`, `kind` | kind = `rate\|concurrent` |
| `mantis_action_total` | counter | `tenant_id`, `step_kind`, `outcome` | step_kind = MicroIntent.type (`navigate\|click\|paginate\|extract_url\|extract_data\|submit\|fill_field\|select_option\|scroll\|navigate_back`); outcome = `success\|failed\|duplicate\|filters_not_applied` |
| `mantis_brain_escalation_total` | counter | `tenant_id`, `from_brain`, `to_brain` | One increment per `BrainLadder.think()`. `to_brain` = `primary\|fallback` |
| `mantis_loop_termination_total` | counter | `tenant_id`, `reason` | One increment per run. reason = `completed\|halted\|cancelled\|paused\|budget_cap\|time_cap` |
| `mantis_plan_branch_total` | counter | `tenant_id`, `branch`, `outcome` | Special-case dispatch routes: `gate_verify\|claude_only\|navigate_back_close_tab\|click_listings\|click_single_element` × `taken\|skipped\|aborted` |
| `mantis_step_latency_seconds` | histogram | `tenant_id`, `phase` | phase = `perceive\|think\|act\|settle`. Buckets: 50ms, 100ms, 250ms, 500ms, 1s, 2s, 5s, 10s, 20s, 30s |

## Setup

The `prometheus-client` Python package is in the `[server]` and `[metrics]` extras. The Baseten Truss build step pip-installs it; for self-hosted Docker images make sure your `Dockerfile` includes it.

Without `prometheus-client`, `/metrics` returns 503 and all metric handles become no-ops — the rest of the API is unaffected.

## Scraping

=== "kube-prometheus / kubectl-prometheus"

    ```yaml
    apiVersion: monitoring.coreos.com/v1
    kind: ServiceMonitor
    metadata:
      name: mantis-holo3-server
    spec:
      selector:
        matchLabels: { app: mantis-holo3-server }
      endpoints:
        - port: http
          path: /metrics
          interval: 30s
    ```

=== "Plain Prometheus config"

    ```yaml
    scrape_configs:
      - job_name: mantis-holo3
        scrape_interval: 30s
        metrics_path: /metrics
        static_configs:
          - targets: ['mantis.example.com:80']
    ```

=== "Datadog Agent"

    ```yaml
    instances:
      - openmetrics_endpoint: http://mantis.example.com/metrics
        namespace: mantis
        metrics:
          - mantis_predict_requests_total
          - mantis_run_duration_seconds
          - mantis_run_cost_usd
          - mantis_concurrent_runs
          - mantis_rate_limit_rejections_total
    ```

## Alerts that pay rent

```yaml
- alert: MantisHighFailureRate
  expr: |
    sum by (tenant_id) (rate(mantis_predict_requests_total{outcome="error"}[10m]))
    / sum by (tenant_id) (rate(mantis_predict_requests_total{mode="run"}[10m])) > 0.1
  for: 15m
  annotations:
    summary: "Mantis tenant {{ $labels.tenant_id }} run failure rate >10%"

- alert: MantisRateLimitChronic
  expr: rate(mantis_rate_limit_rejections_total[10m]) > 0.5
  for: 30m
  annotations:
    summary: "Tenant {{ $labels.tenant_id }} hitting rate limits sustainedly — consider raising caps"

- alert: MantisCostBudgetSpike
  expr: |
    sum by (tenant_id) (
      increase(mantis_run_cost_usd_sum[1h])
    ) > 100
  annotations:
    summary: "Tenant {{ $labels.tenant_id }} burned >$100 in the last hour"

- alert: MantisStuckConcurrency
  expr: max_over_time(mantis_concurrent_runs[1h]) >= 5
  for: 30m
  annotations:
    summary: "Tenant {{ $labels.tenant_id }} stuck at concurrency cap — investigate stuck runs"
```

## Useful queries

```promql
# Top 10 tenants by run cost in the last hour
topk(10,
  sum by (tenant_id) (
    increase(mantis_run_cost_usd_sum[1h])
  )
)

# p95 run duration per tenant
histogram_quantile(0.95,
  sum by (tenant_id, le) (
    rate(mantis_run_duration_seconds_bucket[10m])
  )
)

# Inference passthrough success rate
sum(rate(mantis_chat_completions_total{outcome="ok"}[5m]))
  / sum(rate(mantis_chat_completions_total[5m]))
```

## Per-action triage queries (#156)

```promql
# Per-action success rate, last 15m, broken down by step_kind
sum by (step_kind) (rate(mantis_action_total{outcome="success"}[15m]))
  / sum by (step_kind) (rate(mantis_action_total[15m]))

# Top failing step types per tenant — surfaces tenant-specific regressions
topk(5,
  sum by (tenant_id, step_kind) (
    rate(mantis_action_total{outcome="failed"}[1h])
  )
)

# Brain escalation rate — what fraction of think() calls fall back?
sum(rate(mantis_brain_escalation_total{to_brain="fallback"}[10m]))
  / sum(rate(mantis_brain_escalation_total[10m]))

# Run-termination breakdown — how often do runs hit caps vs complete cleanly?
sum by (reason) (increase(mantis_loop_termination_total[1h]))

# Plan-branch hit rate — gate verify usage, listings vs single-element click split
sum by (branch, outcome) (rate(mantis_plan_branch_total[15m]))

# p95 step latency by phase — pinpoint where the time goes
histogram_quantile(0.95,
  sum by (phase, le) (rate(mantis_step_latency_seconds_bucket[10m]))
)
```

## See also

- [Client / Errors](../client/errors.md) — what each error outcome means
- [Rate limits](rate-limits.md) — what `kind=rate\|concurrent` rejection means
- [Per-action observability dashboard](per-action-dashboard.json) — Grafana JSON
