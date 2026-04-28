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

## See also

- [Client / Errors](../client/errors.md) — what each error outcome means
- [Rate limits](rate-limits.md) — what `kind=rate\|concurrent` rejection means
