# Authentication

Mantis uses **two layers of auth** when deployed behind a platform gateway (Baseten / GCLB / ALB):

| Header | Layer | Set by | Required |
|---|---|---|---|
| `Authorization: Api-Key <BASETEN_API_KEY>` | gateway | platform admin | Baseten only |
| `X-Mantis-Token: <tenant_token>` | container | operator (per tenant) | Always |

Self-hosted deployments (your own Docker / EKS / GKE without a gateway in front) only need `X-Mantis-Token`.

## Getting a tenant token

Tokens are issued by your operator. The flow:

1. **Operator** generates a token (`openssl rand -hex 32`) and adds an entry to the deployment's tenant keys file with your `tenant_id`, scopes, and caps. See [Tenant keys](../operations/tenant-keys.md) for the operator side.
2. **Operator** shares the token with you out-of-band (1Password / Vault / signed email — never in plaintext over Slack).
3. **You** keep it in your secrets store (AWS Secrets Manager, GCP Secret Manager, Vault, etc.) and pass it via `X-Mantis-Token` on every call.

Single-tenant deployments use the `MANTIS_API_TOKEN` env var on the server side; the same token works for everyone.

## Request shape

```http
POST /v1/predict HTTP/1.1
Host: model-qvvgkneq.api.baseten.co
Authorization: Api-Key bsk_live_…           ← platform (Baseten only)
X-Mantis-Token: 9f3e1b2a4c8d…                ← container, always
Content-Type: application/json

{ "detached": true, "micro": "plans/...", ... }
```

The same headers go on every endpoint:
- `POST /v1/predict`
- `POST /v1/chat/completions`
- `GET /v1/runs/{run_id}/video`

Open / un-auth'd endpoints (no tokens needed):
- `GET /health`, `GET /v1/health` — platform liveness probes
- `GET /v1/models` — model discovery
- `GET /metrics` — Prometheus scrape

## Scopes

Each tenant has a list of allowed scopes:

| Scope | What it lets you do |
|---|---|
| `run` | `POST /v1/predict` to start a new run |
| `status` | Poll an existing run for status |
| `result` | Fetch the final result of a completed run |
| `logs` | Fetch the live event log |

A read-only key (e.g., a dashboard scraping run history) can be issued with just `["status", "result"]` and will get **403** on `POST /v1/predict { ... new run ... }`.

## Error responses

| Status | Cause | What to do |
|---|---|---|
| **401** missing | No `X-Mantis-Token` header sent | Add it |
| **401** invalid | Token doesn't match any tenant key | Check the token wasn't truncated; ask operator to re-issue |
| **403** scope | Token valid but tenant lacks the required scope | Operator needs to add the scope to your tenant config |
| **429** rate | Tenant exceeded `rate_limit_per_minute` | Honor the `Retry-After` header |
| **429** concurrent | Tenant at `max_concurrent_runs` | Wait or have operator raise the cap |
| **503** auth-not-configured | Server has neither `MANTIS_API_TOKEN` nor `MANTIS_TENANT_KEYS_PATH` set | Operator misconfiguration |

## Token rotation

Tokens are hot-reloaded from the server's keys file with a 5-second cache. Rotation is operator-side:

1. Operator updates the JSON keys file (replaces your entry's key string).
2. Within 5 seconds, the new token works and the old one is rejected.
3. Update your secrets store with the new value.

No pod restart is needed.

## What `X-Mantis-Token` reveals on the server

Once the server validates your token, it resolves a `TenantConfig` with these fields. Caller-side you don't need to know the internals — but understanding what's enforced helps debug 403/429/etc:

```python
TenantConfig(
    tenant_id="tenant_a",
    scopes=("run", "status", "result", "logs"),
    max_concurrent_runs=3,
    max_cost_per_run=5.0,
    max_time_minutes_per_run=30,
    rate_limit_per_minute=60,
    anthropic_secret_name="anthropic_api_key_tenant_a",
    allowed_domains=("*.marketplace.example.com", "crm.example.com"),
    webhook_url="https://callbacks.example.com/mantis",
    webhook_secret_name="webhook_secret_tenant_a",
)
```

The server uses these to clamp your `max_cost` and `max_time_minutes` to *whichever is smaller* — its config or your request. So if you ask for `max_cost: 50` and your tenant cap is `5`, you get `5`.

## See also

- [Tenant keys](../operations/tenant-keys.md) — operator-side: provisioning + rotation
- [Errors](errors.md) — full error code reference
- [Reference / HTTP API](../api.md) — endpoint-level detail
